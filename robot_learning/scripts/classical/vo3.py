from collections import namedtuple, deque
from filterpy.kalman import InformationFilter
from tf import transformations as tx
import cv2
import numpy as np

from vo_common import recover_pose, drawMatches

class Landmark(object):
    def __init__(self,
            pos,
            var,
            des,
            kpt,
            trk
            ):
        self.pos_ = pos
        self.var_ = var
        self.des_ = des
        self.kpt_ = kpt
        self.trk_ = trk

    def update(self):
        pass

class Conversions(object):
    """
    Utilities class that deal with representations.
    """
    def __init__(self, K, D,
            det=None,
            des=None,
            match=None
            ):
        self.K_ = K
        self.D_ = D

        if (det is None) and (des is None):
            # default detector+descriptor=orb
            orb = cv2.ORB_create(
                    nfeatures=2048
                    )
            det = orb
            des = orb

        self.det_ = det
        self.des_ = des
        # NOTE : des must be assigned prior to
        # self._build_matcher()
        if match is None:
            match = self._build_matcher()
            self.match_ = match

    def _build_matcher(self):
        # define un-exported enums from OpenCV
        FLANN_INDEX_KDTREE = 0
        FLANN_INDEX_LSH = 6

        # TODO : figure out what to set for
        # search_params
        search_params = dict(checks=50)
        # or pass empty dictionary

        # build flann matcher
        fn = None
        if isinstance(self.des_, cv2.ORB):
            # HAMMING
            index_params= dict(algorithm = FLANN_INDEX_LSH,
                       table_number = 6, # 12
                       key_size = 12,     # 20
                       multi_probe_level = 1) #2
            flann = cv2.FlannBasedMatcher(index_params,search_params)
            fn = lambda a,b : flann.knnMatch(np.uint8(a), np.uint8(b), k=2)
        else:
            index_params = dict(
                    algorithm = FLANN_INDEX_KDTREE,
                    trees = 5)
            flann = cv2.FlannBasedMatcher(index_params,search_params)
            fn = lambda a,b : flann.knnMatch(np.float32(a), np.float32(b), k=2)
        return fn

    @staticmethod
    def kpt_to_pt(kpt):
        # k->p
        return cv2.KeyPoint.convert(kpt)

    @staticmethod
    def pt_to_kpt(pt):
        # p->k
        raise NotImplementedError("Point To Keypoint not supported")

    def img_to_kpt(self, img, subpix=True):
        # i->k
        kpt = self.det_.detect(img)
        if subpix:
            # sub-pixel corner refinement
            crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.01)

            p_in = cv2.KeyPoint.convert(kpt)
            img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            spx_kpt = cv2.cornerSubPix(img_g, p_in, (3,3), (-1,-1),
                    criteria = crit)
            for k, pt in zip(kpt, spx_kpt):
                k.pt = tuple(pt)
        return kpt

    def img_kpt_to_kpt_des(self, img, kpt):
        # i,k->k,d
        # -- extract descriptors --
        kpt, des = self.des_.compute(img, kpt)

        if des is None:
            return None

        kpt = np.array(kpt)
        des = np.float32(des)
        return [kpt, des]

    def pt2_to_pt2u(self, pt2):
        pt2 = cv2.undistortPoints(pt2[None,...],
                self.K_,
                self.D_,
                P=self.K_)[0]
        return pt2

    def pt3_pose_to_pt2(self, pt3, pose):
        # pose = (x,y,h)
        x,y,h = pose

        tvec = [x,y,0]

        R = tx.euler_matrix(0, 0, h)
        rvec = cv2.Rodrigues(R)[0]

        pt2, _ = cv2.projectPoints(
                pt3,
                rvec, # or does it require inverse rvec/tvec?
                tvec,
                cameraMatrix=self.K_,
                distCoeffs=self.D_,
                )
        pt2 = np.squeeze(pt2, axis=1)
        return pt2

    def des_des_to_match(self, des1, des2,
            lowe=0.75,
            maxd=64.0
            ):
        # TODO : support arbitrary matchers or something
        # currently only supports wrapper around FLANN
        match = self.match_(des1, des2) # cv2.DMatch
        i1, i2 = np.int32([(m.queryIdx, m.trainIdx) for m in match]).T

        ## apply lowe + distance filter
        good = []
        for e in match:
            if not len(e) == 2:
                continue
            (m, n) = e
            # TODO : set threshold for lowe's filter
            # TODO : set reasonable maxd for GFTT, for instance.
            c_lowe = (m.distance <= lowe * n.distance)
            c_maxd = (m.distance <= maxd)
            if (c_lowe and c_maxd):
                good.append(m)

        # extract indices
        i1, i2 = np.int32([
            (m.queryIdx, m.trainIdx) for m in good
            ]).reshape(-1,2).T

        return i1, i2

    def pt_to_pth(self, pt):
        # copied
        return np.pad(pt, [(0,0),(0,1)],
                mode='constant',
                constant_values=1.0
                )

    def pth_to_pt(self, pth):
        # returns NOT copied
        return np.asarray(pt)[:,:2]

    def __call__(self, ftype, *a, **k):
        return self.f_[ftype](*a, **k)

class ClassicalVO(object):
    def __init__(self):
        # define constant parameters
        Ks = (1.0 / 2.0)
        self.K_ = np.reshape([
            499.114583 * Ks, 0.000000, 325.589216 * Ks,
            0.000000, 498.996093 * Ks, 238.001597 * Ks,
            0.000000, 0.000000, 1.000000], (3,3))
        self.D_ = np.float32([0.158661, -0.249478, -0.000564, 0.000157, 0.000000])

        # define "system" parameters
        self.pEM_ = dict(method=cv2.FM_RANSAC, prob=0.999, threshold=0.1)

        # conversions
        self.cvt_ = Conversions(self.K_, self.D_)

        # data cache + flags
        self.landmarks_ = []
        self.hist_ = deque(maxlen=100)

    def track(self, img1, img2, pt1, pt2=None):
        flags = cv2.OPTFLOW_LK_GET_MIN_EIGENVALS
        if pt2 is not None:
            flags += cv2.OPTFLOW_USE_INITIAL_FLOW

        lk_params = dict( winSize  = (51,13),
                maxLevel = 100,
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 0.001),
                flags = flags,
                minEigThreshold = 1e-2
                )
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        pt2, st, err = cv2.calcOpticalFlowPyrLK(
                img1_gray, img2_gray, pt1, pt2,
                **lk_params)
        #print 'err stats : ', err.min(), err.max(), err.std(), err.mean()

        h, w = np.shape(img2)[:2]

        # apply mask
        msk_in = np.all(np.logical_and(
                np.greater_equal(pt2, [0,0]),
                np.less(pt2, [w,h])), axis=-1)
        msk_st = st[:,0].astype(np.bool)
        msk = np.logical_and.reduce([msk_in, msk_st])#, msk_ef])

        return pt2, msk

    def __call__(self, img, pose, s=1.0):
        # suffix designations:
        # o/0 = origin (i=0)
        # p = previous (i=t-1)
        # c = current  (i=t)

        # process current frame
        img_c, pose_c = img, pose
        kpt_c = self.cvt_.img_to_kpt(img_c)
        kpt_c, des_c = self.cvt_.img_kpt_to_kpt_des(img_c, kpt_c)

        # update history
        self.hist_.append( (kpt_c, des_c, img_c, pose_c) )
        if len(self.hist_) <= 1:
            return True, None
        kpt_p, des_p, img_p, pose_p = self.hist_[-2] # query data from previous time-frame

        # frame-to-frame processing
        pt2_p = self.cvt_.kpt_to_pt(kpt_p)
        pt2_c, msk_t = self.track(img_p, img_c, pt2_p)
        print 'mean delta', np.mean(pt2_c - pt2_p, axis=0) # -14 px

        track_ratio = float(msk_t.sum()) / msk_t.size # logging/status
        print('track : {}/{}'.format(msk_t.sum(), msk_t.size))

        pt2_u_p = self.cvt_.pt2_to_pt2u(pt2_p[msk_t])
        pt2_u_c = self.cvt_.pt2_to_pt2u(pt2_c[msk_t])

        E, msk_e = cv2.findEssentialMat(pt2_u_c, pt2_u_p, self.K_,
                **self.pEM_)
        msk_e = msk_e[:,0].astype(np.bool)

        n_in, R, t, msk_r, _ = recover_pose(E, self.K_,
                pt2_u_c[msk_e], pt2_u_p[msk_e], log=False)

        print 'Rpart'
        print np.round(np.rad2deg(tx.euler_from_matrix(R)), 3)
        print 'Tpart'
        print np.round( t.ravel() , 3)

        msk = np.zeros(len(pt2_p), dtype=np.bool)
        msk[msk_t][msk_e][msk_r] = True

        mim = drawMatches(img_p, img_c, pt2_p, pt2_c, msk)

        # dh/dx in pose_p frame
        x, y, h = pose_p

        dh = -tx.euler_from_matrix(R)[1]
        dx = s * np.float32([ np.abs(t[2]), 0*-t[1] ])

        c, s = np.cos(h), np.sin(h)
        R2_p = np.reshape([c,-s,s,c], [2,2]) # [2,2,N]
        dp = R2_p.dot(dx)

        x_c = x + dp
        h_c = (h + dh + np.pi) % (2*np.pi) - np.pi

        return True, (mim, h_c, x_c, pt2_p, np.empty((0,3)), '')
