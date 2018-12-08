import numpy as np
import sys
import cv2
from scipy.optimize import linear_sum_assignment
from tf import transformations as tx
from collections import deque

class ClassicalVO(object):
    def __init__(self):
        self.hist_ = deque(maxlen=100)
        self.landmark_ = []

        # build detector
        self.gftt_ = cv2.GFTTDetector.create()
        self.brisk_ = cv2.BRISK_create()
        self.orb_ = cv2.ORB_create(nfeatures=4096, scaleFactor=1.2, WTA_K=2)

        # build flann matcher
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        self.flann_ = flann

        # camera matrix parameters
        # orig scale camera matrix (from calibration)
        self.K_ = np.reshape([
            499.114583, 0.000000, 325.589216,
            0.000000, 498.996093, 238.001597,
            0.000000, 0.000000, 1.000000], (3,3))
        # half scale camera matrix (data)
        self.K2_ = np.reshape([
            499.114583 / 2.0, 0.000000, 325.589216 / 2.0,
            0.000000, 498.996093 / 2.0, 238.001597 / 2.0,
            0.000000, 0.000000, 1.000000], (3,3))
        # distortion coefficient
        self.dC_ = np.float32([0.158661, -0.249478, -0.000564, 0.000157, 0.000000])

    def match_v2(self, img1, img2, kp1):
        lk_params = dict( winSize  = (47,3),
                maxLevel = 20,
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.001))
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # p0 = cv2.goodFeaturesToTrack(src_gray, mask = None, **feature_params)
        # calculate optical flow

        #print kp1[0], kp1[0].angle, kp1[0].octave, kp1[0].overlap, kp1[0].response, kp1[0].size
        p1 = np.float32([k.pt for k in kp1])
        p2, st, err = cv2.calcOpticalFlowPyrLK(
                img1_gray, img2_gray, p1,
                None, **lk_params)

        #kp1 = [cv2.KeyPoint(pt=p,size=3) for p in p1]
        #p1_ = cv2.KeyPoint_convert(kp1)#, size=3.0)
        print p2.shape
        kp2 = [cv2.KeyPoint(x=p[0],y=p[1],_size=3.0) for p in p2]#, size=3.0)
        kp2 = np.asarray(kp2, dtype=cv2.KeyPoint)

        h, w = img1.shape[:2]

        msk_in = np.all(np.logical_and(
                np.greater_equal(p1, [0,0]),
                np.less(p1, [w,h])), axis=-1)

        st = st[:,0].astype(np.bool)

        #print st.shape, msk_in.shape
        print msk_in.shape, msk_in.dtype
        print st.shape, st.dtype
        ef = np.isfinite(err[:,0])
        ec = np.less(err[:,0], 10.0)

        st = np.logical_and.reduce([msk_in, st, ef, ec])
        
        p1, p2 = np.float32(p1[st]), np.float32(p2[st])
        kp1, kp2 = kp1[st], kp2[st]
        matches = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=e)
                for i,e in enumerate(err[st])]

        n = len(p1)

        return matches, p1, p2, kp1, kp2

    def match_v1(self, des1, des2, thresh=150.0):
        # apply flann
        flann = self.flann_
        if len(des2) < 2:
            return None

        try:
            matches = flann.knnMatch(des1, des2, k=2)
        except Exception as e:
            print 'Exception during match : {}'.format(e)
            print np.shape(des1)
            print np.shape(des2)
            return None

        # lowe filter
        good = []
        for i,(m,n) in enumerate(matches):
            # TODO : set threshold for lowe's filter
            if m.distance < 0.9*n.distance:
                # and m.distance < thresh
                good.append(m)
        good = np.asarray(good, dtype=cv2.DMatch)
        return good

    def match(self, des1, des2, thresh=150.0):
        # (inefficient & slow right now)
        # hungarian algorithm based matching
        nax = np.newaxis
        c = np.linalg.norm(des1[:,nax] - des2[nax,:], axis=-1)

        # cost mask with outliers removed
        i_good = np.where(np.min(c, axis=1) < 150.0)[0]
        j_good = np.where(np.min(c, axis=0) < 150.0)[0]

        c = c[i_good[:,nax], j_good]

        mi, mj = linear_sum_assignment(c)

        # matching cost
        c      = c[mi,mj]

        # real match index
        mi, mj = i_good[mi], j_good[mj]

        matches = [cv2.DMatch(
            _queryIdx = e1,
            _trainIdx = e2,
            _distance = e3)
            for (e1,e2,e3) in zip(mi, mj, c)]
        return matches
        #return mi, mj


        ## homography
        #M, mask = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
        #mask = np.cast(mask, np.bool)
        #return good[mask]

    def detect(self, img):
        """ detect feature points """

        kp = self.gftt_.detect(img)
        #kp = self.orb_.detect(img)

        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.01)

        # -- refine keypoint corners --
        p_in = cv2.KeyPoint.convert(kp)
        img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp_spt = cv2.cornerSubPix(img_g, p_in, (3,3), (-1,-1),
                criteria = crit)
        for k, pt in zip(kp, kp_spt):
            k.pt = tuple(pt)

        # -- extract descriptors --
        kp2, des2 = self.brisk_.compute(img, kp)
        #kp2, des2 = self.orb_.compute(img, kp)

        if des2 is None:
            return None

        kp2 = np.array(kp2)
        des2 = np.float32(des2)
        img2 = img

        return (kp2, des2, img2)

    def __call__(self, img,
            in_thresh = 16,
            s = 0.1
            ):

        # == 0 :  unroll parameters ==
        Kmat_o = self.K_
        Kmat = self.K2_
        distCoeffs = self.dC_
        # ============================

        # detect features + query/update history
        kp2, des2, img2 = self.detect(img)
        if len(self.hist_) <= 0:
            self.hist_.append( (kp2, des2, img2) )
            return True, None
        kp1, des1, img1 = self.hist_[-1]
        self.hist_.append((kp2, des2, img2))
        # ==============

        # == 3 : match ==
        #matches = self.match(des1, des2)

        # v1 proc
        matches = self.match_v1(des1, des2)
        #matches = sorted(matches, key=lambda e:e.distance)
        if (matches is None) or (len(matches) <= 8):
            # no matches or insufficient # of matches
            return True, None
        # extract points
        i1, i2 = np.int32([(m.queryIdx, m.trainIdx) for m in matches]).T
        p1 = np.float32(cv2.KeyPoint.convert(kp1[i1]))
        p2 = np.float32(cv2.KeyPoint.convert(kp2[i2]))
        midx = np.arange(len(matches))
        # -----------
        # v2 proc
        #matches, p1, p2, kp1, kp2 = self.match_v2(img1, img2, kp1)
        #if (matches is None) or (len(matches) <= 8):
        #    # no matches or insufficient # of matches
        #    return True, None
        #i1 = np.arange(len(p1))
        #i2 = np.arange(len(p1))
        #midx = np.arange(len(matches))
        # ============

        # TODO : expose these parameters

        # == 4 : undistort + correct matches ==

        # undistort
        p1 = cv2.undistortPoints(2.0*p1[None,...], Kmat_o, distCoeffs, P=Kmat_o)[0]/2.0
        p2 = cv2.undistortPoints(2.0*p2[None,...], Kmat_o, distCoeffs, P=Kmat_o)[0]/2.0

        # correct matches
        Fmat, mask = cv2.findFundamentalMat(p2, p1, method=cv2.FM_LMEDS,
                param1=0.1, param2=0.999)
        mask = np.asarray(mask[:,0]).astype(np.bool)

        # filter p1 + bookkeeping mask
        p1 = p1[None, mask]
        p2 = p2[None, mask]
        midx = midx[np.where(mask)[0]]

        p2, p1 = cv2.correctMatches(Fmat, p2, p1)
        p1 = p1[0, ...]
        p2 = p2[0, ...]

        # filter NaN
        msk = np.logical_and(np.isfinite(p1), np.isfinite(p2))
        msk = np.all(msk, axis=-1)

        # filter p1 + bookkeeping mask
        p1 = p1[msk].astype(np.float32)
        p2 = p2[msk].astype(np.float32)
        midx = midx[np.where(msk)[0]]

        if len(p1) <= 8:
            # ?? TODO : sideways translation??
            return True, None

        #Emat0 = (Kmat.T).dot(Fmat).dot(Kmat)
        #print "emat", Emat0
        Emat, mask = cv2.findEssentialMat(p2, p1, Kmat,
                method=cv2.FM_LMEDS, prob=0.999, threshold=0.1)
        #Ki = np.linalg.inv(Kmat)
        #Fmat = Ki.T.dot(Emat).dot(Ki)
        #print 'emat2', Emat

        #print 'EE', Emat, (Kmat.T).dot(Fmat).dot(Kmat)
        #cmat = np.float32([focal,0,pp[0], 0, focal, pp[1], 0,0,1]).reshape(3,3)
        n_in, R, t, msk, _ = cv2.recoverPose(Emat,
                np.float32(p2),
                np.float32(p1),
                cameraMatrix=Kmat,
                distanceThresh=100.0) # TODO : or something like 10.0/s ??

        # TODO : do the correct scale estimation

        # validate triangulation
        pts_h = cv2.triangulatePoints(
                Kmat.dot(np.eye(3,4)),
                Kmat.dot(np.concatenate([R, t], axis=1)),
                p2[None,...],
                p1[None,...]).astype(np.float32)

        # apply mask
        msk = np.logical_and(msk, np.all(np.isfinite(pts_h),axis=0)[:,None])
        midx = midx[np.where(msk)[0]]

        pts_h = pts_h[:, msk[:,0]]
        # homogeneous --> 3d
        pts3 = pts_h[:3] / pts_h[3:]

        # compute reprojection
        pts2_rec, _ = cv2.projectPoints(
                pts3.T[...,None],
                rvec=cv2.Rodrigues(R)[0],
                tvec=t.ravel(),
                #rvec=np.zeros(3),
                #tvec=np.zeros(3),
                cameraMatrix=Kmat,
                distCoeffs=distCoeffs,
                )
        pts2_rec = np.squeeze(pts2_rec, axis=1)

        # points: homogeneous --> 3d coordinates

        # apply scale factor and re-orient to align with base coord
        pts3 = s * np.stack([pts3[2], -pts3[0], -pts3[1]], axis=-1)

        #pts3 = linear_LS_triangulation(
        #        Kmat.dot(np.eye(3,4)),
        #        Kmat.dot(np.concatenate([R, t], axis=1)),
        #        p2, p1)
        #pts3 = s * np.stack([pts3[:,2], -pts3[:,0], -pts3[:,1]], axis=-1)

        # opt2 : custom
        #pts3 = triangulatePoints(
        #        Kmat.dot(np.eye(3,4)),
        #        Kmat.dot(np.concatenate([R, t], axis=1)),
        #        p2, p1)
        #pts3 = pts3[msk[:,0]>0]
        #pts3 = s * np.stack([pts3[:,2], -pts3[:,0], -pts3[:,1]], axis=-1)

        # TODO : will this take care of scale?
        # convert to base_link coordinates

        #pts3 = pts3[:16] # select 16 points to draw
        #msk[np.where(msk)[0][16:]] = 0

        #pts2 = pts3[:, :2]
        # filter by large displacement
        #pts2 = pts2[np.linalg.norm(pts2, axis=-1) < 10.0] #filter by radius=10.0m
        #pts2 = pts2[np.sign(pts2[:,0]) == 1]

        if n_in < in_thresh:
            # insufficient number of inliers to recover pose
            return True, None

        h = tx.euler_from_matrix(R)
        #h = np.round(np.rad2deg(tx.euler_from_matrix(R)), 2)
        #t = np.round(t, 2)
        #print ('h-z', -h[1])
        #print ('t-xy', t[2,0], -t[0,0])

        h = -h[1]
        #print('dh', np.round(np.rad2deg(h), 2))

        # no-slip
        # TODO : why is t[0,0] so bad???
        # TODO : relax the no-slip constraint by being better at triangulating or something
        t = np.asarray([t[2,0], 0.0 * -t[0,0]])

        # magnitude of t is always 1.0! can we do anything about this?
        # something about overall scale
        t *= s

        matchesMask = np.zeros(len(matches), dtype=np.bool)
        matchesMask[midx] = 1

        draw_params = dict(
                matchColor = (0,255,0),
                singlePointColor = (255,0,0),
                flags = 0,
                matchesMask=matchesMask.ravel().tolist()
                )
        mim = cv2.drawMatches(
                img1,kp1,img2,kp2,
                matches,None,**draw_params)
        mim = cv2.addWeighted(np.concatenate([img1,img2],axis=1), 0.5, mim, 0.5, 0.0)
        cv2.drawKeypoints(mim, kp1[i1][matchesMask], mim, color=(0,0,255))
        cv2.drawKeypoints(mim[:,320:], kp2[i2][matchesMask], mim[:,320:], color=(0,0,255))
        print('---')

        return True, (mim, h, t, pts2_rec, pts3)

        #print 'm', m
