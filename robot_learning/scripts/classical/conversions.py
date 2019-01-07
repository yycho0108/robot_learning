import numpy as np
import cv2
from tf import transformations as tx

class Conversions(object):
    """
    Utilities class that deal with representations.
    """
    def __init__(self, K, D,
            T_c2b,
            det=None,
            des=None,
            match=None
            ):
        self.K_ = K
        self.Ki_ = np.linalg.inv(K)
        self.D_ = D
        self.T_c2b_ = T_c2b
        self.T_b2c_ = tx.inverse_matrix(T_c2b)
        if (det is None) and (des is None):
            # default detector+descriptor=orb
            orb = cv2.ORB_create(
                    nfeatures=4096,
                    scaleFactor=1.2,
                    nlevels=8,
                    scoreType=cv2.ORB_FAST_SCORE,
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

        # fields completed post-initialization with first image
        self.h_ = None
        self.w_ = None
        self.U_ = None 

    def initialize(self, shape):
        h, w = shape[:2]
        self.h_ = h
        self.w_ = w
        self.U_ = cv2.initUndistortRectifyMap(self.K_, self.D_,
                R=None, newCameraMatrix=self.K_,
                size=(w,h),
                m1type=cv2.CV_16SC2
                )

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

    def img_to_imgu(self, img):
        return cv2.remap(img, self.U_[0], self.U_[1],
                interpolation=cv2.INTER_LINEAR)

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

        #plt.figure()
        #plt.hist([e.response for e in kpt])
        #plt.show()

        return kpt

    def img_kpt_to_kpt_des(self, img, kpt):
        # i,k->k,d
        # -- extract descriptors --
        kpt, des = self.des_.compute(img, kpt)

        if des is None:
            return None

        kpt = np.asarray(kpt)
        des = np.asarray(des)
        return [kpt, des]

    def pt2_to_pt2u(self, pt2):
        pt2 = cv2.undistortPoints(pt2[None,...],
                self.K_,
                self.D_,
                P=self.K_)[0]
        return pt2

    def pt3_pose_to_pt2_msk(self, pt3, pose, distort=False):
        # pose = (x,y,h)
        # NOTE: pose is specified w.r.t base_link, not camera.
        D = self.D_
        if not distort:
            D = D * 0.0

        # soft fail for empty pt3
        if pt3.size <= 0:
            pt2 = np.empty((0,2), dtype=np.float32)
            msk = np.empty((0,), dtype=np.bool)
            return pt2, msk
        
        pt3_cam = self.map_to_cam(pt3, pose)

        pt2, _ = cv2.projectPoints(
                pt3_cam,
                np.zeros(3),
                np.zeros(3), # zeros, because conversions happened above
                cameraMatrix=self.K_,
                distCoeffs=D,
                # TODO : verify if it is appropriate to apply distortion
                )
        pt2 = np.squeeze(pt2, axis=1)

        # valid visibility mask
        lm_msk = np.logical_and.reduce([
            pt3_cam[:,2] > 1e-3, # z-positive
            0 <= pt2[:,0],
            pt2[:,0] < self.w_, 
            0 <= pt2[:,1],
            pt2[:,1] < self.h_
            ])
        return pt2, lm_msk

    def des_des_to_match(self, des1, des2,
            lowe=0.75,
            maxd=64.0,
            cross=True
            ):
        if cross:
            # check bidirectional
            i1_ab, i2_ab = self.des_des_to_match(des1, des2,
                    lowe, maxd, cross=False)
            i2_ba, i1_ba = self.des_des_to_match(des2, des1,
                    lowe, maxd, cross=False)
            m1 = np.stack([i1_ab, i2_ab], axis=-1)
            m2 = np.stack([i1_ba, i2_ba], axis=-1)
            m  = intersect2d(m1, m2)
            i1, i2 = m.T
            return i1, i2
        else:
            # TODO : support arbitrary matchers or something
            # currently only supports wrapper around FLANN
            if len(des1) <= 0 or len(des2) <= 0:
                return np.int32([]), np.int32([])
            match = self.match_(des1, des2) # cv2.DMatch

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

    @staticmethod
    def pt_to_pth(pt):
        # copied
        return np.pad(pt, [(0,0),(0,1)],
                mode='constant',
                constant_values=1.0
                )

    @staticmethod
    def pth_to_pt(pth):
        # copied
        return (pth[..., :-1] / pth[..., -1:])

    @staticmethod
    def pose_to_T(pose):
        x, y, h = pose
        # transform points from base_link to origin coordinate system.
        return tx.compose_matrix(
                angles=[0.0, 0.0, h],
                translate=[x, y, 0.0])

    def map_to_cam(self, pt, pose):
        # convert map-frame points to cam-frame points
        # NOTE: pose is specified w.r.t base_link, not camera.
        pt_h = self.pt_to_pth(pt)

        T_b2o = self.pose_to_T(pose)
        T_o2b = tx.inverse_matrix(T_b2o)

        pt_cam_h = np.linalg.multi_dot([
            pt_h,
            self.T_c2b_.T, # now base0 coordinates
            T_o2b.T, # now base1 coordinates
            self.T_b2c_.T # now cam1 coordinates
            ])
        pt_cam = self.pth_to_pt(pt_cam_h)
        return pt_cam

    def cam_to_map(self, pt, pose):
        # convert cam-frame points to map-frame points
        # NOTE: pose is specified w.r.t base_link, not camera.
        pt_h = self.pt_to_pth(pt)
        T_b2o = self.pose_to_T(pose)
        T_o2b = tx.inverse_matrix(T_b2o)

        pt_map_h = np.linalg.multi_dot([
            pt_h,
            self.T_c2b_.T,
            T_b2o.T,
            self.T_b2c_.T
            ])
        pt_map = self.pth_to_pt(pt_map_h)
        return pt_map

    def E_to_F(self, E):
        return np.linalg.multi_dot([
            self.Ki_.T, E, self.Ki_])

    def F_to_E(self, F):
        return np.linalg.multi_dot([
            self.K_.T, F, self.K_])

    def H_to_Rtn(self, H,
            eps=1e-5
            #eps=np.finfo(np.float32).eps
            ):
        """ from 
        https://github.com/raulmur/ORB_SLAM/
        vblob/ce199650a25653808f96b83557333bce3461d29f/
        src/Initializer.cc#L570
        """
        A = np.linalg.multi_dot([
            self.Ki_, H, self.K_])
        U, w, Vt = np.linalg.svd(A)
        V = Vt.T
        # A = U . diag(s) . V
        d1, d2, d3 = w
        if (d1/d2 < (1+eps) or d2/d3 < (1+eps)):
            return False
        s = np.linalg.det(U) * np.linalg.det(Vt)

        aux1 = np.sqrt( (d1*d1-d2*d2) / (d1*d1-d3*d3) )
        aux3 = np.sqrt( (d2*d2-d3*d3) / (d1*d1-d3*d3) )
        x1   = [aux1,aux1,-aux1,-aux1]
        x3   = [aux3,-aux3,aux3,-aux3]
        auxs = np.sqrt( (d1*d1-d2*d2)*(d2*d2-d3*d3)) / ((d1+d3)*d2)
        ct   = (d2*d2+d1*d3) / ((d1+d3)*d2)
        st   = [auxs,-auxs,-auxs,auxs]

        vR, vt, vn = [], [], []

        for i in range(4):
            # rot
            Rp = np.float32([
                ct,0,-st[i],
                0,1,0,
                st[i],0,ct]).reshape(3,3)
            R  = s * np.linalg.multi_dot([U,Rp,Vt])
            vR.append(R)

            # trans
            tp = np.float32([x1[i], 0, -x3[i]]) * (d1-d3)
            tp = tp.reshape(3,1)
            t  = U.dot(tp)
            t /= np.linalg.norm(t)
            vt.append(t)

            # norm
            n_p = np.float32([x1[i],0,x3[i]]).reshape(3,1)
            n  = V.dot(n_p)
            if n[2] < 0:
                n *= -1
            vn.append(n)

        aux_sp = np.sqrt( (d1*d1-d2*d2)*(d2*d2-d3*d3)) / ((d1-d3)*d2)
        cp     = (d1*d3-d2*d2) / ((d1-d3)*d2)
        sp     = [aux_sp, -aux_sp, -aux_sp, aux_sp]

        for i in range(4):
            # rot
            Rp = np.float32([
                cp, 0, sp[i],
                0, -1, 0,
                sp[i], 0, -cp]).reshape(3,3)
            R = s * np.linalg.multi_dot([U,Rp,Vt])
            vR.append(R)

            # trans
            tp = np.float32([x1[i], 0, x3[i]]) * (d1+d3)
            tp = tp.reshape(3,1)
            t  = U.dot(tp)
            t /= np.linalg.norm(t)
            vt.append(t)

            # norm
            n_p = np.float32([x1[i],0,x3[i]]).reshape(3,1)
            n   = V.dot(n_p)
            if n[2] < 0:
                n *= -1
            vn.append(n)

        return vR, vt, vn

    def __call__(self, ftype, *a, **k):
        return self.f_[ftype](*a, **k)
