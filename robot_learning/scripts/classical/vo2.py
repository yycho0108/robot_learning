import numpy as np
import sys
import cv2
from scipy.optimize import linear_sum_assignment
from tf import transformations as tx
from collections import deque

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

def drawMatches(img1, img2, pt1, pt2, msk,
        radius = 3
        ):
    h,w = np.shape(img1)[:2]
    pt1 = np.round(pt1).astype(np.int32)
    pt2 = np.round(pt2 + [[w,0]]).astype(np.int32)

    mim = np.concatenate([img1, img2], axis=1)
    mim0 = mim.copy()

    for (p1, p2) in zip(pt1[msk], pt2[msk]):
        p1 = tuple(p1)
        p2 = tuple(p2)
        col = tuple(np.random.randint(255, size=4))
        cv2.line(mim, p1, p2, col, 2)
        cv2.circle(mim, tuple(p1), radius, col, 2)
        cv2.circle(mim, tuple(p2), radius, col, 2)

    for p in pt1[~msk]:
        cv2.circle(mim, tuple(p), radius, (255,0,0), 1)

    for p in pt2[~msk]:
        cv2.circle(mim, tuple(p), radius, (255,0,0), 1)

    mim = cv2.addWeighted(mim0, 0.5, mim, 0.5, 0.0)

    return mim

class ClassicalVO(object):
    def __init__(self):
        # params
        self.method_ = cv2.FM_RANSAC

        # global data cache
        self.hist_ = deque(maxlen=100)
        self.lm_pt2_ = np.empty(shape=(0,2), dtype=np.float32)
        self.lm_pt3_ = np.empty(shape=(0,3), dtype=np.float32)
        self.lm_msk_ = None # previous landmark keypoints mask
        self.lm_des_ = None # landmark descriptors

        # build detector + descriptor
        #self.det_ = cv2.GFTTDetector.create(
        #        maxCorners=4096,
        #        qualityLevel=0.01,
        #        minDistance=1.0,
        #        blockSize=3,
        #        #useHarrisDetector=True,
        #        #k=0.04
        #        ) # keypoints detector
        #self.des_ = cv2.BRISK_create() # descriptor
        orb = cv2.ORB_create(
                nfeatures=1024,
                #scaleFactor=1.1,
                #edgeThreshold=15,
                #patchSize=15,
                #WTA_K=4, # does WTA_K matter much?
                )
        self.det_ = orb
        self.des_ = orb

        # build flann matcher
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        self.flann_ = flann

        # camera matrix parameters
        # orig scale camera matrix (from calibration)
        K_full = np.reshape([
            499.114583, 0.000000, 325.589216,
            0.000000, 498.996093, 238.001597,
            0.000000, 0.000000, 1.000000], (3,3))
        # half scale camera matrix (data)
        K_half = np.reshape([
            499.114583 / 2.0, 0.000000, 325.589216 / 2.0,
            0.000000, 498.996093 / 2.0, 238.001597 / 2.0,
            0.000000, 0.000000, 1.000000], (3,3))
        self.K_ = K_full # select K based on incoming data.
        # distortion coefficient
        self.dC_ = np.float32([0.158661, -0.249478, -0.000564, 0.000157, 0.000000])

        # base_link<->camera transforms cache
        # TODO : extract transforms from input or URDF
        self.T_c2b_ = tx.compose_matrix(
                angles=[-np.pi/2,0.0,-np.pi/2],
                translate=[0.15,0,0.1]) # camera frame to base_link frame
        self.T_b2c_ = tx.inverse_matrix(self.T_c2b_) # base_link frame to camera frame

    def track(self, img1, img2, pt1):
        """
        Track points from source img to destination img.

        Arguments:
            img1(np.ndarray): Source Image
            img2(np.ndarray): Target Image
            pt1(np.ndarray):  Points to track, [N,2]

        Returns:
            pt2(np.ndarray): Tracked Points in Target Image, [N,2]
            msk(np.ndarray): Boolean mask array of valid track points.
        """
        lk_params = dict( winSize  = (51,13),
                maxLevel = 100,
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 0.001),
                flags = cv2.OPTFLOW_LK_GET_MIN_EIGENVALS,
                minEigThreshold = 1e-4
                )
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        pt2, st, err = cv2.calcOpticalFlowPyrLK(
                img1_gray, img2_gray, pt1,
                None, **lk_params)
        #print pt2[:5]
        #print 'std', (pt2 - pt1)[:,1].std()
        print 'err stats : ', err.min(), err.max(), err.std(), err.mean()

        h, w = np.shape(img2)[:2]

        # apply mask
        msk_in = np.all(np.logical_and(
                np.greater_equal(pt2, [0,0]),
                np.less(pt2, [w,h])), axis=-1)
        msk_st = st[:,0].astype(np.bool)
        #msk_ef = np.isfinite(err[:,0])
        msk_ef = np.less(err[:,0], 0.1)
        #msk_hu = np.less(np.abs(pt2[:,1]-pt1[:,1]), 5.0) # heuristic
        #msk_ef = np.less(err[:,0], 10.0)
        msk = np.logical_and.reduce([msk_in, msk_st, msk_ef])

        return pt2, msk

    def match(self, des1, des2, thresh=150.0):
        """
        Compute plausible match between two descriptors.

        Arguments:
            des1(np.ndarray): Source Descriptor
            des2(np.ndarray): Target Descriptor
            thresh(float):    Match distance threshold (TODO: currently unused)

        Returns:
            good(np.ndarray): List of cv2.DMatch objects,
                which contain correspondence information from des1 to des2.
        """
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
        for i, (m,n) in enumerate(matches):
            # TODO : set threshold for lowe's filter
            if m.distance < 0.75*n.distance:
                # and m.distance < thresh
                good.append(m)
        good = np.asarray(good, dtype=cv2.DMatch)

        return good

    def detect(self, img):
        """
        Detect feature points from Image.
        
        """

        # partition keypoints
        kp = []
        h,w = np.shape(img)[:2]
        n_p = 1 # number of partitions
        i_p = np.linspace(0, h, n_p+1).astype(np.int32)
        j_p = np.linspace(0, w, n_p+1).astype(np.int32)
        for (i0, i1) in zip(i_p[:-1], i_p[1:]):
            for (j0, j1) in zip(j_p[:-1], j_p[1:]):
                kp_p = self.det_.detect(img[i0:i1,j0:j1])
                for k in kp_p:
                    k.pt = tuple([k.pt[0]+j0, k.pt[1]+i0])
                kp.extend(kp_p)
        #kp = self.det_.detect(img)

        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.01)

        # -- refine keypoint corners --
        p_in = cv2.KeyPoint.convert(kp)
        #plt.plot(p_in[:,0], p_in[:,1], '+')
        #plt.show()
        img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp_spt = cv2.cornerSubPix(img_g, p_in, (3,3), (-1,-1),
                criteria = crit)
        for k, pt in zip(kp, kp_spt):
            k.pt = tuple(pt)

        # -- extract descriptors --
        kp, des = self.des_.compute(img, kp)

        if des is None:
            return None

        kp = np.array(kp)
        des = np.float32(des)
        return [kp, des]

    def getPoseAndPoints(self, pt1, pt2, midx, method):
        # == 0 :  unroll parameters ==
        Kmat = self.K_
        distCoeffs = self.dC_

        # rectify input
        pt1 = np.float32(pt1)
        pt2 = np.float32(pt2)

        # find fundamental matrix
        Fmat, msk = cv2.findFundamentalMat(pt1, pt2,
                method=method,
                param1=0.1, param2=0.999) # TODO : expose these thresholds
        msk = np.asarray(msk[:,0]).astype(np.bool)

        # filter points + bookkeeping mask
        pt1 = pt1[msk]
        pt2 = pt2[msk]
        midx = midx[msk]

        # correct matches
        pt1 = pt1[None, ...] # add axis 0
        pt2 = pt2[None, ...]
        pt2, pt1 = cv2.correctMatches(Fmat, pt1, pt2) # TODO : not sure if this is necessary
        pt1 = pt1[0, ...] # remove axis 0
        pt2 = pt2[0, ...]

        # filter NaN
        msk = np.logical_and(np.isfinite(pt1), np.isfinite(pt2))
        msk = np.all(msk, axis=-1)

        # filter points + bookkeeping mask
        pt1 = pt1[msk]
        pt2 = pt2[msk]
        midx = midx[msk]

        if len(pt1) <= 8:
            # Insufficient # of points
            # TODO : expose this threshold
            return None

        # TODO : expose these thresholds
        Emat, msk = cv2.findEssentialMat(pt1, pt2, Kmat,
                method=method, prob=0.999, threshold=0.1)
        msk = np.asarray(msk[:,0]).astype(np.bool)

        # filter points + bookkeeping mask
        pt1 = pt1[msk]
        pt2 = pt2[msk]
        midx = midx[msk]

        n_in, R, t, msk, _ = cv2.recoverPose(Emat,
                pt1,
                pt2,
                cameraMatrix=Kmat,
                distanceThresh=1000.0)#np.inf) # TODO : or something like 10.0/s ??
        msk = np.asarray(msk[:,0]).astype(np.bool)

        # filter points + bookkeeping mask
        pt1 = pt1[msk]
        pt2 = pt2[msk]
        midx = midx[msk]

        # validate triangulation
        pts_h = cv2.triangulatePoints(
                Kmat.dot(np.eye(3,4)),
                Kmat.dot(np.concatenate([R, t], axis=1)),
                pt1[None,...],
                pt2[None,...]).astype(np.float32)

        # PnP Validation
        #pts3 = (pts_h[:3] / pts_h[3:]).T
        #_, rvec, tvec, inliers = res = cv2.solvePnPRansac(
        #        pts3, pt2, self.K2_, self.dC_,
        #        useExtrinsicGuess=False,
        #        iterationsCount=1000,
        #        reprojectionError=2.0
        #        )
        #print 'PnP Validation'
        #print tx.euler_from_matrix(R), t
        #print rvec, tvec

        # Apply NaN/Inf Check
        msk = np.all(np.isfinite(pts_h),axis=0)
        pt1 = pt1[msk]
        pt2 = pt2[msk]
        midx = midx[msk]
        pts_h = pts_h[:, msk]

        return [R, t, pt1, pt2, midx, pts_h]

    def undistort(self, pt):
        #pt_u = cv2.undistortPoints(2.0*pt[None,...], self.K2_, self.dC_, P=self.K2_)[0]/2.0
        pt_u = cv2.undistortPoints(pt[None,...], self.K_, self.dC_, P=self.K_)[0]
        return pt_u

    #def add_keyframe(self,
    #        img1, img2,
    #        des1, des2,
    #        kpt1, kpt2
    #        ):
    #    # == 0 :  unroll parameters ==
    #    Kmat = self.K_
    #    distCoeffs = self.dC_
    #    method = self.method_ # TODO : configure
    #    # ============================

    #    # match descriptors
    #    matches = self.match_v1(des1, des2)
    #    i1, i2 = np.int32([(m.queryIdx, m.trainIdx) for m in matches]).T

    #    # grab relevant keypoints + points
    #    kpt1_n = kpt1[i1]
    #    kpt2_n = kpt2[i2]
    #    pt1 = np.float32(cv2.KeyPoint.convert(kpt1_n))
    #    pt2 = np.float32(cv2.KeyPoint.convert(kpt2_n))
    #    midx = np.arange(len(matches)) # track match indices

    #    # undistort
    #    pt1_u = self.undistort(pt1)
    #    pt2_u = self.undistort(pt2)
    #    
    #    res = self.getPoseAndPoints(pt1_u, pt2_u, midx, method)
    #    if res is None:
    #        return None

    #    # unroll result
    #    [R, t, pt1_r, pt2_r, midx, pts_h] = res
    #    t = t.ravel()

    #    # homogeneous --> 3d
    #    pts3 = pts_h[:3] / pts_h[3:]

    #    # format results
    #    lm_pt2 = pt2[midx] # 2d Landmark Image Coordinates
    #    lm_des = des2[i2[midx]] # Landmark Feature Descriptor
    #    lm_pt3 = pts3.T.copy() # 3D Landmark Coordinates
    #    lm_msk = np.zeros(len(kpt2), dtype=np.bool)
    #    lm_msk[i2[midx]] = True # P@(t=2) Keypoint Mask

    #    pp_res = [R, t, midx, matches]
    #    lm_res = [lm_pt2, lm_des, lm_pt3, lm_msk]

    #    return pp_res, lm_res

    def T_b2o(self, pose):
        b_x, b_y, b_h = pose
        # transform points from base_link to origin coordinate system.
        return tx.compose_matrix(
                angles=[0.0, 0.0, b_h],
                translate=[b_x, b_y, 0.0])

    def invTC0(self, pose):
        """
        Inverse camera transform.

        Computes the transformation matrix that would
        take points in current camera frame and
        convert them to the coordinate frame of the camera origin.
        """
        b_x, b_y, b_h = pose
        T_o2b = tx.compose_matrix(
                angles=[0.0, 0.0, b_h],
                translate=[b_x, b_y, 0.0]) # origin -> base_link
        T_b2o = np.linalg.inv(T_o2b) # base_link -> origin
        T_b2c = self.T_b2c_
        T_c2b = self.T_c2b_
        T_c = T_b2c.dot(T_b2o).dot(T_c2b)

        return T_c

    def __call__(self, img, pose,
            in_thresh = 16,
            s = 0.1
            ):
        # == 0 :  unroll parameters ==
        Kmat = self.K_
        distCoeffs = self.dC_
        method = self.method_ # TODO : configure
        # ============================

        # detect features + query/update history
        img2 = img
        kpt2, des2 = self.detect(img2)
        if len(self.hist_) <= 0:
            self.lm_msk_ = np.zeros(len(kpt2), dtype=np.bool)
            self.hist_.append( (kpt2, des2, img2, pose) )
            return True, None
        kpt1, des1, img1, pose1 = self.hist_[-1]
        self.hist_.append((kpt2, des2, img2, pose))
        # ==============

        reinit_thresh = 100

        # TODO : potentially incorporate descriptors match information at some point.
        if len(self.lm_pt2_) >= reinit_thresh:
            # has sufficient # of landmarks to track
            # TODO : configure number of required landmark points
            pt1 = self.lm_pt2_ # use most recent tracking points
            pt2, msk = self.track(img1, img2, pt1)

            track_ratio = float(msk.sum()) / msk.size
            print 'track status : {}%'.format(track_ratio * 100)

            if track_ratio > 0.6:
                self.lm_pt3_ = self.lm_pt3_[msk]
                self.lm_des_ = self.lm_des_[msk]
                self.lm_pt2_ = pt2[msk] # update tracking point

                ## # update landmark positions
                #res = self.getPoseAndPoints(
                #        self.undistort(pt2[msk]),
                #        self.undistort(pt1[msk]),
                #        np.arange(msk.sum()), method)
                #R, t, _, _, midx, pt3_h = res

                ## pt3_h w.r.t pose

                #lm_pt3_c0 = self.lm_pt3_[midx] # landmark w.r.t c0
                ## to estimate scale, c0 -> c1 required
                #lm_pt3_c0_h = cv2.convertPointsToHomogeneous(lm_pt3_c0)
                #lm_pt3_c0_h = np.squeeze(lm_pt3_c0_h, axis=1)
                #lm_pt3_c1 = np.linalg.multi_dot([
                #    lm_pt3_c0_h,
                #    self.T_c2b_.T, # now base0 coordinates
                #    tx.inverse_matrix(self.T_b2o(pose)).T, # now base1 coordinates
                #    self.T_b2c_.T # now cam1 coordinates
                #    ])

                #prv = cv2.convertPointsFromHomogeneous(lm_pt3_c1)
                #prv = np.squeeze(prv, axis=1)
                #cur = cv2.convertPointsFromHomogeneous(pt3_h.T)
                #cur = np.squeeze(cur, axis=1)

                #d_prv = np.linalg.norm(prv, axis=-1)
                #d_cur = np.linalg.norm(cur, axis=-1)

                ## override input scale information
                #ss = d_prv / d_cur
                ##print('scale estimates : {}'.format(ss))
                #print('input scale : {}'.format(s))
                #s = np.median(ss)
                #print('computed scale : {}'.format(s))
                #cur *= s

                #cur_h = cv2.convertPointsToHomogeneous(cur)
                #cur_h = np.squeeze(cur_h, axis=1)

                #lm_pt3 = np.linalg.multi_dot([
                #    cur_h,
                #    self.T_c2b_.T, # now base1 coordinates
                #    self.T_b2o(pose).T, # now base0 coordinates
                #    self.T_b2c_.T # now cam0 coordinates
                #])[:,:3]

                #print 'update landmark position'
                #alpha = 0.95
                #self.lm_pt3_[midx] = (
                #        alpha * self.lm_pt3_[midx] + (1.0-alpha) * lm_pt3)

            else:
                # force landmarks re-initialization
                print('Force Reinitialization')
                self.lm_pt2_ = np.empty((0,2), dtype=np.float32)

        if len(self.lm_pt2_) < reinit_thresh:

            print('re-initializing landmarks')
            # requires 3D landmarks initialization

            # extract points from keypoints
            pt1 = cv2.KeyPoint.convert(kpt1)
            #pt2 = cv2.KeyPoint.convert(kpt2)

            # requires track landmarks initialization
            pt2, msk = self.track(img1, img2, pt1)
            track_ratio = float(msk.sum()) / msk.size
            print 'track status : {}%'.format(track_ratio * 100)

            if track_ratio < 0.8:
                # track failed - attempt match
                print('attempting match ...')
                matches = self.match(des1, des2)

                i1, i2 = np.int32([(m.queryIdx, m.trainIdx) for m in matches]).T

                # grab relevant keypoints + points
                pt1 = np.float32(cv2.KeyPoint.convert(kpt1[i1]))
                pt2 = np.float32(cv2.KeyPoint.convert(kpt2[i2]))
                print('match result : {}/{}'.format(len(matches), len(kpt1)))
                des1 = des1[i1]
                des2 = des2[i2]
                msk = np.ones(len(pt1), dtype=np.bool)

            # apply mask prior to triangulation
            pt1_l, pt2_l = pt1[msk], pt2[msk]
            pt1_u, pt2_u = self.undistort(pt1_l), self.undistort(pt2_l)
            midx = np.arange(len(pt1_l))

            # landmarks triangulation
            res = self.getPoseAndPoints(pt1_u, pt2_u, midx, method)
            R, t, _, _, midx, pt3_h = res

            lm_pt4 = (pt3_h / pt3_h[3:]).T

            if self.lm_des_ is not None:
                # estimate scale from prior landmark matches
                match_lm = self.match(self.lm_des_, des1[msk][midx])
                print('prv landmarks match : {}'.format( len(match_lm) ))
                if len(match_lm) > 10: 

                    i1, i2 = np.int32([(m.queryIdx, m.trainIdx) for m in match_lm]).T
                    # sufficient landmark matches are required to apply scale

                    lm_pt3_c0 = self.lm_pt3_[i1] # landmark w.r.t c0
                    # to estimate scale, c0 -> c1 required
                    lm_pt3_c0_h = cv2.convertPointsToHomogeneous(lm_pt3_c0)
                    lm_pt3_c0_h = np.squeeze(lm_pt3_c0_h, axis=1)
                    lm_pt3_c1 = np.linalg.multi_dot([
                        lm_pt3_c0_h,
                        self.T_c2b_.T, # now base0 coordinates
                        tx.inverse_matrix(self.T_b2o(pose1)).T, # now base1 coordinates
                        self.T_b2c_.T # now cam1 coordinates
                        ])

                    prv = lm_pt3_c1[:,:3] # previous landmark locations
                    cur = lm_pt4[i2,:3] # persistent identified landmark locations

                    d_prv = np.linalg.norm(prv, axis=-1)
                    d_cur = np.linalg.norm(cur, axis=-1)

                    # override input scale information
                    ss = d_prv / d_cur
                    #print('scale estimates : {}'.format(ss))
                    print('input scale : {}'.format(s))
                    s = np.median(ss)
                    print('computed scale : {}'.format(s))
                    lm_pt4[:,:3] *= s
                else:
                    print 'using input scale due to failure : {}'.format(s)
                    lm_pt4[:,:3] *= s
            else:
                # apply input scale (probably only called on initialization)
                print 'input scale : {}'.format(s)
                #s = 0.02
                lm_pt4[:,:3] *= s # HERE is where scale is applied.

            # lm_pt3 currently w.r.t cam1 coordinates
            lm_pt3 = np.linalg.multi_dot([
                lm_pt4,
                self.T_c2b_.T, # now base1 coordinates
                self.T_b2o(pose1).T, # now base0 coordinates
                self.T_b2c_.T # now cam0 coordinates
                ])[:,:3]

            self.lm_pt3_ = lm_pt3
            self.lm_pt2_ = pt2_l[midx] # important: use undistorted version (for tracking)
            # most recent points observation
            self.lm_des_ = des1[msk][midx]

        # construct extrinsic guess
        T_b2b0_est = tx.compose_matrix(
                translate=[ pose[0], pose[1], 0 ],
                angles=[0, 0, pose[2]]
                )
        T_c0c2_est = np.linalg.multi_dot([
            self.T_b2c_,
            tx.inverse_matrix(T_b2b0_est), # T_b0b2
            self.T_c2b_])

        rvec0 = cv2.Rodrigues(T_c0c2_est[:3,:3])[0]
        tvec0 = T_c0c2_est[:3, 3:]

        # compute cam2 pose
        res = cv2.solvePnPRansac(
            self.lm_pt3_, self.lm_pt2_, self.K_, self.dC_,
            useExtrinsicGuess=False,
            #useExtrinsicGuess=True,
            #rvec = rvec0,
            #tvec = tvec0,
            iterationsCount=1000,
            reprojectionError=2.0, # TODO : tune these params
            confidence=0.9999
            )
        dbg, rvec, tvec, inliers = res
        print 'inliers : {}/{}'.format( len(inliers), len(self.lm_pt2_) )
        # TODO : filter by inliers

        # T_c0c2 transforms points in camera origin coordinates
        # to camera2 origin coordinates
        T_c0c2 = np.eye(4, dtype=np.float32)
        T_c0c2[:3,:3] = cv2.Rodrigues(rvec)[0]
        T_c0c2[:3,3:] = tvec
        T_c2c0 = tx.inverse_matrix(T_c0c2)

        T_b2b0 = np.linalg.multi_dot([
                self.T_c2b_,
                T_c2c0,
                self.T_b2c_]) # base2 in base0 coordinate system
        _,_,ang,lin,_ = tx.decompose_matrix(T_b2b0)
        #ang = tx.euler_from_matrix(T_b2b0)
        #lin = tx.translation_from_matrix(T_b2b0)

        base_h1 = ang[-1] # w.r.t z-ax
        base_t1 = [lin[0], lin[1]] # x-y components

        #print base_h1, base_t1

        # from here, visualization + validation

        # compute reprojection
        pts2_rec, _ = cv2.projectPoints(
                np.float32(self.lm_pt3_),
                np.float32(rvec), # or does it require inverse rvec/tvec?
                np.float32(tvec),
                cameraMatrix=Kmat,
                distCoeffs=distCoeffs,
                )
        pts2_rec = np.squeeze(pts2_rec, axis=1)
        # TODO : compute relative scale factor

        #if len(pts3) < in_thresh:
        #    # insufficient number of inliers to recover pose
        #    return True, None

        # no-slip
        # TODO : why is t[0,0] so bad???
        # TODO : relax the no-slip constraint by being better at triangulating or something

        mim = drawMatches(img1, img2, pt1, pt2, msk)
        #mim = np.concatenate([img1,img2], axis=1)

        #cv2.drawKeypoints(mim, kpt1[i1][matchesMask], mim, color=(0,0,255))
        #cv2.drawKeypoints(mim[:,320:], kpt2[i2][matchesMask], mim[:,320:], color=(0,0,255))

        pts3 = self.lm_pt3_.dot(self.T_c2b_[:3,:3].T) + self.T_c2b_[:3,3]

        #matchesMask = np.zeros(len(matches), dtype=np.bool)
        #matchesMask[midx] = 1

        #draw_params = dict(
        #        matchColor = (0,255,0),
        #        singlePointColor = (255,0,0),
        #        flags = 0,
        #        matchesMask=matchesMask.ravel().tolist()
        #        )
        #mim = cv2.drawMatches(
        #        img1,kpt1,img2,kpt2,
        #        matches,None,**draw_params)
        #mim = cv2.addWeighted(np.concatenate([img1,img2],axis=1), 0.5, mim, 0.5, 0.0)
        #cv2.drawKeypoints(mim, kpt1[i1][matchesMask], mim, color=(0,0,255))
        #cv2.drawKeypoints(mim[:,320:], kpt2[i2][matchesMask], mim[:,320:], color=(0,0,255))
        print('---')

        return True, (mim, base_h1, base_t1, pts2_rec, pts3)

        #print 'm', m
