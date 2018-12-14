import numpy as np
import sys
import cv2
from scipy.optimize import linear_sum_assignment
from tf import transformations as tx
from collections import deque

class ClassicalVO(object):
    def __init__(self):
        self.hist_ = deque(maxlen=100)
        self.lm_pt2_ = None
        self.lm_pt3_ = None # structure: [ pt3, des ]
        self.lm_msk_ = None # previous landmark keypoints mask
        self.lm_des_ = None # landmark descriptors

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

    def track(self, img1, img2, pt1):
        lk_params = dict( winSize  = (47,3),
                maxLevel = 20,
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.001))
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        pt2, st, err = cv2.calcOpticalFlowPyrLK(
                img1_gray, img2_gray, pt1,
                None, **lk_params)

        h, w = np.shape(img2)[:2]

        # apply mask
        msk_in = np.all(np.logical_and(
                np.greater_equal(pt2, [0,0]),
                np.less(pt2, [w,h])), axis=-1)
        msk_st = st[:,0].astype(np.bool)
        msk_ef = np.isfinite(err[:,0])
        #ec = np.less(err[:,0], 10.0)
        msk = np.logical_and.reduce([msk_in, msk_st, msk_ef])

        return pt2, msk

    #def match_v2(self, img1, img2, kp1):
    #    lk_params = dict( winSize  = (47,3),
    #            maxLevel = 20,
    #            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.001))
    #    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    #    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    #    # p0 = cv2.goodFeaturesToTrack(src_gray, mask = None, **feature_params)
    #    # calculate optical flow

    #    #print kp1[0], kp1[0].angle, kp1[0].octave, kp1[0].overlap, kp1[0].response, kp1[0].size
    #    p1 = np.float32([k.pt for k in kp1])
    #    p2, st, err = cv2.calcOpticalFlowPyrLK(
    #            img1_gray, img2_gray, p1,
    #            None, **lk_params)

    #    #kp1 = [cv2.KeyPoint(pt=p,size=3) for p in p1]
    #    #p1_ = cv2.KeyPoint_convert(kp1)#, size=3.0)
    #    print p2.shape
    #    kp2 = [cv2.KeyPoint(x=p[0],y=p[1],_size=3.0) for p in p2]#, size=3.0)
    #    kp2 = np.asarray(kp2, dtype=cv2.KeyPoint)

    #    h, w = img1.shape[:2]

    #    msk_in = np.all(np.logical_and(
    #            np.greater_equal(p1, [0,0]),
    #            np.less(p1, [w,h])), axis=-1)

    #    st = st[:,0].astype(np.bool)

    #    #print st.shape, msk_in.shape
    #    print msk_in.shape, msk_in.dtype
    #    print st.shape, st.dtype
    #    ef = np.isfinite(err[:,0])
    #    ec = np.less(err[:,0], 10.0)

    #    st = np.logical_and.reduce([msk_in, st, ef, ec])
    #    
    #    p1, p2 = np.float32(p1[st]), np.float32(p2[st])
    #    kp1, kp2 = kp1[st], kp2[st]
    #    matches = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=e)
    #            for i,e in enumerate(err[st])]

    #    n = len(p1)

    #    return matches, p1, p2, kp1, kp2

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

    def getPoseAndPoints(self, pt1, pt2, midx, method):
        # == 0 :  unroll parameters ==
        Kmat_o = self.K_
        Kmat = self.K2_
        distCoeffs = self.dC_

        # rectify input
        pt1 = np.float32(pt1)
        pt2 = np.float32(pt2)

        # find fundamental matrix
        Fmat, msk = cv2.findFundamentalMat(pt2, pt1,
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
        pt2, pt1 = cv2.correctMatches(Fmat, pt2, pt1) # TODO : not sure if this is necessary
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
        Emat, msk = cv2.findEssentialMat(pt2, pt1, Kmat,
                method=method, prob=0.999, threshold=0.1)
        msk = np.asarray(msk[:,0]).astype(np.bool)

        # filter points + bookkeeping mask
        pt1 = pt1[msk]
        pt2 = pt2[msk]
        midx = midx[msk]

        n_in, R, t, msk, _ = cv2.recoverPose(Emat,
                pt2,
                pt1,
                cameraMatrix=Kmat,
                distanceThresh=100.0) # TODO : or something like 10.0/s ??
        msk = np.asarray(msk[:,0]).astype(np.bool)

        # filter points + bookkeeping mask
        pt1 = pt1[msk]
        pt2 = pt2[msk]
        midx = midx[msk]

        # validate triangulation
        pts_h = cv2.triangulatePoints(
                Kmat.dot(np.eye(3,4)),
                Kmat.dot(np.concatenate([R, t], axis=1)),
                pt2[None,...],
                pt1[None,...]).astype(np.float32)
        
        msk = np.all(np.isfinite(pts_h),axis=0)
        pt1 = pt1[msk]
        pt2 = pt2[msk]
        midx = midx[msk]

        return [R, t, pt1, pt2, midx, pts_h]

    def undistort(self, pt):
        pt_u = cv2.undistortPoints(2.0*pt[None,...], self.K2_, self.dC_, P=self.K2_)[0]/2.0
        return pt_u

    def __call__(self, img, pose,
            in_thresh = 16,
            s = 0.1
            ):

        # == 0 :  unroll parameters ==
        Kmat_o = self.K_
        Kmat = self.K2_
        distCoeffs = self.dC_
        method = cv2.FM_LMEDS # TODO : configure
        # ============================

        # detect features + query/update history
        kpt2, des2, img2 = self.detect(img)
        if len(self.hist_) <= 0:
            self.lm_msk_ = np.zeros(len(kpt2), dtype=np.bool)
            self.hist_.append( (kpt2, des2, img2) )
            return True, None
        kpt1, des1, img1 = self.hist_[-1]
        self.hist_.append((kpt2, des2, img2))
        # ==============

        if (self.lm_pt3_ is not None):
            print(' # Landmarks Tracking : {}'.format(len(self.lm_pt3_)))
            # update + track existing landmark positions
            pt2, msk = self.track(img1, img2, self.lm_pt2_)

            # filter out landmarks by current tracking match

            # TODO : update lm_pt3 with recent tracking match info
            #pt1_u = self.undistort(self.lm_pt2_[msk])
            #pt2_u = self.undistort(pts[msk])
            #res = self.getPoseAndPoints(pt1_u, pt2_u, np.arange(len(pt1_u)), method)

            self.lm_pt2_ = pt2[msk]
            self.lm_pt3_ = self.lm_pt3_[msk]
            self.lm_des_ = self.lm_des_[msk]

            # solve PnP and get pose information
	    #solvePnPRansac(landmarks_ref, featurePoints_ref, K, dist_coeffs,rvec, tvec,false, 100, 8.0, 0.99, inliers);// inliers);
            _, rvec, tvec, inliers = res = cv2.solvePnPRansac(
                    self.lm_pt3_, self.lm_pt2_, self.K2_, self.dC_,
                    useExtrinsicGuess=False,
                    iterationsCount=1000,
                    reprojectionError=5.0
                    )

            # construct transformation matrix from PnP results
            T_m2i = np.eye(4, 4) # pt3 -> pt2
            R, _ = cv2.Rodrigues(rvec)
            t    = tvec.ravel()
            print('R-t', R, t)
            T_m2i[:3,:3] = R
            T_m2i[:3,3]  = t

            # image to map transform
            T_i2m = np.linalg.inv(T_m2i)

            # also match descriptor with landmarks
            # TODO : not sure if this is necessary at all.
            # if self.lm_des_ is not None:
            #     lm_match = self.match_v1(self.lm_des_, des2) # match with landmarks
            #     unsel_msk = np.ones(len(des2), dtype=np.bool) # un-selected
            #     i1, i2 = np.int32([(m.queryIdx, m.trainIdx) for m in lm_match]).T

            #     # update landmark descriptors
            #     # TODO : handle this more intelligently?
            #     self.lm_des_[i1] = des2[i2]

            #     unsel_msk[i2] = 0
            #     kpt2_n = kpt2[unsel_msk]
            #     des2_n = des2[unsel_msk] # now, only select points that do not match with landmarks
            # else:

            kpt2_n = kpt2
            des2_n = des2

            # keypoints from previous frame
            # that are not part of existing landmarks
            msk_n  = np.logical_not(self.lm_msk_)
            kpt1_n = kpt1[msk_n]
            des1_n = des1[msk_n]

            # compute matches with current frame
            matches = self.match_v1(des1_n, des2_n)
            i1, i2 = np.int32([(m.queryIdx, m.trainIdx) for m in matches]).T

            # grab relevant keypoints + points
            kpt1_nlm = kpt1_n[i1]
            kpt2_nlm = kpt2_n[i2]
            des1_nlm = des1_n[i1]
            des2_nlm = des2_n[i2]
            pt1 = np.float32(cv2.KeyPoint.convert(kpt1_nlm))
            pt2 = np.float32(cv2.KeyPoint.convert(kpt2_nlm))
            midx = np.arange(len(matches)) # track match indices

            pt1_u = self.undistort(pt1)
            pt2_u = self.undistort(pt2)

            res = self.getPoseAndPoints(pt1_u, pt2_u, midx, method)

            if res is None:
                # no new landmarks to add
                pass
            else:
                # add new landmarks with rel. scale
                [_, _, pt1_r, pt2_r, midx, pts_h] = res
                # TODO : use R,t info from ^^ ??????
                s    = np.linalg.norm(t) # apply relative scale
                pts3 = pts_h.copy()
                pts3 /= pts3[3:] # bring to scale
                pts3[:3] *= s # multiply points norm by s # ... 4xN
                #pts3 = np.dot(T_i2m[:3], pts3) # 3x4 * 4xN --> 3xN
                pts3 = pts3[:3]
                # bring lm_pt3 to coord frame of most recent observation
                self.lm_pt3_ = np.dot(
                        self.lm_pt3_, T_m2i[:3,:3].T
                        ) + T_m2i[:3,3:].T

                # extend landmarks
                self.lm_pt2_ = np.concatenate((self.lm_pt2_, pt2[midx]), axis=0) # undistorted!
                self.lm_pt3_ = np.concatenate((self.lm_pt3_, pts3.T), axis=0)
                self.lm_des_ = np.concatenate((self.lm_des_, des2_nlm[midx]), axis=0)
                # setup keypoint mask
                self.lm_msk_ = np.zeros(len(kpt2), dtype=np.bool)
                self.lm_msk_[i2[midx]] = True
        else:
            # initialize landmarks
            matches = self.match_v1(des1, des2)
            i1, i2 = np.int32([(m.queryIdx, m.trainIdx) for m in matches]).T
            # grab relevant keypoints + points
            kpt1_n = kpt1[i1]
            kpt2_n = kpt2[i2]
            pt1 = np.float32(cv2.KeyPoint.convert(kpt1_n))
            pt2 = np.float32(cv2.KeyPoint.convert(kpt2_n))
            midx = np.arange(len(matches)) # track match indices

            # undistort
            pt1_u = cv2.undistortPoints(2.0*pt1[None,...], Kmat_o, distCoeffs, P=Kmat_o)[0]/2.0
            pt2_u = cv2.undistortPoints(2.0*pt2[None,...], Kmat_o, distCoeffs, P=Kmat_o)[0]/2.0
            
            res = self.getPoseAndPoints(pt1_u, pt2_u, midx, method)
            if res is None:
                return True, None

            # unroll result
            [R, t, pt1_r, pt2_r, midx, pts_h] = res
            t = t.ravel()
            t = s * t # use input scale

            # homogeneous --> 3d
            pts3 = pts_h[:3] / pts_h[3:]
            pts3 = s * pts3 # use input scale

            # initialize landmarks
            self.lm_pt2_ = pt2[midx]
            # TODO : should compute offsets and such
            print 'R-t init', R, t
            self.lm_pt3_ = np.copy(pts3.T)
            self.lm_msk_ = np.zeros(len(kpt2), dtype=np.bool)
            self.lm_msk_[i2[midx]] = True
            self.lm_des_ = des2[i2[midx]] # landmark descriptors

        # compute reprojection
        pts2_rec, _ = cv2.projectPoints(
                pts3.T[...,None],
                rvec=cv2.Rodrigues(R)[0],
                tvec=t,
                #rvec=np.zeros(3),
                #tvec=np.zeros(3),
                cameraMatrix=Kmat,
                distCoeffs=distCoeffs,
                )
        pts2_rec = np.squeeze(pts2_rec, axis=1)

        # TODO : compute relative scale factor

        # apply scale factor and re-orient to align with base coordinates
        pts3 = np.stack([pts3[2], -pts3[0], -pts3[1]], axis=-1)

        if len(pts3) < in_thresh:
            # insufficient number of inliers to recover pose
            return True, None

        rpy_cam = tx.euler_from_matrix(R)
        h = -rpy_cam[1] # base_link angular displacement

        # no-slip
        # TODO : why is t[0,0] so bad???
        # TODO : relax the no-slip constraint by being better at triangulating or something
        t = np.asarray([t[2], 0.0 * -t[0]])

        matchesMask = np.zeros(len(matches), dtype=np.bool)
        matchesMask[midx] = 1

        draw_params = dict(
                matchColor = (0,255,0),
                singlePointColor = (255,0,0),
                flags = 0,
                matchesMask=matchesMask.ravel().tolist()
                )
        mim = cv2.drawMatches(
                img1,kpt1,img2,kpt2,
                matches,None,**draw_params)
        mim = cv2.addWeighted(np.concatenate([img1,img2],axis=1), 0.5, mim, 0.5, 0.0)
        cv2.drawKeypoints(mim, kpt1[i1][matchesMask], mim, color=(0,0,255))
        cv2.drawKeypoints(mim[:,320:], kpt2[i2][matchesMask], mim[:,320:], color=(0,0,255))
        print('---')

        return True, (mim, h, t, pts2_rec, pts3)

        #print 'm', m
