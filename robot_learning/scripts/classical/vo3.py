from collections import namedtuple, deque
from filterpy.kalman import InformationFilter
from tf import transformations as tx
import cv2
import numpy as np

from vo_common import recover_pose, drawMatches
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors


class Landmark(object):
    def __init__(self,
            pos,
            var,
            des,
            kpt,
            trk,
            ang
            ):
        self.pos_ = pos # 3D Position = [3]
        self.var_ = var # Variance = [3x3]
        self.des_ = des # Descriptor = [32?]
        self.kpt_ = kpt # Keypoint = [cv2.KeyPoint]
        self.trk_ = trk # Tracking = Bool
        self.ang_ = ang # View Angle = Float - useful for loop closure

    def update(self):
        pass

# manage multiple landmarks
# for easier queries.

class Landmarks(object):
    def __init__(self, n_des=32):
        self.pos_ = np.empty((0,3), dtype=np.float32)
        self.var_ = np.empty((0,3,3), dtype=np.float32)
        self.des_ = np.empty((0,n_des), dtype=np.int32)
        self.ang_ = np.empty((0,1), dtype=np.float32)
        #self.kpt_
        #self.trk_

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
        self.D_ = D
        self.T_c2b_ = T_c2b
        self.T_b2c_ = tx.inverse_matrix(T_c2b)

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
        #kpt = self.det_.detect(img[240:])
        #for k in kpt:
        #    k.pt = tuple(k.pt[0]+240, k.pt[1])

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

        kpt = np.asarray(kpt)
        des = np.asarray(des)
        return [kpt, des]

    def pt2_to_pt2u(self, pt2):
        pt2 = cv2.undistortPoints(pt2[None,...],
                self.K_,
                self.D_,
                P=self.K_)[0]
        return pt2

    def pt3_pose_to_pt2_msk(self, pt3, pose):
        # pose = (x,y,h)
        # NOTE: pose is specified w.r.t base_link, not camera.
        
        pt3_cam = self.map_to_cam(pt3, pose)

        pt2, _ = cv2.projectPoints(
                pt3_cam,
                np.zeros(3),
                np.zeros(3), # zeros, because conversions happened above
                cameraMatrix=self.K_,
                distCoeffs=0*self.D_,
                # TODO : verify if it is appropriate to apply distortion
                )
        pt2 = np.squeeze(pt2, axis=1)

        # valid visibility mask
        lm_msk = (pt3_cam[:,2] > 1e-3) # z-positive
        lm_msk = np.logical_and.reduce([
            pt3_cam[:,2] > 1e-3, # z-positive
            0 <= pt2[:,0],
            pt2[:,0] < 640, # TODO : hardcoded
            0 <= pt2[:,1],
            pt2[:,1] < 480 # TODO : hardcoded
            ])
        return pt2, lm_msk

    def des_des_to_match(self, des1, des2,
            lowe=0.75,
            maxd=64.0
            ):
        # TODO : support arbitrary matchers or something
        # currently only supports wrapper around FLANN
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
        return (pth[:, :-1] / pth[:, -1:])

    def pose_to_T(self, pose):
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

    def __call__(self, ftype, *a, **k):
        return self.f_[ftype](*a, **k)

class ClassicalVO(object):
    def __init__(self):
        # define constant parameters
        Ks = (1.0 / 1.0)
        self.K_ = np.reshape([
            499.114583 * Ks, 0.000000, 325.589216 * Ks,
            0.000000, 498.996093 * Ks, 238.001597 * Ks,
            0.000000, 0.000000, 1.000000], (3,3))
        self.D_ = np.float32([0.158661, -0.249478, -0.000564, 0.000157, 0.000000])

        # conversion from camera frame to base_link frame
        self.T_c2b_ = tx.compose_matrix(
                angles=[-np.pi/2 - np.deg2rad(30),0.0,-np.pi/2],
                translate=[0.25,0,0.1])


        # define "system" parameters
        self.pEM_ = dict(method=cv2.FM_RANSAC, prob=0.999, threshold=1.0)
        self.pLK_ = dict(winSize = (51,13),
                maxLevel = 16,
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
                flags = 0,
                minEigThreshold = 1e-3 # TODO : disable eig?
                )

        # conversions
        self.cvt_ = Conversions(
                self.K_, self.D_,
                self.T_c2b_
                )

        # data cache + flags
        #self.landmarks_ = []
        self.landmarks_ = Landmarks()
        self.hist_ = deque(maxlen=100)

    def track(self, img1, img2, pt1, pt2=None):
        # stat img
        h, w = np.shape(img2)[:2]

        # copy LK Params
        pLK = self.pLK_.copy()

        # convert to grayscale
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # forward flow
        if pt2 is not None:
            # set initial flow flags
            pLK['flags'] |= cv2.OPTFLOW_USE_INITIAL_FLOW
        pt2, st, err = cv2.calcOpticalFlowPyrLK(
                img1_gray, img2_gray, pt1, pt2,
                **pLK)

        # backward flow
        # unset initial flow flags
        pLK['flags'] &= ~cv2.OPTFLOW_USE_INITIAL_FLOW
        pt1_r, st, err = cv2.calcOpticalFlowPyrLK(
                img2_gray, img1_gray, pt2, None,
                **pLK)

        # override error with reprojection error
        # (default error doesn't make much sense anyways)
        err = np.linalg.norm(pt1 - pt1_r, axis=-1)

        # apply mask
        msk_in = np.all(np.logical_and(
                np.greater_equal(pt2, [0,0]),
                np.less(pt2, [w,h])), axis=-1)
        msk_st = st[:,0].astype(np.bool)
        msk_err = (err < 1.0)
        msk = np.logical_and.reduce([
            msk_err,
            msk_in,
            msk_st
            ])#, msk_ef])

        return pt2, msk

    def proc_f2m(self, img, pose, lmk):
        # 1. 
        res = cv2.solvePnPRansac(
            self.lm_pt3_, self.lm_pt2_, self.K_, self.dC_,
            useExtrinsicGuess=False,
            #useExtrinsicGuess=True,
            #rvec = rvec0,
            #tvec = tvec0,
            iterationsCount=1000,
            reprojectionError=.1, # TODO : tune these params
            confidence=0.9999,
            #flags = cv2.SOLVEPNP_EPNP
            #flags = cv2.SOLVEPNP_DLS # << WORKS PRETTY WELL (SLOW?)
            #flags = cv2.SOLVEPNP_AP3P
            flags = cv2.SOLVEPNP_ITERATIVE # << default
            #flags = cv2.SOLVEPNP_P3P
            #flags = cv2.SOLVEPNP_UPNP
            )

    def __call__(self, img, pose, scale=1.0):
        # suffix designations:
        # o/0 = origin (i=0)
        # p = previous (i=t-1)
        # c = current  (i=t)

        # process current frame
        # TODO : enable lazy evaluation
        # (currently very much eager)
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

        # == obtain next-frame keypoints ==
        # opt1 : points by track
        pt2_c, msk_t = self.track(img_p, img_c, pt2_p)

        # opt2 : points by match
        # i1, i2 = self.cvt_.des_des_to_match(des_p, des_c)
        # msk_t = np.zeros(len(pt2_p), dtype=np.bool)
        # msk_t[i1] = True
        # pt2_c = np.zeros_like(pt2_p)
        # pt2_c[i1] = self.cvt_.kpt_to_pt(kpt_c[i2])
        # =================================

        #print 'mean delta', np.mean(pt2_c - pt2_p, axis=0) # -14 px

        track_ratio = float(msk_t.sum()) / msk_t.size # logging/status
        print('track : {}/{}'.format(msk_t.sum(), msk_t.size))

        pt2_u_p = self.cvt_.pt2_to_pt2u(pt2_p[msk_t])
        pt2_u_c = self.cvt_.pt2_to_pt2u(pt2_c[msk_t])

        E, msk_e = cv2.findEssentialMat(pt2_u_c, pt2_u_p, self.K_,
                **self.pEM_)
        msk_e = msk_e[:,0].astype(np.bool)
        #print('em : {}/{}'.format(msk_e.sum(), msk_e.size))

        n_in, R, t, msk_r, pt3 = recover_pose(E, self.K_,
                pt2_u_c[msk_e], pt2_u_p[msk_e], log=False)
        pt3 = pt3.T
        #print( 'mr : {}/{}'.format(msk_r.sum(), msk_r.size))

        if True:
            if len(self.landmarks_.pos_) > 0:
                # enter landmark processing
                pt2_lm_c, lm_msk = self.cvt_.pt3_pose_to_pt2_msk(
                        self.landmarks_.pos_, pose)
                # note that pt2_lm_c is undistorted.
                print('visible landmarks : {}/{}'.format(lm_msk.sum(), lm_msk.size))
            else:
                lm_msk = np.ones((0), dtype=np.bool)

            # select useful descriptor by current match
            des_p_m = des_p[msk_t][msk_e][msk_r]

            i1, i2 = self.cvt_.des_des_to_match(
                    self.landmarks_.des_[lm_msk],
                    des_p_m)

            if lm_msk.sum() > 0:
                # filter correspondences by Emat consensus
                # TODO : take advantage of the Emat here to some use?
                _, lm_msk_e = cv2.findEssentialMat(
                        pt2_lm_c[lm_msk][i1],
                        pt2_u_c[msk_e][msk_r][i2],
                        self.K_,
                        **self.pEM_)
                lm_msk_e = lm_msk_e[:,0].astype(np.bool)
                print('landmark concensus : {}/{}'.format( lm_msk_e.sum(), lm_msk_e.size))

                ## == visualize projection error ==
                #ax = plt.gca()
                #ax.cla()
                #viz_lmk = pt2_lm_c[lm_msk][i1] # landmark projections to current pose
                #viz_cam = pt2_u_c[msk_e][msk_r][i2] # camera correspondences
                #ax.plot(viz_lmk[:,0],viz_lmk[:,1], 'ko', alpha=0.2) # where landmarks are supposed to be
                #ax.plot(viz_cam[:,0],viz_cam[:,1], 'r+', alpha=0.2)
                #ax.quiver(
                #        viz_lmk[:,0], viz_lmk[:,1],
                #        viz_cam[:,0]-viz_lmk[:,0], viz_cam[:,1]-viz_lmk[:,1],
                #        scale_units='xy',
                #        angles='xy',
                #        scale=1,
                #        color='b',
                #        alpha=0.2
                #        )
                ## apply consensus
                #viz_lmk = viz_lmk[lm_msk_e]
                #viz_cam = viz_cam[lm_msk_e]

                ##plt.hist(viz_cam[:,0] - viz_lmk[:,0],
                ##        bins=np.linspace(-100,100)
                ##        )
                #ax.plot(viz_lmk[:,0],viz_lmk[:,1], 'ko') # where landmarks are supposed to be
                #ax.plot(viz_cam[:,0],viz_cam[:,1], 'r+')
                #ax.quiver(
                #        viz_lmk[:,0], viz_lmk[:,1],
                #        viz_cam[:,0]-viz_lmk[:,0], viz_cam[:,1]-viz_lmk[:,1],
                #        scale_units='xy',
                #        angles='xy',
                #        scale=1,
                #        color='g'
                #        )
                #if not ax.yaxis_inverted():
                #    ax.invert_yaxis()
                # ====================================
            else:
                lm_msk_e = np.ones(len(i1), dtype=np.bool)

            # landmark correspondences
            p_lm_0 = self.landmarks_.pos_[lm_msk][i1][lm_msk_e] # map-frame lm pos
            p_lm_c = self.cvt_.map_to_cam(p_lm_0, pose) # TODO : use rectified pose?

            p_lm_v2_c = pt3[i2][lm_msk_e] # current camera frame lm pos

            # estimate scale from landmark correspondences
            d_lm_old = np.linalg.norm(p_lm_c, axis=-1)
            d_lm_new = np.linalg.norm(p_lm_v2_c, axis=-1)

            # TODO : update landmarks from computed correspondences

            if len(d_lm_old) > 0:
                scale_rel = (d_lm_old / d_lm_new)
                #plt.gca().cla()
                #plt.hist(scale_rel)
                #plt.show()

                #print 'scale_rel'
                #print scale_rel

                scale_est = np.median(scale_rel, axis=-1)
                print('estimated scale ratio : {}/{} = {}'.format(
                    scale_est, scale, scale_est/scale))

                # override scale here
                scale = scale_est

                #res = cv2.solvePnPRansac(
                #        p_lm_0, pt2_u_c[msk_e][msk_r][i2], self.K_, 0*self.D_,
                #        useExtrinsicGuess = False,
                #        iterationsCount=1000,
                #        reprojectionError=1.0,
                #        confidence=0.9999,
                #        #flags = cv2.SOLVEPNP_EPNP
                #        #flags = cv2.SOLVEPNP_DLS # << WORKS PRETTY WELL (SLOW?)
                #        #flags = cv2.SOLVEPNP_AP3P
                #        flags = cv2.SOLVEPNP_ITERATIVE # << default
                #        #flags = cv2.SOLVEPNP_P3P
                #        #flags = cv2.SOLVEPNP_UPNP
                #        )
                #dbg, rvec, tvec, inliers = res
                #print('pnp {}/{}'.format( len(inliers), len(p_lm_0)))
                #print('tvec-pnp', tvec.ravel())
            else:
                # implicit : scale = scale
                pass

            # insert unselected landmarks
            lm_sel_msk = np.zeros(len(des_p_m), dtype=np.bool)
            lm_sel_msk[i2] = True
            lm_new_msk = ~lm_sel_msk


            n_new = (lm_new_msk).sum()
            msk_n = np.ones(n_new, dtype=np.bool)

            if len(d_lm_old) > 0:
                # filter insertion by proximity to existing landmarks
                neigh = NearestNeighbors(n_neighbors=1)
                neigh.fit(pt2_lm_c[lm_msk])
                d, _ = neigh.kneighbors(pt2_u_c[msk_e][msk_r][lm_new_msk], return_distance=True)
                msk_knn = (d < 20.0)[:,0] # TODO : magic number

                # dist to nearest landmark, less than 20px
                msk_n[msk_knn] = False
                n_new = msk_n.sum()

            print('adding {} landmarks : {}->{}'.format(n_new,
                len(self.landmarks_.pos_), len(self.landmarks_.pos_)+n_new
                ))

            pos_new = self.cvt_.cam_to_map(
                    scale * pt3[lm_new_msk][msk_n],
                    pose) # TODO : use rectified pose here if available
            des_new = des_p_m[lm_new_msk][msk_n]
            ang_new = np.full((n_new,1), pose[-1], dtype=np.float32)

            # append new landmarks ...
            self.landmarks_.ang_ = np.concatenate(
                    [self.landmarks_.ang_, ang_new], axis=0)
            self.landmarks_.des_ = np.concatenate(
                    [self.landmarks_.des_, des_new], axis=0)
            self.landmarks_.pos_ = np.concatenate(
                    [self.landmarks_.pos_, pos_new], axis=0) 

        msk = np.zeros(len(pt2_p), dtype=np.bool)
        midx = np.where(msk_t)[0][
                np.where(msk_e)[0]][
                        np.where(msk_r)[0]]
        msk[midx] = True
        #print('final msk : {}/{}'.format(msk.sum(), msk.size))

        mim = drawMatches(img_p, img_c, pt2_p, pt2_c, msk)

        # dh/dx in pose_p frame
        x, y, h = pose_p

        dh = -tx.euler_from_matrix(R)[1]
        #dx = s * np.float32([ np.abs(t[2]), 0*-t[1] ])
        dx = scale * np.float32([ t[2], -t[1] ])

        c, s = np.cos(h), np.sin(h)
        R2_p = np.reshape([c,-s,s,c], [2,2]) # [2,2,N]
        dp = R2_p.dot(dx).ravel()

        x_c = [x+dp[0],y+dp[1]]
        h_c = (h + dh + np.pi) % (2*np.pi) - np.pi
        print('tvec-f2f', x_c)

        # TODO : return correct values for reconstructed 3d points in map frame
        # as well as reprojected 2d points based on such landmarks.

        return True, (mim, h_c, x_c, pt2_p, np.empty((0,3)), '')

def main():
    K = np.float32([500,0,320,0,500,240,0,0,1]).reshape(3,3)
    D = np.zeros(5, dtype=np.float32)
    T_c2b = tx.compose_matrix(
            angles=[-np.pi/2 - np.deg2rad(30),0.0,-np.pi/2],
            translate=[0.25,0,0.1])

    cvt = Conversions(K,D,T_c2b)

    pt = np.random.uniform(-3, -3, size=(100,3))
    pose = np.random.uniform(-np.pi, np.pi, size=3)

    pt_m = cvt.cam_to_map(pt, pose)
    print (pt_m - pt).sum()
    pt_c = cvt.map_to_cam(pt_m, pose)
    print np.sum(pt - pt_c)

if __name__ == "__main__":
    main()
