"""
Semi-Urgent TODOs:
    - memory management (masks/fancy indexing creates copies; reuse same-sized arrays etc.)
"""
from collections import namedtuple, deque
from filterpy.kalman import InformationFilter
from tf import transformations as tx
import cv2
import numpy as np

from vo_common import recover_pose, drawMatches, recover_pose_from_RT
from vo_common import robust_mean, oriented_cov, show_landmark_2d
from vo_common import Landmarks, Conversions
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D

from scipy.sparse import lil_matrix
from scipy.optimize import least_squares

from ukf import build_ukf, build_ekf, get_QR
from ba import ba_J

tfig = None

def lerp(a,b,w):
    return (a*w) + (b*(1.0-w))

def axisEqual3D(ax):
    """ from https://stackoverflow.com/a/19248731 """
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

def ransac_update_num_iters(p, ep, mpt, it,
        eps = np.finfo(np.float32).eps
        ):
    """
    based on opencv calib3d/ptsetreg.cpp
    Still have no idea why this is valid.
    """
    p  = np.clip(p, 0, 1)
    ep = np.clip(ep, 0, 1)

    nmr = max(1.0 - p, eps)
    dmr = 1.0 - np.power(1.0-ep, mpt)

    if dmr < eps:
        return 0

    nmr = np.log(nmr)
    dmr = np.log(dmr)

    res = it if (dmr >= 0 or -nmr >= it * -dmr) else np.round(nmr/dmr).astype(np.int32)
    return res

def estimate_plane_ransac(pts,
        max_it=1000,
        conf = 0.99,
        thresh = 0.1,
        nvec=None
        ):

    best_fit = None
    best_err = np.inf
    best_msk = None

    n_it = max(max_it, 1)
    i = 0

    while i < n_it:
        ## select three points that define a plane and go from there.
        #sel = np.random.randint(len(pts), size=3)
        sel = np.random.choice(len(pts), size=3, replace=False)

        c = np.mean(pts[sel], axis=0, keepdims=True) # plane center

        if nvec is None:
            pa, pb, pc = pts[sel]

            ba = tx.unit_vector(pb-pa)
            ca = tx.unit_vector(pc-pa)

            n = tx.unit_vector(np.cross(ba, ca)) # plane normal
        else:
            n = nvec

        err = (pts - c).dot(n.reshape(-1,1)) # Nx3 . 3x1
        err = np.abs(err)

        msk = (err < thresh)
        n_in = msk.sum()
        err = err[msk].sum()

        n_it = ransac_update_num_iters(conf,
                float(msk.size - n_in) / msk.size, # idk what ep is
                3, # 3 points required to define a plane
                n_it)

        if err < best_err:
            best_err = err
            best_fit = (c, n)
            best_msk = msk

        i += 1

    #print('completed in {} iterations'.format(i))

    return best_fit, best_err, best_msk


def get_points_color(img, pts, w=1):
    # iterative method

    n, m = img.shape[:2]
    pis, pjs = np.round(pts[:,::-1]).T.reshape(2,-1).astype(np.int32)
    oi, oj = np.mgrid[-w:w+1,-w:w+1]
    iw, jw = pis[:,None,None] + oi, pjs[:,None,None] + oj
    iw = np.clip(iw, 0, n-1)
    jw = np.clip(jw, 0, m-1)

    cols_w = img[iw, jw] # n,2*w+1,2*w+1,3

    # opt 1 : naive mean
    # cols = np.mean(cols_w, axis=(1,2))
    cols = cols_w.astype(np.float32)
    # opt 2 : rms
    cols = np.sqrt(np.mean(np.square(cols),axis=(1,2)))
    #cs = np.clip(cs, 0, 255) # TODO : evaluate if necessary
    return np.asarray(cols, dtype=img.dtype)

def score_H(pt1, pt2, H, cvt, sigma=1.0):
    """ Homography model symmetric transfer error. """
    score = 0.0
    th = 5.991 # ??? TODO : magic number
    iss = (1.0 / (sigma*sigma))

    Hi = np.linalg.inv(H)
    pt2_r = cvt.pth_to_pt(cvt.pt_to_pth(pt1).dot(H.T))
    pt1_r = cvt.pth_to_pt(cvt.pt_to_pth(pt2).dot(Hi.T))
    e1 = np.square(pt1 - pt1_r).sum(axis=-1)
    e2 = np.square(pt2 - pt2_r).sum(axis=-1)

    #score = 1.0 / (e1.mean() + e2.mean())
    chi_sq1 = e1 * iss
    msk1 = (chi_sq1 <= th)
    score += ((th - chi_sq1) * msk1).sum()

    chi_sq2 = e2 * iss
    msk2 = (chi_sq2 <= th)
    score += ((th - chi_sq2) * msk2).sum()
    return score, (msk1 & msk2)

def score_F(pt1, pt2, F, cvt, sigma=1.0):
    """
    Fundamental Matrix symmetric transfer error.
    reference:
        https://github.com/opencv/opencv/blob/master/modules/calib3d/src/fundam.cpp#L728
    """
    score = 0.0
    th = 3.841 # ??
    th_score = 5.991 # ?? TODO : magic number
    iss = (1.0 / (sigma*sigma))

    pt1_h = cvt.pt_to_pth(pt1)
    pt2_h = cvt.pt_to_pth(pt2)

    x1, y1 = pt1.T
    x2, y2 = pt2.T

    a, b, c = pt1_h.dot(F.T).T # Nx3
    s2 = 1./(a*a + b*b);
    d2 = a * x2 + b * y2 + c
    e2 = d2*d2*s2

    a, b, c = pt2_h.dot(F).T
    s1 = 1./(a*a + b*b);
    d1 = a * x1 + b * y1 + c
    e1 = d1*d1*s1

    #score = 1.0 / (e1.mean() + e2.mean())
    chi_sq2 = e2 * iss
    msk2 = (chi_sq2 <= th)
    score += ((th_score - chi_sq2) * msk2).sum()

    chi_sq1 = e1* iss
    msk1 = (chi_sq1 <= th)
    score += ((th_score - chi_sq1) * msk1).sum()

    return score, (msk1 & msk2)

class ClassicalVO(object):
    # define flags
    VO_USE_FM_COR    = 1<<0  # Enable correctMatches() (NOTE: time-consuming)
    VO_USE_TRACK     = 1<<1  # Correspondences by track vs. descriptor match
    VO_USE_SCALE_A3D = 1<<2  # Estimate Scale from Affine3D
    VO_USE_SCALE_GP  = 1<<3  # Estimate Scale from Ground-Plane
    VO_USE_PNP       = 1<<4  # Compute Pose from PNP (TODO: NOT SUPPORTED)
    VO_USE_BA        = 1<<5  # Use Bundle Adjustment
    VO_USE_HOMO      = 1<<6  # Use Homography Fallback
    VO_USE_F2M       = 1<<7  # Use Frame-To-Map Information
    VO_USE_LM_KF     = 1<<8  # Use Landmark Kalman Filter
    VO_USE_KPT_SPX   = 1<<9  # Sub-pixel refinement (NOTE: time-consuming)
    VO_USE_MXCHECK   = 1<<10 # Cross-Check Matches
    VO_USE_GP_RSC    = 1<<11 # Enable RANSAC Plane estimation for ground plane

    VO_DEFAULT = VO_USE_FM_COR | VO_USE_TRACK | VO_USE_SCALE_GP | \
            VO_USE_BA | VO_USE_HOMO | VO_USE_F2M | \
            VO_USE_LM_KF | VO_USE_KPT_SPX | VO_USE_MXCHECK | \
            VO_USE_GP_RSC

    def __init__(self):
        # define configuration
        self.flag_ = ClassicalVO.VO_DEFAULT
        self.flag_ &= ~ClassicalVO.VO_USE_HOMO # TODO : doesn't really work?
        #self.flag_ &= ~ClassicalVO.VO_USE_GP_RSC
        #self.flag_ &= ~ClassicalVO.VO_USE_FM_COR # performance

        # define constant parameters
        Ks = (1.0 / 1.0)
        self.K_ = np.reshape([
            499.114583 * Ks, 0.000000, 325.589216 * Ks,
            0.000000, 498.996093 * Ks, 238.001597 * Ks,
            0.000000, 0.000000, 1.000000], (3,3))
        self.D_ = np.float32([0.158661, -0.249478, -0.000564, 0.000157, 0.000000])

        # conversion from camera frame to base_link frame
        # NOTE : extrinsic parameter anchor
        self.T_c2b_ = tx.compose_matrix(
                angles=[-np.pi/2-np.deg2rad(10),0.0,-np.pi/2],
                #angles=[-np.pi/2,0.0,-np.pi/2],
                translate=[0.174,0,0.113])

        # Note that camera intrinsic+extrinsic parameters
        # i.e. K, D, T_c2b
        # are coupled with the data, rather than the algorithm.

        # define "system" parameters
        self.pEM_ = dict(method=cv2.FM_RANSAC, prob=0.999, threshold=1.0)
        self.pLK_ = dict(winSize = (51,13),
                maxLevel = 16,
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
                flags = 0,
                minEigThreshold = 1e-3 # TODO : disable eig?
                )

        self.pPNP_ = dict(
                iterationsCount=1000,
                reprojectionError=2.0,
                confidence=0.999,
                #flags = cv2.SOLVEPNP_EPNP
                #flags = cv2.SOLVEPNP_DLS
                #flags = cv2.SOLVEPNP_AP3P
                flags = cv2.SOLVEPNP_ITERATIVE
                #flags = cv2.SOLVEPNP_P3P
                #flags = cv2.SOLVEPNP_UPNP
                )
        # TODO : what is FAST threshold?
        orb = cv2.ORB_create(
                nfeatures=2048,
                scaleFactor=1.2,
                nlevels=8,
                scoreType=cv2.ORB_FAST_SCORE,
                )
        det = orb
        #det = cv2.FastFeatureDetector_create(
        #        threshold=20, # I think this is the default
        #        nonmaxSuppression=True
        #        )
        #det = cv2.MSER_create()
        #det = cv2.GFTTDetector.create(
        #        maxCorners=4096,
        #        qualityLevel=0.01,
        #        minDistance=1.0,
        #        blockSize=3,
        #        #useHarrisDetector=True,
        #        #k=0.04
        #        ) # keypoints detector
        des = orb

        # conversions
        self.cvt_ = Conversions(
                self.K_, self.D_,
                self.T_c2b_,
                det=det,
                des=des
                )

        # data cache + flags
        self.landmarks_ = Landmarks()
        self.hist_ = deque(maxlen=3)

        # pnp
        self.pnp_p_ = None
        self.pnp_h_ = None

        # UKF
        #self.ukf_l_  = build_ukf()
        #self.ukf_g_  = build_ukf()
        self.ukf_l_  = build_ekf()
        self.ukf_g_  = build_ekf()
        self.ukf_dt_ = []

        # bundle adjustment
        self.ba_freq_ = 16# empirically pretty good
        self.ba_pos_ = []
        self.ba_ci_ = []
        self.ba_li_ = []
        self.ba_p2_ = []

    def track(self, img1, img2, pt1, pt2=None,
            thresh=1.0
            ):
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
        idx = np.arange(len(pt1))
        msk_in = np.all(np.logical_and(
                np.greater_equal(pt2, [0,0]),
                np.less(pt2, [w,h])), axis=-1)
        msk_st = st[:,0].astype(np.bool)
        msk_err = (err < 1.0) # track reprojection error
        msk = np.logical_and.reduce([
            msk_err,
            msk_in,
            msk_st
            ])
        idx = idx[msk]
        return pt2, idx

    def initialize_landmark_variance(self, pt3_c, pose):
        # initialize variance
        var_rel = np.square([0.02, 0.02, 1.0]) # expected landmark variance @ ~ 1m
        # TODO : ^ is a ballpark estimate.
        var_rel = np.diag(var_rel) # 3x3
        var_c = oriented_cov(pt3_c, var_rel) # Nx3x3

        # now rotate camera-coordinate variance to map-coord variance
        T_b2o = self.cvt_.pose_to_T(pose)
        T_c2m = np.linalg.multi_dot([
            self.cvt_.T_b2c_,
            T_b2o,
            self.cvt_.T_c2b_
            ])
        R_c2m = T_c2m[:3,:3]

        var_m = np.matmul(
                np.matmul(R_c2m[None,...], var_c),
                R_c2m.T[None,...]
                )

        return var_m

    def proc_f2m(self, pose, scale,
            des_p, des_c,
            idx_t, idx_e, idx_r,
            pt2_u_p, pt2_u_c,
            pt3,
            img_c, pt2_c,
            h_override,
            msg
            ):

        # build index combinationss
        # TODO : tracking these indices are really getting quite ridiculous.
        idx_te = idx_t[idx_e]
        idx_ter = idx_te[idx_r]
        idx_er = idx_e[idx_r]
        pt2_lm_c = None # ???

        # frame-to-map processing
        # (i.e. uses landmark data)

        if len(self.landmarks_.pos) > 0:
            # enter landmark processing

            # filter by distance (<10m for match)
            pose_tmp = self.cvt_.T_b2c_.dot([pose[0], pose[1], 0, 1]).ravel()[:3]
            delta    = np.linalg.norm(self.landmarks_.pos - pose_tmp[None,:], axis=-1)
            d_msk    = (delta < 10.0 / scale)

            # TODO : add preliminary filter by view angle

            # filter by visibility
            pt2_lm_c, lm_msk = self.cvt_.pt3_pose_to_pt2_msk(
                    self.landmarks_.pos, pose)
            lm_msk = np.logical_and.reduce([
                lm_msk,
                d_msk
                ])
            lm_idx = np.where(lm_msk)[0]
        else:
            # ==> empty
            lm_msk = np.ones((0), dtype=np.bool)
            lm_idx = np.where(lm_msk)[0]

        print('visible landmarks : {}/{}'.format(len(lm_idx), self.landmarks_.size_))

        # select useful descriptor based on current viewpoint
        des_p_m = des_p[idx_ter]
        i1, i2 = self.cvt_.des_des_to_match(
                self.landmarks_.des[lm_idx],
                des_p_m, cross=(self.flag_ & ClassicalVO.VO_USE_MXCHECK)
                )

        lm_msk_e = np.ones(len(i1), dtype=np.bool)
        lm_idx_e = np.where(lm_msk_e)[0]

        if len(lm_idx) > 16:
            # filter correspondences by Emat consensus
            # TODO : take advantage of the Emat here to some use?

            # first-order estimate: image-coordinate distance-based filter
            cor_delta = (pt2_lm_c[lm_idx][i1] - pt2_u_c[idx_er][i2])
            cor_delta = np.linalg.norm(cor_delta, axis=-1)
            lm_msk_d = (cor_delta < 64.0) 
            lm_idx_d = np.where(lm_msk_d)[0]

            # second estimate
            try:
                # TODO : maybe not the most efficient way to
                # check landmark consensus?
                _, lm_msk_e = cv2.findEssentialMat(
                        pt2_lm_c[lm_idx][i1][lm_idx_d],
                        pt2_u_c[idx_er][i2][lm_idx_d],
                        self.K_,
                        **self.pEM_)
            except Exception as e:
                lm_msk_e = None

            if lm_msk_e is not None:
                # refine by Emat
                lm_msk_e = lm_msk_e[:,0].astype(np.bool)
                lm_idx_e = np.where(lm_msk_e)[0]
                lm_msk_e = lm_msk_d[lm_idx_e]
                lm_idx_e = lm_idx_d[lm_idx_e]
            else:
                lm_msk_e = lm_msk_d
                lm_idx_e = lm_idx_d

            print('landmark concensus : {}/{}'.format(len(lm_idx_e), lm_msk_e.size))


            ## == visualize projection error ==
            # global tfig
            # if tfig is None:
            #     tfig = plt.figure()
            #     ax  = tfig.add_subplot(1,1,1)

            # ax = tfig.gca()
            # ax.cla()
            # viz_lmk = pt2_lm_c[lm_idx][i1][lm_idx_e] # landmark projections to current pose
            # viz_cam = pt2_u_c[idx_er][i2][lm_idx_e] # camera correspondences
            # ax.plot(viz_lmk[:,0],viz_lmk[:,1], 'ko', alpha=0.2) # where landmarks are supposed to be
            # ax.plot(viz_cam[:,0],viz_cam[:,1], 'r+', alpha=0.2)
            # ax.quiver(
            #         viz_lmk[:,0], viz_lmk[:,1],
            #         viz_cam[:,0]-viz_lmk[:,0], viz_cam[:,1]-viz_lmk[:,1],
            #         scale_units='xy',
            #         angles='xy',
            #         scale=1,
            #         color='b',
            #         alpha=0.2
            #         )
            # tfig.canvas.draw()
            # plt.pause(0.001)

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

        # landmark correspondences
        p_lm_0 = self.landmarks_.pos[lm_idx][i1][lm_idx_e] # map-frame lm pos
        p_lm_c = self.cvt_.map_to_cam(p_lm_0, pose) # TODO : use rectified pose?

        p_lm_v2_c = pt3[i2][lm_idx_e] # current camera frame lm pos

        # estimate scale from landmark correspondences
        # opt1 : norm
        #d_lm_old = np.linalg.norm(p_lm_c, axis=-1)
        #d_lm_new = np.linalg.norm(p_lm_v2_c, axis=-1)
        # opt2 : take z-value in camera coordinates
        # z-value much more stable than norm?
        d_lm_old = p_lm_c[:,2]
        d_lm_new = p_lm_v2_c[:,2]

        # validation : dot product (=cos(theta))
        #uv_lm_c = p_lm_c / d_lm_old[:, None] # Nx3
        #uv_lm_v2_c = p_lm_v2_c / d_lm_new[:, None] #Nx3
        #if uv_lm_c.size > 0:
        #    ang = np.sum(uv_lm_c * uv_lm_v2_c, axis=-1)
        #    print('angle stats :', ang.min(), ang.mean(), ang.max(), ang.std())

        scale_rel = (d_lm_old / d_lm_new).reshape(-1,1)
        scale_rel_std = scale_rel.std()
        print('estimated scale stability', scale_rel_std)

        if self.flag_ & ClassicalVO.VO_USE_SCALE_A3D:
            if len(p_lm_v2_c) > 0:
                res_a3, T_a3, inl_a3 = cv2.estimateAffine3D(
                        p_lm_v2_c[...], p_lm_c[...],
                        ransacThreshold=0.1,
                        confidence=0.999
                        )
                T_a3 = np.concatenate([T_a3, [[0,0,0,1]]], axis=0)
                scale_est = tx.scale_from_matrix(T_a3)[0]
        else:
            if scale_rel_std < 0.3:
                # scale weight by landmark variance
                scale_w = (self.landmarks_.var[lm_idx][i1][lm_idx_e][:,(0,1,2),(0,1,2)])
                scale_w = np.linalg.norm(scale_w, axis=-1) 
                scale_w = np.sum(scale_w) / scale_w
                scale_est = robust_mean(scale_rel, weight=scale_w)
                #scale_est_2 = robust_mean(scale_rel)
                #print('weighting diff : {} vs {}'.format(scale_est, scale_est_2))
            else:
                # TODO : why does this happen?
                # scale estimates are anticipated to be unstable.
                # use input scale
                scale_est = scale

        if len(d_lm_old) > 0:
            print('estimated scale ratio : {}/{} = {}'.format(
                scale_est, scale, scale_est/scale))
            # TODO : tune scale interpolation alpha
            alpha = 0.5
            # override scale here
            # will smoothing over time hopefully prevent scale drift?
            """
            There are currently three methods to estimate scale:
            1. baseline=ukf prediction based estimate
            2. ground-plane projection based estimate
            3. landmark correspondence based estimate
            all of these estimates are un-intelligently
            combined to produce the final result.
            """
            scale = lerp(scale, scale_est, alpha)
        else:
            # implicit : scale = scale
            pass

        update_lm = self.flag_ & ClassicalVO.VO_USE_LM_KF

        if update_lm and (not h_override):
            # update landmarks from computed correspondences
            # TODO: apply scale_est (aggregate) or rel (individual)?
            p_lm_v2_c_s = p_lm_v2_c * scale
            var_lm_new = self.initialize_landmark_variance(p_lm_v2_c_s, pose)
            p_lm_v2_0 = self.cvt_.cam_to_map(p_lm_v2_c_s, pose)

            midx = np.arange(self.landmarks_.size_)
            midx = midx[lm_idx][i1][lm_idx_e]
            self.landmarks_.update(midx, p_lm_v2_0, var_lm_new)

            # Add correspondences to BA Cache
            self.cache_BA(
                    lmk_idx=lm_idx[i1][lm_idx_e],
                    lmk_pt2=pt2_u_c[idx_er][i2][lm_idx_e]
                    )

        if self.flag_ & ClassicalVO.VO_USE_PNP: # == if USE_PNP
            pt_world = self.landmarks_.pos[lm_idx][i1]#[lm_msk_e]
            pt_cam   = pt2_u_c[idx_e][idx_r][i2]#[lm_msk_e]

            if pt_world.size>0 and pt_cam.size>0:
                # PNP is super unreliable
                # use guess?

                T_b2o = self.cvt_.pose_to_T(pose)
                T_c0c2_est = np.linalg.multi_dot([
                    self.cvt_.T_b2c_,
                    tx.inverse_matrix(T_b2o), # T_b0b2
                    self.cvt_.T_c2b_])

                rvec0 = cv2.Rodrigues(T_c0c2_est[:3,:3])[0]
                tvec0 = T_c0c2_est[:3, 3:]

                res = cv2.solvePnPRansac(
                        pt_world, pt_cam,
                        self.K_, 0*self.D_,
                        #useExtrinsicGuess = False,
                        useExtrinsicGuess = True,
                        rvec=rvec0,
                        tvec=tvec0,
                        **self.pPNP_
                        )

                suc, rvec, tvec, inliers = res
                if suc:
                    T = np.eye(4, dtype=np.float64)
                    T[:3,:3] = cv2.Rodrigues(rvec)[0]
                    T[:3,3:] = tvec.reshape(3,1)
                    T = tx.inverse_matrix(T)

                    pnp_p = T[2,3], -T[0,3]
                    pnp_h = -tx.euler_from_matrix(T)[1]
                    #print('pnp : {}, {}, ({}/{})'.format(
                    #    pnp_p, pnp_h, len(inliers), len(pt_world)))
                    msg += '(pnp:{}/{})'.format( len(inliers), len(pt_world))
                    # NOTE : uncomment this to revive pnp visualization
                    self.pnp_p_ = pnp_p
                    self.pnp_h_ = pnp_h
                else:
                    self.pnp_p_ = None
                    self.pnp_h_ = None


        # == visualize filtering process ==
        # n_show = 1
        # colors = np.random.uniform(size=(n_show,3))
        # show_landmark_2d(p_lm_0[:n_show], var_lm_old[:n_show],
        #         clear=True, draw=False,
        #         style='k.', colors=colors, label='lm_pre'
        #         )
        # show_landmark_2d(p_lm_v2_0[:n_show], var_lm_new[:n_show],
        #         clear=False, draw=False,
        #         style='r+', colors=colors, label='lm_obs'
        #         )
        # show_landmark_2d(
        #         self.landmarks_.pos_[midx][:n_show],
        #         self.landmarks_.var_[midx][:n_show],
        #         clear=False, draw=True,
        #         style='b*', colors=colors, label='lm_post'
        #         )
        # =================================

        # insert unselected landmarks

        # apply a lot more lenient matcher
        i1_lax, i2_lax = self.cvt_.des_des_to_match(
                self.landmarks_.des[lm_idx],
                des_p_m,
                lowe=1.0,
                maxd=128.0,
                cross=False
                )

        # update "invisible" landmarks that should have been visible
        lm_filter_msk = np.ones(len(lm_idx), dtype=np.bool)
        lm_filter_msk[i1_lax] = False
        self.landmarks_.cnt[lm_idx][i1_lax] -= 1

        if not h_override:
            # do not insert landmarks on h_override
            lm_sel_msk = np.zeros(len(des_p_m), dtype=np.bool)
            lm_sel_msk[i2_lax] = True
            lm_new_msk = ~lm_sel_msk
            lm_new_idx = np.where(lm_new_msk)[0]

            n_new = len(lm_new_idx)
            msk_n = np.ones(n_new, dtype=np.bool)

            if len(d_lm_old) > 0:
                # filter insertion by proximity to existing landmarks
                if len(lm_new_idx) > 0:
                    neigh = NearestNeighbors(n_neighbors=1)
                    neigh.fit(pt2_lm_c)
                    d, _ = neigh.kneighbors(pt2_u_c[idx_e][idx_r][lm_new_idx], return_distance=True)
                    msk_knn = (d < 16.0)[:,0] # TODO : magic number

                    # dist to nearest landmark, less than 20px
                    msk_n[msk_knn] = False
                    n_new = msk_n.sum()

            idx_n = np.where(msk_n)[0]

            print('adding {} landmarks : {}->{}'.format(n_new,
                len(self.landmarks_.pos), len(self.landmarks_.pos)+n_new
                ))
            pt3_new_c = scale * pt3[lm_new_idx][idx_n]

            pos_new = self.cvt_.cam_to_map(
                    pt3_new_c,
                    pose) # TODO : use rectified pose here if available

            des_new = des_p_m[lm_new_idx][idx_n]
            ang_new = np.full((n_new,1), pose[-1], dtype=np.float32)

            var_new = self.initialize_landmark_variance(
                    pt3_new_c,
                    pose)

            col_new = get_points_color(img_c, pt2_c[idx_t][idx_e][idx_r][lm_new_idx][idx_n], w=1)

            # append new landmarks ...
            # TODO : the problem here is that the landmarks are inserted eagerly,
            # and therefore interferes with pose rectification that requires
            # offsets to be consistent.

            li_0 = self.landmarks_.size_
            self.landmarks_.append(
                    pos_new, var_new,
                    des_new, ang_new,
                    col_new)#, kpt_new)
            li_1 = self.landmarks_.size_

            # NOTE : using undistorted version of pt2.
            # WARN : pt2_u_c = undistort( pt2_c[idx_t])
            # for whatever reason, indexing was somewhat messed up.
            self.cache_BA(
                    lmk_idx=np.arange(li_0, li_1),
                    lmk_pt2=pt2_u_c[idx_e][idx_r][lm_new_idx][idx_n]
                    )

        return scale, msg

    def pRt2pose(self, p, R, t):
        x, y, h = p

        dh = tx.euler_from_matrix(R)[2]
        dx = np.float32([t[0], t[1]])

        c, s = np.cos(h), np.sin(h)
        R2_p = np.reshape([c,-s,s,c], [2,2]) # [2,2,N]
        dp = R2_p.dot(dx).ravel()

        x_c = x+dp[0]
        y_c = y+dp[1]
        h_c = (h + dh + np.pi) % (2*np.pi) - np.pi

        return np.float32([x_c,y_c,h_c])

    """ all BA Stuff """
    def cache_BA(self, lmk_idx, lmk_pt2):
        if not (self.flag_ & ClassicalVO.VO_USE_BA):
            return
        # called for both newly registered landmarks
        # and re-observed landmarks.

        i = len(self.ba_pos_) # current pose index for BA
        n = len(lmk_idx)

        # --> convert to [i,i,i,i,...], [l0,l1,l2,...], [pt0,pt1,...]
        c_i = np.full(n, i) # == (n,)
        l_i = lmk_idx # == (n,)
        pt2 = lmk_pt2 # == (n,2) array of points
        # NOTE : prefer to pass in undistorted points to avoid
        # repeated applications of distortions in reprojections

        self.ba_ci_.append( c_i )
        self.ba_li_.append( l_i )
        self.ba_p2_.append( pt2 )

    def project_BA(self, cam, lmk, return_msk=False):
        """
        cam = np.array(Nx3) camera 2d pose (x,y,h) (WARN: actually base_link pose)
        lmk = np.array(Nx3) landmark position (x,y,z) in <map> coordinates
        """
        n = len(cam)

        x = cam[:,0]
        y = cam[:,1]
        h = cam[:,2]

        # z-axis heading
        c = np.cos(h)
        s = np.sin(h)

        # directly construct batchwise T_o2b
        T_o2b = np.zeros((n,4,4), dtype=np.float32)

        # Rotation Part
        T_o2b[:,0,0] = c
        T_o2b[:,0,1] = s
        T_o2b[:,1,0] = -s
        T_o2b[:,1,1] = c
        T_o2b[:,2,2] = 1

        # Translation part
        T_o2b[:,0,3] = -y*s - x*c
        T_o2b[:,1,3] = x*s - y*c

        # Homogeneous part
        T_o2b[:,3,3] = 1

        lmk_h = self.cvt_.pt_to_pth(lmk)
        pt2_h = reduce(np.matmul,[
            self.K_,
            self.cvt_.T_b2c_[:3],
            T_o2b,
            self.cvt_.T_c2b_,
            lmk_h[...,None]])[...,0]
        pt2 = self.cvt_.pth_to_pt(pt2_h)
        if return_msk:
            # NOTE: mask only implements depth check,
            # as out-of-bounds pixel coordinates are still meaningful
            # for bundle adjustment.
            msk = (pt2_h[...,-1] > 0.0)
            return pt2, msk
        return pt2

    def residual_BA(self, params,
            n_camera, n_landmark,
            c_i, l_i,
            obs_pt2):
        pos = params[:n_camera*3].reshape(-1, 3) # camera 2d pose (x,y,h)
        lmk = params[n_camera*3:].reshape(-1, 3) # landmark positions
        # c_i = [N_obs] array of camera indices
        # l_i = [N_obs] array of landmark indices
        # obs_pt2 = [N_obsx3] array of projected landmark points
        # pos[c_i] --> [N_obsx3]
        # lmk[c_i] --> [N_obsx3]

        prj_pt2, msk = self.project_BA(pos[c_i], lmk[l_i], return_msk=True)
        err = (prj_pt2 - obs_pt2)
        # TODO : is it actually necessary to apply the mask?
        #i_null = np.where(~msk)[0]
        #err[i_null] = 0
        return err.ravel()

    def sparsity_BA(self, n_c, n_l, ci, li):
        m = len(ci) * 2 # number of observations x projected observation size
        n = n_c * 3 + n_l * 3 # number of parameters
        A = lil_matrix((m,n), dtype=int) # TODO: dtype=bool?

        ci3 = ci*3
        li3 = li*3 # offsets for interleaving

        i = np.arange(len(ci))
        for s in range(3):
            A[2*i,   ci3+s] = 1
            A[2*i+1, ci3+s] = 1
            A[2*i,   li3+(n_c*3+s)] = 1
            A[2*i+1, li3+(n_c*3+s)] = 1
        # from here, the constraint is expressed
        # that the format of parameters is
        # [cx0,cy0,ch0, cx1,cy1,ch1, ... lx0,ly0,lz0, lx1,ly1,lz1, ...]
        # and the format of output is
        # [px0,py0,px1,py1, ...] (flattened points2d array)
        return A

    def jac_BA(self, params,
            n_camera, n_landmark,
            c_i, l_i,
            obs_pt2):
        """
        Currently unused due to being slow;
        Also, I'm not quite sure if the current implementation is correct.
        TODO : optimize and/or validate
        """
        pos = params[:n_camera*3].reshape(-1, 3) # camera 2d pose (x,y,h)
        lmk = params[n_camera*3:].reshape(-1, 3) # landmark positions
        J = ba_J(pos[c_i], lmk[l_i], self.K_, self.cvt_.T_b2c_, self.cvt_.T_c2b_)

        n_obs = len(c_i)
        J_res = np.zeros((2*n_obs, len(params))).reshape(
                2*n_obs, -1, 3)

        print len(c_i), n_obs
        J_res[:, c_i, :] += J[:, :3*n_obs].reshape(2*n_obs,-1,3)
        J_res[:, l_i, :] += J[:, 3*n_obs:].reshape(2*n_obs,-1,3)
        return J_res.reshape(2*n_obs, -1)

    def run_BA(self):
        """
        Sources:
            https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html
            https://github.com/jahdiel/pySBA/blob/master/PySBA.py
        """

        if not self.flag_ & ClassicalVO.VO_USE_BA:
            return

        if len(self.ba_ci_) <= 0 or len(self.ba_pos_) <= 0:
            # unable to run BA (NO DATA!)
            return

        # create np arrays
        p0 = np.stack(self.ba_pos_, axis=0)
        ci = np.concatenate(self.ba_ci_, axis=0)
        li = np.concatenate(self.ba_li_, axis=0)
        p2 = np.concatenate(self.ba_p2_, axis=0)

        # gather stats
        n_c = len(self.ba_pos_)
        n_l = self.landmarks_.size_

        # filter by landmarks that were actually observed
        # TODO : filter by cnt>=2?
        li_u, li = np.unique(li, return_inverse=True) # WARN: li override
        n_l = len(li_u)
        x0 = np.concatenate([
            p0.ravel(),
            self.landmarks_.pos[li_u].ravel()
            ])

        # compute BA sparsity structure
        A  = self.sparsity_BA(n_c, n_l, ci, li)

        # actually run BA
        # TODO : use x_scale='jac'???
        res = least_squares(
                self.residual_BA, x0,
                jac_sparsity=A, verbose=2,
                ftol=1e-4,
                #x_scale='jac', # -- landmark @ pose should be about equivalent
                method='trf',
                args=(n_c, n_l, ci, li, p2) )

        # format ...
        pos_opt = res.x[:n_c*3].reshape(-1,3)
        lmk_opt = res.x[n_c*3:].reshape(-1,3)

        #print('initial latest pose',  self.ba_pos_[-1])
        #print('optimized latest pose', pos_opt[-1])

        # apply BA results
        # TODO : revive this
        self.landmarks_.pos[li_u] = lmk_opt

        # reset BA cache
        self.ba_pos_ = []
        self.ba_ci_ = []
        self.ba_li_ = []
        self.ba_p2_ = []

        return pos_opt

    def run_ukf(self, dt, scale=None):
        """ motion-based prediction """
        pose_p = self.ukf_l_.x[:3].copy()
        Q, R = get_QR(pose_p, dt)
        self.ukf_l_.Q = Q
        self.ukf_l_.R = R
        self.ukf_l_.predict(dt)
        self.ukf_dt_.append(dt)
        pose_c = self.ukf_l_.x[:3].copy()
        if scale is None:
            # initialize scale from UKF
            scale = np.linalg.norm(pose_c[:2] - pose_p[:2])
        return pose_p, pose_c, scale

    def run_gp(self, pt_c, pt_p, pt3=None, scale=None):
        """
        Scale estimation based on locating the ground plane.
        if scale:=None, scale based on best z-plane will be returned.
        """
        if not (self.flag_ & ClassicalVO.VO_USE_SCALE_GP):
            return scale
        camera_height = self.T_c2b_[2, 3]

        if self.flag_ & ClassicalVO.VO_USE_GP_RSC:
            # opt1 : estimate ground-plane for projection
            # unfortunately, there's far too few points on the ground plane
            # to compute a reasonable estimate.

            # camera pitch filter (NOTE: not 100% robust, but works in 2D)
            y_min = np.linalg.multi_dot([
                self.K_,
                self.cvt_.T_b2c_[:3],
                np.reshape([1,0,0,1], (4,1))])[1]

            gp_msk = np.logical_and.reduce([
                pt_c[:,1] >= y_min,
                pt_p[:,1] >= y_min])

            gp_idx = np.where(gp_msk)[0]

            # NOTE: debug; show gp plane points correspondences
            # gp_idx = np.random.choice(gp_idx, size=32)
            # try:
            #     fig = self.gfig_
            #     ax  = fig.gca()
            # except Exception:
            #     self.gfig_ = plt.figure()
            #     ax = self.gfig_.gca()
            # col = np.random.uniform(0.0, 1.0, size=(len(gp_idx), 3)).astype(np.float32)
            # ax.cla()
            # ax.scatter(pt_c[gp_idx,0], pt_c[gp_idx,1], color=col)
            # ax.scatter(pt_p[gp_idx,0], pt_p[gp_idx,1], color=col)
            # if not ax.yaxis_inverted():
            #     ax.invert_yaxis()

            # ground plane is a plane, so homography can (and should) be applied here
            H, msk_h = cv2.findHomography(pt_c[gp_idx], pt_p[gp_idx],
                    method=self.pEM_['method'],
                    ransacReprojThreshold=self.pEM_['threshold']
                    )
            idx_h = np.where(msk_h)[0]

            print 'Ground Plane Homography : {}/{}'.format(msk_h.sum(), msk_h.size)

            #Hr, Ht, Hn = self.cvt_.H_to_Rtn(H)
            #Hn = np.float32(Hn)
            #gp_z = (Hn[...,0].dot(self.T_c2b_[:3,:3].T))
            #print 'custom', gp_z
            #print np.ravel(Ht)

            res_h, Hr, Ht, Hn = cv2.decomposeHomographyMat(H, self.K_)
            Hn = np.float32(Hn)
            Ht = np.float32(Ht)
            Ht /= np.linalg.norm(Ht, axis=1, keepdims=True) # NOTE: Ht is N,3,1
            gp_z = (Hn[...,0].dot(self.T_c2b_[:3,:3].T))

            # filter by estimated plane z-norm
            z_val = ( np.abs(np.dot(gp_z, [0,0,1])) > 0.9 )
            z_idx = np.where(z_val)[0]
            # NOTE: honestly don't know why I need to pre-filter by z-norm at all
            perm = zip(Hr,Ht)
            perm = [perm[i] for i in z_idx]
            n_in, R, t, msk_r, gpt3, sel = recover_pose_from_RT(perm, self.K_,
                    pt_c[idx_h], pt_p[idx_h], return_index=True, log=False)
            gpt3 = gpt3.T

            # convert w.r.t base_link
            gpt3_base = gpt3.dot(self.cvt_.T_c2b_[:3,:3].T)
            h_gp = robust_mean(-gpt3_base[:,2])
            scale_gp = (camera_height / h_gp)
            print 'gp-ransac scale', scale_gp
            scale = scale_gp
        else:
            # opt2 : directly estimate ground plane by simple height filter

            # only apply rotation: pt3_base still w.r.t camera offset @ base orientation
            pt3_base = pt3.dot(self.cvt_.T_c2b_[:3,:3].T)

            dh_thresh = 0.1
            gp_msk = np.logical_and.reduce([
                pt3_base[:,2] < (-camera_height + dh_thresh) / scale, # only filter for down-ness
                (-camera_height -dh_thresh)/scale < pt3_base[:,2], # sanity check with large-ish height value
                pt3_base[:,0] < 50.0 / scale  # sanity check with large-ish depth value
                ])
            gp_idx = np.where(gp_msk)[0]
            pt_gp = pt3_base[gp_idx]

            print 'gp inl : {}/{}'.format(len(gp_idx), gp_msk.size)

            if len(gp_idx) > 3: # at least 3 points
                h_gp = robust_mean(-pt_gp[:,2])
                if not np.isnan(h_gp):
                    scale_gp = camera_height / h_gp
                    print 'scale_gp : {:.4f}/{:.4f}={:.2f}%'.format(scale_gp, scale,
                            100 * scale_gp / scale)
                    # use gp scale instead
                    scale = scale_gp
        return scale

    def __call__(self, img, dt, scale=None):
        msg = ''
        # suffix designations:
        # o/0 = origin (i=0)
        # p = previous (i=t-1)
        # c = current  (i=t)

        # process current frame
        # TODO : enable lazy evaluation
        # (currently very much eager)

        img_c = img
        kpt_c = self.cvt_.img_to_kpt(img_c,
                subpix=(self.flag_ & ClassicalVO.VO_USE_KPT_SPX))
        kpt_c, des_c = self.cvt_.img_kpt_to_kpt_des(img_c, kpt_c)

        # update history
        self.hist_.append( [kpt_c, des_c, img_c] )
        if len(self.hist_) <= 1:
            return None
        # query data from previous time-frame
        # NOTE : -2 since -1 = current
        kpt_p, des_p, img_p = self.hist_[-2]

        # ukf
        pose_p, pose_c, scale = self.run_ukf(dt, scale)

        # frame-to-frame processing
        pt2_p = self.cvt_.kpt_to_pt(kpt_p)

        # == obtain next-frame keypoints ==
        if self.flag_ & ClassicalVO.VO_USE_TRACK:
            # opt1 : points by track
            pt2_c, idx_t = self.track(img_p, img_c, pt2_p)
        else:
            # opt2 : points by match
            i1, i2 = self.cvt_.des_des_to_match(des_p, des_c,
                    cross=(self.flag_ & ClassicalVO.VO_USE_MXCHECK)
                    )
            msk_t = np.zeros(len(pt2_p), dtype=np.bool)
            msk_t[i1] = True
            pt2_c = np.zeros_like(pt2_p)
            pt2_c[i1] = self.cvt_.kpt_to_pt(kpt_c[i2])

        # apply additional constraints
        # TODO : evaluate if the >1px constraint is necessary
        # msk_d = (np.max(np.abs(pt2_p - pt2_c), axis=-1) > 1.0) # enforce >1px difference
        # msk_t &= msk_d
        print('track : {}/{}'.format(len(idx_t), len(pt2_p)))
        # TODO : also track landmark points?
        # =================================

        pt2_u_p = self.cvt_.pt2_to_pt2u(pt2_p[idx_t])
        pt2_u_c = self.cvt_.pt2_to_pt2u(pt2_c[idx_t])

        # NOTE : experimental
        if self.flag_ & ClassicalVO.VO_USE_FM_COR:
            # correct Matches by RANSAC consensus
            # NOTE : cannot apply undistort() after correction
            F, msk_f = cv2.findFundamentalMat(
                    pt2_u_c,
                    pt2_u_p,
                    method=self.pEM_['method'],
                    param1=self.pEM_['threshold'],
                    param2=self.pEM_['prob'],
                    )
            msk_f = msk_f[:,0].astype(np.bool)
            idx_f = np.where(msk_f)[0]
            idx_t = idx_t[idx_f]

            pt2_u_c, pt2_u_p = cv2.correctMatches(F,
                    pt2_u_c[idx_f][None,...],
                    pt2_u_p[idx_f][None,...])
            pt2_u_c = np.squeeze(pt2_u_c, axis=0)
            pt2_u_p = np.squeeze(pt2_u_p, axis=0)

        # opt 1 : essential
        E, msk_e = cv2.findEssentialMat(pt2_u_c, pt2_u_p, self.K_,
                **self.pEM_)
        msk_e = msk_e[:,0].astype(np.bool)
        idx_e = np.where(msk_e)[0]
        print('e_in : {}/{}'.format(len(idx_e), msk_e.size))
        F = self.cvt_.E_to_F(E)

        if self.flag_ & ClassicalVO.VO_USE_HOMO:
            # opt 2 : homography
            H, msk_h = cv2.findHomography(pt2_u_c, pt2_u_p,
                    method=self.pEM_['method'],
                    ransacReprojThreshold=self.pEM_['threshold']
                    )
            msk_h = msk_h[:,0].astype(np.bool)
            idx_h = np.where(msk_h)[0]
            print('h_in : {}/{}'.format(len(idx_h), msk_h.size))

            # compare errors
            sH, msk_sh = score_H(pt2_u_c[idx_h], pt2_u_p[idx_h], H, self.cvt_)
            sF, msk_sf = score_F(pt2_u_c[idx_e], pt2_u_p[idx_e], F, self.cvt_)

            r_H = (sH / (sH + sF))
            print('score determinant : {} -------------------------------> {}'.format(r_H, 'H' if r_H > 0.45 else 'E'))

        h_override = False
        if self.flag_ & ClassicalVO.VO_USE_HOMO:
            h_override = (r_H > 0.45)# and ( len(idx_h) > len(idx_e) )

        if h_override:
            ## homography
            #idx_h = idx_h[msk_sh]
            res_h, Hr, Ht, Hn = cv2.decomposeHomographyMat(H, self.K_)
            print Hr[0], Ht[0], np.linalg.norm(Hr[0]), np.linalg.norm(Ht[0])
            Ht = np.float32(Ht)

            perm = zip(Hr,Ht)
            n_in, R, t, msk_r, pt3 = recover_pose_from_RT(perm, self.K_,
                    pt2_u_c[idx_h], pt2_u_p[idx_h], log=False)
            #t /= np.linalg.norm(t)
            print('homography : {}/{}/{}'.format(n_in, len(idx_h), len(msk_h) ))

            # TODO : fix legacy variable name
            idx_e = idx_h
        else:
            #idx_e = idx_e[msk_sf]
            n_in, R, t, msk_r, pt3 = recover_pose(E, self.K_,
                    pt2_u_c[idx_e], pt2_u_p[idx_e], log=False,
                    #z_min = 0.01 / scale,
                    #z_max = 100.0 / scale
                    #z_max = 5000.0
                    # = usually ~10m
                    )
            print('essentialmat : {}/{}/{}'.format(n_in, len(idx_e), len(msk_e) ))

        # convert R,t to base_link frame
        R = np.linalg.multi_dot([
            self.cvt_.T_c2b_[:3,:3],
            R,
            self.cvt_.T_c2b_[:3,:3].T
            ])
        t = np.linalg.multi_dot([
            self.cvt_.T_c2b_[:3,:3],
            t.reshape(3,1),
            ]).ravel()

        idx_r = np.where(msk_r)[0]
        pt3 = pt3.T
        print('triangulation : {}/{}'.format(len(idx_r), msk_r.size))

        scale = self.run_gp(pt2_u_c[idx_e], pt2_u_p[idx_e], pt3, scale)

        msk = np.zeros(len(pt2_p), dtype=np.bool)
        msk[idx_t[idx_e[idx_r]]] = True
        #print('final msk : {}/{}'.format(msk.sum(), msk.size))

        mim = drawMatches(img_p, img_c, pt2_p, pt2_c, msk)

        # dh/dx in pose_p frame
        pose_c_r = self.pRt2pose(pose_p, R, scale*t)

        if self.flag_ & ClassicalVO.VO_USE_F2M:
            # TODO : smarter way to incorporate ground-plane scale information??
            # estimate scale based on current pose guess
            scale, msg = self.proc_f2m(pose_c_r, scale,
                    des_p, des_c,
                    idx_t, idx_e, idx_r,
                    pt2_u_p, pt2_u_c,
                    pt3,
                    img_c, pt2_c,
                    h_override,
                    msg
                    )
            # recompute rectified pose_c_r
            pose_c_r = self.pRt2pose(pose_p, R, scale*t)

        self.ukf_l_.update(pose_c_r)
        pose_c_r = self.ukf_l_.x[:3].copy() # TODO : chronologically correct?

        # NOTE!! this vvvv must be called after all cache_BA calls
        # have been completed.
        if self.flag_ & ClassicalVO.VO_USE_BA:
            self.ba_pos_.append( pose_c_r.copy() ) # cache BA
            if len(self.ba_pos_) >= self.ba_freq_: # TODO: configure BA frame
                # run BA every 16 frames
                ba_res = self.run_BA()
                if ba_res is not None:
                    # naive version : just take last value

                    # update global ukf
                    for (g_dt, g_z) in zip(
                            self.ukf_dt_[-self.ba_freq_:],
                            ba_res):
                        ukf_Q, ukf_R = get_QR(self.ukf_g_.x[:3], dt)
                        self.ukf_g_.Q = ukf_Q
                        self.ukf_g_.R = ukf_R
                        self.ukf_g_.predict(g_dt)
                        self.ukf_g_.update(g_z)

                    # copy to local ukf
                    self.ukf_l_.x = self.ukf_g_.x.copy()
                    self.ukf_l_.P = self.ukf_g_.P.copy()

                    # clear history (currently, just time data)
                    self.ukf_dt_ = []

                    # result
                    pose_c_r = self.ukf_l_.x[:3].copy()

                # TODO : currently pruning happens with BA
                # in order to not mess up landmark indices.
                self.landmarks_.prune_nmx()
                #self.landmarks_.prune()

        print('\t\t pose-f2f : {}'.format(pose_c_r))
        ## === FROM THIS POINT ALL VIZ === 

        # construct visualizations

        # = opt1 : show currently tracked points =
        # pt3_m = self.cvt_.cam_to_map(pt3 * scale, pose)
        # pt2_c_rec, _ = self.cvt_.pt3_pose_to_pt2_msk(pt3_m, pose)

        # = opt2 : show landmark instead =
        # override pt3_m with all currently tracked landmarks
        # filter by 'high' confidence
        #pt3_lm_c = self.cvt_.map_to_cam(self.landmarks_.pos_, pose)
        #d_lm_c   = np.linalg.norm(pt3_lm_c, axis=-1)
        #v_xy = self.landmarks_.var_[:, (2,1), (2,1)]
        #s_xy = np.linalg.norm(np.sqrt(v_xy), axis=-1)
        #s_xy_r = s_xy / d_lm_c # relative conf.
        #idx  = np.argsort(s_xy_r)
        #pt3_m = self.landmarks_.pos_[idx[:512]]
        pt3_m = self.landmarks_.pos
        col_m = self.landmarks_.col

        # filter by height
        # convert to base_link coordinates
        # pt3_m_b = pt3_m.dot(self.T_c2b_[:3,:3].T) + self.T_c2b_[:3,3]
        #pt3_viz_msk = np.logical_and.reduce([
        #    -0.2 <= pt3_m_b[:,2],
        #    pt3_m_b[:,2] < 5.0])
        #pt3_m = pt3_m[pt3_viz_msk]
        #col_m = col_m[pt3_viz_msk]

        # NOTE: pt2_c_rec will not be 100% accurate of the tracked positions,
        # as (due to possible accuracy reasons) distortions have been disabled.
        pt2_c_rec, front_msk = self.project_BA(np.asarray([pose_c_r]), pt3_m,
                return_msk=True
                )
        rec_msk = np.logical_and.reduce([
            front_msk,
            0 <= pt2_c_rec[:,0],
            pt2_c_rec[:,0] < 640,
            0 <= pt2_c_rec[:,1],
            pt2_c_rec[:,1] < 480
            ])
        #pt2_c_rec, rec_msk = self.cvt_.pt3_pose_to_pt2_msk(pt3_m, pose_c_r, distort=False)
        rec_idx = np.where(rec_msk)[0]
        pt2_c_rec = pt2_c_rec[rec_idx]
        pt3_m = pt3_m[rec_idx]
        col_m = col_m[rec_idx]
        
        #if False: # == VIZ_ALL
        #    # sort points by variance?
        #    lmk_var = np.linalg.norm(
        #            self.landmarks_.var[:, (0,1,2), (0,1,2)],
        #            axis=-1)
        #    lm_idx_s = np.argsort(-lmk_var)
        #    # small variance listed last
        #    # hopefully also gets drawn last
        #    pt3_m = pt3_m[lm_idx_s]
        #    col_m = col_m[lm_idx_s]
        #elif False: # == VIZ_FOV
        #    # filter by visibility
        #    pt3_m = pt3_m[rec_idx]
        #    col_m = col_m[rec_idx]
        #elif False: # == VIZ_SUMSAMPLE
        #    # subsample points to show
        #    n_show = min(len(pt3_m), 128)
        #    if n_show <= 0:
        #        sel = np.empty(0, dtype=np.int32)
        #    else:
        #        # opt1 : random
        #        sel = np.random.choice(len(pt3_m), size=n_show, replace=(len(pt3_m) > n_show))
        #        # opt2 : high confidence
        #        #lmk_var = np.linalg.norm(
        #        #        self.landmarks_.var[:, (0,1,2), (0,1,2)],
        #        #        axis=-1)
        #        #idx = np.argsort(lmk_var)
        #        #print 'idx', idx
        #        #sel = idx[:n_show]
        #    pt3_m = pt3_m[sel]
        #    col_m = col_m[sel]
        # ================================

        # convert to base_link coordinates
        pt3_m = pt3_m.dot(self.T_c2b_[:3,:3].T) + self.T_c2b_[:3,3]

        # TODO : propagate status messages for the GUI
        # namely, track status, pnp status(?),
        # ground-plane projection status,
        # landmark correspondence scale estimation status,
        # landmark updates (#additions), etc.

        return [mim, pose_c_r, pt2_c_rec, pt3_m, col_m, msg]

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
