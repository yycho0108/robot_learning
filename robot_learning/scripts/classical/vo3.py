"""
Semi-Urgent TODOs:
    - memory management (masks/fancy indexing creates copies; reuse same-sized arrays etc.)
"""
from collections import namedtuple, deque
from filterpy.kalman import InformationFilter
from tf import transformations as tx
import cv2
import numpy as np

from vo_common import recover_pose, drawMatches
from vo_common import robust_mean, oriented_cov, show_landmark_2d
from vo_common import Landmarks, Conversions
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D

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
    cols = np.sqrt(np.mean(np.square(cols),axis=(1,2)))
    #cols = np.linalg.norm(cols_w, axis=(1,2))


    # opt 2 : rms
    #cols = np.sqrt(np.mean(np.square(cols_w),axis=(1,2)))
    #print 'stat-pre', cols_w.std(axis=(0,1,2))
    #cols = np.linalg.norm(cols_w, axis=(1,2))
    #csq  = np.square(cols_w) #R
    #csum = np.mean(csq, axis=(1,2)) #M
    #cols = np.sqrt(csum) #S
    #print 'stat-post', cols.std(axis=0)
    #cols = np.linalg.norm(cols_w, axis=(1,2)).astype(np.float32) # n,3

    #for oi in range(-w,w+1):
    #    for oj in range(-w,w+1):
    #        img[pis-oi,pjs-oj]
    #cs = []
    #for (pi,pj) in zip(pis, pjs):
    #    c = np.linalg.norm(img[pi-w:pi+w, pj-w:pj+w], axis=(0,1))
    #    cs.append(c)

    # vectorized method
    #cs = np.clip(cs, 0, 255) # TODO : evaluate if necessary
    #print 'cmax(pre)', cols_w.max()
    #print 'cmax(post)', cols.max()
    return np.asarray(cols, dtype=img.dtype)

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
                angles=[-np.pi/2 - np.deg2rad(10),0.0,-np.pi/2],
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
                nfeatures=4096,
                scaleFactor=1.2,
                nlevels=8,
                scoreType=cv2.ORB_FAST_SCORE,
                )
        det = cv2.FastFeatureDetector_create(
                threshold=20, # I think this is the default
                nonmaxSuppression=True
                )
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
        #self.landmarks_ = []
        self.landmarks_ = Landmarks()
        self.hist_ = deque(maxlen=100)

        self.pnp_p_ = None
        self.pnp_h_ = None

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
        # TODO : ^ is a ballpart estimate.
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
            msg
            ):

        # build index combinationss
        idx_te = idx_t[idx_e]
        idx_ter = idx_te[idx_r]
        idx_er = idx_e[idx_r]

        # frame-to-map processing
        # (i.e. uses landmark data)

        if len(self.landmarks_.pos) > 0:
            # enter landmark processing
            # TODO : add preliminary filter by view angle
            pt2_lm_c, lm_msk = self.cvt_.pt3_pose_to_pt2_msk(
                    self.landmarks_.pos, pose)
            # note that pt2_lm_c is undistorted.
            print('visible landmarks : {}/{}'.format(lm_msk.sum(), lm_msk.size))
        else:
            lm_msk = np.ones((0), dtype=np.bool)

        lm_idx = np.where(lm_msk)[0]

        # select useful descriptor based on current viewpoint
        des_p_m = des_p[idx_ter]

        i1, i2 = self.cvt_.des_des_to_match(
                self.landmarks_.des[lm_idx],
                des_p_m)

        if len(lm_idx) > 16:
            # filter correspondences by Emat consensus

            # TODO : take advantage of the Emat here to some use?

            _, lm_msk_e = cv2.findEssentialMat(
                    pt2_lm_c[lm_idx][i1],
                    pt2_u_c[idx_er][i2],
                    self.K_,
                    **self.pEM_)
            lm_msk_e = lm_msk_e[:,0].astype(np.bool)
            lm_idx_e = np.where(lm_msk_e)[0]

            #cor_delta = (pt2_lm_c[lm_msk][i1] - pt2_u_c[msk_e][msk_r][i2])
            #cor_delta = np.linalg.norm(cor_delta, axis=-1)
            #lm_msk_e = (cor_delta < 64.0) # distance-based filter

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
        else:
            lm_msk_e = np.ones(len(i1), dtype=np.bool)
            lm_idx_e = np.where(lm_msk_e)[0]

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

        if False: # == USE_SCALE_A3D
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

        if True: # == if USE_LM_KF
            # update landmarks from computed correspondences
            p_lm_v2_c_s = p_lm_v2_c * scale_rel # apply est or rel ???
            var_lm_old = self.landmarks_.var[lm_idx][i1][lm_idx_e]
            var_lm_new = self.initialize_landmark_variance(p_lm_v2_c_s, pose)
            p_lm_v2_0 = self.cvt_.cam_to_map(p_lm_v2_c_s, pose)

            # apply kalman filter
            # (with transition & observation matrices F=I, H=I)
            y_k = (p_lm_v2_0 - p_lm_0).reshape(-1,3,1)
            S_k = var_lm_new + var_lm_old # I think R_k = var_lm_new (measurement noise)
            K_k = np.matmul(var_lm_old, np.linalg.inv(S_k))
            x_k = p_lm_0.reshape(-1,3,1) + np.matmul(K_k, y_k)
            I = np.eye(3)[None,...] # (1,3,3)
            P_k = np.matmul(I - K_k, var_lm_old)

            # now try to apply the same transform to to-be-inserted landmarks
            # cv2.estimateAffine3D(src, dst[, out[, inliers[, ransacThreshold[, confidence]]]])
            # --> retval, out, inliers

            midx = np.arange(len(self.landmarks_.pos))
            midx = midx[lm_idx][i1][lm_idx_e]

            self.landmarks_.pos[midx] = x_k[...,0]
            self.landmarks_.var[midx] = P_k

        if False: # == if USE_PNP
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
                maxd=128.0
                )

        lm_sel_msk = np.zeros(len(des_p_m), dtype=np.bool)
        lm_sel_msk[i2_lax] = True
        lm_new_msk = ~lm_sel_msk
        lm_new_idx = np.where(lm_new_msk)[0]

        n_new = len(lm_new_idx)
        msk_n = np.ones(n_new, dtype=np.bool)

        if len(d_lm_old) > 0:
            # filter insertion by proximity to existing landmarks
            neigh = NearestNeighbors(n_neighbors=1)
            neigh.fit(pt2_lm_c[lm_idx])
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
        self.landmarks_.append(
                pos_new, var_new,
                des_new, ang_new,
                col_new)
        return scale, msg

    def pRt2pose(self, p, R, t):
        x, y, h = p

        dh = -tx.euler_from_matrix(R)[1]
        #dx = s * np.float32([ np.abs(t[2]), 0*-t[1] ])
        dx = np.float32([ t[2], -t[1] ])

        c, s = np.cos(h), np.sin(h)
        R2_p = np.reshape([c,-s,s,c], [2,2]) # [2,2,N]
        dp = R2_p.dot(dx).ravel()

        x_c = x+dp[0]
        y_c = y+dp[1]
        h_c = (h + dh + np.pi) % (2*np.pi) - np.pi

        return np.float32([x_c,y_c,h_c])

    def __call__(self, img, pose, scale=1.0):
        msg = ''
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
        pt2_c, idx_t = self.track(img_p, img_c, pt2_p)

        # opt2 : points by match
        # i1, i2 = self.cvt_.des_des_to_match(des_p, des_c)
        # msk_t = np.zeros(len(pt2_p), dtype=np.bool)
        # msk_t[i1] = True
        # pt2_c = np.zeros_like(pt2_p)
        # pt2_c[i1] = self.cvt_.kpt_to_pt(kpt_c[i2])

        # apply additional constraints
        # TODO : evaluate if the >1px constraint is necessary
        # msk_d = (np.max(np.abs(pt2_p - pt2_c), axis=-1) > 1.0) # enforce >1px difference
        # msk_t &= msk_d
        # =================================

        #print 'mean delta', np.mean(pt2_c - pt2_p, axis=0) # -14 px
        print('track : {}/{}'.format(len(pt2_c), len(idx_t)))

        pt2_u_p = self.cvt_.pt2_to_pt2u(pt2_p[idx_t])
        pt2_u_c = self.cvt_.pt2_to_pt2u(pt2_c[idx_t])

        # NOTE : experimental
        if False:#True: # == USE_FM_COR
            # correct Matches
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

        E, msk_e = cv2.findEssentialMat(pt2_u_c, pt2_u_p, self.K_,
                **self.pEM_)
        msk_e = msk_e[:,0].astype(np.bool)
        idx_e = np.where(msk_e)[0]
        print('em : {}/{}'.format(len(idx_e), msk_e.size))

        n_in, R, t, msk_r, pt3 = recover_pose(E, self.K_,
                pt2_u_c[idx_e], pt2_u_p[idx_e], log=False,
                #z_min = 0.01 / scale,
                #z_max = 100.0 / scale
                #z_max = 5000.0
                # = usually ~10m
                )
        idx_r = np.where(msk_r)[0]
        #print( 'mr : {}/{}'.format(msk_r.sum(), msk_r.size))
        pt3 = pt3.T

        # == process ground plane ==
        camera_height = 0.113 # TODO : hardcoded
        pt3_base = pt3.dot(self.cvt_.T_c2b_[:3,:3].T)
        # only apply rotation: pt3_base still w.r.t camera offset @ base orientation

        # opt1 : directly estimate ground plane by filtering height
        dh_thresh = 0.1
        gp_msk = np.logical_and.reduce([
            pt3_base[:,2] < (-camera_height + dh_thresh) / scale, # only filter for down-ness
            (-camera_height -dh_thresh)/scale < pt3_base[:,2], # sanity check with large-ish height value
            pt3_base[:,0] < 50.0 / scale  # sanity check with large-ish depth value
            ])
        gp_idx = np.where(gp_msk)[0]
        pt_gp = pt3_base[gp_idx]

        # opt2 : estimate ground-plane for projection
        # unfortunately, there's far too few points on the ground plane
        # to compute a reasonable estimate.

        # dh_thresh = 0.3
        # gp_msk_lax = np.logical_and.reduce([
        #     pt3_base[:,0] < (10.0 / scale),
        #     pt3_base[:,2] < (-camera_height + dh_thresh) / scale,
        #     (-camera_height - dh_thresh)/scale < pt3_base[:,2]
        #     ])

        # # get mask from plane estimation
        # gp_fit, gp_err, gp_msk = estimate_plane_ransac(
        #         pt3_base[gp_msk_lax],
        #         1000,
        #         0.999,
        #         0.1 / scale, # ~ apply 10cm tolerance for ground
        #         nvec = np.float32([0.0, 0.0, 1.0])
        #         )
        # #print 'gp dist', gp_fit[0].dot(gp_fit[1])
        # #print 'gp nvec', gp_fit[1]
        # gp_msk = gp_msk[:,0].astype(np.bool)
        # pt_gp = pt3_base[gp_msk_lax][gp_msk]

        # visualize 
        #tmp = pt3_base[gp_msk_lax][gp_msk]
        #global tfig
        #if tfig is None:
        #    tfig = plt.figure()
        #    tax  = tfig.add_subplot(1,1,1, projection='3d')

        #tfig.gca().cla()
        #tfig.gca().plot(tmp[:,0],tmp[:,1],tmp[:,2],'.')
        #axisEqual3D(tfig.gca())
        #tfig.gca().set_xlabel('x')
        #tfig.gca().set_ylabel('y')
        #tfig.gca().set_zlabel('z')

        scale_gp = scale
        print 'gp inl : {}/{}'.format(len(gp_idx), gp_msk.size)
        if len(gp_idx) > 0:
            h_gp = robust_mean(-pt_gp[:,2])
            #h_gp = - gp_fit[0].dot(gp_fit[1])[0]
            #print 'gp height?', h_gp
            #h_gp = np.median(pt_gp[:,1])

            #print 'ground-plane {}/{}'.format(gp_msk.sum(), gp_msk.size)
            #print (pt_gp.min(axis=0), pt_gp.max(axis=0), pt_gp.mean(axis=0))
            if not np.isnan(h_gp):
                scale_gp = camera_height / h_gp
            print 'scale_gp : {:.4f}/{:.4f}={:.2f}%'.format(scale_gp, scale,
                    100 * scale_gp / scale)

            # use gp scale instead
            scale = scale_gp
        # ========================== 

        msk = np.zeros(len(pt2_p), dtype=np.bool)
        msk[idx_t[idx_e[idx_r]]] = True
        #print('final msk : {}/{}'.format(msk.sum(), msk.size))

        mim = drawMatches(img_p, img_c, pt2_p, pt2_c, msk)

        # dh/dx in pose_p frame
        pose_c_r = self.pRt2pose(pose_p, R, scale*t)
        if True:
            # TODO : smarter way to incorporate ground-plane scale information??

            # estimate scale based on current pose guess
            scale, msg = self.proc_f2m(pose_c_r, scale,
                    des_p, des_c,
                    idx_t, idx_e, idx_r,
                    pt2_u_p, pt2_u_c,
                    pt3,
                    img_c, pt2_c,
                    msg
                    )
            # recompute rectified pose_c_r
            pose_c_r = self.pRt2pose(pose_p, R, scale*t)

        x_c = pose_c_r[:2]
        h_c = pose_c_r[2]
            
        print('\t\t pose-f2f : {}.{}'.format(x_c, h_c))

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
        pt3_m = self.landmarks_.pos_
        col_m = self.landmarks_.col_

        # filter by height
        # convert to base_link coordinates
        # pt3_m_b = pt3_m.dot(self.T_c2b_[:3,:3].T) + self.T_c2b_[:3,3]
        #pt3_viz_msk = np.logical_and.reduce([
        #    -0.2 <= pt3_m_b[:,2],
        #    pt3_m_b[:,2] < 5.0])
        #pt3_m = pt3_m[pt3_viz_msk]
        #col_m = col_m[pt3_viz_msk]

        pt2_c_rec, rec_msk = self.cvt_.pt3_pose_to_pt2_msk(pt3_m, pose_c_r, distort=True)
        rec_idx = np.where(rec_msk)[0]

        # filter by visibility
        pt3_m = pt3_m[rec_idx]
        col_m = col_m[rec_idx]
        pt2_c_rec = pt2_c_rec[rec_idx]

        # subsample points to show
        n_show = min(len(pt3_m), 128)
        sel = np.random.choice(len(pt3_m), size=n_show, replace=(len(pt3_m) > n_show))
        pt3_m = pt3_m[sel]
        col_m = col_m[sel]

        # ================================

        # convert to base_link coordinates
        pt3_m = pt3_m.dot(self.T_c2b_[:3,:3].T) + self.T_c2b_[:3,3]

        # TODO : propagate status messages for the GUI
        # namely, track status, pnp status(?),
        # ground-plane projection status,
        # landmark correspondence scale estimation status,
        # landmark updates (#additions), etc.

        return True, (mim, h_c, x_c, pt2_c_rec, pt3_m, col_m, msg)

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
