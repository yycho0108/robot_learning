"""
Semi-Urgent TODOs:
    - memory management (masks/fancy indexing creates copies; reuse same-sized arrays etc.)
    - Loop Closure!!
    - Try to apply the homography model from ORB_SLAM??
    - Keyframes?
    - Incorporate Variance information in BA?
    - Cross-Frame (i.e. temporal displacement >1) Matching + Tracking
"""

from collections import namedtuple, deque, defaultdict
from tf import transformations as tx
import cv2
import numpy as np
import sys

from vo_common import recover_pose, drawMatches, recover_pose_from_RT
from vo_common import robust_mean, oriented_cov, show_landmark_2d
from vo_common import Landmarks, Conversions
from vo_common import print_Rt
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D

from scipy.sparse import lil_matrix
from scipy.optimize import least_squares

from ukf import build_ukf, build_ekf, get_QR
from ba import ba_J, ba_J_v2

def print_ratio(msg, a, b):
    as_int = np.issubdtype(type(a), np.integer)
    if b == 0:
        q = np.nan
    else:
        q = float(a) / b

    if as_int:
        print '{} : {}/{} = {:.2f}%'.format(
                msg, a, b, 100 * q)
    else:
        print '{} : {:.4f}/{:.4f} = {:.2f}%'.format(
                msg, a, b, 100 * q)

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


def get_points_color(img, pts, w=3):
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

class VGraph(object):
    """ Simple wrapper for visibility graph """
    # TODO : unify dynamic container interface or something.
    # (i.e. with Landmarks() )
    def __init__(self, cap0=1024):
        # list of poses (somewhat special)
        self.pose_ = []

        # pose index
        self.pi_ = np.empty( (cap0,), dtype=np.int32 )
        # landmark index
        self.li_ = np.empty( (cap0,), dtype=np.int32 )
        # point observation
        self.p2_ = np.empty( (cap0, 2), dtype=np.float32)

        self.size_     = 0
        self.capacity_ = cap0
        self.fields_ = ['pi_', 'li_', 'p2_']

        # TODO : handle persistent coloring more intelligently ...
        self.viz_cols_ = np.random.uniform(size=(16384,3))

    def resize(self, c_new):
        d = vars(self)
        c_old = self.size_
        for f in self.fields_:
            d_old = d[f][:c_old]
            s_old = d_old.shape
            
            s_new = (c_new,) + tuple(s_old[1:])
            d_new = np.empty(s_new, dtype=d[f].dtype)
            d_new[:c_old] = d_old[:c_old]

            d[f] = d_new
        self.capacity_ = c_new

    def append(self, li, p2):
        """
        Add observation to lmk[li]
        WARNING: .append() should be called AFTER add_pos().
        """
        # automatically figure out pose index.
        # TODO : validate
        pi = len(self.pose_)
        n = len(li) # NOTE: in general, cannot use len(pi).
        if self.size_ + n > self.capacity_:
            self.resize(self.capacity_ * 2)
            self.append(li, p2)
        else:
            i = np.s_[self.size_:self.size_+n]
            self.pi_[i] = pi
            self.li_[i] = li
            self.p2_[i] = p2
            self.size_ += n

    def prune(self, keep_idx):
        """ reindex graph with keep_idx """
        # naive iterative version
        #for (i_new, i_prv) in enumerate(keep_idx):
        #    self.li_[ np.where(self.li_ == i_prv)[0] ] = i_new

        # NOTE: lmk[li0[i0]] == lmk[keep_idx][i1]

        # TODO : is the below code super inefficient?
        keep_idx = np.int32(keep_idx)
        i0, i1 = np.where(self.li[:,None] == keep_idx[None, :])

        n = len(i0)
        self.li_[:n] = i1
        self.pi_[:n] = self.pi[i0].copy()
        self.p2_[:n] = self.p2[i0].copy()
        self.size_ = n

    def query(self, t, return_A=False):
        """ t = how much to look back """
        # assume (0...,1...,2...,3...,4..,5..,6...)
        # then self.pi[-1] = 6
        # if t = 3, then desired result is [4...,5...,6...]
        mx  = self.pi[-1]
        msk = (self.pi >= (mx+1-t))
        idx = np.where(msk)[0]

        p0 = np.asarray(self.pose_[-t:])
        pi = self.pi[idx] - self.pi[idx[0]] # return offset index for p0
        li = self.li[idx]
        p2 = self.p2[idx]

        if return_A:
            # validation: anticipated sparsity structure
            n_o = len( idx ) # number of observations
            n_p = len( p0  )
            n_l = 1 + li.max()   # assume un-pruned n_l
            n_x = n_p + n_l

            # raw sparsity matrix
            A = np.zeros((n_o,2,n_x,3), dtype=int)
            A[np.arange(n_o), :, pi, :] = 1
            A[np.arange(n_o), :, n_p+li, :] = 1

            # pruned sparsity matrix
            li_u = np.unique( li )
            i    = np.r_[:n_p, n_p+li_u]
            Ap = A[:, :, i, :] # pruned
            Ap = Ap.reshape(n_o*2, -1)

            return p0, pi, li, p2, Ap

        return p0, pi, li, p2

    def update(self, t, pos):
        self.pose_[-t:] = pos
    
    def add_pos(self, v):
        self.pose_.append(v)

    def draw(self, ax, cvt, lmk):
        p0, pi, li, p2 = self.query(t=len(self.pi))

        # convert to map coord
        lmk_m = lmk.pos.dot(cvt.T_c2b_[:3,:3].T) + cvt.T_c2b_[:3,3:].T
        lmk_m = lmk_m[:, :2] # extract x-y indices

        # count cutoff
        # filter by the number of li appearances in history
        # most "well-observed" landmarks
        li_u, li_i, cnt = np.unique(li, return_inverse=True, return_counts=True)

        # filter by > 16 connections
        # TODO : probably requires better filtering for better visualization
        # idx = np.where( cnt >= np.percentile(cnt, 99.0) )[0]

        #if len(idx) >= 32:
        #    idx = np.random.choice(idx, 32)

        #if len(idx) <= 0:
        #    # fallback to best 32
        #    idx = np.argsort(cnt)[-32:] # or other heuristics
        idx = np.argsort(cnt)[-32:] # or other heuristics

        _, c_idx = np.where(li_i[None,:] == idx[:,None])

        # apply cutoff
        p = p0[pi][c_idx]
        l = lmk_m[li][c_idx]
        pcol = lmk.col[li][c_idx]
        col = self.viz_cols_[li][c_idx]

        # validation : x-y reprojection check
        #x = p2[c_idx]
        #y = [cvt.pt3_pose_to_pt2_msk(lmk.pos[li][c_idx], p_)[0][i]
        #        for (i,p_) in enumerate(p)]
        #print 'x', np.int32(x).reshape(-1,2)
        #print 'y', np.int32(y).reshape(-1,2)

        # where they are 'supposed to be'
        # Nx2, image coordinates
        g = p2[c_idx]
        # Nx3, camera coordinates
        g = cvt.pt_to_pth(g).dot(cvt.Ki_.T)
        # Nx3, Normalize + apply depth
        g /= np.linalg.norm(g, axis=-1, keepdims=True)
        d = np.linalg.norm(l[:,:2] - p[:,:2], axis=-1, keepdims=True)
        g *= d
        # Nx3, camera->base coordinates
        g = g.dot(cvt.T_c2b_[:3,:3].T) + cvt.T_c2b_[:3,3:].T
        # Construct Z-axis rotation matrix
        h = p[:,2]
        c, s = np.cos(h), np.sin(h)
        Rz = np.zeros((len(c),3,3), dtype=np.float32)
        Rz[:,0,0] = c
        Rz[:,0,1] = -s
        Rz[:,1,0] = s
        Rz[:,1,1] = c # Nx3x3
        Rz[:,2,2] = 1
        # Nx3, base -> map coordinates
        g = np.matmul(Rz, g[...,None])[...,0]

        ax.quiver(
                p[:,0], p[:,1],
                l[:,0]-p[:,0], l[:,1]-p[:,1],
                angles='xy',
                scale=1.0,
                scale_units='xy',
                color=col,
                width=0.005,
                alpha=0.25
                )
        ax.quiver(
                p[:,0], p[:,1],
                g[:,0], g[:,1],
                angles='xy',
                scale=1.0,
                scale_units='xy',
                color=col,
                linestyle=':',
                dashes=(0,(10, 20)),
                width=0.0025,
                alpha=0.5,
                edgecolor=col,
                facecolor='none',
                linewidth=1
                )
        ax.plot(p[:,0], p[:,1], 'ro', fillstyle='none')
        #ax.plot(l[:,0], l[:,1], 'bo', fillstyle='none')
        ax.scatter(l[:,0], l[:,1],
                color=pcol[...,::-1] / 255.0,
                marker='^'
                )
        ax.set_aspect('equal', 'datalim')

    @property
    def pi(self):
        return self.pi_[:self.size_]
    @property
    def li(self):
        return self.li_[:self.size_]
    @property
    def p2(self):
        return self.p2_[:self.size_]
    @property
    def index(self):
        return len(self.pose_)

class StateHistory(object):
    """
    Circular Buffer of State History.
    Minor Thing : Will NOT reject invalid queries,
    such as ones without sufficient initialization
    or overflow.
    """
    def __init__(self, maxlen=32):
        self.i_ = 0
        self.n_ = maxlen
        self.x_ = np.empty((maxlen, 6))
        self.P_ = np.empty((maxlen, 6, 6))
        self.dt_ = np.empty((maxlen,))

    def query(self, t):
        i = np.arange(self.i_-t, self.i_) % self.n_
        return self.x_[i], self.P_[i], self.dt_[i]

    def update(self, t, x, P, dt):
        i = np.arange(self.i_-t, self.i_) % self.n_
        self.x_[i] = x
        self.P_[i] = P
        self.dt_[i] = dt

    def append(self, x, P, dt):
        i = self.i_
        self.x_[i] = x
        self.P_[i] = P
        self.dt_[i] = dt
        self.i_ = (self.i_ + 1) % self.n_

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
            VO_USE_BA | VO_USE_F2M | VO_USE_LM_KF | \
            VO_USE_KPT_SPX | VO_USE_MXCHECK | VO_USE_GP_RSC

    def __init__(self, cinfo=None):
        # define configuration
        self.flag_ = ClassicalVO.VO_DEFAULT
        self.flag_ &= ~ClassicalVO.VO_USE_HOMO # TODO : doesn't really work?
        #self.flag_ &= ~ClassicalVO.VO_USE_BA
        #self.flag_ |= ClassicalVO.VO_USE_PNP
        #self.flag_ &= ~ClassicalVO.VO_USE_FM_COR # performance

        # TODO : control stage-level verbosity

        # Note that camera intrinsic+extrinsic parameters
        # i.e. K, D, T_c2b
        # are coupled with the data, rather than the algorithm.
        if cinfo is not None:
            self.K_ = cinfo['K']
            self.D_ = cinfo['D']
            self.T_c2b_ = cinfo['T']
        else:
            # define default constant parameters
            Ks = (1.0 / 1.0)
            self.K_ = np.reshape([
                499.114583 * Ks, 0.000000, 325.589216 * Ks,
                0.000000, 498.996093 * Ks, 238.001597 * Ks,
                0.000000, 0.000000, 1.000000], (3,3))
            self.D_ = np.float32([0.158661, -0.249478, -0.000564, 0.000157, 0.000000])
            # NOTE: the following transform describes a frontal-looking camera,
            # pitched 10 degrees downwards, offset +x=0.174m amd +z=0.113m from the base.
            self.T_c2b_ = tx.compose_matrix(
                    angles=[-np.pi/2-np.deg2rad(10),0.0,-np.pi/2],
                    #angles=[-np.pi/2,0.0,-np.pi/2],
                    translate=[0.174,0,0.113])

        # define "system" parameters
        self.pEM_ = dict(
                method=cv2.FM_RANSAC,
                prob=0.999,
                threshold=1.0)
        self.pLK_ = dict(winSize = (32,16),
                maxLevel = 4, # == effective winsize up to 32*(2**4) = 512x256
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.003),
                flags = 0,
                minEigThreshold = 1e-3 # TODO : disable eig?
                )
        self.pBA_ = dict(
                ftol=1e-4,
                #xtol=np.finfo(float).eps,
                loss='huber',
                max_nfev=1024,
                method='trf',
                verbose=2,
                tr_solver='lsmr',
                )
        self.pPNP_ = dict(
                iterationsCount=10000,
                reprojectionError=0.5,
                confidence=0.99,
                #flags = cv2.SOLVEPNP_EPNP
                #flags = cv2.SOLVEPNP_DLS
                #flags = cv2.SOLVEPNP_AP3P
                flags = cv2.SOLVEPNP_ITERATIVE
                #flags = cv2.SOLVEPNP_P3P
                #flags = cv2.SOLVEPNP_UPNP
                )
        # TODO : what is FAST threshold?
        # TODO : tune nfeatures; empirically 2048 is pretty good
        orb = cv2.ORB_create(
                nfeatures=2048,
                scaleFactor=1.2,#np.sqrt(2),??
                nlevels=8,
                # IMPORTANT : scoretype here influences response-based filters.
                scoreType=cv2.ORB_FAST_SCORE,
                #scoreType=cv2.ORB_HARRIS_SCORE,
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
        self.landmarks_ = Landmarks(des)
        self.hist_ = deque(maxlen=3)
        self.prune_freq_ = 16

        # pnp (NOTE : temporary for viz + should be deleted)
        self.pnp_p_ = None
        self.pnp_h_ = None

        # bundle adjustment + loop closure
        # sort ba pyramid by largest first
        ba_pyr = [16] # [16,64]
        self.ba_pyr_  = np.sort(ba_pyr)[::-1]
        self.graph_ = VGraph()

        # UKF
        # NOTE : ALWAYS enforce s_hist_.maxlen >= self.ba_pyr_.max()
        # TODO : evaluate EKF vs. UKF? empirically similar.
        # TODO : rename such that ukf/ekf naming is not confusing.
        # to avoid bugs due to circular buffer.
        self.ukf_l_  = build_ekf() # local incremental UKF
        self.ukf_h_  = build_ekf() # global historic UKF

        # State History
        if len(ba_pyr) >= 1:
            slen = self.ba_pyr_.max() + 1
        else:
            slen = 1
        self.s_hist_ = StateHistory(maxlen=slen) # stores cache of prior.

        # logging / visualization
        self.lm_cnt_ = []
        self.scales_ = defaultdict(lambda:[])

    def pts_nmx(self,
            pt2_c, pt2_l,
            rsp_c, rsp_l,
            k=4,
            radius=16.0
            ):
        # NOTE : somewhat confusing;
        # here suffix c=camera, l=landmark.
        # TODO : is it necessary / proper to take octaves into account?
        if pt2_l.size <= 0:
            # No registered landmarks to filter with.
            return np.arange(len(pt2_c))

        # compute nearest neighbors
        neigh = NearestNeighbors(n_neighbors=k, radius=radius)
        neigh.fit(pt2_l)

        # NOTE : 
        # radius_neighbors would be nice, but indexing is difficult to use
        # res = neigh.radius_neighbors(pt2_c, return_distance=False)
        d, i = neigh.kneighbors(pt2_c, return_distance=True)

        # too far from other landmarks to apply non-max
        msk_d = (d.min(axis=1) >= radius)
        # passed non-max
        msk_v = np.all(rsp_l[i] <= rsp_c[:,None], axis=1) # 

        # format + return results
        msk = (msk_d | msk_v)
        idx = np.where(msk)[0]
        print_ratio('non-max', len(idx), msk.size)
        return idx

    def track(self, img1, img2, pt1, pt2=None,
            thresh=1.0
            ):

        if pt1.size <= 0:
            # soft fail
            pt2 = np.empty([0,2], dtype=np.float32)
            idx = np.empty([0], dtype=np.int32)
            return pt2, idx

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
        # track reprojection error
        msk_err = (err < thresh)
        msk = np.logical_and.reduce([
            msk_err,
            msk_in,
            msk_st
            ])
        idx = idx[msk]
        return pt2, idx

    def proc_f2m_old(self,
            pose, scale,
            pt3_new_c, lm_idx,
            pt2_new,
            pt2_new_u
            ):
        """
        NOTE : pt3_new_c MUST BE in camera coordinates.
        """
        # mostly targetted at updating existing landmarks.

        # estimate scale from landmark correspondences
        # TODO : evaluate whether or not to use z-depthvalue or the full distance
        # may not matter too much given that scale from either SHOULD be consistent.
        # opt1 : norm

        p_lm_0 = self.landmarks_.pos[lm_idx] # landmark points in cam-map coord.
        p_lm_c = self.cvt_.map_to_cam(p_lm_0, pose)
        d_lm_old = np.linalg.norm(p_lm_c, axis=-1)
        d_lm_new = np.linalg.norm(pt3_new_c, axis=-1)

        scale_rel = (d_lm_old / d_lm_new).reshape(-1,1)
        scale_rel_std = scale_rel.std()
        print('estimated scale stability', scale_rel_std)

        # acquire scale corrections ...
        if len(lm_idx) > 8 and scale_rel_std < 0.3:
            # TODO : is it better to scale the weights?
            # scale_w = (var_lm[i1][lm_idx_e][:,(0,1,2),(0,1,2)])
            # scale_w = np.linalg.norm(scale_w, axis=-1) 
            # scale_w = np.sum(scale_w) / scale_w
            # TODO : is logarithmic mean better than naive mean?
            scale_est = np.exp(robust_mean(np.log(scale_rel)))
        else:
            # scale estimates are anticipated to be unstable.
            # use input scale
            scale_est = scale

        if len(d_lm_old) > 0:
            print_ratio('estimated scale ratio', scale_est, scale)
            # TODO : tune scale interpolation alpha
            alpha = 0.8 # high trust in ground-plane/ukf based estimate
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
            if (scale < 5e-3) and (scale_est / scale) > 2.0:
                # TODO : Magic ^^
                # disable scale interpolation
                # most likely running into pure rotation
                # implicit: scale=scale
                pass
            else:
                # logarithmic interpolation
                scale = np.exp(lerp(np.log(scale), np.log(scale_est), alpha))
        else:
            # implicit : scale = scale
            pass

        # == scale_is_believable @ >= 5e-3m translation
        # TODO : figure out better heuristic?
        run_lm = (scale >= 5e-3) # TODO : magic-ish

        # update landmarks
        update_lm = (run_lm and self.flag_ & ClassicalVO.VO_USE_LM_KF)
        if update_lm:
            # update landmarks from computed correspondences
            # TODO: apply scale_est (aggregate) or rel (individual)?
            pt3_new_c_s = pt3_new_c * scale
            var_lm_new = self.landmarks_.lm_var(self.cvt_,
                    pose, pt3_new_c_s)
            pt3_new_c_0 = self.cvt_.cam_to_map(pt3_new_c_s, pose)

            # == VIZ ==
            # try:
            #     fig = self.tfig_
            #     ax = fig.gca()
            # except Exception as e:
            #     self.tfig_ = plt.figure()
            #     fig = self.tfig_
            #     ax  = fig.add_subplot(1,1,1)#, projection='3d')

            # ax.cla()


            # pt3_old = self.landmarks_.pos[lm_idx]
            # pt3_new = pt3_new_c_0
            # pt2_old, v_msk = self.cvt_.pt3_pose_to_pt2_msk(
            #         pt3_old, pose)
            # pt2_new, v_msk = self.cvt_.pt3_pose_to_pt2_msk(
            #         pt3_new, pose)

            # nsel = 16
            # cols = np.random.uniform(size=(nsel,3))
            # if len(pt3_new) > 0:
            #     i = np.random.randint(len(pt2_new), size=nsel)
            #     print cols.shape, cols.dtype
            #     print pt2_new.shape, pt2_new.dtype
            #     print pt2_old.shape, pt2_old.dtype
            #     print '?????????', pt2_new[i,0].shape
            #     ax.scatter(pt2_new[i,0], pt2_new[i,1], c=cols)
            #     ax.scatter(pt2_old[i,0], pt2_old[i,1], c=cols)
            #     plt.pause(0.001)
            # == VIZ END ==

            self.landmarks_.update(lm_idx, pt3_new_c_0, var_lm_new)

            # Add correspondences to BA Cache
            # wow, that's a lot of chained indices.
            self.graph_.append(lm_idx, pt2_new_u)

        # == PNP ===================
        ## flag to decide whether to run PNP
        #run_pnp = bool(self.flag_ & ClassicalVO.VO_USE_PNP)
        #run_pnp &= lm_idx_e.size >= 16 # use at least > 16 points
        # TODO : REVIVE PNP AT SOME POINT
        ## reset PNP data no matter what
        #self.pnp_p_ = None
        #self.pnp_h_ = None
        #if run_pnp:
        #    # either landmarks are wrong, or poses are wrong, which influences PNP performance.
        #    # The issue is that both of them must be simultaneously optimized.
        #    pt_map = pos_lm[i1[lm_idx_e]] # --> "REAL" map from old observations
        #    #pt_map = p_lm_v2_0 # --> "FAKE" map from current observation
        #    #pt_map = lerp(pos_lm[i1[lm_idx_e]], p_lm_v2_0, 0.15) # compromise?
        #    pt_cam = pt2_u_c[i2[lm_idx_e]]

        #    # filter by above-GP features
        #    # (TODO : hack because ground plane tends to be somewhat homogeneous
        #    # in the test cases)
        #    pt_map_b = pt_map.dot(self.cvt_.T_c2b_[:3,:3].T) + self.cvt_.T_c2b_[:3,3:].T
        #    ngp_msk =  (pt_cam[:,1] < self.y_GP) # based on camera
        #    ngp_msk &= (pt_map_b[:,2] > 0.0)     # based on map
        #    ngp_idx = np.where(ngp_msk)[0]
        #    pt_map = pt_map[ngp_idx]
        #    pt_cam = pt_cam[ngp_idx]
        #    #pt_cam_rec = pt2_lm[i1[lm_idx_e[ngp_idx]]] # un-updated
        #    pt_cam_rec, _ = self.cvt_.pt3_pose_to_pt2_msk(
        #            pt_map, pose) # updated version

        #    # == debugging ==
        #    # TODO : make slot for pnp
        #    #try:
        #    #    fig = self.pfig_
        #    #    ax  = fig.gca()
        #    #except Exception:
        #    #    self.pfig_ = plt.figure()
        #    #    fig = self.pfig_
        #    #    ax  = fig.gca()
        #    #ax.cla()
        #    #ax.imshow(img_c[...,::-1])
        #    #ax.plot(pt_cam_rec[:,0], pt_cam_rec[:,1], 'r*', alpha=0.5)
        #    #ax.plot(pt_cam[:,0], pt_cam[:,1], 'b+', alpha=0.5)
        #    #if not ax.yaxis_inverted():
        #    #    ax.invert_yaxis()
        #    # == debugging ==
        #    msg = self.run_PNP(pt_map, pt_cam, pose, msg=msg)
        # ========================

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

        return scale

    def proc_f2m_new(self,
            pose, scale, pt3,
            kpt_p, des_p,
            pt2_c, pt2_u_c,
            img_c,
            msg=''
            ):

        # mostly targetted at processing new points.
        # should not really return scale, since
        # only a few landmarks should have correspondences here.

        # query visible points from landmarks database
        # atol here, chosen based on fov
        # TODO : avoid hardcoding or figure out better heuristic
        # dtol chosen based on depth uncertainty guess
        qres = self.landmarks_.query(pose, self.cvt_,
                atol=np.deg2rad(60.0),
                dtol=2.0 # NOTE: not meters; defines depth ratio [0.5, 2.0]
                )
        pt2_lm, pos_lm, des_lm, var_lm, cnt_lm, lm_idx = qres

        print_ratio('visible landmarks', len(lm_idx), self.landmarks_.size_)

        # NOTE : currently disabled re-matching with
        # un-tracked landmarks.
        # potentially useful to revive at some point.

        # # select useful descriptor based on current viewpoint
        # i1, i2 = self.cvt_.des_des_to_match(
        #         des_lm,
        #         des_p, cross=(self.flag_ & ClassicalVO.VO_USE_MXCHECK)
        #         )

        # if len(lm_idx) > 16: # TODO : MAGIC
        #     # filter correspondences by Emat consensus
        #     # first-order estimate: image-coordinate distance-based filter
        #     cor_delta = (pt2_lm[i1] - pt2_u_c[i2])
        #     cor_delta = np.linalg.norm(cor_delta, axis=-1)
        #     lm_msk_d = (cor_delta < 128.0)  # TODO : MAGIC
        #     lm_idx_d = np.where(lm_msk_d)[0]

        #     # second estimate
        #     try:
        #         # TODO : maybe not the most efficient way to
        #         # check landmark consensus?
        #         # TODO : take advantage of the Emat here to some use?
        #         _, lm_msk_e = cv2.findEssentialMat(
        #                 pt2_lm[i1][lm_idx_d],
        #                 pt2_u_c[i2][lm_idx_d],
        #                 self.K_,
        #                 **self.pEM_)
        #     except Exception as e:
        #         lm_msk_e = None

        #     if lm_msk_e is not None:
        #         # refine by Emat
        #         lm_msk_e = lm_msk_e[:,0].astype(np.bool)
        #         lm_idx_e = np.where(lm_msk_e)[0]
        #         lm_msk_e = lm_msk_d[lm_idx_e]
        #         lm_idx_e = lm_idx_d[lm_idx_e]
        #     else:
        #         lm_msk_e = lm_msk_d
        #         lm_idx_e = lm_idx_d

        #     print_ratio('landmark concensus', len(lm_idx_e), lm_msk_e.size)
        # else:
        #     # use all available data, at the cost of maybe noise
        #     # TODO : verify if abort is necessary instead
        #     lm_msk_e = np.ones(len(i1), dtype=np.bool)
        #     lm_idx_e = np.where(lm_msk_e)[0]

        # # landmark correspondences
        # p_lm_0 = pos_lm[i1][lm_idx_e] # map-frame lm pos
        # p_lm_c = self.cvt_.map_to_cam(p_lm_0, pose) # TODO : use rectified pose?
        # p_lm_v2_c = pt3[i2][lm_idx_e] # current camera frame lm pos

        # apply lenient matcher, to prevent re-inserting the same landmark
        i1_lax, i2_lax = self.cvt_.des_des_to_match(
                des_lm,
                des_p,
                lowe=1.0,
                maxd=128.0,
                cross=False
                )

        # update "invisible" landmarks that should have been visible
        # TODO : something more than just count decrementing?
        # TODO : also, some landmarks may be invisible due to obstacles.
        # == (probably) filtering by view angle would help
        # TODO : revive cnt_lm if it becomes useful
        # TODO : note that cnt_lm is a copy, not a view.
        # cnt_lm[i1_lax] -= 1

        # insert new landmarks
        insert_lm = (scale >= 5e-3) # TODO : magic-ish

        if insert_lm:
            # filter points loosely matched with existing landmarks
            match_msk = np.zeros(len(des_p), dtype=np.bool)
            match_msk[i2_lax] = True
            insert_msk = ~match_msk
            insert_idx = np.where(insert_msk)[0]

            n_new = len(insert_idx)
            msk_n = np.ones(n_new, dtype=np.bool)

            if len(pt2_lm) > 0 and len(insert_idx) > 0:
                # filter insertion by proximity to existing landmarks
                neigh = NearestNeighbors(n_neighbors=1)
                neigh.fit(pt2_lm)
                d, _ = neigh.kneighbors(pt2_u_c[insert_idx], return_distance=True)
                msk_knn = (d < 16.0)[:,0] # TODO : magic number
                #print_ratio('msk_knn', msk_knn.sum(), msk_knn.size)

                # dist to nearest landmark, less than 20px
                msk_n[msk_knn] = False

            # apply filter
            idx_n = np.where(msk_n)[0]
            insert_idx = insert_idx[idx_n]

            # filter by map point distance
            lm_d  = np.linalg.norm(pt3[insert_idx], axis=-1)
            msk_d = (lm_d*scale < 20.0) # NOTE : heuristic to suppress super-far points

            # apply filter
            idx_d = np.where(msk_d)[0]
            insert_idx = insert_idx[idx_d]

            # finalize insertion
            n_new = insert_idx.size
            print('adding {} landmarks : {}->{}'.format(n_new,
                len(self.landmarks_.pos), len(self.landmarks_.pos)+n_new
                ))
            pt3_new_c = scale * pt3[insert_idx]
            des_new = des_p[insert_idx]
            kpt_new = kpt_p[insert_idx]
            col_new = get_points_color(img_c, pt2_c[insert_idx], w=1)

            # append new landmarks ...
            # TODO : the problem here is that the landmarks are inserted eagerly,
            # and therefore interferes with pose rectification that requires
            # offsets to be consistent.

            li_0 = self.landmarks_.size_
            self.landmarks_.append_from(
                    self.cvt_, # requires Conversions handle
                    pose, pt3_new_c,
                    des_new, col_new, kpt_new)
            li_1 = self.landmarks_.size_

            # NOTE : using undistorted version of pt2.
            # WARN : pt2_u_c = undistort( pt2_c[idx_t])
            # for whatever reason, indexing was somewhat messed up.
            self.graph_.append(np.arange(li_0, li_1),
                    pt2_u_c[insert_idx]
                    )
        # TODO : evaluate what to return from here
        # really shouldn't return scale?
        return scale

    def proc_f2m(self,
            pose, scale, pt3,
            kpt_p, des_p,
            pt2_c, pt2_u_c,
            img_c,
            msg=''
            ):
        """
        """
        # frame-to-map processing
        # (i.e. uses landmark data)

        # control flags
        # update existing landmarks

        return scale, msg

    def pRt2pose(self, p, R, t):
        """
        returns pose updated with input rotation + translation.
        """
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
    def project_BA(self, cam, lmk, 
            return_h=False,
            return_msk=False
            ):
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
        T_o2b[:,0,1] = s # important : note transposed
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
            self.K_, # 3x3
            self.cvt_.T_b2c_[:3], # 3x4
            T_o2b, # 4x4
            self.cvt_.T_c2b_, # 4x4
            lmk_h[...,None]])[...,0] # 4x1
        if return_h:
            return pt2_h

        pt2 = self.cvt_.pth_to_pt(pt2_h)
        if return_msk:
            #simple depth check
            msk = (pt2_h[..., -1] >= 0)
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
        err = obs_pt2 - prj_pt2 # == y-x
        #err = prj_pt2 - obs_pt2 # == x-y
        # TODO : is it actually necessary to apply the mask?
        #i_null = np.where(~msk)[0]
        #err[i_null] = 0
        return err.ravel()

    def sparsity_BA(self, n_c, n_l, ci, li):
        s_o = 2 # observation state size
        s_c = 3 # camera state size
        s_l = 3 # landmark state size
        m = len(ci) * s_o # flat # of observations
        n = n_c * s_c + n_l * s_l # flat # of parameters
        A = lil_matrix((m,n), dtype=int) # TODO: dtype=bool?

        # pre-compute offsets for interleaving
        ci0 = ci*s_c
        li0 = li*s_l

        i = np.arange(len(ci))
        for s in range(s_c):
            A[2*i,   ci0+s] = 1
            A[2*i+1, ci0+s] = 1
        # experimental : link cameras
        for s in range(s_l):
            A[2*i,   n_c*s_c + li0+s] = 1
            A[2*i+1, n_c*s_c + li0+s] = 1
        # from here, the constraint is expressed
        # that the format of parameters is
        # [cx0,cy0,ch0, cx1,cy1,ch1, ... lx0,ly0,lz0, lx1,ly1,lz1, ...]
        # and the format of output is
        # [px0,py0,px1,py1, ...] (flattened points2d array)
        return A

    def jac_BA(self, x, n_c, n_l, c_i, l_i, obs_pt2):
        """
        Currently unused due to being slow;
        Also, I'm not quite sure if the current implementation is correct.
        TODO : optimize and/or validate
        """
        # shape information
        n_o = len(c_i)

        # unroll input params
        pos = x[:n_c*3].reshape(-1, 3) # camera 2d pose (x,y,h)
        lmk = x[n_c*3:].reshape(-1, 3) # landmark positions

        # TODO : cache results for pt2_h to avoid calcing twice
        pt2_h = self.project_BA(pos[c_i], lmk[l_i], return_h=True)
        #err   = obs_pt2 - pt2_h

        J = -ba_J_v2(pos[c_i], lmk[l_i], self.K_,
                self.cvt_.T_b2c_[:3,:3], self.cvt_.T_b2c_[:3,3:], pt2_h) # Nx2x6

        J_c = J[:,:,:3]
        J_l = J[:,:,3:]

        J_res = lil_matrix((2*n_o, n_c*3+n_l*3))

        o_i0 = np.arange(n_o)
        for i_o in range(2): # iterate over point (x,y)
            for i_c in range(3): # iterate over pose (x,y,h)
                J_res[o_i0*2+i_o, c_i*3+i_c] = J_c[:,i_o,i_c]
            for i_l in range(3): # iterate over landmark (x,y,z)
                J_res[o_i0*2+i_o, n_c*3 + l_i*3+i_l] = J_l[:,i_o,i_l]
        # quick validation.
        #zmsk = (self.sparsity_BA(n_c,n_l,c_i,l_i).todense() == 0)
        #print 'jacobian validation', np.abs(J_res[zmsk]).sum()

        return J_res

    def run_BA(self, win, ax=None):
        """
        Sources:
            https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html
            https://github.com/jahdiel/pySBA/blob/master/PySBA.py
        """

        if not self.flag_ & ClassicalVO.VO_USE_BA:
            return

        if self.graph_.size_ <= 0:
            return

        # create np arrays
        p0, ci, li, p2 = self.graph_.query(win)
        _, vp, _ = self.s_hist_.query(win)

        # gather stats
        n_c = len(p0)
        n_l = self.landmarks_.size_

        # filter by landmarks that were actually observed
        # TODO : filter by cnt >= 2?
        li_u, li = np.unique(li, return_inverse=True) # WARN: li override
        n_l = len(li_u)
        x0 = np.concatenate([
            p0.ravel(),
            self.landmarks_.pos[li_u].ravel()
            ])

        # compute BA sparsity structure
        A  = self.sparsity_BA(n_c, n_l, ci, li)
        #J_test = self.jac_BA(x0, n_c, n_l, ci, li, p2)
        #print 'JTEST!!'
        #print J_test.todense()[A.todense() == 1]

        # EXPERIMENTAL: setting x_scale
        # "characteristic scale", standard deviation
        si = np.arange(3)
        vp = vp[:,si, si]
        vl = self.landmarks_.var[li_u][:, si, si]

        # scales -- avoid hyper-strong outliers
        # NOTE : lower scale needs to be regulated, not high.
        # TODO : x_scale=sc is very very unstable.
        sp = np.sqrt(vp)
        sl = np.sqrt(vl)
        #sl_lo = np.percentile(sp, 50)
        #sl_lo = np.percentile(sl, 20)
        #sl_hi = np.percentile(sl, 80)
        #sl = np.clip(sl, sl_lo, a_max=None)
        sc = np.concatenate([sp.ravel(), sl.ravel()])

        if ax is not None:
            ax['ba_1'].cla()
            ax['ba_2'].cla()
            ax['ba_1'].set_title('scale-pos')
            ax['ba_2'].set_title('scale-lmk')

            # prep data for viz
            w = (1.0 / sp) # << is how it will actually be reflected
            hi = np.full(len(sp), np.percentile(w, 80))
            lo = np.full(len(sp), np.percentile(w, 20))

            ax['ba_1'].plot(w, '+')
            ax['ba_1'].plot(hi, '--')
            ax['ba_1'].plot(lo, '--')
            #ax['ba_1'].hist(sc[:n_c], alpha=0.5, label='cam')

            w = (1.0 / sl) # << is how it will actually be reflected
            hi = np.full(len(sl), np.percentile(w, 80))
            lo = np.full(len(sl), np.percentile(w, 20))

            ax['ba_2'].plot(w, '+')
            ax['ba_2'].plot(hi, '--')
            ax['ba_2'].plot(lo, '--')
            #ax['ba_2'].set_title('lmk-sc')
            #ax['ba_2'].hist(sc[n_c:], alpha=0.5, label='lmk')

        # mean reprojection error
        err0 = np.sqrt(np.square(self.residual_BA(x0,
            n_c, n_l,
            ci, li, p2)).mean())

        # actually run BA
        res = least_squares(
                self.residual_BA, x0,
                #jac_sparsity=A,
                jac=self.jac_BA,
                #x_scale = sc,
                x_scale='jac',
                args=(n_c, n_l, ci, li, p2),
                **self.pBA_
                )

        # format ...
        pos_opt = res.x[:n_c*3].reshape(-1,3)
        lmk_opt = res.x[n_c*3:].reshape(-1,3)

        err1 = np.sqrt(np.square(self.residual_BA(res.x,
            n_c, n_l,
            ci, li, p2)).mean())
        #print err0, err1

        if ax is not None:
            ax['ba_0'].cla()
            ax['ba_0'].plot(p0[:,0], p0[:,1], 'ko-', label='initial')
            ax['ba_0'].plot(pos_opt[:,0], pos_opt[:,1], 'r+-', label='optimized')
            ax['ba_0'].set_title('BA : {:.3e}->{:.3e}'.format( err0, err1 ))
            ax['ba_0'].axis('equal')
            ax['ba_0'].set_aspect('equal', 'datalim')
            ax['ba_0'].legend()

        # apply BA results
        # TODO : can we INPUT variance information to scipy.least_squares?
        # TODO : can we extract variance information out of least_squares?
        self.graph_.update(win, pos_opt) # TODO : or result from UKF? may not matter.
        self.landmarks_.pos[li_u] = lmk_opt

        return pos_opt

    def run_UKF(self, dt, scale=None):
        """ motion-based prediction """
        pose_p = self.ukf_l_.x[:3].copy()
        Q, R = get_QR(pose_p, dt)
        self.ukf_l_.Q = Q
        self.ukf_l_.R = R
        self.ukf_l_.predict(dt)
        pose_c = self.ukf_l_.x[:3].copy()
        if scale is None:
            # initialize scale from UKF
            scale = np.linalg.norm(pose_c[:2] - pose_p[:2])
        else:
            # implicit : scale is kept
            pass
        return pose_p, pose_c, scale
    
    @property
    def y_GP(self):
        """
        Minimum Ground Plane value in Image Coordinates.
        probably super inefficient but doesn't really matter.

        returns y_GP such that
        z.T.(R_c2b.(K^{-1}.([[0,ymin,1]].T)) + t_c2b) == 0

        NOTE: assumes camera roll w.r.t ground plane = 0
        NOTE: operates on undistorted coordinates.
        """
        # TODO : super rough implementation
        try:
            # try to return existing cache
            return self.y_gp_
        except Exception:
            z = np.reshape([0,0,1], (3,1))
            R = self.cvt_.T_c2b_[:3,:3]
            t = self.cvt_.T_c2b_[:3,3:]
            Ki = self.cvt_.Ki_

            A_part = np.linalg.multi_dot([z.T, R, Ki])
            b_part = -z.T.dot(t)
            y_gp = (b_part[0,0] - A_part[0,2]) / A_part[0,1]

            # validation
            #x_part = np.reshape([0,y_gp,1], (3,1))
            #print z.T.dot(R.dot(Ki.dot(x_part)) + t)

            self.y_gp_ = y_gp
        return self.y_gp_

    def run_GP(self, pt_c, pt_p, pt3=None,
            scale=None,
            guess=None
            ):
        """
        Scale estimation based on locating the ground plane.
        if scale:=None, scale based on best z-plane will be returned.
        """
        if not (self.flag_ & ClassicalVO.VO_USE_SCALE_GP):
            return None, scale, guess

        camera_height = self.T_c2b_[2, 3]

        if self.flag_ & ClassicalVO.VO_USE_GP_RSC:
            # opt1 : estimate ground-plane for projection
            # unfortunately, there's far too few points on the ground plane
            # to compute a reasonable estimate.

            y_min = self.y_GP

            gp_msk = np.logical_and.reduce([
                pt_c[:,1] >= y_min,
                pt_p[:,1] >= y_min])

            gp_idx = np.where(gp_msk)[0]

            if len(gp_idx) <= 3: # TODO : magic
                # too few points, abort gp estimate
                return None, scale, guess

            # update pt_c and pt_p
            pt_c = pt_c[gp_idx]
            pt_p = pt_p[gp_idx]

            # NOTE: debug; show gp plane points correspondences
            # vsel = np.random.choice(len(gp_idx), size=32)
            # try:
            #     fig = self.gfig_
            #     ax  = fig.gca()
            # except Exception:
            #     self.gfig_ = plt.figure()
            #     ax = self.gfig_.gca()
            # col = np.random.uniform(0.0, 1.0, size=(len(vsel), 3)).astype(np.float32)
            # ax.cla()
            # ax.scatter(pt_c[vsel,0], pt_c[vsel,1], color=col)
            # ax.scatter(pt_p[vsel,0], pt_p[vsel,1], color=col)
            # ax.quiver(
            #         pt_p[vsel, 0], pt_p[vsel, 1],
            #         pt_c[vsel, 0] - pt_p[vsel,0],
            #         pt_c[vsel, 1] - pt_p[vsel,1],
            #         angles='xy',
            #         scale=1,
            #         scale_units='xy',
            #         color='g',
            #         alpha=0.5
            #         )
            # ax.set_xlim(0, 640)
            # ax.set_ylim(0, 480)
            # if not ax.yaxis_inverted():
            #     ax.invert_yaxis()

            # ground plane is a plane, so homography can (and should) be applied here
            H, msk_h = cv2.findHomography(pt_c, pt_p,
                    method=self.pEM_['method'],
                    ransacReprojThreshold=self.pEM_['threshold']
                    )
            idx_h = np.where(msk_h)[0]
            print_ratio('Ground-plane Homography', len(idx_h), msk_h.size)

            if len(idx_h) < 16: # TODO : magic number
                # insufficient # of points -- abort
                return None, scale, guess


            # update pt_c and pt_p
            pt_c = pt_c[idx_h]
            pt_p = pt_p[idx_h]

            # TODO : lots of information is discarded here,
            # Such as R/T from homography and the reconstructed 3D Points.
            # Only Scale is propagated.

            res_h, Hr, Ht, Hn = cv2.decomposeHomographyMat(H, self.K_)
            Hn = np.float32(Hn)
            Ht = np.float32(Ht)
            Ht /= np.linalg.norm(Ht, axis=1, keepdims=True) # NOTE: Ht is N,3,1
            gp_z = (Hn[...,0].dot(self.T_c2b_[:3,:3].T))

            # filter by estimated plane z-norm
            # ~15-degree deviation from the anticipated z-vector (0,0,1)
            # TODO : collect these heuristics params
            z_val = ( np.abs(np.dot(gp_z, [0,0,1])) > np.cos(np.deg2rad(15)) )
            z_idx = np.where(z_val)[0]
            if len(z_idx) <= 0:
                # abort ground-plane estimation.
                return None, scale, guess
            # NOTE: honestly don't know why I need to pre-filter by z-norm at all
            perm = zip(Hr,Ht)
            perm = [perm[i] for i in z_idx]
            n_in, R, t, msk_r, gpt3, sel = recover_pose_from_RT(perm, self.K_,
                    pt_c, pt_p, return_index=True, guess=guess, log=False)
            gpt3 = gpt3.T # TODO : gpt3 not used

            # convert w.r.t base_link
            gpt3_base = gpt3.dot(self.cvt_.T_c2b_[:3,:3].T)
            h_gp = robust_mean(-gpt3_base[:,2])
            scale_gp = (camera_height / h_gp)
            print 'gp-ransac scale', scale_gp
            if np.isfinite(scale_gp) and scale_gp > 0:
                # project just in case scale < 0...
                scale = scale_gp

            # this is functionally the only time it's considered "success".
            return H, scale, (R, t)
        else:
            # opt2 : directly estimate ground plane by simple height filter
            # only works with "reasonable" initial scale guess.
            # all it does is refine a good scale estimate to a potentially "better" one.

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

            print_ratio('GP Inlier', len(gp_idx), gp_msk.size)

            if len(gp_idx) > 3: # at least 3 points
                h_gp = robust_mean(-pt_gp[:,2])
                if not np.isnan(h_gp):
                    scale_gp = camera_height / h_gp
                    print_ratio('scale_gp', scale_gp, scale)
                    # use gp scale instead
                    scale = scale_gp
        return None, scale, guess

    def run_PNP(self, pt3_map, pt2_cam, pose,
            p_min=16, msg=''):

        if len(pt3_map) < p_min or len(pt2_cam) < p_min:
            return msg

        try:
            # construct extrinsic guess
            T_b2o = self.cvt_.pose_to_T(pose)
            T_c2m = np.linalg.multi_dot([
                self.cvt_.T_b2c_,
                T_b2o,
                self.cvt_.T_c2b_
                ])
            T_m2c = tx.inverse_matrix(T_c2m)
            T_src = T_m2c  # Model (map) --> Camera Coord.

            rvec0 = cv2.Rodrigues(T_src[:3,:3])[0]
            tvec0 = T_src[:3, 3:].ravel()
            #res = cv2.solvePnPRansac(
            #        pt_map, pt_cam,
            #        self.K_, 0*self.D_,
            #        useExtrinsicGuess = True,
            #        rvec=rvec0.copy(),
            #        tvec=tvec0.copy(),
            #        **self.pPNP_
            #        )
            #suc, rvec, tvec, inliers = res
            #rvec = rvec0
            #tvec = tvec0
            #idx = np.random.choice(len(pt_map), size=4, replace=False)
            #idx = np.s_[:len(pt_map)]
            res = cv2.solvePnP(
                    pt3_map, pt2_cam,
                    self.K_, 0*self.D_,
                    useExtrinsicGuess = True,
                    rvec=rvec0.copy(),
                    tvec=tvec0.copy(),
                    flags=self.pPNP_['flags']
                    )
            suc, rvec, tvec = res
            inliers = np.arange(len(pt3_map))

            if suc:
                # parse output from solvePnP
                T_m2c = np.eye(4, dtype=np.float64)
                T_m2c[:3,:3] = cv2.Rodrigues(rvec)[0]
                T_m2c[:3,3:] = tvec.reshape(3,1)
                T_c2m = tx.inverse_matrix(T_m2c)

                T_b2o = np.linalg.multi_dot([
                        self.cvt_.T_c2b_, 
                        T_c2m,
                        self.cvt_.T_b2c_
                        ])
                t_b = tx.translation_from_matrix(T_b2o)
                r_b = tx.euler_from_matrix(T_b2o)
                
                pnp_p = (t_b[0], t_b[1])
                pnp_h = r_b[-1]
                print('pnp : {}, {}'.format(pnp_p, pnp_h))
                print_ratio('PNP Ratio', len(inliers), len(pt3_map))
                msg += '(pnp:{}/{})'.format( len(inliers), len(pt3_map))
                # NOTE : uncomment this to revive pnp visualization
                # NOTE : standardize visualizations
                self.pnp_p_ = pnp_p
                self.pnp_h_ = pnp_h
        except Exception as e:
            # ignore exception
            print('PNP Error : {}'.format(e))
            pass

        return msg

    def scale_c(self, s_b, R_c, t_c, guess=None):
        """
        obtain scale of camera-frame translation from
        s_b (base-frame translation scale)
        R_c (camera-frame rotation matrix)
        t_c (camera-frame translation vector)

        refer to test_scale.py for validation.
        """

        R_c2b = self.cvt_.T_c2b_[:3,:3]
        t_c2b = self.cvt_.T_c2b_[:3,3:]
        R_b2c = self.cvt_.T_b2c_[:3,:3]
        t_b2c = self.cvt_.T_b2c_[:3,3:]

        R_b = np.linalg.multi_dot([
            R_c2b, R_c, R_b2c])

        v1 = R_c2b.dot(t_c)
        v2 = t_c2b-R_b.dot(t_c2b)

        # c_a*s_c^2 + c_b*s_c^1 + c_c = s_b**2
        c_a = (v1*v1).sum() # v1.T.dot(v1)[0,0]
        c_b = 2 * (v1*v2).sum() #2*v1.T.dot(v2)[0,0]
        c_c = (v2*v2).sum() - s_b**2 #v2.T.dot(v2)[0,0] - s_b**2

        det = c_b**2-4*c_a*c_c # determinant part
        if det < 0:
            print('det', det)
            # this is usually due to numerical error.
            # raise ValueError("Determinant is invalid! {}".format(det))
            det = 0.0
        sol_1 = (-c_b + np.sqrt(det) ) / (2*c_a)
        sol_2 = (-c_b - np.sqrt(det) ) / (2*c_a)

        if sol_1 < 0.0 and sol_2 < 0.0:
            #raise ValueError("solution somehow does not exist")
            #print sol_1
            #print sol_2
            #return 0.0
            # WHAT SHOULD I DO? rely on the fact that scale_b ~~ scale_c?
            # NOTE : this is wrong.
            # TODO : this is wrong.
            # WARN : this is wrong.
            return s_b

        if sol_1 < 0.0:
            return sol_2
        if sol_2 < 0.0:
            return sol_1

        if guess is None:
            # return a random "valid" solution.
            # beware of the possible flipped signs.
            return np.random.choice([sol_1,sol_2])
        else:
            # tie-breaker with guess
            # tb1 and tb2 are mostly negatives of each other.
            # choose the one that best aligns with the current estimate.
            t_b = guess
            stb1 = np.matmul(R_c2b, sol_1 * t_c) - np.matmul(R_b, t_c2b) + t_c2b
            stb2 = np.matmul(R_c2b, sol_2 * t_c) - np.matmul(R_b, t_c2b) + t_c2b
            score_1 = (t_b * stb1).sum() / np.linalg.norm(stb1)
            score_2 = (t_b * stb2).sum() / np.linalg.norm(stb2)
            idx = np.argmax([score_1,score_2])
            return [sol_1, sol_2][idx]
        return sol_1

    def run_EM(self, 
            pt2_u_c, pt2_u_p,
            no_gp = True,
            guess=None
            ):

        if no_gp:
            # pre-filter by ymin
            y_gp = self.y_GP
            # NOTE: is it necessary to also check pt2_u_p?
            # probably gives similar results; skip.
            ngp_msk = (pt2_u_c[:,1] <= y_gp)
            ngp_idx = np.where(ngp_msk)[0]
            pt2_u_c = pt2_u_c[ngp_idx]
            pt2_u_p = pt2_u_p[ngp_idx]

        # == opt 1 : essential ==
        # NOTE ::: findEssentialMat() is run on ngp_idx (Not tracking Ground Plane)
        # Because the texture in the test cases were repeatd,
        # and was prone to mis-identification of transforms.
        E, msk_e = cv2.findEssentialMat(pt2_u_c, pt2_u_p, self.K_,
                **self.pEM_)
        msk_e = msk_e[:,0].astype(np.bool)
        idx_e = np.where(msk_e)[0]
        print_ratio('e_in', len(idx_e), msk_e.size)
        F = self.cvt_.E_to_F(E)
        # == essential over ==

        if self.flag_ & ClassicalVO.VO_USE_HOMO:
            # opt 2 : homography
            H, msk_h = cv2.findHomography(pt2_u_c, pt2_u_p,
                    method=self.pEM_['method'],
                    ransacReprojThreshold=self.pEM_['threshold']
                    )
            msk_h = msk_h[:,0].astype(np.bool)
            idx_h = np.where(msk_h)[0]
            print_ratio('h_in', len(idx_h), msk_h.size)

            # compare errors
            sH, msk_sh = score_H(pt2_u_c, pt2_u_p, H, self.cvt_)
            sF, msk_sf = score_F(pt2_u_c, pt2_u_p, F, self.cvt_)

            r_H = (sH / (sH + sF))
            print_ratio('RH', sH, sH+sF)

        use_h = False
        if self.flag_ & ClassicalVO.VO_USE_HOMO:
            # TODO : "magic" determinant number based on ORB-SLAM Paper
            use_h = (r_H > 0.45)

        # TODO : option : filter based on input guess R,t
        # maybe a good idea.

        idx_p = None
        if use_h:
            # use Homography Matrix for pose
            #idx_h = idx_h[msk_sh]
            res_h, Hr, Ht, Hn = cv2.decomposeHomographyMat(H, self.K_)
            print Hr[0], Ht[0], np.linalg.norm(Hr[0]), np.linalg.norm(Ht[0])
            Ht = np.float32(Ht)

            # NOTE: still don't know why Ht is not normalized.
            Ht /= np.linalg.norm(Ht, axis=(1,2), keepdims=True)

            perm = zip(Hr,Ht)
            n_in, R, t, msk_r, pt3 = recover_pose_from_RT(perm, self.K_,
                    pt2_u_c[idx_h], pt2_u_p[idx_h], guess=guess,
                    log=False)
            print_ratio('homography', len(idx_h), msk_h.size)

            idx_p = idx_h
        else:
            # use Essential Matrix for pose
            # TODO : specify z_min/z_max?
            n_in, R, t, msk_r, pt3 = recover_pose(E, self.K_,
                    pt2_u_c[idx_e], pt2_u_p[idx_e], guess=guess,
                    log=False
                    )
            print_ratio('essentialmat', len(idx_e), msk_e.size)
            idx_p = idx_e

        # idx_r = which points were used for pose reconstruction
        pt3 = pt3.T
        idx_r = np.where(msk_r)[0]
        print_ratio('triangulation', len(idx_r), msk_r.size)

        idx_in = idx_p[idx_r] # overall, which indices were used?

        return idx_in, pt3, R, t

    def __call__(self, img, dt, scale=None,
            ax=None
            ):
        index = self.graph_.index
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
        # store priors
        self.s_hist_.append(
                self.ukf_l_.x,
                self.ukf_l_.P,
                dt)

        pose_p, pose_c, scale = self.run_UKF(dt, scale)

        # Estimate #0 : Based on UKF Motion
        # TODO : is it possible to pass R_c/t_c
        # as initial guesses to recoverPose()?

        # construct t_b/t_c guess
        dx, dy, dh = (pose_c - pose_p)
        R_b = tx.euler_matrix(0, 0, dh)[:3,:3]
        t_b = np.reshape([dx, dy, 0], (3,1))

        T_b2b1 = np.eye(4)
        T_b2b1[:3,:3] = R_b
        T_b2b1[:3,3:] = t_b
        T_c2c1 = np.linalg.multi_dot([
            self.cvt_.T_b2c_,
            T_b2b1,
            self.cvt_.T_c2b_
            ])
        R_c = T_c2c1[:3,:3]
        t_c = T_c2c1[:3,3:]

        # cache results
        R_c0 = R_c
        R_b0 = R_b
        t_c0 = t_c
        t_b0 = t_b
        self.scales_['c0'].append([index, np.linalg.norm(t_c0)])
        self.scales_['b0'].append([index, np.linalg.norm(t_b0)])
        # == #0 finished ==

        # frame-to-frame processing
        pt2_p = self.cvt_.kpt_to_pt(kpt_p)
        rsp_p = np.float32([e.response for e in kpt_p])

        #try:
        #    fig = self.rfig_
        #except Exception:
        #    self.rfig_ = plt.figure()
        #    fig = self.rfig_
        #rax = fig.gca()
        #rax.hist(rsp_p)

        idx_l, pt2_l = self.landmarks_.track_points()
        rsp_p_l = self.landmarks_.kpt[idx_l, 2]

        # == obtain next-frame keypoints ==
        if self.flag_ & ClassicalVO.VO_USE_TRACK:
            # opt1 : points by track
            # TODO : support pyramid
            #pyr_p = cv2.buildOpticalFlowPyramid(img_p,
            #        self.pLK_['winSize'],
            #        self.pLK_['maxLevel']
            #        )
            #pyr_c = cv2.buildOpticalFlowPyramid(img_p,
            #        self.pLK_['winSize'],
            #        self.pLK_['maxLevel']
            #        )
            pt2_c,   idx_t   = self.track(img_p, img_c, pt2_p)
            pt2_c_l, idx_t_l = self.track(img_p, img_c, pt2_l)

            # mark points from landmarks that failed to track
            nidx_l = np.ones(len(pt2_l), dtype=np.bool)
            nidx_l[idx_t_l] = False
            print_ratio('track', len(idx_t), len(pt2_p))
            print_ratio('lmk track', len(idx_t_l), len(pt2_l))
            self.landmarks_.untrack(idx_l[nidx_l])

            # non-max suppression on remaining points
            # NOTE: requires refactoring indexing architecture
            idx_s = self.pts_nmx(
                    pt2_c[idx_t], pt2_c_l[idx_t_l],
                    rsp_p[idx_t], rsp_p_l[idx_t_l],
                    )

            # apply suppression indices
            idx_t = idx_t[idx_s]
        else:
            # opt2 : points by match
            # TODO : not currently supporting landmarks extension
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
        # TODO : also track landmark points?
        # =================================

        # merge data
        cam_lim = len(idx_s)
        pt2_p_all = np.concatenate([
            pt2_p[idx_t],
            pt2_l[idx_t_l],
            ], axis=0) # [N+M,2]
        pt2_c_all = np.concatenate([
            pt2_c[idx_t], # new first
            pt2_c_l[idx_t_l] # old last
            ], axis=0)

        # update points tracking reference
        self.landmarks_.kpt[idx_l[idx_t_l], :2] = pt2_c_l[idx_t_l]

        # undistort
        pt2_u_p = self.cvt_.pt2_to_pt2u(pt2_p_all)
        pt2_u_c = self.cvt_.pt2_to_pt2u(pt2_c_all)

        F = None
        if self.flag_ & ClassicalVO.VO_USE_FM_COR:
            # correct Matches by RANSAC consensus
            # NOTE : probably invalid to apply undistort() after correction
            F, msk_f = cv2.findFundamentalMat(
                    pt2_u_c,
                    pt2_u_p,
                    method=self.pEM_['method'],
                    param1=self.pEM_['threshold'],
                    param2=self.pEM_['prob'],
                    )
            msk_f = msk_f[:,0].astype(np.bool)
            print_ratio('FM correction', msk_f.sum(), msk_f.size)

            ## compute masks on corresponding indices
            idx_f_c = np.where(msk_f[:cam_lim])[0]
            idx_f_l = np.where(msk_f[cam_lim:])[0]
            idx_f   = np.concatenate([idx_f_c, cam_lim + idx_f_l],axis=0)

            # retro-update corresponding indices
            # to where pt2_u_p will be, based on idx_f
            idx_t   = idx_t[idx_f_c]
            cam_lim = len(idx_t)
            idx_t_l = idx_t_l[idx_f_l]

            # update pt2_u_c definitions
            pt2_u_c = pt2_u_c[idx_f]
            pt2_u_p = pt2_u_p[idx_f]

            pt2_p_all = pt2_p_all[idx_f] # log-only
            pt2_c_all = pt2_c_all[idx_f]

            #pt2_u_c2, pt2_u_p2 = cv2.correctMatches(F,
            #        pt2_u_c[None,...],
            #        pt2_u_p[None,...])

            #pt2_u_c2 = np.squeeze(pt2_u_c2, axis=0)
            #pt2_u_p2 = np.squeeze(pt2_u_p2, axis=0)

            ## -- will sometimes return NaN.
            #check_c = np.all(np.isfinite(pt2_u_c2))
            #check_p = np.all(np.isfinite(pt2_u_p2))

            #if check_c and check_p:
            #    ## -- viz begin --
            #    #try:
            #    #    fig = self.cfig_
            #    #    cax = fig.gca()
            #    #except Exception as e:
            #    #    self.cfig_ = plt.figure()
            #    #    fig = self.cfig_
            #    #    cax  = fig.add_subplot(1,1,1)#, projection='3d')
            #    #cax.cla()

            #    #pp = pt2_u_p
            #    #pc = pt2_u_c
            #    #print pp.shape
            #    #print pc.shape

            #    #col = np.random.uniform(size = (len(pp), 3))
            #    #cax.quiver(
            #    #        pp[:,0], pp[:,1],
            #    #        pc[:,0]-pp[:,0], pc[:,1]-pp[:,1],
            #    #        angles='xy',
            #    #        scale=1.0,
            #    #        scale_units='xy',
            #    #        color='r',#col,
            #    #        alpha=0.5,
            #    #        label='prv',
            #    #        )
            #    #pp = pt2_u_p2
            #    #pc = pt2_u_c2

            #    #print pp.shape
            #    #print pc.shape
            #    #cax.quiver(
            #    #        pp[:,0], pp[:,1],
            #    #        pc[:,0]-pp[:,0], pc[:,1]-pp[:,1],
            #    #        angles='xy',
            #    #        scale=1.0,
            #    #        scale_units='xy',
            #    #        color='b',#col,
            #    #        alpha=0.5,
            #    #        label='cor'
            #    #        )
            #    #cax.legend()
            #    #plt.pause(0.001)
            #    ## -- viz end --

            #    pt2_u_c = pt2_u_c2
            #    pt2_u_p = pt2_u_c2

        # unscaled camera pose + reconstructed points from triangulation
        idx_e, pt3, R_em, t_em = self.run_EM(
                pt2_u_c, pt2_u_p,
                no_gp=False,
                guess=(R_c0, t_c0)
                )

        # index of idx_e that applies to new (camera) points
        idx_e_msk_cam = (idx_e < cam_lim)
        idx_e_msk_lmk = ~idx_e_msk_cam

        idx_idx_e_cam = np.where(idx_e_msk_cam)[0]
        idx_idx_e_lmk = np.where(idx_e_msk_lmk)[0]

        idx_e_cam = idx_e[idx_idx_e_cam]
        idx_e_lmk = idx_e[idx_idx_e_lmk]

        # collect used points for matches + draw
        msk = np.zeros(len(pt2_p_all), dtype=np.bool)
        msk[idx_e] = True
        mim = drawMatches(img_p, img_c, pt2_p_all, pt2_c_all, msk)

        # Estimate #1 : Based on EM Results
        R_c, t_c_u = R_em, t_em # Camera-frame u-trans
        scale_b = np.linalg.norm(t_b) # current scale estimate
        print 'scale_b #0', scale_b
        scale_c = self.scale_c(scale_b, R_c, t_c_u,
                guess = t_b0
                )
        t_c = t_c_u * scale_c

        T_c2c1 = np.eye(4)
        T_c2c1[:3,:3] = R_c
        T_c2c1[:3,3:] = t_c.reshape(3,1)

        T_b2b1 = np.linalg.multi_dot([
            self.cvt_.T_c2b_,
            T_c2c1,
            self.cvt_.T_b2c_
            ])
        R_b, t_b = T_b2b1[:3,:3], T_b2b1[:3,3:]

        scale_b = np.linalg.norm(t_b)
        scale_c = np.linalg.norm(t_c)
        print 'scale_b #1', scale_b
        print 'scale_c #1', scale_c

        # cache results
        R_c1 = R_c
        R_b1 = R_b
        t_c1 = t_c
        t_b1 = t_b
        self.scales_['c1'].append([index, np.linalg.norm(t_c1)])
        self.scales_['b1'].append([index, np.linalg.norm(t_b1)])
        # == #1 finished ==

        # Estimate #2 : Based on Ground-Plane Estimation
        # <<-- initial guess, provided defaults in case of abort
        t_c1_u = (t_c1 / scale_c if scale_c >= np.finfo(np.float32).eps else t_c1)

        H, scale_c2, (R_c, t_c_u) = self.run_GP(
                pt2_u_c, pt2_u_p, pt3,
                scale_c, guess=(R_c1, t_c1_u)
                ) # note returned t_c is uvec
        t_c = t_c_u * scale_c2 # apply scale
        print 'scale_c #2', scale_c2

        rh = 0.25
        if (H is not None) and (F is not None):
            sF, msk_sF = score_F(pt2_u_c, pt2_u_p, F, self.cvt_)
            sH, msk_sH = score_H(pt2_u_c, pt2_u_p, H, self.cvt_)
            rh2 = float(sH) / (sH+sF)
            print_ratio('RH-gp', sH, sH+sF)
            if np.isfinite(rh2):
                rh = rh2
            else:
                print pt2_u_c.shape
                print pt2_u_p.shape
                print F
                print H
                print sH
                print sF
                raise ValueError(rh2, "Something is off.")
        
        # NOTE: estimate #2 does not produce R_b1 / t_b1
        # because it does not need to.
        # cache results
        R_c2 = R_c
        t_c2 = t_c
        self.scales_['c2'].append([index, np.linalg.norm(t_c2)])
        # == #2 finished ==

        # Estimate #3 : un-intelligently resolve two measurements.
        # TODO : figure out confidence scaling or variance.
        # NOTE/TODO : does NOT check for inconsistency in rotation.

        if np.dot(t_c1.ravel(), t_c2.ravel()) < 0:
            # facing opposite directions
            # most often happens in pure-rotation scenarios
            print '\t\t\t>>>>>>>>>>>>>>>>>>>>>>>>>>>WARNING : GP and EM Systems Disagree.'
            print_Rt(R_c1, t_c1)
            print_Rt(R_c2, t_c2)
            print '============================'

            # choose t_c consisent with current motion guess t_c0.
            score_c1 = (t_c0 * t_c1).sum() / ( np.linalg.norm(t_c0) * np.linalg.norm(t_c1))
            score_c2 = (t_c0 * t_c2).sum() / ( np.linalg.norm(t_c0) * np.linalg.norm(t_c2))
            print 'scores', score_c1, score_c2
            idx = np.argmax([score_c1, score_c2])
            print('selected : {}'.format(idx))
            R_c = [R_c1, R_c2][idx]
            t_c = [t_c1, t_c2][idx]
        else:
            # facing the same direction
            # interpolate the results for time-domain scale stability.
            #alpha = 0.25 # TODO : tune
            alpha = 1.0 - lerp(rh, 1.0, 0.5)
            # USE score_F/score_H results
            # heuristic : generally in favor of GP
            r_c1  = np.ravel(tx.euler_from_matrix(R_c1))
            r_c2 = np.ravel(tx.euler_from_matrix(R_c2))
            R_c = tx.euler_matrix(*lerp(r_c1, r_c2, alpha))[:3,:3]
            t_c = lerp(t_c1, t_c2, alpha) # TODO : better way to fuse?
            scale_c = np.linalg.norm(t_c)
        print 'scale_c #3 (lerp)', scale_c

        T_c2c1 = np.eye(4)
        T_c2c1[:3,:3] = R_c
        T_c2c1[:3,3:] = t_c.reshape(3,1)
        T_b2b1 = np.linalg.multi_dot([
            self.cvt_.T_c2b_,
            T_c2c1,
            self.cvt_.T_b2c_
            ])
        R_b, t_b = T_b2b1[:3,:3], T_b2b1[:3,3:]
        scale_b = np.linalg.norm(t_b)
        print 'scale_b #2', scale_b

        # cache results
        R_c3 = R_c
        R_b3 = R_b
        t_c3 = t_c
        t_b3 = t_b

        self.scales_['c3'].append([index, np.linalg.norm(t_c3)])
        self.scales_['b3'].append([index, np.linalg.norm(t_b3)])
        # == #3 finished ==

        pose_c_r = self.pRt2pose(pose_p, R_b, t_b)

        if self.flag_ & ClassicalVO.VO_USE_F2M:
            # Estimate #4 : Based on Landmarks
            # TODO : smarter way to incorporate ground-plane scale information??
            # estimate scale based on current pose guess
            # and recompute rectified pose_c_r

            # process old points first and obtain camera motion scale from correspondences
            scale_c = self.proc_f2m_old(
                    pose_c_r, scale_c,
                    pt3[idx_idx_e_lmk], # new points3
                    idx_l[idx_t_l[idx_e_lmk - cam_lim]], # landmark indices; NOTE : apply lmk index start offset
                    pt2_c_all[idx_e_lmk], # points here are needed to update kpt location
                    pt2_u_c[idx_e_lmk], # undistorted observations, required for obs. adding
                    )

            # new points
            kpt_p_cam = kpt_p[idx_t[idx_e_cam]]
            des_p_cam = des_p[idx_t[idx_e_cam]]
            # process new points (insert + update untracked old(?))
            scale_c = self.proc_f2m_new(
                    pose_c_r, scale_c, pt3[idx_idx_e_cam],
                    kpt_p_cam, des_p_cam,
                    pt2_c_all[idx_e_cam], pt2_u_c[idx_e_cam],
                    img_c, 
                    msg
                    )

            # update transforms with new scale_c information from proc_f2m.
            # (scale_c fusion happenes automatically within proc_f2m)
            t_c = t_c / np.linalg.norm(t_c) * scale_c
            T_c2c1 = np.eye(4)
            T_c2c1[:3,:3] = R_c
            T_c2c1[:3,3:] = t_c
            T_b2b1 = np.linalg.multi_dot([
                self.cvt_.T_c2b_,
                T_c2c1,
                self.cvt_.T_b2c_
                ])
            R_b, t_b = T_b2b1[:3,:3], T_b2b1[:3,3:]
            print 'scale_b #3', scale_b
            pose_c_r = self.pRt2pose(pose_p, R_b, t_b)

            # cache results
            R_c4 = R_c
            R_b4 = R_b
            t_c4 = t_c
            t_b4 = t_b

            self.scales_['c4'].append([index, np.linalg.norm(t_c4)])
            self.scales_['b4'].append([index, np.linalg.norm(t_b4)])
            # == #4 finished ==

        # Estimate #5 : return Post-filter results as UKF Posterior
        # NOTE: self.graph_ contains pose posterior.
        self.ukf_l_.update(pose_c_r)
        pose_c_r = self.ukf_l_.x[:3].copy()
        self.graph_.add_pos(pose_c_r.copy())
        # NOTE: estimate #5 does not produce R_c, t_c, R_b, t_c
        # because it is not needed.
        # == #5 finished ==

        # prune
        # NOTE : indexing offset here is 100% because of visualization.
        # TODO : make more robust
        if (index+1 >= self.prune_freq_) and ( (index+1) % self.prune_freq_)==0 :
            # 1. prep viz (if available)
            if ax is not None:
                ax['prune_0'].cla()
                ax['prune_1'].cla()
            # 2. draw pre-prune
            if ax is not None:
                self.graph_.draw(ax['prune_0'], self.cvt_, self.landmarks_) # pre-prune
            # 3. prune
            keep_idx = self.landmarks_.prune()
            self.graph_.prune(keep_idx)
            # 4. draw post-prune
            if ax is not None:
                self.graph_.draw(ax['prune_1'], self.cvt_, self.landmarks_) # post-prune

        # show landmark statistics
        self.lm_cnt_.append( self.landmarks_.size_ )
        if ax is not None:
            ax['lmk_cnt'].cla()
            ax['lmk_cnt'].plot(self.lm_cnt_)
            ax['lmk_cnt'].set_title('# Landmarks Stat')

        # NOTE!! this vvvv must be called after all cache_BA calls
        # have been completed.
        if self.flag_ & ClassicalVO.VO_USE_BA:
            ba_win = None
            run_ba = False

            for win in self.ba_pyr_:
                # Survey list of BA frequencies (windows)
                # And run the largest possible BA
                # if multiple windows satisfy the condition.
                s_check = (len(self.graph_.pose_) >= win)
                f_check = (len(self.graph_.pose_) % win == 0)
                if s_check and f_check:
                    ba_win = win
                    run_ba = True
                    break

            if run_ba:
                # run BA every [win] frames
                print('Running BA @ scale={}'.format(ba_win))
                ba_res = self.run_BA(ba_win, ax=ax)
                if ba_res is not None:
                    # "historic" UKF
                    xs, Ps, dts = self.s_hist_.query(ba_win)
                    self.ukf_h_.x = xs[0]
                    self.ukf_h_.P = Ps[0]

                    xs = []
                    Ps = []
                    for (h_dt, h_z) in zip(dts, ba_res):
                        xs.append( self.ukf_h_.x.copy() )
                        Ps.append( self.ukf_h_.P.copy() )
                        ukf_Q, ukf_R = get_QR(self.ukf_h_.x[:3], dt)
                        self.ukf_h_.Q = ukf_Q
                        self.ukf_h_.R = ukf_R
                        self.ukf_h_.predict(h_dt)
                        self.ukf_h_.update(h_z)

                    # overwrite state history
                    # note that self.s_hist_ only stores priors. (prv)
                    # self.graph_ stores posteriors. (current)
                    self.s_hist_.update(ba_win, xs, Ps, dts)

                    # copy to local ukf posterior
                    self.ukf_l_.x = self.ukf_h_.x.copy()
                    self.ukf_l_.P = self.ukf_h_.P.copy()

                    # result
                    pose_c_r = self.ukf_l_.x[:3].copy()

        # plot scale

        if ax is not None:
            ax['scale'].cla()
            for (k, v) in self.scales_.items():
                s_i, s_v = zip(*v)
                ax['scale'].plot(s_i, s_v, label=k)
            ax['scale'].legend()

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
        pt2_c_rec, front_msk = self.project_BA(
                np.asarray([pose_c_r]), pt3_m,
                return_msk=True
                )
        rec_msk = np.logical_and.reduce([
            front_msk,
            0 <= pt2_c_rec[:,0],
            pt2_c_rec[:,0] < 640, #TODO: hardcoded
            0 <= pt2_c_rec[:,1],
            pt2_c_rec[:,1] < 480
            ])
        #pt2_c_rec, rec_msk = self.cvt_.pt3_pose_to_pt2_msk(pt3_m, pose_c_r, distort=False)
        rec_idx = np.where(rec_msk)[0]
        pt2_c_rec = pt2_c_rec[rec_idx]

        # apply filter by project-able landmarks
        #pt3_m = pt3_m[rec_idx]
        #col_m = col_m[rec_idx]

        # option : query only visible points
        #qres = self.landmarks_.query(pose_c_r, self.cvt_, atol=np.deg2rad(60.0) )
        #pt2_c_rec, pt3_m, _, _, _, lm_idx = qres
        #col_m = self.landmarks_.col[lm_idx]
        pt3_m = pt3_m.dot(self.T_c2b_[:3,:3].T) + self.T_c2b_[:3,3]
        
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

        # TODO : propagate status messages for the GUI title
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
