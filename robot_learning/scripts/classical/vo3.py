"""
Semi-Urgent TODOs:
    - memory management (masks/fancy indexing creates copies; reuse same-sized arrays etc.)
    - Loop Closure!!
    - Try to apply the homography model from ORB_SLAM??
    - Keyframes?
    - Incorporate Variance information in BA?
    - Cross-Frame (i.e. temporal displacement >1) Matching + Tracking
    - Currently, All frames are processed twice?
"""

from collections import namedtuple, deque, defaultdict
from tf import transformations as tx
import cv2
import numpy as np
import sys
import time

from vo_common import recover_pose, drawMatches, recover_pose_from_RT
from vo_common import robust_mean, oriented_cov, show_landmark_2d
from vo_common import Landmarks, Conversions
from vo_common import print_Rt
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D

from scipy.sparse import lil_matrix, csr_matrix, bsr_matrix
from scipy.optimize import least_squares
from scipy.optimize._lsq.common import scale_for_robust_loss_function
from scipy.optimize._lsq.least_squares import construct_loss_function

from ukf import build_ukf, build_ekf, get_QR
from ba import ba_J, ba_J_v2, schur_trick
from opt import solve_PNP, solve_TRI, solve_TRI_fast

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
    return (a*(1.0-w)) + (b*w)

def resolve_Rt(R0, R1, t0, t1, alpha=0.5, guess=None):
    # TODO : deal with R0/R1 disagreement?
    # usually not a big issue.
    if np.dot(t0.ravel(), t1.ravel()) < 0:
        # disagreement : choose
        if guess is not None:
            # reference
            R_ref, t_ref = guess

            # precompute norms
            d_ref = np.linalg.norm(t_ref)
            d0 = np.linalg.norm(t0)
            d1 = np.linalg.norm(t1)

            # compute alignment score
            score0 = np.dot(t_ref.ravel(), t0.ravel()) / (d_ref * d0)
            score1 = np.dot(t_ref.ravel(), t1.ravel()) / (d_ref * d1)
        else:
            # reference does not exist, choose smaller t
            score0 = np.linalg.norm(t0)
            score1 = np.linalg.norm(t1)

        idx = np.argmax([score0, score1])

        return [(R0,t0), (R1,t1)][idx]
    else:
        # agreement : interpolate
        # rotation part
        R0h = np.eye(4, dtype=np.float32)
        R0h[:3,:3] = R0
        R1h = np.eye(4, dtype=np.float32)
        R1h[:3,:3] = R1
        q0 = tx.quaternion_from_matrix(R0h)
        q1 = tx.quaternion_from_matrix(R1h)
        q = tx.quaternion_slerp(q0, q1, alpha)
        R = tx.quaternion_matrix(q)[:3,:3]

        # translation part
        t = lerp(t0, t1, alpha)
        return (R, t)

def p2vec(p2, p, l, cvt):
    # where they are 'supposed to be'
    # Nx2, image coordinates
    g = p2
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
    return g

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

def lsq(f, x0, jac, args,
        ftol=1e-8, xtol=1e-8,
        max_nfev=1000,
        # below, currently ignored parameters
        loss=None,
        method=None,
        verbose=None,
        W=None,
        tr_solver=None,
        f_scale=1.0,
        ):

    def f_cost(x):
        return 0.5 * np.dot(x,x)

    # initial values
    x = x0
    F = f(x, *args)
    J = jac(x, *args)
    cost = f_cost(F)

    for i in range(max_nfev):
        ts = []
        ts.append( time.time())
        # compute deltas
        J = jac(x, *args)
        ts.append( time.time()) 
        # NOTE: *args[:2] is a hack to get n_c & n_l populated
        dx = schur_trick(J, F, W=W, *args[:2])
        ts.append( time.time()) 
        # TODO: check xtol
        x += dx
        ts.append( time.time()) 
        F1 = f(x, *args)
        ts.append( time.time()) 
        cost1 = f_cost(F1)
        ts.append( time.time()) 

        # check ftol
        d_cost = cost - cost1 
        print('[{}] ref : {}'.format(i, d_cost / cost))
        if d_cost < ftol * cost:
            print('satisfied ftol')
            break

        # cache data
        F = F1
        cost = cost1

        dt = np.diff(ts)

        #print 'net time summary'
        #print dt
        #print dt / dt.max()

    return x

from scipy.sparse.linalg import svds as sparse_svd

def cov_from_jac(J):
    """ from scipy/optimize/minpack.py#L739 """
    #_, s, VT = np.linalg.svd(J, full_matrices=False)
    res = sparse_svd(J, return_singular_vectors='vh')
    _, s, VT = res
    thresh = np.finfo(np.float32).eps * max(J.shape) * s[0]
    s = s[s > thresh]
    VT = VT[:s.size]
    pcov = np.dot(VT.T / s**2, VT)
    return pcov

def extract_block_diag(a, n, k=0):
    a = np.asarray(a)
    if a.ndim != 2:
        raise ValueError("Only 2-D arrays handled")
    if not (n > 0):
        raise ValueError("Must have n >= 0")

    if k > 0:
        a = a[:,n*k:] 
    else:
        a = a[-n*k:]

    n_blocks = min(a.shape[0]//n, a.shape[1]//n)

    new_shape = (n_blocks, n, n)
    new_strides = (n*a.strides[0] + n*a.strides[1],
                   a.strides[0], a.strides[1])
    return np.lib.stride_tricks.as_strided(a, new_shape, new_strides)

class VGraph(object):
    """ Simple wrapper for visibility graph """
    # TODO : unify dynamic container interface or something.
    # (i.e. with Landmarks() )
    def __init__(self, cvt, cap0=1024):
        # processing handle
        self.cvt_ = cvt

        # pose data
        # self.index_ points to current pose index.
        # current pose is empty (invalid), so setting to -1
        self.index_ = -1
        self.pos_ = []
        self.cov_ = [] # TODO : is it necessary to store "non-diagonal" pos->lmk cov?
        self.dat_ = [] # Extra frame data (img/kpt/des)
        self.kf_ = build_ekf() # << sets x0,P0 to modest defaults.
        # NOTE: landmark data is stored in ClassicalVO.landmarks_

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

    def set_data(self, dat, i=None):
        # 0. get index
        if i is None:
            # automatically set index
            i = self.index
        # 1. append or set data
        if i == len(self.dat_):
            self.dat_.append(dat)
        else:
            # this won't usually happen,
            # but override data at index if requsted
            self.dat_[i] = dat

    def set_data_from(self, img, i=None):
        # 1. process data
        img = self.cvt_.img_to_imgu(img) # undistort
        kpt = self.cvt_.img_to_kpt(img,
                subpix=True
                )
        kpt, des = self.cvt_.img_kpt_to_kpt_des(img, kpt)
        pt2 = self.cvt_.kpt_to_pt(kpt)
        rsp = np.float32([e.response for e in kpt])
        dat = (img, kpt, des, pt2, rsp)

        # 2. delegate to set_data routine
        self.set_data(dat, i)

    def get_data(self, i=None):
        if i is None:
            # automatically set index
            i = self.index

        if i < 0:
            # rectify index for bounds checking below
            i += len(self.dat_)

        if (i<0) or (i>self.index):
            # check bounds; self.index should be the last valid index
            return None

        return self.dat_[i]

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

    def add_obs(self, li, p2, pi=None):
        """
        Add observation to lmk[li]
        WARN: .add_obs() should be called AFTER predict() if pi=None.
        """
        if pi is None:
            # automatically figure out current pose index.
            pi = self.index_

        n = len(li) # NOTE: in general, cannot use len(pi).
        if self.size_ + n > self.capacity_:
            self.resize(self.capacity_ * 2)
            self.add_obs(li, p2)
        else:
            i = np.s_[self.size_:self.size_+n]
            self.pi_[i] = pi
            self.li_[i] = li
            self.p2_[i] = p2
            self.size_ += n

    def prune(self, keep_idx):
        """
        Reindex observation graph with keep_idx.
        i.e. assume lmk_c = lmk_p[keep_idx] due to pruning.
        """
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

    def query(self, t):
        """
        t = how much to look back.
        
        Note that by the time query() is called,
        VGraph.predict(append=True) must have been called
        such that current pose at VGraph.pos_[VGraph.index] is valid.
        """
        # assume (0...,1...,2...,3...,4..,5..,6...)
        # then self.pi[-1] = 6
        # if t = 3, then desired result is [4...,5...,6...]

        i1 = self.index + 1 # == retrieve up to self.index
        msk = (self.pi >= (i1 - t)) # expresses range [i0,i1)
        idx = np.where(msk)[0]

        p0 = np.asarray(self.pos_[-t:])
        v0 = np.asarray(self.cov_[-t:])
        pi = self.pi[idx] 
        pi -= pi.min() # normalized p0 index, with offset removed

        li = self.li[idx]
        p2 = self.p2[idx]
        return p0[:,:3], v0, pi, li, p2

    """ pose-related """
    def initialize(self, x=None, P=None):
        # pose[i] = {x, P, dt}, where dt[i] = t[i] - t[i-1]
        if x is not None:
            self.kf_.x = x
        if P is not None:
            self.kf_.P = P

        # initialize cache
        self.pos_ = [self.kf_.x.copy()]
        self.cov_ = [self.kf_.P.copy()]
        self.dt_  = [0.0] # dt[i] = 0, dt[1] = t[1] - t[0] ...
        self.index_ = 0 # < now points to a valid location

    def predict(self, dt, commit=True):
        """
        Predict what the next state will be given the current state.
        with commit:=True, the state information will be updated with the predictions.
        """
        if not commit:
            # look-only; system state must not be changed.
            # need to restore everything afterwards.
            state0 = (
                    self.kf_.x.copy(), self.kf_.P.copy(),
                    self.kf_.Q.copy(), self.kf_.R.copy()
                    )

        prv = self.kf_.x[:3].copy()
        Q, R = get_QR(prv, dt)
        self.kf_.Q = Q
        self.kf_.R = R
        self.kf_.predict(dt) # self.kf_.x, P holds new pose
        cur = self.kf_.x[:3].copy() # TODO : support reset
        scale = np.linalg.norm(cur[:2] - prv[:2])

        if commit:
            # update + save cache
            self.index_ += 1
            self.pos_.append( self.kf_.x.copy() )
            self.cov_.append( self.kf_.P.copy() )
            self.dt_.append( dt )
        else:
            # restore
            self.kf_.x = state0[0]
            self.kf_.P = state0[1]
            self.kf_.Q = state0[2]
            self.kf_.R = state0[3]

        return prv, cur, scale

    def update(self, win, pos,
            cov=None, hard=False,
            dw=1):
        """
        Update pose[-win:] part of VGraph.
        if hard:=True, then the values are completely overwritten.
        Otherwise, kf-filtered results are written instead.
        """
        if hard:
            # hard update
            self.pos_[-win:] = pos
            if cov is not None:
                self.cov_[-win:] = cov

            # also apply results to KF
            self.kf_.x[:3] = pos[-1]
            self.kf_.P[:3,:3] = cov[-1]
        else:
            # soft update; filter

            # initialize from cache
            self.kf_.x = self.pos_[-win-1] # set to prior
            self.kf_.P = self.cov_[-win-1]

            if cov is None:
                # make into array
                cov = [None for _ in range(win)]

            u_idx = np.arange(-win, 0)
            for (i, x, R) in zip(u_idx, pos, cov):
                # TODO : check if dt indexing is valid
                dt = self.dt_[i]

                Q, R_ = get_QR(self.kf_.x[:3], dt)
                if R is None:
                    R = R_
                self.kf_.Q = Q
                self.kf_.R = R

                self.kf_.predict(dt) # NOTE: dt[i] = t[i] - t[i-1]
                self.kf_.update(x)

                # set filtered data
                self.pos_[i] = self.kf_.x
                self.cov_[i] = self.kf_.P

        return self.pos_[-1][:3] # x-y-h components, for convenience

    def draw(self, ax, lmk):
        cvt = self.cvt_

        p0, v0, pi, li, p2 = self.query(t=self.index+1)

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
        """
        NOTE: current pose index.
        WARN: VGraph.predict(commit=True) will increment the index.
        """
        return self.index_

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

    VO_DEFAULT = VO_USE_FM_COR | VO_USE_TRACK | VO_USE_SCALE_GP | \
            VO_USE_BA | VO_USE_F2M | VO_USE_LM_KF | \
            VO_USE_KPT_SPX | VO_USE_MXCHECK

    def __init__(self, cinfo=None):
        # define configuration
        self.flag_ = ClassicalVO.VO_DEFAULT
        self.flag_ &= ~ClassicalVO.VO_USE_HOMO # TODO : doesn't really work?
        self.flag_ &= ~ClassicalVO.VO_USE_BA
        #self.flag_ |= ClassicalVO.VO_USE_PNP
        self.flag_ &= ~ClassicalVO.VO_USE_FM_COR # doesn't really work anymore?
        #self.flag_ &= ~ClassicalVO.VO_USE_SCALE_GP

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

        # all heuristics that are used outside of __init__
        #self.pH_ = dict(
        #        rad_nmx2=16.0, # 2D Image Coordinates Non-max Radius
        #        rad_nmx3=0.05, # 3D Physical Coordinates non-max Radius
        #        max_ferr=2.0,  # Bidirectional Flow Check Threshold for Tracking
        #        min_s=1e-4, # Minimum SCale for Landmark Processing
        #        min_np=16   # Minimum # Points required for processing
        #        k_nmx2=2,   # Max Number of neighbors to suppress for 2D Non-Max
        #        k_nmx3=2,   # Max Number of neighbors to suppress for 3D Non-Max
        #        w_g2e =0.5,  # GP vs. EM estimate interpolation weight
        #        w_f2m =0.2,  # Frame-To-Map Weight




        # define "system" parameters
        self.pEM_ = dict(
                method=cv2.FM_RANSAC,
                prob=0.999,
                threshold=1.0)
        self.pLK_ = dict(
                winSize = (12,6),
                maxLevel = 4, # == effective winsize up to 32*(2**4) = 512x256
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.003),
                flags = 0,
                minEigThreshold = 1e-3 # TODO : disable eig?
                )
        self.pBA_ = dict(
                ftol=1e-4,
                xtol=np.finfo(float).eps,
                loss='huber',
                max_nfev=1024,
                method='trf',
                verbose=2,
                tr_solver='lsmr',
                f_scale=1.0
                )
        self.pPNP_ = dict(
                iterationsCount=10000,
                reprojectionError=2.0,
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
                nfeatures=1024,
                scaleFactor=1.2,
                nlevels=8,
                # NOTE : scoretype here influences response-based filters.
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
        self.prune_freq_ = 16

        # bundle adjustment + loop closure
        # sort ba pyramid by largest first
        ba_pyr = [4,16,64,256,1024]#[4,16,64,256,1024]#[2,4,16,64,256,1024]
        self.ba_pyr_  = np.sort(ba_pyr)[::-1]
        self.graph_ = VGraph(self.cvt_)

        # reference scale
        self.scale0_ = 1.0 # set to 1 by default
        self.use_s0_ = True

        # logging / visualization
        self.tmp_ = defaultdict(lambda:[])
        self.fig_ = defaultdict(lambda: plt.figure())
        self.cnt_lm_tot_ = []
        self.cnt_lm_trk_ = []
        self.cnt_lm_new_ = []
        self.scales_ = defaultdict(lambda:[])

    def pts_nmx(self,
            pt_new, pt_ref,
            rsp_new, rsp_ref,
            k=16,
            radius=1.0 # << NOTE : supply valid radius here when dealing with 2D Data
            ):

        # NOTE : somewhat confusing;
        # here suffix c=camera, l=landmark.
        # TODO : is it necessary / proper to take octaves into account?
        if len(pt_ref) < k:
            # Not enough references to apply non-max with.
            return np.arange(len(pt_new))

        # compute nearest neighbors
        neigh = NearestNeighbors(n_neighbors=k, radius=radius)
        neigh.fit(pt_ref)

        # NOTE : 
        # radius_neighbors would be nice, but indexing is difficult to use
        # res = neigh.radius_neighbors(pt_new, return_distance=False)
        d, i = neigh.kneighbors(pt_new, return_distance=True)

        # too far from other landmarks to apply non-max
        msk_d = (d.min(axis=1) >= radius)
        # passed non-max
        msk_v = np.all(rsp_ref[i] < rsp_new[:,None], axis=1) # 

        # format + return results
        msk = (msk_d | msk_v)
        idx = np.where(msk)[0]
        print_ratio('non-max', len(idx), msk.size)
        return idx

    def track(self, img1, img2, pt1, pt2=None,
            thresh=2.0
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

        # == VIZ TRACK STAT BEGIN ==
        # delta = np.abs(pt2[idx] - pt1[idx])
        # self.tmp_['optflow'].append(delta)
        # ofig = self.fig_['optflow']
        # oax = ofig.gca()
        # oax.cla()
        # oax.hist(np.concatenate(self.tmp_['optflow'], axis=0))
        # plt.pause(0.001)
        # print 'optflow log', delta.max(axis=0)
        # == VIZ TRACK STAT END ==

        return pt2, idx

    def proc_f2m_old(self,
            pose, scale,
            pt3_new_c, lm_idx,
            pt2_new,
            pt2_new_u,
            img_c,
            ax,
            msg=''
            ):
        """
        NOTE : pt3_new_c MUST BE in camera coordinates.
        """
        # mostly targetted at updating existing landmarks.

        # estimate scale from landmark correspondences
        # TODO : evaluate whether or not to use z-depthvalue or the full distance
        # may not matter too much given that scale from either SHOULD be consistent.
        # opt1 : norm

        # save input scale ( which has been applied to pt3_new_c )
        scale0 = scale
        print 'scale0', scale0

        p_lm_0 = self.landmarks_.pos[lm_idx] # landmark points in cam-map coord.

        ## override input pose with PnP information
        #_, pose_pnp = self.run_PNP(p_lm_0, pt2_new, pose)
        #if pose_pnp is not None:
        #    # TODO : instead of full override, consider merging them
        #    # intelligently?
        #    print 'pose-in', pose.ravel()
        #    print 'pose-pnp', pose_pnp
        #    pose = pose_pnp
        #    #print 'pnp scale', np.linalg.norm(np.subtract(pose[:2], self.graph_.pos_[-2][:2]))

        p_lm_c = self.cvt_.map_to_cam(p_lm_0, pose)

        d_lm_old = p_lm_c[:,2]#np.linalg.norm(p_lm_c, axis=-1)
        d_lm_new = pt3_new_c[:,2]#np.linalg.norm(pt3_new_c, axis=-1)

        scale_rel = (d_lm_old / d_lm_new).reshape(-1,1)
        scale_rel = scale_rel[
                np.logical_and.reduce([
                    np.isfinite(scale_rel),
                    scale_rel > 0.0
                    ])]

        #if len(d_lm_old) > 0:
        #    sax = self.fig_['scale'].gca()
        #    sax.cla()
        #    #sax.hist(scale_rel)
        #    sidx = np.random.randint(0, len(d_lm_old), min(len(d_lm_old), 32))
        #    sax.plot(d_lm_old[sidx], 'r+', label='old')
        #    #sax.plot(d_lm_new, 'bx', label='new')
        #    sax.plot(d_lm_new[sidx] * scale_rel.mean(), 'c.', label='new-s')
        #    sax.legend()
        #    print 'setting limits'
        #    lo = np.percentile(d_lm_old, 20)
        #    hi = np.percentile(d_lm_old, 80)
        #    sax.set_ylim(lo, hi)
        #    print 'scale_rel', robust_mean(scale_rel)
        #    plt.pause(0.001)

        #scale_rel_std = np.exp(np.log(scale_rel).std()) / scale0
        #print('estimated scale stability', scale_rel_std)
        #print(np.exp(robust_mean(np.log(scale_rel))))

        # acquire scale corrections ...
        if len(lm_idx) > 8 and len(scale_rel) > 0:
            scale_est = scale * robust_mean(scale_rel)
        else:
            # scale estimates are anticipated to be unstable.
            # use input scale
            scale_est = scale

        if len(d_lm_old) > 0:
            print_ratio('estimated scale ratio', scale_est, scale)
            # TODO : tune scale interpolation alpha
            alpha = 0.75 # high trust in ground-plane/ukf based estimate
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
            scale = np.exp(lerp(np.log(scale), np.log(scale_est), alpha))
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
            """
        else:
            # implicit : scale = scale
            pass

        # == scale_is_believable @ >= 5e-3m translation
        # TODO : figure out better heuristic?
        run_lm = (scale >= 1e-4) # TODO : magic-ish

        # update landmarks
        update_lm = (run_lm and self.flag_ & ClassicalVO.VO_USE_LM_KF)
        if update_lm:
            # update landmarks from computed correspondences
            # TODO : compute appropriate landmark variances from
            # all the currently associated poses.

            # TODO : apply updated scale corrections
            pt3_new_c = pt3_new_c * scale / scale0

            var_lm_new = self.landmarks_.lm_var(self.cvt_,
                    pose, pt3_new_c)
            pt3_new_c_0 = self.cvt_.cam_to_map(pt3_new_c, pose)

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
            self.graph_.add_obs(lm_idx, pt2_new_u)

        # == PNP ===================
        # flag to decide whether to run PNP
        run_pnp = bool(self.flag_ & ClassicalVO.VO_USE_PNP)
        run_pnp &= lm_idx.size >= 16 # use at least > 16 points
        # TODO : REVIVE PNP AT SOME POINT
        # reset PNP data no matter what
        if run_pnp:
            # either landmarks are wrong, or poses are wrong, which influences PNP performance.
            # The issue is that both of them must be simultaneously optimized.

            #pt_map = p_lm_v2_0 # --> "FAKE" map from current observation
            #pt_map = lerp(pos_lm[i1[lm_idx_e]], p_lm_v2_0, 0.15) # compromise?
            pt_map = self.landmarks_.pos[lm_idx]
            pt_cam = pt2_new_u

            # filter by above-GP features
            # (TODO : hack because ground plane tends to be somewhat homogeneous
            # in the test cases)
            pt_map_b = pt_map.dot(self.cvt_.T_c2b_[:3,:3].T) + self.cvt_.T_c2b_[:3,3:].T
            ngp_msk =  (pt_cam[:,1] < self.y_GP) # based on camera
            ngp_msk &= (pt_map_b[:,2] > 0.0)     # based on map
            ngp_idx = np.where(ngp_msk)[0]
            pt_map = pt_map[ngp_idx]
            pt_cam = pt_cam[ngp_idx]

            #pt_cam_rec = pt2_lm[i1[lm_idx_e[ngp_idx]]] # un-updated
            pt_cam_rec, _ = self.cvt_.pt3_pose_to_pt2_msk(
                    pt_map, pose) # updated version

            # == debugging ==
            if ax is not None:
                ax['pnp'].cla()
                ax['pnp'].imshow(img_c[...,::-1])
                ax['pnp'].plot(pt_cam_rec[:,0], pt_cam_rec[:,1], 'rx', alpha=0.5)
                ax['pnp'].plot(pt_cam[:,0], pt_cam[:,1], 'b+', alpha=0.5)

                ax['pnp'].set_xlim(0, img_c.shape[1])
                ax['pnp'].set_ylim(0, img_c.shape[0])
                if not ax['pnp'].yaxis_inverted():
                    ax['pnp'].invert_yaxis()
                ax['pnp'].set_title('lmk overlay')
            # == debugging ==
            msg = self.run_PNP(pt_map, pt_cam, pose, ax=ax, msg=msg)
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
        return scale, msg

    def filter_pts(
            self,
            pose,
            pt3_new, pt2_new,
            des_new, rsp_new,

            k2=4,
            r2=16.0,
            k3=4,
            r3=0.05
            ):
        """
        Apply a general filter from the landmarks
        to exclude redundant points from the candidate point set.
        """

        # 0: query() applies a light filter to returned references.
        qres = self.landmarks_.query(pose, self.cvt_,
                atol=np.deg2rad(60.0),
                dtol=1.2, # NOTE: not meters; defines depth ratio [0.5, 2.0]
                trk=True
                )
        pt2_ref, pt3_ref, des_ref, var_ref, cnt_ref, lm_idx = qres
        rsp_ref = self.landmarks_.kpt[lm_idx, 2]

        # initialize index
        insert_idx = np.arange(len(pt3_new))

        if len(pt3_ref) >= k2: # match with k below
            # 1: 3D Non-Max Suppression in euclidean coordinates
            fi1 = self.pts_nmx(
                    pt3_new, pt3_ref,
                    rsp_new, rsp_ref,
                    k=k2,
                    radius=0.05
                    )

            # apply filter
            pt3_new = pt3_new[fi1]
            pt2_new = pt2_new[fi1]
            des_new = des_new[fi1]
            rsp_new = rsp_new[fi1]
            insert_idx = insert_idx[fi1]

        if len(pt3_ref) >= k3: # match with k below
            # 2: 2D Non-Max Suppression in angular projected coordinates
            fi2 = self.pts_nmx(
                    pt2_new, pt2_ref,
                    rsp_new, rsp_ref,
                    k=k3,
                    radius=16
                    )

            # apply filter
            pt3_new = pt3_new[fi2]
            pt2_new = pt2_new[fi2]
            des_new = des_new[fi2]
            rsp_new = rsp_new[fi2]
            insert_idx = insert_idx[fi2]

        # 3: apply lenient matcher, to prevent re-inserting the same landmark
        _, nfi3 = self.cvt_.des_des_to_match(
                des_ref,
                des_new,
                lowe=1.0,
                maxd=128.0,
                cross=False
                )

        # invert index
        fi3_msk = np.ones(len(des_new), dtype=np.bool)
        fi3_msk[nfi3] = False
        fi3 = np.where(fi3_msk)[0]

        # apply filter
        pt3_new = pt3_new[fi3]
        pt2_new = pt2_new[fi3]
        des_new = des_new[fi3]
        rsp_new = rsp_new[fi3]
        insert_idx = insert_idx[fi3]

        # 4: filter by map point distance
        # note that this is permissible because pt3_new is in the camera coord.
        d_pt = np.linalg.norm(pt3_new, axis=-1)
        #print d_pt.max(), d_pt.min(), d_pt.mean()
        #print pt3_new[:,2].max(), pt3_new[:,2].min(), pt3_new[:,2].mean()
        fi4_msk = np.logical_and.reduce([
            0.0 < pt3_new[:,2], # z-check
            0.0 < d_pt, # TODO: min depth check??
            d_pt < 100.0
            ])
        fi4 = np.where(fi4_msk)[0]

        # apply filter
        pt3_new = pt3_new[fi4]
        pt2_new = pt2_new[fi4]
        des_new = des_new[fi4]
        rsp_new = rsp_new[fi4]
        insert_idx = insert_idx[fi4]

        # alignment validation

        #if len(pt3_new) > 0:
        #    pt2_rec, msk = self.cvt_.pt3_pose_to_pt2_msk(
        #            self.cvt_.cam_to_map(pt3_new, pose), pose)
        #    print_ratio('rec_msk', msk.sum(), msk.size)
        #    print 'rec_err', np.sqrt(np.percentile(np.square(pt2_rec - pt2_new), 50))

        #    try:
        #        tfig = self.tfig_
        #    except Exception as e:
        #        self.tfig_ = plt.figure()
        #        tfig = self.tfig_
        #    tax = tfig.gca()
        #    tax.cla()
        #    tax.plot(pt2_rec[msk,0], pt2_rec[msk,1], 'b+', label='pt3')
        #    tax.plot(pt2_new[msk,0], pt2_new[msk,1], 'rx', label='obs')
        #    if not tax.yaxis_inverted():
        #        tax.invert_yaxis()

        #    plt.pause(0.001)


        return insert_idx


    def proc_f2m_new(self,
            pose, scale, pt3,
            kpt_p, des_p,
            pt2_c, pt2_u_c,
            pt2_p, pt2_u_p,
            img_c,
            msg=''
            ):
        """
        Frame-To-Map Processing: New points.
        Mostly focused on pre-insertion filtering.

        TODO: should not really return scale, since
        only a few landmarks should have correspondences here.
        """

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
        n_new = 0
        insert_lm = (scale >= 1e-4) # TODO : magic-ish

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
            msk_d = (lm_d < 20.0) # NOTE : heuristic to suppress super-far points

            # apply filter
            idx_d = np.where(msk_d)[0]
            insert_idx = insert_idx[idx_d]

            # finalize insertion
            n_new = insert_idx.size
            print('adding {} landmarks : {}->{}'.format(n_new,
                len(self.landmarks_.pos), len(self.landmarks_.pos)+n_new
                ))
            pt3_new_c = pt3[insert_idx]
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
            # add edge from previous obs. <<NEW!!
            self.graph_.add_obs(np.arange(li_0, li_1),
                    pt2_u_p[insert_idx],
                    pi=self.graph_.index-1
                    )
            # add edge from current obs.
            self.graph_.add_obs(np.arange(li_0, li_1),
                    pt2_u_c[insert_idx],
                    pi=self.graph_.index
                    )
        # TODO : evaluate what to return from here
        # really shouldn't return scale?
        return scale, n_new

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
        T_o2b[:,0,1] = s # NOTE: transposed z-axis rotation.
        T_o2b[:,1,0] = -s
        T_o2b[:,1,1] = c
        T_o2b[:,2,2] = 1

        # Translation part
        T_o2b[:,0,3] = -y*s - x*c
        T_o2b[:,1,3] = x*s - y*c

        # Homogeneous part
        T_o2b[:,3,3] = 1

        lmk_h = self.cvt_.pt_to_pth(lmk)

        # NOTE : einsum was not faster.
        pt2_h = reduce(np.matmul,[
            self.K_, # 3x3
            self.cvt_.T_b2c_[:3], # 3x4
            T_o2b, # Nx4x4
            self.cvt_.T_c2b_, # 4x4
            lmk_h[...,None]])[...,0] # Nx4x1
        
        if return_h:
            return pt2_h

        pt2 = self.cvt_.pth_to_pt(pt2_h)
        if return_msk:
            #simple depth check
            #m_h = 64*2
            #m_w = 48*2
            msk = np.logical_and.reduce([
                pt2_h[..., -1] >= 0, # positive depth check
                #-m_h <= pt2[:,0],
                #pt2[:,0] < 640+m_h,
                #-m_w <= pt2[:,1],
                #pt2[:,1] < 480+m_w
                ])
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
        #print_ratio('invalid BA', msk.sum(), msk.size)

        err = obs_pt2 - prj_pt2 # == y-x
        #err[msk] = 0
        #err = prj_pt2 - obs_pt2 # == x-y
        # TODO : is it actually necessary to apply the mask?
        #i_null = np.where(~msk)[0]
        #err[i_null] = 0
        return err.ravel()

    def residual_BA2(self, params, n_c, n_l, ci, li, obs_pt2):
        pos = params[:n_c*3].reshape(-1, 3) # camera 2d pose (x,y,h)
        lmk = params[n_c*3:].reshape(-1, 3) # landmark positions

        # unpack parameters
        # lx, ly, lz = lmk[li].T
        x, y, h = pos[ci].T

        # observation vector
        p = self.cvt_.pt_to_pth(obs_pt2)
        p = p.dot(self.cvt_.Ki_.T)

        n = p / np.linalg.norm(p, axis=-1, keepdims=True) # normalize
        n = np.nan_to_num(n)

        # convert lmk w.r.t. pos
        c = np.cos(h)
        s = np.sin(h)

        T_o2b = np.zeros((len(h),4,4), dtype=np.float32)

        # Rotation Part
        T_o2b[:,0,0] = c
        T_o2b[:,0,1] = s # NOTE: transposed z-axis rotation.
        T_o2b[:,1,0] = -s
        T_o2b[:,1,1] = c
        T_o2b[:,2,2] = 1

        # Translation part
        T_o2b[:,0,3] = -y*s - x*c
        T_o2b[:,1,3] = x*s - y*c

        # Homogeneous part
        T_o2b[:,3,3] = 1

        lmk_h = self.cvt_.pt_to_pth(lmk[li])
        lmk_h = reduce(np.matmul, [
            self.cvt_.T_b2c_,
            T_o2b,
            self.cvt_.T_c2b_,
            lmk_h[...,None]])[...,0] # Nx4
        lmk_3 = self.cvt_.pth_to_pt(lmk_h)
        nd_lmk = lmk_3 / np.linalg.norm(lmk_3, axis=-1, keepdims=True)
        nd_lmk = np.nan_to_num(nd_lmk)

        #err = 1.0 - (n * nd_lmk).sum(axis=1)
        #err = np.arccos((n * nd_lmk).sum(axis=-1))
        err = 1.0 - np.sqrt( (n * nd_lmk).sum(axis=-1) )

        return err.ravel()


    def sparsity_BA(self, n_c, n_l, ci, li,
            s_o=2, # observation state size
            s_c=3, # camera state size
            s_l=3  # landmark state size
            ):
        m = len(ci) * s_o # flat # of observations
        n = n_c * s_c + n_l * s_l # flat # of parameters
        A = lil_matrix((m,n), dtype=int) # TODO: dtype=bool?

        # pre-compute offsets for interleaving
        ci0 = ci*s_c
        li0 = li*s_l

        i = np.arange(len(ci))

        for i_o in range(s_o):
            for s in range(s_c):
                A[s_o*i+i_o,   ci0+s] = 1
            for s in range(s_l):
                A[s_o*i+i_o, n_c*s_c+li0+s] = 1

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
        ts = []
        n_o = len(c_i)

        # unroll input params
        pos = x[:n_c*3].reshape(-1, 3) # camera 2d pose (x,y,h)
        lmk = x[n_c*3:].reshape(-1, 3) # landmark positions

        # TODO : cache results for pt2_h to avoid calcing twice
        ts.append(time.time())
        pt2_h = self.project_BA(pos[c_i], lmk[l_i], return_h=True)
        ts.append(time.time())
        #err   = obs_pt2 - pt2_h
        # Note : minus sign here is critical if residual_BA returns y - f(x).
        J = -ba_J_v2(pos[c_i], lmk[l_i], self.K_,
                self.cvt_.T_b2c_[:3,:3], self.cvt_.T_b2c_[:3,3:], pt2_h) # Nx2x6
        ts.append(time.time())

        J_c = J[:,:,:3]
        J_l = J[:,:,3:]

        o_i0 = np.arange(n_o)

        # TODO : better format?
        ri = np.r_[
                o_i0*2, o_i0*2, o_i0*2,
                o_i0*2, o_i0*2, o_i0*2,
                o_i0*2+1, o_i0*2+1, o_i0*2+1,
                o_i0*2+1, o_i0*2+1, o_i0*2+1
                ]
        ci = np.r_[
                c_i*3,
                c_i*3+1,
                c_i*3+2,
                n_c*3+l_i*3,
                n_c*3+l_i*3+1,
                n_c*3+l_i*3+2,
                c_i*3,
                c_i*3+1,
                c_i*3+2,
                n_c*3+l_i*3,
                n_c*3+l_i*3+1,
                n_c*3+l_i*3+2
                ]

        data = np.transpose(J, [1,2,0]).ravel()
        #print 'can i just...?', np.square(data - data1.ravel()).sum()

        ts.append(time.time())
        J_res = csr_matrix(
                (data, (ri, ci)),
                shape=(2*n_o, n_c*3+n_l*3)
                )

        # legacy reference : lil_matrix
        #J_res = lil_matrix((2*n_o, n_c*3+n_l*3))
        #for i_o in range(2): # iterate over point (x,y)
        #    for i_c in range(3): # iterate over pose (x,y,h)
        #        J_res[o_i0*2+i_o, c_i*3+i_c] = J_c[:,i_o,i_c]
        #    for i_l in range(3): # iterate over landmark (x,y,z)
        #        J_res[o_i0*2+i_o, n_c*3 + l_i*3+i_l] = J_l[:,i_o,i_l]
        #J_res = J_res.tocsr()

        ts.append(time.time())

        dt = np.diff(ts)

        #print 'jac_BA dt', dt
        #print 'jac_BA dt(normalized)', dt / dt.sum()

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
        p0, vp, ci, li, p2 = self.graph_.query(win)

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

        vl = self.landmarks_.var[li_u] # n_lx3x3

        # Construct Vx, parameter covariance
        # This will help compute Vy, error covariance
        Wdata  = np.concatenate([vp[:, :3,:3],vl], axis=0)
        wip = np.arange( len(Wdata) + 1)
        wix = np.arange( len(Wdata) )
        W   = bsr_matrix((Wdata, wix, wip),
                shape=( len(Wdata)*3, len(Wdata)*3))

        # compute BA sparsity structure : deprecated with analytical jac_BA()
        A = self.sparsity_BA(n_c, n_l, ci, li, s_o=2)

        if ax is not None:
            ## prep data for viz
            # pose standard deviation
            vp_d = vp[:,(0,1,2),(0,1,2)]
            sp   = np.sqrt(vp_d)
            hip = np.full(len(sp), np.percentile(sp, 80))
            lop = np.full(len(sp), np.percentile(sp, 20))

            ax['ba_3'].cla()
            ax['ba_3'].set_title('std-pos')
            ax['ba_3'].plot(sp, '*')
            ax['ba_3'].plot(hip, '--')
            ax['ba_3'].plot(lop, '--')

            # landmark standard deviation
            vl_d = vl[:, (0,1,2), (0,1,2)]
            sl = np.sqrt(vl_d)

            hil = np.full(len(sl), np.percentile(sl, 80))
            lol = np.full(len(sl), np.percentile(sl, 20))

            ax['ba_2'].cla()
            ax['ba_2'].set_title('std-lmk')
            ax['ba_2'].plot(sl, '*')
            ax['ba_2'].plot(hil, '--')
            ax['ba_2'].plot(lol, '--')

        # mean reprojection error

        err0s = np.square(self.residual_BA(x0,
            n_c, n_l,
            ci, li, p2))
        err0 = np.sqrt(err0s.mean())

        ## actually run BA
        # -- opt1 : custom --
        #x1 = lsq(self.residual_BA, x0,
        #        jac=self.jac_BA,
        #        args=(n_c, n_l, ci, li, p2),
        #        **self.pBA_
        #        )
        # -- opt2 : scipy --

        res = least_squares(
                self.residual_BA, x0,
                jac_sparsity=A,
                jac=self.jac_BA,
                x_scale='jac',
                args=(n_c, n_l, ci, li, p2),
                **self.pBA_
                )

        x1 = res.x
        # ------------------

        # format ...
        pos_opt = x1[:n_c*3].reshape(-1,3)
        lmk_opt = x1[n_c*3:].reshape(-1,3)

        err1s = np.square(self.residual_BA(x1,
            n_c, n_l,
            ci, li, p2))
        err1 = np.sqrt(err1s.mean())
        #print err0, err1

        if ax is not None:
            # visualize best improvements
            n_viz = 16
            imp = np.sqrt(err0s.reshape(-1,2).sum(axis=1)) - np.sqrt(err1s.reshape(-1,2).sum(axis=1))

            print 'improvement min', imp.min()
            print 'improvement max', imp.max()

            v_idx = np.argsort(imp)[-n_viz:]

            print 'most improved landmarks', li[v_idx]
            pre_l  = self.landmarks_.pos[li_u][li[v_idx]]
            post_l = lmk_opt[li[v_idx]]

            print 'most improved poses', ci[v_idx]
            pre_p  = p0[ci[v_idx]]
            post_p = pos_opt[ci[v_idx]]

            # points
            pre_p2  = p2[v_idx]
            post_p2 = p2[v_idx] # NOTE: same

            vp0 = pre_p
            vl0 = pre_l.dot(self.cvt_.T_c2b_[:3,:3].T) + self.cvt_.T_c2b_[:3,3:].T # map coord
            d0  = p2vec(pre_p2, vp0, vl0, self.cvt_)

            vp1 = post_p
            vl1 = post_l.dot(self.cvt_.T_c2b_[:3,:3].T) + self.cvt_.T_c2b_[:3,3:].T # map coord
            d1  = p2vec(post_p2, vp1, vl1, self.cvt_)

            elems = [vp0, vl0, vp1, vl1]

            mnlim = np.min([e.min(axis=0) for e in elems], axis=0)
            mxlim = np.max([e.max(axis=0) for e in elems], axis=0)
            xlim, ylim, _ = zip(*[mnlim, mxlim])

            # apply padding
            pad = 1.2
            xmean = np.mean(xlim)
            ymean = np.mean(ylim)
            xlim = [xmean + pad * (xlim[0] - xmean), xmean + pad * (xlim[1] - xmean)]
            ylim = [ymean + pad * (ylim[0] - ymean), ymean + pad * (ylim[1] - ymean)]

            ax['ba_0'].cla()
            ax['ba_0'].plot(vp0[:,0], vp0[:,1], 'bo', label='pos0')
            ax['ba_0'].plot(vl0[:,0], vl0[:,1], 'c^', label='lmk0')
            ax['ba_0'].quiver(
                    vp0[:,0], vp0[:,1],
                    #vl0[:,0] - vp0[:,0], vl0[:,1] - vp0[:,1],
                    d0[:,0], d0[:,1],
                    angles='xy',
                    scale=1.0,
                    scale_units='xy',
                    color='b',
                    linestyle=':',
                    dashes=(0,(10, 20)),
                    alpha=0.5
                    )
            ax['ba_0'].plot(vp1[:,0], vp1[:,1], 'r+', label='pos1')
            ax['ba_0'].plot(vl1[:,0], vl1[:,1], 'm^', label='lmk1')
            ax['ba_0'].quiver(
                    vp1[:,0], vp1[:,1],
                    #vl1[:,0] - vp1[:,0], vl1[:,1]-vp1[:,1],
                    d1[:,0], d1[:,1],
                    angles='xy',
                    scale=1.0,
                    scale_units='xy',
                    color='r',
                    linestyle=':',
                    dashes=(0,(10, 20)),
                    alpha=0.5
                    )
            ax['ba_0'].set_title('BA : {:.3e}->{:.3e}'.format( err0, err1 ))
            ax['ba_0'].set_xlim(xlim)
            ax['ba_0'].set_ylim(ylim)
            ax['ba_0'].set_aspect('equal', 'datalim')
            ax['ba_0'].legend()
            #ax['ba_0'].plot(p0[:,0], p0[:,1], 'ko-', label='pos-pre')
            #ax['ba_0'].plot(p0[:,0], p0[:,1], 'ko-', label='lmk-pre')

        if ax is not None:
            ax['ba_1'].cla()
            ax['ba_1'].plot(p0[:,0], p0[:,1], 'ko-', label='initial')
            ax['ba_1'].plot(pos_opt[:,0], pos_opt[:,1], 'r+-', label='optimized')
            ax['ba_1'].set_title('BA Pose'.format( err0, err1 ))
            ax['ba_1'].axis('equal')
            ax['ba_1'].set_aspect('equal', 'datalim')
            ax['ba_1'].legend()

        if ax is not None:
            ax['pnp'].cla()
            ax['pnp'].plot(np.sqrt(err0s), '+', label='err0')
            ax['pnp'].plot(np.sqrt(err1s), 'x', label='err1')
            ax['pnp'].legend()

        # == apply BA results ==
        ba_cov = cov_from_jac(res.jac)
        ba_cov = extract_block_diag(ba_cov, 3)

        # == camera pose updates : currently "soft" ==
        # TODO : evaluate whether or not hard updates are better
        ba_cov_c = ba_cov[:n_c]
        ba_cov_c[:, (0,1,2), (0,1,2)] += (3e-2)**2 # regularization, ~3cm / ~1.7 deg.
        pose_c_r = self.graph_.update(win, pos_opt,
                #cov=ba_cov_c, hard=False
                cov=None, hard=False
                )

        # == landmark updates : hard vs. soft ==
        ba_cov_l = ba_cov[n_c:]
        ba_cov_l[:, (0,1,2), (0,1,2)] += (1e-1)**2 # regularization, ~10cm / ~5.7 deg.
        #self.landmarks_.update(li_u, lmk_opt, hard=True) # opt1:hard update
        self.landmarks_.update(li_u, lmk_opt,
                #ba_cov_l,
                #hard=False
                hard=True
                ) # opt2:soft update
        # == BA updates complete ==

        return pose_c_r

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

    def run_GP(self, pt_c, pt_p,
            scale=None,
            guess=None
            ):
        """
        Scale estimation based on locating the ground plane.
        if scale:=None, scale based on best z-plane will be returned.
        """
        null_result = (None, scale, guess, (None,None))
        if not (self.flag_ & ClassicalVO.VO_USE_SCALE_GP):
            return null_result

        camera_height = self.T_c2b_[2, 3]

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
            return null_result

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
            return null_result

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
            return null_result
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
        #print 'gp std', (gpt3_base[:,2] * scale_gp).std()
        print 'gp-ransac scale', scale_gp
        if np.isfinite(scale_gp) and scale_gp > 0:
            # project just in case scale < 0...
            scale = scale_gp

        # this is functionally the only time it's considered "success".
        return H, scale, (R, t), (gpt3, gp_idx[idx_h][msk_r])

    def run_PNP(self, pt3_map, pt2_cam, pose,
            p_min=16, ax=None, msg=''):

        if len(pt3_map) < p_min or len(pt2_cam) < p_min:
            return msg, None

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

            res = cv2.solvePnP(
                    pt3_map, pt2_cam,
                    self.K_, 0*self.D_,
                    useExtrinsicGuess = True,
                    rvec=rvec0.copy(),
                    tvec=tvec0.copy(),
                    flags=self.pPNP_['flags']
                    )
            suc, rvec01, tvec01 = res
            if suc:
                rvec0, tvec0 = rvec01, tvec01

            res = cv2.solvePnPRansac(
                    pt3_map[:,None], pt2_cam[:,None],
                    self.K_, 0*self.D_,
                    useExtrinsicGuess = True,
                    rvec=rvec0.copy(),
                    tvec=tvec0.copy(),
                    **self.pPNP_
                    )
            suc, rvec, tvec, inliers = res
            #inliers = np.arange(len(pt3_map))

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

                if ax is not None:
                    ax['main'].plot(
                            [pnp_p[0]],
                            [pnp_p[1]],
                            'go',
                            label='pnp',
                            alpha=1.0
                            )
                    ax['main'].quiver(
                            [pnp_p[0]],
                            [pnp_p[1]],
                            [np.cos(pnp_h)],
                            [np.sin(pnp_h)],
                            angles='xy',
                            #scale=1,
                            color='g',
                            alpha=0.75
                            )
            else:
                return msg, None
        except Exception as e:
            # ignore exception
            print('PNP Error : {}'.format(e))
            return msg, None

        pose_pnp = np.asarray([t_b[0], t_b[1], r_b[-1]])
        delta = pose - pose_pnp
        if np.linalg.norm(delta[:2]) > 0.2:
            print 'rejecting pnp due to jump : {}'.format(delta)
            return msg, None

        return msg, pose_pnp

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
        null_result = (
                np.empty((0,), dtype=np.int32), # index
                np.empty((0,3), dtype=np.float32), # points
                guess # transformation
                )

        if no_gp:
            # pre-filter by ymin
            y_gp = self.y_GP
            # NOTE: is it necessary to also check pt2_u_p?
            # probably gives similar results; skip.
            ngp_msk = (pt2_u_c[:,1] <= y_gp)
            ngp_idx = np.where(ngp_msk)[0]
            pt2_u_c = pt2_u_c[ngp_idx]
            pt2_u_p = pt2_u_p[ngp_idx]

        if len(pt2_u_c) <= 5:
            return null_result

        # EXPERIMENTAL : least-squares
        (R, t), pt3 = solve_TRI_fast(pt2_u_p, pt2_u_c,
                self.cvt_.K_, self.cvt_.Ki_,
                self.cvt_.T_b2c_, self.cvt_.T_c2b_,
                guess)
        return np.arange(len(pt2_u_p)), pt3, (R,t)

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

        if len(idx_e) < 16: #TODO : magic number
            # insufficient # of points -- abort
            return null_result

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

        return idx_in, pt3, (R, t)

    def initialize(self, img0, scale0,
            x0=None, P0=None
            ):
        # 0. initialize processing handles
        self.cvt_.initialize(img0.shape)

        # 1. initialize graph + data cache
        self.graph_.initialize(x0, P0)
        self.graph_.set_data_from(img0)

        # 2. initialize scale
        self.scale0_ = scale0
        self.use_s0_ = True

    def pp2RcTc(self, p0, p1):
        # convert p1 - p0 to R, T
        dx, dy, dh = (p1 - p0)
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

        return R_c, t_c
    
    def Rctc2Rbtb(self, R_c, t_c):
        T_c2c1 = np.eye(4)
        T_c2c1[:3,:3] = R_c
        T_c2c1[:3,3:] = t_c.reshape(3,1)
        T_b2b1 = np.linalg.multi_dot([
            self.cvt_.T_c2b_,
            T_c2c1,
            self.cvt_.T_b2c_
            ])
        R_b = T_b2b1[:3,:3]
        t_b = T_b2b1[:3,3:]
        return R_b, t_b

    def proc_f2f_i(self, fi0, fi1, pt3=None, msk3=None):
        """ call proc_f2f from indices """
        # NOTE : pt3/msk3 references fi1.
        # NOTE: currently only img1 data is used
        # query frame data
        img0, _, _, _, _ = self.graph_.get_data( fi0 )
        img1, _, des1, pt21, rsp1= self.graph_.get_data( fi1 )

        # query initial pose (may be guesses)
        pose0 = self.graph_.pos_[fi0]
        pose1 = self.graph_.pos_[fi1]

        return self.proc_f2f(
                img0, img1,
                pose0, pose1,
                des1, pt21, pt3=pt3, msk3=msk3
                )

    def proc_f2f(self, 
            img0, img1,
            pose0, pose1,
            des1, pt21,
            pt3=None,
            msk3=None,
            alpha=0.5,
            ref=1,
            scale=None
            ):
        """
        High alpha = bias toward new estimates
        """
        if ref == 0:
            # (des-pt2), (pt3-msk) all refer to img0

            # NOTE : because it is flipped,
            # pt3 input must be in coord0

            # flip
            res = self.proc_f2f(img1, img0, pose1, pose0,
                    des1, pt21, pt3, msk3, alpha, ref=1, scale=scale)

            sc, pt3, msk3, (o_pt21, o_pt20, o_idx), (o_R, o_t) = res
            # o_R/o_t is a transform from coord0 to coord1

            # pt3 is in coord. frame of 0; must convert to coord1
            pt3 = pt3.dot(o_R.T) + o_t.T

            # NOW flip o_R/o_t (NOTE : this must come after pt3 inversion)
            # TODO : replace below with more efficient computation
            T = np.eye(4)
            T[:3, :3] = o_R
            T[:3, 3:] = o_t.reshape(3,1)
            Ti = tx.inverse_matrix(T)
            o_R = Ti[:3, :3]
            o_t = Ti[:3, 3:] #.ravel() necessary?

            return sc, pt3, msk3, (o_pt20, o_pt21, o_idx), (o_R, o_t)

        # ^^ TODO : also input pt3 cov?
        # NOTE :  pt21 **MAY** contain landmark information later.
        # I think that might be a better idea.

        if pt3 is None:
            # construct initial pt3 guess if it doesn't exist
            pt3  = np.zeros((len(pt21),3), dtype=np.float32 )
            msk3 = np.zeros((len(pt21)), dtype=np.bool)
        idx3 = np.where(msk3)[0]
        # compose initial dR&dt guess
        R_c0, t_c0 = self.pp2RcTc(pose0, pose1)
        sc0 = np.linalg.norm(t_c0)

        # track
        # NOTE: distort=false assuming images and points are all pre-undistorted
        pt20_G = pt21.copy() # expected pt2 locations @ pose0

        if len(idx3) > 0:
            # fill in guesses if 3D information for pts exists
            pt20_G[idx3] = cv2.projectPoints(
                    # project w.r.t pose0.
                    # this works because T_c0 (R_c0|t_c0)
                    # represents the transform that takes everything to pose0 coordinates.
                    pt3[idx3],
                    cv2.Rodrigues(R_c0)[0], # rvec needs to be formatted as such.
                    t_c0.ravel(), # TODO : ravel needed?
                    cameraMatrix=self.K_,
                    distCoeffs=self.D_*0,
                    )[0][:,0]

        pt20, idx_t = self.track(img1, img0, pt21, pt2=pt20_G) # NOTE: track backwards

        # TODO : FM Correction with self.run_fm_cor() ??
        F = None
        if self.flag_ & ClassicalVO.VO_USE_FM_COR:
            # correct Matches by RANSAC consensus
            F, msk_f = cv2.findFundamentalMat(
                    pt21[idx_t],
                    pt20[idx_t],
                    method=self.pEM_['method'],
                    param1=self.pEM_['threshold'],
                    param2=self.pEM_['prob'],
                    )
            msk_f = msk_f[:,0].astype(np.bool)
            print_ratio('FM correction', msk_f.sum(), msk_f.size)

            # retro-update corresponding indices
            # to where pt2_u_p will be, based on idx_f
            idx_t   = idx_t[msk_f]

            # NOTE : invalid to apply undistort() after correction
            # NOTE : below code will work, but validity is questionable.

            #pt21_f, pt20_f = cv2.correctMatches(F,
            #        pt21[None,idx_t],
            #        pt20[None,idx_t])
            #pt21_f = np.squeeze(pt21_f, axis=0)
            #pt20_f = np.squeeze(pt20_f, axis=0)

            ### -- will sometimes return NaN.
            #ck0 = np.all(np.isfinite(pt20_f))
            #ck1 = np.all(np.isfinite(pt21_f))

            #if ck0 and ck1:
            #    pt20[idx_t] = pt20_f
            #    pt21[idx_t] = pt21_f

        # stage 1 : EM
        res = self.run_EM(pt21[idx_t], pt20[idx_t], no_gp=False, guess=(R_c0, t_c0) )
        idx_e, pt3_em_u, (R_em, t_em_u) = res # parse run_EM, no scale info
        t_em_u /= np.linalg.norm(t_em_u) # make sure uvec
        idx_e = idx_t[idx_e]
        # ^ note pt3_em_u in camera (pose1) coord

        # stage 2 : GP
        # guess based on em or c0 ?? Is it double-dipping to use R_em/t_em for GP?
        res = self.run_GP(pt21[idx_t], pt20[idx_t], sc0, guess=(R_em, t_em_u) )
        H, sc2, (R_gp, t_gp_u), (pt3_gp_u, idx_g) = res # parse run_GP
        t_gp_u /= np.linalg.norm(t_gp_u)
        if idx_g is not None:
            idx_g = idx_t[idx_g]
        # ^ note pt3_gp_u also in camera (pose1) coord

        # stage 3 : resolve scale based on guess + GP measurement


        # interpolation factor between
        # ground-plane estimates vs. essentialmat estimates
        # high value = high GP trust
        alpha_gve = 0.5

        # resolve scale based on EM / GP results
        # NOTE : sc0 based on ekf/ukf; sc2 based on ground plane.

        sc = lerp(sc0, sc2, alpha)
        if scale is not None:
            # incorporate input scale information
            sc = lerp(scale, sc, alpha)

        # prepare observation mask
        o_msk = np.zeros((len(pt21)), dtype=np.bool)

        # 1. resolve pose observation
        o_R, o_t = resolve_Rt(R_em, R_gp, t_em_u*sc, t_gp_u*sc,
                alpha=alpha_gve,
                guess=(R_c0, t_c0))

        # 2. fill in pt3 information + mark indices
        # ( >> TODO << : incorporate confidence information )
        # NOTE: can't use msk3 for o_msk, which is aggregate info.
        if pt3_em_u is not None:
            print('applied scale', sc)
            pt3_em = pt3_em_u * sc
            pt3[idx_e] = np.where(
                    msk3[idx_e,None],
                    lerp(pt3[idx_e], pt3_em, alpha),
                    pt3_em)
            msk3[idx_e] = True
            o_msk[idx_e] = True

        if pt3_gp_u is not None:
            # NOTE: overwrites idx_e results with idx_g
            pt3_gp = pt3_gp_u * sc
            pt3[idx_g] = pt3_gp 
            pt3[idx_g] = np.where(
                    msk3[idx_g,None],
                    lerp(pt3[idx_g], pt3_gp, alpha),
                    pt3_gp)
            msk3[idx_g] = True
            o_msk[idx_g] = True


        # 2.  parse indices
        o_idx = np.where(o_msk)[0]
        o_pt20, o_pt21 = pt20[o_idx], pt21[o_idx]

        # NOTE : final result
        # 1. refined 3d positions + masks,
        # 2. observation of the points at the respective poses,
        # 3. observation of the relative pose from p0->p1. NOTE: specified in camera coordinates.
        return sc, pt3, msk3, (o_pt20, o_pt21, o_idx), (o_R, o_t)

    def __call__(self, img, dt, ax=None):
        msg = ''
        # suffix designations:
        # o/0 = origin (i=0)
        # p = previous (i=t-1)
        # c = current  (i=t)

        # estimate current pose and update state to current index
        pose_p, pose_c, sc = self.graph_.predict(dt, commit=True) # with commit:=True, current pose index will also be updated
        index = self.graph_.index # << index should NOT change after predict()
        self.graph_.set_data_from( img ) # << this must be called after predict()

        # query graph for processing data (default : neighboring frames only)
        # TODO : currently preparing architecture from cross-frame matching/tracking
        img_p, kpt_p, des_p, pt2_p, rsp_p = self.graph_.get_data(-2) # previous
        img_c, _, _, _, _ = self.graph_.get_data(-1) # current
        # NOTE : as of right now, pt2_c, rsp_c are propagated from pt2_p and rsp_p
        # and are not the results from self.graph_.get_data()

        if self.use_s0_:
            # use recently supplied reference scale
            scale = self.scale0_
            scale_alpha = 0.0
            self.use_s0_ = False
        else:
            # scale will be automatically figured out
            scale = None
            scale_alpha = 0.5 # = high trust towards measurements

        # query LMK
        idx_l, pt2_l = self.landmarks_.track_points()
        rsp_l = self.landmarks_.kpt[idx_l, 2] # query responses for nmx
        pt3_l = self.cvt_.map_to_cam(
                self.landmarks_.pos[idx_l],
                pose_p
                ) # pt3 references coord0.
        des_l = self.landmarks_.des[idx_l]

        # format inputs
        li1 = len(pt2_l) # = end of landmark points

        # get refined pose_c_r results from track + PnP
        o_nmsk = np.ones(li1, dtype=np.bool)
        pt2_l_current, ti = self.track(img_p, img_c, pt2_l)
        if len(ti) > 16: # TODO : magic
            pose_c_pnp = solve_PNP(
                    self.landmarks_.pos[idx_l[ti]],
                    pt2_l_current[ti],
                    self.cvt_.K_,
                    self.cvt_.T_b2c_,
                    self.cvt_.T_c2b_,
                    guess = pose_c
                    )
            #_, pose_c_pnp = self.run_PNP(
            #        self.landmarks_.pos[idx_l[ti]],
            #        pt2_l_current[ti],
            #        pose_c, ax=ax)
            if pose_c_pnp is not None:
                pose_c = lerp(pose_c, pose_c_pnp, 0.8)

            if ax is not None:
                ax['main'].plot(
                        [pose_c_pnp[0]],
                        [pose_c_pnp[1]],
                        'go',
                        label='pnp',
                        alpha=1.0
                        )
                ax['main'].quiver(
                        [pose_c_pnp[0]],
                        [pose_c_pnp[1]],
                        [np.cos(pose_c_pnp[-1])],
                        [np.sin(pose_c_pnp[-1])],
                        angles='xy',
                        #scale=1,
                        color='g',
                        alpha=0.75
                        )

        o_nmsk[ti] = False
        o_nidx = np.where(o_nmsk)[0]
        self.landmarks_.kpt[idx_l[ti],:2] = pt2_l_current[ti]
        self.landmarks_.untrack(idx_l[o_nidx])
        print_ratio('LMK Track', len(ti), li1)

        # collect observations
        obs = [] 

        # step 1 : initialize with previous keypoints + landmarks

        # TODO / NOTE : maybe apply non-max suppression to pt2_p beforehand
        # IFF landmark tracking appears to get lost way too quickly
        # (probably due to matching with other nearby points)

        idx_s = self.pts_nmx(
                pt2_p, pt2_l,
                rsp_p, rsp_l,
                k=4,
                radius=16.0
                )
        # == NMX VIS START ==
        #nfig = self.fig_['nmx']
        #nax  = nfig.gca()
        #nax.cla()
        #nax.imshow(img_p)
        #nax.plot( pt2_p[:,0], pt2_p[:,1], 'rx' ,label='p')
        #nax.plot( pt2_p[idx_s,0], pt2_p[idx_s,1], 'c+' ,label='s')
        #nax.plot( pt2_l[:,0], pt2_l[:,1], 'b.' ,label='l')
        #if not nax.yaxis_inverted():
        #    nax.invert_yaxis()
        #nax.legend()
        #plt.pause(0.001)
        # == NMX VIS END ==


        # NOTE : var-names overwritten with sliced version
        kpt_p = kpt_p[idx_s]
        des_p = des_p[idx_s]
        pt2_p = pt2_p[idx_s]
        rsp_p = rsp_p[idx_s]

        pt2_p_all = np.concatenate([pt2_l, pt2_p], axis=0)
        des_p_all = np.concatenate([des_l, des_p], axis=0)
        pt3  = np.zeros((len(pt2_p_all), 3), dtype=np.float32)
        pt3[:li1] = pt3_l # pre-populate known 3D locations
        msk3 = np.zeros(len(pt2_p_all), dtype=np.bool)
        msk3[:li1] = True # mark indices of known 3D locations
        res = self.proc_f2f(
                img_p, img_c,
                pose_p, pose_c,
                des_p, pt2_p_all,
                pt3=pt3, msk3=msk3,
                alpha=scale_alpha,
                ref=0,
                scale=scale,
                )
        # ^^ proc_f2f with previous-frame configuration.
        # as with "ordinary" calls to proc_f2f,
        # pt3-msk1 will reference pose1
        # o_pt20, o_pt21 will reference pose0, pose1
        # o_R, o_t will be the transform from coord 1 to coord0.
        scale, pt3, msk3, (o_pt20, o_pt21, o_idx), (R_c, t_c) = res

        #print 'proc_f2f results', np.linalg.norm(t_c)

        # index bookkeeping:
        # pt3  : []
        # msk3 : []
        # o_pt20 : [o_idx]
        # o_pt21 : [o_idx]
        # o_idx : []

        # preliminary update for pose_c with obtained (o_R, o_t)
        R_b, t_b = self.Rctc2Rbtb(R_c, t_c)
        pose_c = self.pRt2pose(pose_p, R_b, t_b)

        # parse indices & split data
        o_msk_l = (o_idx < li1)
        o_msk_p = ~o_msk_l
        o_idx_l = o_idx[o_msk_l]
        o_idx_p = o_idx[~o_msk_l]
        #pt3_l = pt3[o_idx][o_msk_l] # avoid splitting pt3_l, as it will get updated
        #pt3_p = pt3[o_idx][~o_msk_l] # avoid splitting pt3_p, as it will get updates
        o_pt20_l = o_pt20[o_msk_l]
        o_pt20_p = o_pt20[~o_msk_l]
        o_pt21_l = o_pt21[o_msk_l]
        o_pt21_p = o_pt21[~o_msk_l]

        o_idx_p0 = o_idx_p - li1 # o_idx_p with offset removed
        n_new_lmk_max = len(o_idx_p0) # ( max possible # of new landmarks : probably less)

        # setup indices
        i_lm_old = idx_l[o_idx_l]

        # unset tracking flag for failed lmk tracks

 

        # == VIZ TRACK BEGIN ==
        # lfig = self.fig_['trk']
        # lax = lfig.gca()

        # lax.cla()
        # lax.imshow(img_c)
        # lax.plot(o_pt21_l[:,0], o_pt21_l[:,1], 'r+', label='new')
        # tmp_old_pt = self.landmarks_.kpt[idx_l[o_idx_l],:2]
        # lax.plot(tmp_old_pt[:,0], tmp_old_pt[:,1], 'bx', label='old')
        # lax.quiver(
        #         tmp_old_pt[:,0], tmp_old_pt[:,1],
        #         o_pt21_l[:,0] - tmp_old_pt[:,0], o_pt21_l[:,1] - tmp_old_pt[:,1],
        #         angles='xy',
        #         scale=1.0,
        #         scale_units='xy'
        #         )
        # if not lax.yaxis_inverted():
        #     lax.invert_yaxis()
        # lax.legend()
        # plt.pause(0.001)
        # == VIZ TRACK END ==

        # update tracking points for successful lmk tracks
        # self.landmarks_.kpt[idx_l[o_idx_l],:2] = o_pt21_l

        # cache candidate new observations to add
        # NOTE : appending to obs is NOT final;
        # will be resolved before calling Landmarks.append_from() / VGraph.add_obs()
        obs.append((np.r_[:n_new_lmk_max], o_pt20_p, index-1) )
        obs.append((np.r_[:n_new_lmk_max], o_pt21_p, index)   )
        # observations of old landmarks will be handled in proc_f2m_old.
        # (including visibility graph management)

        # setup data at current pose index for refinement
        des_c = des_p_all[o_idx] # does not get refined.
        pt2_c = o_pt21 # does not get refined (for obvious reasons)
        pt3   = pt3[o_idx] # should be all set. NOTE: pt3 overwritten
        msk3  = msk3[o_idx] # should be all 1.

        # i=-1 : current frame (already used -- ?)
        # i=-2 : previous frame (already used)
        # look further back for new information.
        for di in [-3]:#[-1, -3]:#[-3, -4, -5]:
            # -1=index, -2=index-1, -3=index-2
            data = self.graph_.get_data(di)
            if data is None:
                # no further history is available
                break

            # parse data
            pose_p_i = self.graph_.pos_[di][:3]
            img_p_i, _, _, _, _ = data

            res = self.proc_f2f(
                    img_p_i, img_c,
                    pose_p_i, pose_c,
                    des_c, pt2_c,
                    pt3, msk3,
                    alpha=0.25, # bias towards old measurements
                    ref=1,
                    )
            # parse output
            _, pt3, msk3, (o_pt20_i, o_pt21_i, o_idx), (R_c_i, t_c_i) = res

            # setup indices
            o_msk_p_i  = (o_idx >= li1)
            o_idx_p_i  = o_idx[o_msk_p_i] # index with offset
            o_idx_p_i0 = o_idx_p_i - li1 # index without offset

            # add to observation cache
            # ( cannot be added to graph yet )
            # TODO : validate index o_idx_p_i0
            obs.append( (o_idx_p_i0, o_pt20_i[o_msk_p_i], index+1+di) )

            # update pose_c (pt3 is updated automatically)
            R_b_i, t_b_i = self.Rctc2Rbtb(R_c_i, t_c_i)
            pose_c_i = self.pRt2pose(pose_p_i, R_b_i, t_b_i)
            #print 'pose_c_i', pose_c_i

            # TODO : intelligently merge multiple pose_c estimates
            pose_c = lerp(pose_c, pose_c_i, 0.25)
        # after look-back, pt3 is finalized here.

        # split pt3 to old and new components; should both be fairly refined by now
        pt3_l = pt3[o_msk_l]
        pt3_p = pt3[~o_msk_l]

        # apply filter to pt3_new to prevent redundant insertions
        # TODO : apply other filters (descriptor match, ...)
        pt3_new = pt3[o_msk_p]
        pt2_new = o_pt21_p # new points, in frame 1
        rsp_new = rsp_p[o_idx_p0]
        des_new = des_p[o_idx_p0]


        if len(o_idx_l) >= 256:
            # enough tracked landmarks; filter
            idx_f = self.filter_pts(
                    pose_c,
                    pt3_new, pt2_new, # << NOTE : in pose_c_r coordinates
                    des_new, rsp_new
                    )
        else:
            # not enough landmarks; keep all
            idx_f = np.arange( len(pt3_new) )

        n_new_lmk = len(idx_f)
        print('adding {} landmarks : {}->{}'.format(n_new_lmk,
            self.landmarks_.size_, self.landmarks_.size_+n_new_lmk
            ))
        #idx_des = self.filter_by_descriptor( ... )

        # At this point : highly refined pt3 & pose_c information

        # TODO : fix naming issues
        scale_c = np.linalg.norm(t_c)

        pose_c_r = pose_c

        if self.flag_ & ClassicalVO.VO_USE_F2M:
            # Estimate #4 : Based on Landmarks
            # TODO : smarter way to incorporate ground-plane scale information??
            # estimate scale based on current pose guess
            # and recompute rectified pose_c_r

            # process old points first and obtain camera motion scale from correspondences

            #try:
            #    tfig = self.tfig_
            #except Exception as e:
            #    self.tfig_ = plt.figure()
            #    tfig = self.tfig_
            #tax = tfig.gca()

            #tmp, tmsk = self.cvt_.pt3_pose_to_pt2_msk(pt3_l, pose_c_r)
            #tax.plot(tmp[tmsk,0], tmp[tmsk,1], 'b+', label='pt3')
            #tax.plot(o_pt21_l[:,0], o_pt21_l[:,1], 'rx', label='obs')
            #plt.pause(0.001)

            scale_c_2, msg = self.proc_f2m_old(
                    pose_c_r, scale_c,
                    pt3_l,
                    i_lm_old, # landmark indices; NOTE : apply lmk index start offset
                    o_pt21_l, # points here are needed to update kpt location
                    o_pt21_l, # undistorted observations, required for obs. adding
                    img_c,
                    ax,
                    msg
                    )

            # scale corrections
            pt3_new *= (scale_c_2 / scale_c)
            scale_c = scale_c_2
            print('scale_c', scale_c)

            # update transforms with new scale_c information from proc_f2m.
            # (scale_c fusion happenes automatically within proc_f2m)
            t_c = t_c / np.linalg.norm(t_c) * scale_c
            #print('t_c', t_c, np.linalg.norm(t_c))
            T_c2c1 = np.eye(4)
            T_c2c1[:3,:3] = R_c
            T_c2c1[:3,3:] = t_c
            T_b2b1 = np.linalg.multi_dot([
                self.cvt_.T_c2b_,
                T_c2c1,
                self.cvt_.T_b2c_
                ])
            R_b, t_b = T_b2b1[:3,:3], T_b2b1[:3,3:]
            #print('t_b', t_b)
            scale_b = np.linalg.norm(t_b)
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
        # NOTE : This finalizes pose_c.
        #print('pose_c_r - update', pose_c_r)
        pose_c_r = self.graph_.update(1, [pose_c_r])
        # NOTE: estimate #5 does not produce R_c, t_c, R_b, t_c
        # because it is not needed.
        # == #5 finished ==


        # == FINALLY ADD NEW LANDMARKS ==
        # NOTE : f2m_new got folded info f2f,
        # as the construction of the map itself could be considered part of
        # frame-to-frame processing; reference of the map (proc_f2m_old) still remains.

        if self.flag_ & ClassicalVO.VO_USE_F2M:
            # output = validated pt3
            pt3_new = pt3_new[idx_f]
            des_new = des_p[o_idx_p0[idx_f]]
            col_new = get_points_color(img_c, pt2_p[o_idx_p0[idx_f]], w=1)
            kpt_new = kpt_p[o_idx_p0[idx_f]]
            for e,v in zip(kpt_new, pt2_new[idx_f]):
                e.pt = tuple(v)

            #print o_pt21_p[idx_f].shape
            #print (rsp_p[o_idx_p0[idx_f]][:,None]).shape
            #kpt_new = np.concatenate([
            #    pt2_new[idx_f], # == pt2_c
            #    rsp_new[idx_f][:,None]], axis=-1)

            # go through obs + add to graph
            idx_f_r = np.arange(len(idx_f))

            #try:
            #    tfig = self.tfig_
            #except Exception as e:
            #    self.tfig_ = plt.figure()
            #    tfig = self.tfig_
            #tax = tfig.gca()
            #tax.cla()

            for li_X, p2_X, pi in obs:
                _, i_X, i_Y = np.intersect1d(idx_f, li_X, return_indices=True)
                di, p2 = idx_f_r[i_X], p2_X[i_Y]

                #tmp, tmsk = self.cvt_.pt3_pose_to_pt2_msk(
                #        self.cvt_.cam_to_map(pt3_new[di], pose_c_r),
                #        self.graph_.pos_[pi][:3]
                #        )
                #print_ratio('tmsk', tmsk.sum(), tmsk.size)

                #tax.plot(tmp[tmsk,0], tmp[tmsk,1], 'b+', label='pt3')
                #tax.plot(p2[:,0], p2[:,1], 'gx', label='obs')
                #plt.pause(0.001)

                self.graph_.add_obs(
                        self.landmarks_.size_ + di, # apply new landmark offset
                        p2,
                        pi)
            # actually append now
            self.landmarks_.append_from(
                    self.cvt_, pose_c_r,
                    pt3_new,
                    des_new,
                    col_new,
                    kpt_new
                    )
            # how many got added?
            n_new_lm = len(idx_f)
        # ===============================

        # prune
        # NOTE : indexing offset here is 100% because of visualization.
        # TODO : make more robust
        if (index >= self.prune_freq_) and (index % self.prune_freq_)==0 :
            # 1. prep viz (if available)
            if ax is not None:
                ax['prune_0'].cla()
                ax['prune_1'].cla()
                ax['prune_0'].set_title('pre-prune')
                ax['prune_1'].set_title('post-prune')
            # 2. draw pre-prune
            if ax is not None:
                self.graph_.draw(ax['prune_0'], self.landmarks_) # pre-prune
            # 3. prune
            keep_idx = self.landmarks_.prune()
            self.graph_.prune(keep_idx)
            # 4. draw post-prune
            if ax is not None:
                self.graph_.draw(ax['prune_1'], self.landmarks_) # post-prune

        if self.flag_ & ClassicalVO.VO_USE_BA:
            ba_win = None
            run_ba = False

            for win in self.ba_pyr_:
                # Survey list of BA frequencies (windows)
                # And run the largest possible BA
                # if multiple windows satisfy the condition.
                s_check = (index >= win)
                f_check = (index % win == 0)
                if s_check and f_check:
                    ba_win = win
                    run_ba = True
                    break

            if run_ba:
                # run BA every [win] frames
                print('Running BA @ scale={}'.format(ba_win))
                pose_c_r = self.run_BA(ba_win, ax=ax)

        ## === FROM THIS POINT ALL VIZ === 
        msk = np.ones(len(o_pt20), dtype=np.bool)
        mim = drawMatches(img_p, img_c, o_pt20, o_pt21, msk=msk)

        # show landmark statistics
        self.cnt_lm_tot_.append( self.landmarks_.size_ )
        self.cnt_lm_trk_.append( len(o_idx_l) )
        self.cnt_lm_new_.append( n_new_lmk )

        if ax is not None:
            ax['cnt'].cla()
            ax['cnt'].plot(self.cnt_lm_tot_, label='lmk-tot')
            ax['cnt'].plot(self.cnt_lm_trk_, label='lmk-trk')
            ax['cnt'].plot(self.cnt_lm_new_, label='lmk-new')
            ax['cnt'].legend()
            ax['cnt'].grid()
            ax['cnt'].set_axisbelow(True)
            ax['cnt'].set_title('Feature Counts Stat')

        # plot scale
        if ax is not None:
            pass
            #ax['scale'].cla()
            ##for (k, v) in self.scales_.items():
            #for k in ['c2','b4','c4']:
            #    v = self.scales_[k]
            #    s_i, s_v = zip(*v)
            #    n_plot = 64
            #    di = max(len(s_i) / n_plot, 1)
            #    ax['scale'].plot(s_i[::di], s_v[::di], '+--', label=k)
            #ax['scale'].legend()
            #ax['scale'].set_title('scale')

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
        pt3_m_b = pt3_m.dot(self.T_c2b_[:3,:3].T) + self.T_c2b_[:3,3]
        #col_m = (pt3_m_b[:,2] - pt3_m_b[:,2].min()) / (np.max(pt3_m_b[:,2]) - pt3_m_b[:,2].min())


        # filter by height
        # convert to base_link coordinates
        #pt3_m_b = pt3_m.dot(self.T_c2b_[:3,:3].T) + self.T_c2b_[:3,3]
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

        #if len(pt3_l) > 0:
        #    rpt3 = cv2.projectPoints(
        #            pt3_l,
        #            np.zeros(3),
        #            np.zeros(3),
        #            cameraMatrix=self.K_,
        #            distCoeffs=0*self.D_
        #            )[0][:,0]
        #    print '??'
        #    print rpt3.shape
        #else:
        #    rpt3 = np.empty((0,2), dtype=np.float32)
        #
        #rfig = self.fig_['rec']
        #rax  = rfig.gca()
        #rax.cla()
        #rax.imshow(img_c)
        #rax.plot(pt2_c_rec[:,0], pt2_c_rec[:,1], 'rx')
        #rpt2 = self.landmarks_.kpt[rec_idx,:2]
        #rax.plot(rpt2[:,0], rpt2[:,1], 'b+')
        #rax.plot(rpt3[:,0], rpt3[:,1], 'g.')
        ##rax.set_xlim(0, 640)
        ##rax.set_ylim(0, 480)
        #if not rax.yaxis_inverted():
        #    rax.invert_yaxis()
        #plt.pause(0.001)

        
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
