import numpy as np
import cv2
from tf import transformations as tx
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt

def print_Rt(R, t):
    print '\tR', np.round(np.rad2deg(tx.euler_from_matrix(R)), 2)
    print '\tt', np.round(t.ravel() / np.linalg.norm(t), 2)

def intersect2d(a, b):
    """
    from https://stackoverflow.com/a/8317403
    """
    nrows, ncols = a.shape
    dtype={'names':['f{}'.format(i) for i in range(ncols)],
        'formats':ncols * [a.dtype]}
    c = np.intersect1d(a.view(dtype), b.view(dtype))
    c = c.view(a.dtype).reshape(-1, ncols)
    return c

def recover_pose_from_RT(perm, K,
        pt1, pt2,
        z_min = np.finfo(np.float32).eps,
        z_max = np.inf,
        return_index=False,
        log=False,
        threshold=0.8,
        guess=None
        ):
    P1 = np.eye(3,4)
    P2 = np.eye(3,4)

    sel   = 0
    scores = [0.0 for _ in perm]
    msks = [None for _ in perm]
    pt3s = [None for _ in perm]
    ctest = -np.inf

    for i, (R, t) in enumerate(perm):
        # Compute Projection Matrix
        P2[:3,:3] = R
        P2[:3,3:] = t.reshape(3,1)
        KP1 = K.dot(P1) # NOTE : this could be unnecessary, idk.
        KP2 = K.dot(P2)

        # Triangulate Points
        pth_a = cv2.triangulatePoints(
                KP1, KP2,
                pt1[None,...],
                pt2[None,...]).astype(np.float32)
        pth_a /= pth_a[3:]

        # transform points into camera coordinates
        pt3_a = P1.dot(pth_a)
        pt3_b = P2.dot(pth_a)

        # apply z-value (depth) filter
        za, zb = pt3_a[2], pt3_b[2]
        msk_i = np.logical_and.reduce([
            z_min < za,
            za < z_max,
            z_min < zb,
            zb < z_max
            ])
        c = msk_i.sum()

        # store data
        pt3s[i] = pt3_a # NOTE: a, not b
        msks[i] = msk_i
        scores[i] = ( float(msk_i.sum()) / msk_i.size)

        if log:
            print('[{}] {}/{}'.format(i, c, msk_i.size))
            print_Rt(R, t)

    # option one: compare best/next-best
    sel = np.argmax(scores)

    if guess is not None:
        # -- option 1 : multiple "good" estimates by score metric
        # here, threshold = score
        # soft_sel = np.greater(scores, threshold)
        # soft_idx = np.where(soft_sel)[0]
        # do_guess = (soft_sel.sum() >= 2)
        # -- option 1 end --

        # -- option 2 : alternative next estimate is also "good" by ratio metric
        # here, threshold = ratio
        next_idx, best_idx = np.argsort(scores)[-2:]
        soft_idx = [next_idx, best_idx]
        if scores[best_idx] >= np.finfo(np.float32).eps:
            do_guess = (scores[next_idx] / scores[best_idx]) > threshold
        else:
            # zero-division protection
            do_guess = False
        # -- option 2 end --

        soft_scores = []
        if do_guess:
            # TODO : currently, R-guess is not supported.
            R_g, t_g = guess
            t_g_u = np.reshape(t_g, 3) / np.linalg.norm(t_g) # convert guess to uvec
            
            for i in soft_idx:
                # filter by alignment with current guess-translational vector
                R_i, t_i = perm[i]
                t_i_u = np.reshape(t_i, 3) / np.linalg.norm(t_i)
                score_i = np.sum(t_g_u * t_i_u) # dot product
                soft_scores.append(score_i)

            # finalize selection
            sel = soft_idx[ np.argmax(soft_scores) ]
            unsel = soft_idx[ np.argmin(soft_scores) ] # NOTE: log-only

            if True: # TODO : swap with if log:
                print('\t\tresolving ambiguity with guess:')
                print('\t\tselected  i={}, {}'.format(sel, perm[sel]))
                print('\t\tdiscarded i={}, {}'.format(unsel, perm[unsel]))

    R, t = perm[sel]
    msk = msks[sel]
    pt3 = pt3s[sel][:,msk]
    n_in = msk.sum()

    if return_index:
        return n_in, R, t, msk, pt3, sel
    else:
        return n_in, R, t, msk, pt3

def recover_pose(E, K,
        pt1, pt2,
        z_min = np.finfo(np.float32).eps,
        z_max = np.inf,
        threshold=0.8,
        guess=None,
        log=False
        ):
    R1, R2, t = cv2.decomposeEssentialMat(E)
    perm = [
            (R1, t),
            (R2, t),
            (R1, -t),
            (R2, -t)]
    return recover_pose_from_RT(perm, K,
            pt1, pt2,
            z_min, z_max,
            threshold=threshold,
            guess=guess,
            log=log
            )

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


def robust_mean(x, margin=10.0, weight=None):
    if len(x) <= 0:
        return np.nan
    x = np.asarray(x, dtype=np.float32)
    s_lo = np.percentile(x, 50.0 - margin)
    s_hi = np.percentile(x, 50.0 + margin)
    msk = np.logical_and.reduce([
        s_lo <= x, x <= s_hi
        ])

    if weight is None:
        return x[msk].mean()
    else:
        # compute normalized weight
        #print 'weighting'
        #print weight.shape
        #print msk.shape
        w = weight[msk[...,0]]
        w = w / w.sum()

        return np.sum(x[msk] * w, axis=-1)

def zu2Rs(u):
    """
    specialization of zu2R for batch u.
    computes R such that R.z = u, where z=(0,0,1)
    requires u to be a unit vector.
    """

    u_z = np.float32([0,0,1]).reshape(1,3)
    c = np.sum(u_z*u, axis=-1) # cos of angle
    sax = np.cross(u_z, u) # sin * axis
    ax = sax / np.linalg.norm(sax, axis=-1, keepdims=True)
    A = ax[:,None,:] * ax[:,:,None]

    sx, sy, sz = sax.T

    b = np.asarray([
        c,-sz,sy,
        sz,c,-sx,
        -sy,sx,c]).T.reshape(-1,3,3)
    R = (1 - c).reshape(-1,1,1) * A + b
    return R

def oriented_cov(
        pt3,
        cov0,
        ):
    """
    According to https://math.stackexchange.com/a/476311
    """

    d = np.linalg.norm(pt3, axis=-1, keepdims=True)
    u_z = pt3 / d # Nx3
    R = zu2Rs(u_z)

    RT = np.transpose(R, [0,2,1])
    C = np.matmul(np.matmul(R, cov0[None,...]), RT)

    # apply depth scale for variance
    C = d[...,None] * C
    return C

lmk_fig = None
def show_landmark_2d(pos, cov, clear=True, draw=True,
        style='k+',
        colors=None,
        label=''
        ):
    """ from https://stackoverflow.com/a/20127387 """

    global lmk_fig
    if lmk_fig is None:
        lmk_fig = plt.figure()
    ax = lmk_fig.gca()
    if clear:
        ax.cla()

    # subsample
    if len(pos) <= 0:
        return

    n = min(256, len(pos))
    idx = np.random.randint(0, len(pos), size=n)
    pos = pos[idx]
    cov = cov[idx]

    x = pos[:,2]
    y = -pos[:,1]


    if colors is None:
        colors = np.random.uniform(size=(n,3))
    
    ax.plot(x, y, style, alpha=0.75,label=label
            )

    for p,c,col in zip(pos, cov, colors):
        # re-orient to the things we care about ...
        x,y = p[2], -p[1]
        c_2d = np.reshape([
                c[2,2], c[2,1],
                c[1,2], c[1,1]], (2,2))

        l, v = np.linalg.eig(c_2d)
        l    = np.sqrt(l)
        ell = Ellipse(xy=(x, y),
                    width=l[0]*2, height=l[1]*2,
                    angle=np.rad2deg(np.arccos(v[0, 0])))
        #ell.set_facecolor('none') #??
        ax.add_artist(ell)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.25)
        ell.set_facecolor(col)

    ax.set_xlim(-1.0, 10.0)
    ax.set_ylim(-5.0, 5.0)
    #ax.set_aspect('equal', 'datalim')
    #ax.autoscale(True)
    #print 'should have added {} ellipses'.format(len(pos))
    if draw:
        ax.legend()
        ax.plot([0],[0],'k+')
        lmk_fig.canvas.draw()

class Landmarks(object):
    """ Landmarks management """
    def __init__(self, descriptor):
        # == auto unroll descriptor parameters ==
        n_des = descriptor.descriptorSize()
        # Not 100% sure, but don't want to take risks with unsigned math
        t_des = (np.uint8 if descriptor.descriptorType() == cv2.CV_8U
                else np.float32)
        s_fac = descriptor.getScaleFactor()
        n_lvl = descriptor.getNLevels()
        # =======================================

        self.capacity_ = c = 1024
        self.size_ = s = 0

        c = self.capacity_
        s = self.size_

        # general data ...
        self.pos_ = np.empty((c,3), dtype=np.float32)
        self.des_ = np.empty((c,n_des), dtype=t_des)
        self.ang_ = np.empty((c,1), dtype=np.float32)
        self.col_ = np.empty((c,3), dtype=np.uint8)

        # landmark "health" ...
        self.var_ = np.empty((c,3,3), dtype=np.float32)
        self.cnt_ = np.empty((c,1), dtype=np.int32)

        # observation source?
        # self.src_ = ...

        # tracking ...
        self.trk_ = np.empty((c,1), dtype=np.bool)
        # kpt will be pre-formatted to hold (x,y,rsp)
        # and maybe oct, idk.
        self.kpt_ = np.empty((c,3), dtype=np.float32)

        # query filtering
        # NOTE : maxd/mind scale filtering may be ORB-specific.
        self.s_fac_ = s_fac
        self.n_lvl_ = n_lvl
        self.s_pyr_ = np.power(self.s_fac_, np.arange(self.n_lvl_))
        self.maxd_ = np.empty((c,1), dtype=np.float32)
        self.mind_ = np.empty((c,1), dtype=np.float32)

        self.fields_ = [
                'pos_', 'des_', 'ang_', 'col_',
                'var_', 'cnt_',
                'trk_', 'kpt_',
                'maxd_', 'mind_']

        # store index for efficient pruning
        self.pidx_ = 0

    def resize(self, c_new):
        print('-------landmarks resizing : {} -> {}'.format(self.capacity_, c_new))

        d = vars(self)
        for f in self.fields_:
            # old data
            c_old = self.size_
            d_old = d[f][:c_old]
            s_old = d_old.shape

            # new data
            s_new = (c_new,) + tuple(s_old[1:])
            d_new = np.empty(s_new, dtype=d[f].dtype)
            d_new[:c_old] = d_old[:c_old]

            # set data
            d[f] = d_new
        self.capacity_ = c_new

    @staticmethod
    def lm_var(cvt, src, pos_c):
        # initialize variance
        var_rel = np.square([0.02, 0.02, 1.0]) # expected landmark variance @ ~ 1m
        # TODO : ^ is a ballpark estimate.
        var_rel = np.diag(var_rel) # 3x3
        var_c = oriented_cov(pos_c, var_rel) # Nx3x3

        # now rotate camera-coordinate variance to map-coord variance
        T_b2o = cvt.pose_to_T(src)
        R_c2m = np.linalg.multi_dot([
            cvt.T_b2c_[:3,:3], #R-part
            T_b2o[:3,:3],
            cvt.T_c2b_[:3,:3]
            ])

        var_m = reduce(np.matmul,[
            R_c2m, var_c, R_c2m.T])
        return var_m

    def append_from(self, cvt,
            src, pos_c,
            des, col, kpt):
        # unroll source
        src_x, src_y, src_h = np.ravel(src)
        # compute expected variance from depth
        # lmk pos w.r.t base
        pos = cvt.cam_to_map(pos_c, src)

        # original pose-based view angle may be sufficient
        # NOTE : if unreliable, consider rigorous validation
        ang    = np.full((len(pos),1), src_h, dtype=np.float32)

        var = self.lm_var(cvt, src, pos_c)
        dis = np.linalg.norm(pos_c, axis=-1)
        self.append(pos, var, des, ang, col, kpt,
                dis)

    def append(self, p, v, d, a, c, k,
            dist=None
            ):
        n = len(p)
        if self.size_ + n > self.capacity_:
            self.resize(self.capacity_ * 2)
            # retry append after resize
            self.append(p,v,d,a,c,k)
        else:
            # assign
            i = np.s_[self.size_:self.size_+n]
            self.pos_[i] = p
            self.var_[i] = v
            self.des_[i] = d
            self.ang_[i] = a
            self.col_[i] = c
            #self.kpt_[i] = k#[:,None]
            self.kpt_[i, :2] = np.reshape([e.pt for e in k], [-1,2])
            self.kpt_[i, 2]  = [e.response for e in k]

            # auto initialized vars
            self.cnt_[i] = 1
            self.trk_[i] = True

            if dist is not None:
                # depth-based visibility according to ORB-SLAM
                lsf = np.float32([self.s_pyr_[e.octave] for e in k])
                mxd = dist * lsf
                mnd = mxd / self.s_pyr_[-1]
                # NOTE : applying a small margin around mind/maxd
                # accounting for error in initial dist estimate.
                self.maxd_[i] = mxd[:,None]# * self.s_fac_
                self.mind_[i] = mnd[:,None]# / self.s_fac_
            else:
                # do not apply visibility
                self.maxd_[i] = np.inf
                self.mind_[i] = 0.0

            # update size
            self.size_ += n

    def query(self, src, cvt,
            atol = np.deg2rad(30.0),
            dtol = 1.2,
            trk=False
            ):
        # unroll map query source (base frame)
        src_x, src_y, src_h = np.ravel(src)

        # filter : by view angle
        a_dif = np.abs((self.ang[:,0] - src_h + np.pi)
                % (2*np.pi) - np.pi)
        a_msk = np.less(a_dif, atol) # TODO : kind-of magic number

        # filter : by min/max ORB distance
        ptmp  = cvt.T_b2c_.dot([src_x, src_y, 0, 1]).ravel()[:3]
        dist = np.linalg.norm(self.pos - ptmp[None,:], axis=-1)
        d_msk = np.logical_and.reduce([
            self.mind[:,0] / dtol <= dist,
            dist <= self.maxd[:,0] * dtol
            ])

        # filter : by visibility
        if self.pos.size <= 0:
            # no data : soft fail
            pt2   = np.empty((0,2), dtype=np.float32)
            v_msk = np.empty((0,),  dtype=np.bool)
        else:
            pt2, v_msk = cvt.pt3_pose_to_pt2_msk(
                    self.pos, src)

        # merge filters
        msk = np.logical_and.reduce([
            v_msk,
            a_msk,
            d_msk
            ])

        if trk:
            msk &= self.trk[:,0]
        idx = np.where(msk)[0]

        # collect results + return
        pt2 = pt2[idx]
        pos = self.pos[idx]
        des = self.des[idx]
        var = self.var[idx]
        cnt = self.cnt[idx]
        return (pt2, pos, des, var, cnt, idx)

    def update(self, idx, pos_new, var_new=None, hard=False):
        if hard:
            # total overwrite
            self.pos[idx] = pos_new
            if var_new is not None:
                self.var[idx] = var_new
        else:
            # incorporate previous information
            pos_old = self.pos[idx]
            var_old = self.var[idx]

            # kalman filter (== multivariate product)
            y_k = (pos_new - pos_old).reshape(-1,3,1)
            S_k = var_new + var_old # I think R_k = var_lm_new (measurement noise)
            K_k = np.matmul(var_old, np.linalg.inv(S_k))
            x_k = pos_old.reshape(-1,3,1) + np.matmul(K_k, y_k)
            I = np.eye(3)[None,...] # (1,3,3)
            P_k = np.matmul(I - K_k, var_old)

            # TODO : maybe also update mind_ and maxd_
            # bookkeeping self.src_ maybe?

            self.pos[idx] = x_k[...,0]
            self.var[idx] = P_k
            np.add.at(self.cnt, idx, 1) # can't trust cnt[idx] to return a view.

    def prune(self, k=8, radius=0.05, keep_last=512):
        """
        Non-max suppression based pruning.
        set k=1 to disable  nmx. --> TODO: verify this
        """
        # TODO : Tune keep_last parameter
        # TODO : sometimes pruning can be too aggressive
        # and get rid of desirable landmarks.
        # TODO : if(num_added_landmarks_since_last > x) == lots of new info
        #             search_and_add_keyframe()

        # NOTE: choose value to suppress with
        #v = 1.0 / np.linalg.norm(self.var[:,(0,1,2),(0,1,2)], axis=-1)
        v = self.kpt[:, 2]

        neigh = NearestNeighbors(n_neighbors=k)
        neigh.fit(self.pos)
        d, i = neigh.kneighbors(return_distance=True)
        # filter results by non-max suppression radius
        msk_d = np.min(d, axis=1) < radius

        # neighborhood count TODO : or np.sum() < 2?
        msk_v = np.all(v[i] <= v[:,None], axis=1)

        # protect recent observations.
        # keep latest N landmarks
        msk_t = np.arange(self.size_) > (self.size_ - keep_last)
        # and keep all landmarks added after last prune
        msk_t |= np.arange(self.size_) >= self.pidx_
        # also keep all currently tracked landmarks
        msk_t |= self.trk[:,0]

        # strong responses are preferrable and will be kept
        rsp = self.kpt[:,2]
        msk_r = np.greater(rsp, 48) # TODO : somewhat magical

        # non-max results + response filter
        msk_n = (msk_d & msk_v) | (~msk_d & msk_r)

        # below expression describes the following heuristic:
        # if (new_landmark) keep;
        # else if (passed_non_max) keep;
        # else if (strong_descriptor) keep;
        #msk = msk_t | (msk_d & msk_v | ~msk_d) | (msk_r & ~msk_d)
        #msk = msk_t | ~msk_d | msk_v
        # msk = msk_t | (msk_n & (np.linalg.norm(self.pos) < 30.0)) 

        msk = msk_t | (msk_n & ( (np.linalg.norm(self.pos, axis=-1) < 30.0) ) ) # +enforce landmark bounds

        sz = msk.sum()
        print('Landmarks Pruning : {}->{}'.format(msk.size, sz))

        d = vars(self)
        for f in self.fields_:
            # NOTE: using msk here instead of index,
            # in order to purposefully make a copy.
            d[f][:sz] = d[f][:self.size_][msk]
        self.size_ = sz
        self.pidx_ = self.size_

        # return pruned indices
        return np.where(msk)[0]

    def track_points(self):
        t_idx = np.where(self.trk)[0]
        return t_idx, self.kpt_[t_idx, :2]

    def untrack(self, idx):
        self.trk[idx] = False

    @property
    def pos(self):
        return self.pos_[:self.size_]
    @property
    def var(self):
        return self.var_[:self.size_]
    @property
    def des(self):
        return self.des_[:self.size_]
    @property
    def ang(self):
        return self.ang_[:self.size_]
    @property
    def col(self):
        return self.col_[:self.size_]
    @property
    def cnt(self):
        return self.cnt_[:self.size_]
    @property
    def trk(self):
        return self.trk_[:self.size_]
    @property
    def kpt(self):
        return self.kpt_[:self.size_]
    @property
    def mind(self):
        return self.mind_[:self.size_]
    @property
    def maxd(self):
        return self.maxd_[:self.size_]

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
        return (pth[:, :-1] / pth[:, -1:])

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
