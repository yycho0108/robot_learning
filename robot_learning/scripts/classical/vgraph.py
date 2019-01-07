import numpy as np
from ukf import build_ekf, build_ukf, get_QR

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


