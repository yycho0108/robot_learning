import numpy as np
import cv2
from vo_common import oriented_cov
from sklearn.neighbors import NearestNeighbors

# TODO : currently in the process
# of refactoring with accord to the new parametrization,
# lmk = src + d * vec.

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

        # container management ...
        self.capacity_ = c = 1024
        self.size_ = s = 0

        # general data ...
        self.src_ = np.empty((c,1), dtype=np.int32) # source pose index
        self.idp_ = np.empty((c,1), dtype=np.float32) # landmarks depth (NOTE: inverse, 1/d)
        self.vec_ = np.empty((c,3), dtype=np.float32) # view-vector from src, eager initialization
        self.des_ = np.empty((c,n_des), dtype=t_des)
        self.ang_ = np.empty((c,1), dtype=np.float32)
        self.col_ = np.empty((c,3), dtype=np.uint8)

        # landmark "health" ...
        self.var_ = np.empty((c,1), dtype=np.float32) # holds depth variance only
        self.cnt_ = np.empty((c,1), dtype=np.int32)

        # tracking ...
        self.trk_ = np.empty((c,1), dtype=np.bool)
        # kpt will be pre-formatted to hold (x,y,rsp,oct)
        self.kpt_ = np.empty((c,4), dtype=np.float32)

        # query filtering
        # NOTE : maxd/mind scale filtering may be ORB-specific.
        self.s_fac_ = s_fac
        self.n_lvl_ = n_lvl
        self.s_pyr_ = np.power(self.s_fac_, np.arange(self.n_lvl_))

        self.fields_ = [
                'src_', 'idp_', 'vec_',
                'des_', 'ang_', 'col_',
                'var_', 'cnt_',
                'trk_', 'kpt_',
                ]

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

    def append_from(self, cvt,
            src, src_i, pos_c,
            des, col, kpt):
        # unroll source
        src_x, src_y, src_h = np.ravel(src)
        # compute expected variance from depth
        # lmk pos w.r.t base
        pos = cvt.cam_to_map(pos_c, src)

        # extract z-depth
        idp = 1.0 / pos_c[:, 2:] # NOTE : inverse depth parametrized
        vec = pos_c * idp 

        # if idp@10m  = 0.1, var = 10.0 --> 0.1
        # if idp@1m   = 1.0, var = 100.0 --> 0.01
        # if idp@0.1m = 10.0, var = 1000.0 --> 0.001
        # anyways, a rough guess. a better estimate can be found from pixel disparity.
        # TODO : initialize variance from pixel disparity
        var = idp * 100.0

        # original pose-based view angle may be sufficient
        # NOTE : if unreliable, consider rigorous validation
        ang = np.full((len(pos),1), src_h, dtype=np.float32)

        self.append(src_i, idp, vec,
                des, ang, col,
                var, kpt)

    def append(self, src_i, idp, vec,
            des, ang, col, var, kpt):
            ):
        n = len(p)
        if self.size_ + n > self.capacity_:
            self.resize(self.capacity_ * 2)
            # retry append after resize
            self.append(p,v,d,a,c,k)
        else:
            # assign
            i = np.s_[self.size_:self.size_+n]

            self.src_[i] = src_i
            self.idp_[i] = idp
            self.vec_[i] = vec
            self.des_[i] = des
            self.ang_[i] = ang
            self.col_[i] = col
            self.var_[i] = var

            self.kpt_[i, :2] = np.reshape([e.pt for e in k], [-1,2])
            self.kpt_[i, 2]  = [e.response for e in k]
            self.kpt_[i, 3]  = [e.octave for e in k]

            # auto initialized vars
            self.cnt_[i] = 1
            self.trk_[i] = True

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

