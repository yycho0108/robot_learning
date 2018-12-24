import numpy as np
import cv2
from tf import transformations as tx

def print_Rt(R, t):
    print '\tR', np.round(np.rad2deg(tx.euler_from_matrix(R)), 2)
    print '\tt', np.round(t.ravel() / np.linalg.norm(t), 2)

def recover_pose_from_RT(perm, K,
        pt1, pt2,
        z_min = np.finfo(np.float32).eps,
        z_max = np.inf,
        log=False
        ):
    P1 = np.eye(3,4)
    P2 = np.eye(3,4)

    msks = [None for _ in range(4)]
    pt3s = [None for _ in range(4)]

    sel   = 0
    ctest = -np.inf

    for i, (R, t) in enumerate(perm):
        #print '==== recover_pose validation ===='
        P2[:3,:3] = R
        P2[:3,3:] = t.reshape(3,1)

        KP1 = K.dot(P1)
        KP2 = K.dot(P2)

        pth_a = cv2.triangulatePoints(
                KP1, KP2,
                pt1[None,...],
                pt2[None,...]).astype(np.float32)
        pth_a /= pth_a[3:]

        pt3_a = P1.dot(pth_a)
        pt3_b = P2.dot(pth_a)

        pt3s[i] = pt3_a

        za, zb = pt3_a[2], pt3_b[2]

        msk_i = np.logical_and.reduce([
            z_min < za,
            za < z_max,
            z_min < zb,
            zb < z_max
            ])
        msks[i] = msk_i
        c = msk_i.sum()
        if log:
            print('[{}] {}/{}'.format(i, c, msk_i.size))
            print_Rt(R, t)

        if c > ctest:
            sel = i
            ctest = c

    R, t = perm[sel]
    msk = msks[sel]
    pt3 = pt3s[sel][:,msk]
    n_in = msk.sum()

    return n_in, R, t, msk, pt3

def recover_pose(E, K,
        pt1, pt2,
        z_min = np.finfo(np.float32).eps,
        z_max = np.inf,
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
            z_min, z_max, log
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
        plt.pause(0.001)

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
        self.n_des_=n_des
        self.capacity_ = c = 1024
        self.size_ = s = 0

        c = self.capacity_
        s = self.size_

        self.pos_ = np.empty((c,3), dtype=np.float32)
        self.var_ = np.empty((c,3,3), dtype=np.float32)
        self.des_ = np.empty((c,n_des), dtype=np.int32)
        # TODO : ^ highly dependent on descriptor
        self.ang_ = np.empty((c,1), dtype=np.float32)
        self.col_ = np.empty((c,3), dtype=np.uint8)
        self.cnt_ = np.empty((c,1), dtype=np.int32)

    def resize(self, c_new):
        print('-------landmarks resizing : {} -> {}'.format(self.capacity_, c_new))
        p = np.empty((c_new,3), dtype=np.float32)
        v = np.empty((c_new,3,3), dtype=np.float32)
        d = np.empty((c_new,self.n_des_), dtype=np.int32)
        a = np.empty((c_new,1), dtype=np.float32)
        c = np.empty((c_new,3), dtype=np.uint8)
        c2 = np.empty((c_new,3), dtype=np.int32)

        p[:self.size_] = self.pos
        v[:self.size_] = self.var
        d[:self.size_] = self.des
        a[:self.size_] = self.ang
        c[:self.size_] = self.col
        c2[:self.size_] = self.cnt

        self.pos_ = p
        self.var_ = v
        self.des_ = d
        self.ang_ = a
        self.col_ = c
        self.cnt_ = c2

        self.capacity_ = c_new

    def append(self, p,v,d,a,c):
        n = len(p)
        if self.size_ + n > self.capacity_:
            self.resize(self.capacity_ * 2)
            # retry append
            self.append(p,v,d,a,c)
        else:
            # assign
            self.pos_[self.size_:self.size_+n] = p
            self.var_[self.size_:self.size_+n] = v
            self.des_[self.size_:self.size_+n] = d
            self.ang_[self.size_:self.size_+n] = a
            self.col_[self.size_:self.size_+n] = c
            self.cnt_[self.size_:self.size_+n] = 1
            # update size
            self.size_ += n

    def prune(self, min_cnt=3, min_keep=512):
        # TODO : prune by confidence, etc.
        msk_c = (self.cnt >= min_cnt)[...,0]
        msk_t = np.arange(self.size_) > (self.size_ - min_keep)
        msk = (msk_c | msk_t)
        sz  = msk.sum()
        self.pos[:sz] = self.pos[msk]
        self.var[:sz] = self.var[msk]
        self.des[:sz] = self.des[msk]
        self.ang[:sz] = self.ang[msk]
        self.col[:sz] = self.col[msk]
        self.size_ = sz

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

    def pt3_pose_to_pt2_msk(self, pt3, pose, distort=False):
        # pose = (x,y,h)
        # NOTE: pose is specified w.r.t base_link, not camera.

        D = self.D_
        if not distort:
            D *= 0.0
        
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

    def E_to_F(self, E):
        return np.linalg.multi_dot([
            self.Ki_.T, E, self.Ki_])

    def F_to_E(self, F):
        return np.linalg.multi_dot([
            self.K_.T, F, self.K_])

    def __call__(self, ftype, *a, **k):
        return self.f_[ftype](*a, **k)
