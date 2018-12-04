import numpy as np
import sys
import cv2
from scipy.optimize import linear_sum_assignment
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from tf import transformations as tx
from utils.vo_utils import add_p3d, sub_p3d

from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints, JulierSigmaPoints

# ukf = (x,y,h,v,w)

def ukf_fx(s, dt):
    x,y,h,v,w = s

    dx = v * dt
    dy = 0.0
    dh = w * dt
    x,y,h = add_p3d([x,y,h], [dx,dy,dh])
    #x += v * np.cos(h) * dt
    #y += v * np.sin(h) * dt
    #h += w * dt
    #v *= 0.999
    return np.asarray([x,y,h,v,w])

def ukf_hx(s):
    return s[:3].copy()

def ukf_mean(xs, wm):
    """
    Runs circular mean for angular states, which is critical to preventing issues related to linear assumptions. 
    WARNING : do not replace with the default mean function
    """
    # Important : SUM! not mean.
    mx = np.sum(xs * np.expand_dims(wm, -1), axis=0)
    ms = np.mean(np.sin(xs[:,2])*wm)
    mc = np.mean(np.cos(xs[:,2])*wm)
    mx[2] = np.arctan2(ms,mc)
    return mx

def ukf_residual(a, b):
    """
    Runs circular residual for angular states, which is critical to preventing issues related to linear assumptions.
    WARNING : do not replace with the default residual function.
    """
    d = np.subtract(a, b)
    d[2] = np.arctan2(np.sin(d[2]), np.cos(d[2]))
    return d

def triangulatePoints(P1, P2, p1, p2):
    """
    Custom impl., as cv2.triangulatePoints resulted in memory Error.
    """

    pt = [p1, p2]
    P = [P1,P2]

    n = len(p1)

    pt_4dh = np.empty((n, 4), dtype=np.float32)
    for i in range(n):
        A = np.empty((4,4), dtype=np.float32)
        for j in range(2):
            x, y = pt[j][i]
            A[j*2+0] = x * P[j][2] - P[j][0]
            A[j*2+1] = y * P[j][2] - P[j][1]
        U,s,V = np.linalg.svd(A) # U.diag(S).V == A
        pt_4dh[i] = V[3,:]
    return pt_4dh[:,:3] / pt_4dh[:, 3:]

def recoverPoseWithPoints(E, p1, p2,
        fx, fy, cx, cy):
    # mostly based on cv2.recoverPose() implementation
    c = np.reshape( (cx,cy), (1,2) ).astype(np.float32)
    f = np.reshape( (fx,fy), (1,2) ).astype(np.float32)

    # normalize points
    p1_n = (p1 - c) / f
    p2_n = (p2 - c) / f

    R1, R2, t = cv2.decomposeEssentialMat(E)
    # choose from argmin(R(z))

    cand = (
            (R1, t),
            (R2, t),
            (R1, -t),
            (R2, -t))

    P0 = np.eye(3, 4, dtype=np.float32)
    max_det = None
    R_res = None
    t_res = None

    # candidates filtering by chirality test
    for (cR,ct) in cand:
        cP = np.concatenate((cR, ct), axis=1)
        Q  = triangulatePoints(P0, cP, p1, p2)

        msk = np.logical_and(
                np.greater(Q[:,:2], 0.0),
                np.less(Q[:,:2], 50.0))
        det = np.sum(msk)
        if (max_det is None) or (det >= max_det):
            max_det = det
            R_res = cR
            t_res = ct

    return max_det, R_res, t_res

class ClassicalVO(object):
    def __init__(self):
        self.prv_ = None # previous image

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
            if m.distance < 0.75*n.distance:
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

    def __call__(self, img,
            in_thresh = 32,
            s = 0.1
            ):

        kp2, des2 = self.orb_.detectAndCompute(img, None)
        #kp = self.gftt_.detect(img)
        #kp2, des2 = self.brisk_.compute(img, kp)

        if des2 is None:
            # something is invalid about the image?
            return True, None

        kp2 = np.array(kp2)
        img2 = img

        des2 = np.float32(des2)
        if self.prv_ is None:
            self.prv_ = (kp2, des2, img2)
            return True, None

        # swap prv
        kp1, des1, img1 = self.prv_
        self.prv_ = (kp2, des2, img2)

        matches = self.match_v1(des1, des2)
        matches = sorted(matches, key=lambda e:e.distance)
        #matches = self.match(des1, des2)

        if (matches is None) or (len(matches) <= 8):
            # no matches or insufficient # of matches
            return True, None

        i1, i2 = np.int32([(m.queryIdx, m.trainIdx) for m in matches]).T
        p1 = np.int32([e.pt for e in kp1[i1]])
        p2 = np.int32([e.pt for e in kp2[i2]])

        # TODO : expose these parameters
        focal = 530.0 / 4.0
        pp    = (160,120)#(0,0)#(160,120)

        eres = cv2.findEssentialMat(p2, p1, focal=focal, pp=pp,
                method=cv2.RANSAC, prob=0.9999, threshold=0.5
                )
        Emat, mask = eres[0], eres[1] # p2 w.r.t. p1
        cmat = np.float32([focal,0,pp[0], 0, focal, pp[1], 0,0,1]).reshape(3,3)

        n_in, R, t, msk, pts = cv2.recoverPose(Emat, p2[:64], p1[:64], cameraMatrix=cmat, distanceThresh=50.0)

        # TODO : do the correct scale estimation

        # points: homogeneous --> 3d coordinates
        pts = pts[:, (msk[:,0]==255)]
        pts3 = pts[:3] / pts[3]

        # TODO : will this take care of scale?

        # convert to base_link coordinates
        pts3 = s * np.stack([pts3[2], -pts3[0], -pts3[1]], axis=-1)

        pts2 = pts3[:, :2]
        # filter by large displacement
        #pts2 = pts2[np.linalg.norm(pts2, axis=-1) < 10.0] #filter by radius=10.0m
        #pts2 = pts2[np.sign(pts2[:,0]) == 1]

        if n_in < in_thresh:
            # insufficient number of inliers to recover pose
            return True, None

        h = tx.euler_from_matrix(R)
        #h = np.round(np.rad2deg(tx.euler_from_matrix(R)), 2)
        #t = np.round(t, 2)
        #print ('h-z', -h[1])
        #print ('t-xy', t[2,0], -t[0,0])

        h = -h[1]
        #print('dh', np.round(np.rad2deg(h), 2))

        # no-slip
        # TODO : why is t[0,0] so bad???
        # TODO : relax the no-slip constraint by being better at triangulating or something
        t = np.asarray([t[2,0], 0.0 * -t[0,0]])

        # magnitude of t is always 1.0! can we do anything about this?
        # something about overall scale
        t *= s

        print len(kp1)
        print len(msk)
        mskp = np.pad(msk, [[0, len(matches)-len(msk)], [0,0]], mode='constant')

        print len(matches), msk.shape

        draw_params = dict(
                matchColor = (0,255,0),
                singlePointColor = (255,0,0),
                flags = 0,
                matchesMask=mskp.ravel().tolist()
                )
        mim = cv2.drawMatches(
                img1,kp1,img2,kp2,
                matches,None,**draw_params)
        mim = cv2.addWeighted(np.concatenate([img1,img2],axis=1), 0.5, mim, 0.5, 0.0)
        cv2.drawKeypoints(mim, kp1[i1][mskp[:,0]==255], mim, color=(0,0,255))
        print('---')

        return True, (mim, h, t, pts2, pts3)

        #print 'm', m

def Rmat(x):
    c,s = np.cos(x), np.sin(x)
    R = np.float32([c,-s,s,c]).reshape(2,2)
    return R

class CVORunner(object):
    def __init__(self, imgs, stamps, odom, scan=None):
        self.index_ = 0
        self.n_ = len(imgs)
        self.imgs_ = imgs
        self.stamps_ = stamps
        self.odom_ = odom
        self.scan_ = scan

        self.fig_ = fig = plt.figure()
        self.ax0_ = fig.add_subplot(2,2,1)
        self.ax1_ = fig.add_subplot(1,2,2)
        self.ax2_ = fig.add_subplot(2,2,3, projection='3d')

        self.map_ = np.empty((0, 2), dtype=np.float32)
        self.vo_ = ClassicalVO()
        self.ukf_ = self._build_ukf()
        self.vo_( imgs[0] )

        self.tx_ = []
        self.ty_ = []
        self.th_ = []
        self.quit_ = False

    def _build_ukf(self):
        # build ukf
        Q = np.diag([1e-4, 1e-4, 1e-4, 1e-2, 1e-2]) #xyhvw
        R = np.diag([1e-3, 1e-3, 1e-3]) # xyh
        x0 = np.zeros(5)
        P0 = np.diag([1e-6,1e-6,1e-6, 1e-3, 1e-3])

        #spts = MerweScaledSigmaPoints(5,1e-3,2,-2,subtract=ukf_residual)
        spts = JulierSigmaPoints(5, 5-2, sqrt_method=np.linalg.cholesky, subtract=ukf_residual)

        ukf = UKF(5, 3, (1.0 / 30.), # dt guess
                ukf_hx, ukf_fx, spts,
                x_mean_fn=ukf_mean,
                z_mean_fn=ukf_mean,
                residual_x=ukf_residual,
                residual_z=ukf_residual)
        ukf.x = x0.copy()
        ukf.P = P0.copy()
        ukf.Q = Q
        ukf.R = R

        return ukf

    def handle_key(self, event):
        k = event.key
        if k in ['n', ' ', 'enter']:
            self.index_ += 1
            if self.index_ < self.n_:
                self.step()
        if k in ['q', 'escape']:
            self.quit_ = True
            sys.exit(0)

    def scan_to_pt(self, scan):
        r, mask = scan[:,0], scan[:,1]
        a = np.linspace(0, 2*np.pi, 361)
        mask = mask.astype(np.bool)
        r, a = r[mask], a[mask]
        #mask = ( np.abs(a) < np.deg2rad(45) )
        #r, a = r[mask], a[mask]
        c, s = np.cos(a), np.sin(a)
        p = r[:,None] * np.stack([c,s], axis=-1)
        return p

    def step(self):
        i = self.index_
        n = self.n_
        print('i : {}/{}'.format(i,n))

        if i >= n:
            return

        # unroll properties
        odom   = self.odom_
        scan   = self.scan_
        stamps = self.stamps_
        imgs  = self.imgs_
        ukf   = self.ukf_
        vo    = self.vo_
        ax0,ax1,ax2 = self.ax0_, self.ax1_, self.ax2_

        # index
        stamp = stamps[i]
        img   = imgs[i]

        # TODO : there was a bug in data_collector that corrupted all time-stamp data!
        # very unfortunate.
        #dt    = (stamps[i] - stamps[i-1])
        dt = 0.1

        # experimental : pass in scale as a parameter
        # TODO : estimate scale
        dps_gt = sub_p3d(odom[i], odom[i-1])
        s = np.linalg.norm(dps_gt[:2])

        prv = ukf.x[:3].copy()

        ukf.predict(dt=dt)
        suc, res = vo(img, s=s)
        if not suc:
            print('Visual Odometry Aborted!')
            return

        if res is None:
            # skip filter updates
            return

        (img, dh, dt, pts, pts3) = res
        dps = np.float32([dt[0], dt[1], dh])
        print('(pred-gt) {} vs {}'.format(dps, dps_gt) )
        pos = add_p3d(prv, dps)
        ukf.update(pos)

        self.tx_.append( float(ukf.x[0]) )
        self.ty_.append( float(ukf.x[1]) )
        self.th_.append( float(ukf.x[2]) )

        ax0.cla()
        ax1.cla()
        ax2.cla()

        ax0.imshow(img[...,::-1])
        ax0.axis('off')

        # TODO : plot err in dps
        #ax2.plot(dpss)
        #ax2.axis('off')

        #ax1.plot(tx, ty, 'r--')
        #ax1.quiver(tx, ty,
        #        np.cos(th), np.sin(th),
        #        scale_units='xy',
        #        angles='xy')

        ax1.plot(odom[:i+1,0], odom[:i+1,1], 'k--')

        # pts2 in the proper coordinate system
        pts_c = pts.dot(Rmat(odom[i,2]).T) + np.reshape(odom[i,:2], (1,2))

        ph = np.rad2deg(np.arctan2(pts[:,1], pts[:,0]))
        print 'fov', np.min(ph), np.max(ph)

        self.map_ = np.concatenate([self.map_, pts_c], axis=0)
        ax1.plot(pts_c[:,0], pts_c[:,1], 'r.', label='visual')

        scan_c = self.scan_to_pt(scan[i]).dot(Rmat(odom[i,2]).T) + np.reshape(odom[i, :2], (1,2))

        ax1.plot(scan_c[:,0], scan_c[:,1], 'b.', label='scan')
        ax1.plot([0],[0],'k+')

        lx = np.linspace(0, 5)
        lx = np.stack([lx,lx*0],axis=-1)

        fov_l = lx.dot(Rmat(odom[i,2]-np.deg2rad(45)).T) + np.reshape(odom[i,:2], (1,2))
        fov_r = lx.dot(Rmat(odom[i,2]+np.deg2rad(45)).T) + np.reshape(odom[i,:2], (1,2))
        h = np.linspace(-np.pi, np.pi)
        radius = 5.0 * np.stack([np.cos(h),np.sin(h)], axis=-1)

        ax1.plot(fov_l[:,0], fov_l[:,1], 'g--')
        ax1.plot(fov_r[:,0], fov_r[:,1], 'g--')
        ax1.plot(radius[:,0], radius[:,1], 'g--')

        cx, cy = odom[i,:2]
        ax1.set_xlim(cx-5.0, cx+5.0)
        ax1.set_ylim(cy-5.0, cy+5.0)

        ax2.scatter(pts3[:,0], pts3[:,1], pts3[:,2])
        mx = np.abs(pts3).max(axis=0)
        ax2.set_xlabel('x')
        #ax2.set_xlim(-mx[0],mx[0])
        ax2.set_ylabel('y')
        #ax2.set_ylim(-mx[1],mx[1])
        ax2.set_zlabel('z')
        plt.legend()

        #ax2.set_zlim(-mx[2],mx[2])
        #ax2.set_aspect('equal')

        #ax1.quiver(tx, ty,
        #        np.cos(th), np.sin(th),
        #        scale_units='xy',
        #        angles='xy')

        self.fig_.canvas.draw()

    def run(self, auto=False):
        self.fig_.canvas.mpl_connect('key_press_event', self.handle_key)

        if auto:
            while not self.quit_:
                if self.index_ < self.n_:
                    self.index_ += 1
                    self.step()
                plt.pause(0.001)
        else:
            plt.show()

def main():
    idx = np.random.choice(8)
    #idx = 13
    print('idx', idx)

    # load data
    imgs   = np.load('../data/train/{}/img.npy'.format(idx))
    stamps = np.load('../data/train/{}/stamp.npy'.format(idx))
    odom   = np.load('../data/train/{}/odom.npy'.format(idx))
    scan   = np.load('../data/train/{}/scan.npy'.format(idx))

    # set odom @ t0= (0,0,0)
    R0 = Rmat(odom[0,2])
    odom -= odom[0]
    odom[:,:2] = odom[:,:2].dot(R0)

    stamps -= stamps[0] # t0 = 0

    app = CVORunner(imgs, stamps, odom, scan)
    app.run()

    #vo = ClassicalVO()

    #fig = plt.figure()
    #ax0 = fig.add_subplot(2,2,1)
    #ax1 = fig.add_subplot(1,2,2)
    #ax2 = fig.add_subplot(2,2,3, projection='3d')

    #pos = np.zeros(shape=3, dtype=np.float32)

    #tx = []
    #ty = []
    #th = []

    ## build ukf
    #Q = np.diag([1e-4, 1e-4, 1e-4, 1e-2, 1e-2]) #xyhvw
    #R = np.diag([1e-3, 1e-3, 1e-3]) # xyh
    #x0 = np.zeros(5)
    #P0 = np.diag([1e-6,1e-6,1e-6, 1e-3, 1e-3])

    ##spts = MerweScaledSigmaPoints(5,1e-3,2,-2,subtract=ukf_residual)
    #spts = JulierSigmaPoints(5, 5-2, sqrt_method=np.linalg.cholesky, subtract=ukf_residual)

    #ukf = UKF(5, 3, (1.0 / 30.), # dt guess
    #        ukf_hx, ukf_fx, spts,
    #        x_mean_fn=ukf_mean,
    #        z_mean_fn=ukf_mean,
    #        residual_x=ukf_residual,
    #        residual_z=ukf_residual)
    #ukf.x = x0.copy()
    #ukf.P = P0.copy()
    #ukf.Q = Q
    #ukf.R = R

    #n = len(stamps)
    #vo(imgs[0]) # initialize vo

    #pt_map = np.empty((0, 2), dtype=np.float32)

    #state = {'index' : 1, 'new' : True}
    #def handle_key(event):
    #    k = event.key
    #    if k in ['n', ' ', 'enter']:
    #        state['index'] += 1
    #    if k in ['q', 'escape']:
    #        sys.exit(0)

    #fig.canvas.mpl_connect('key_press_event', handle_key)

    #while True:
    #    i = state['index']
    #    if i >= n:
    #        break
    #    stamp = stamps[i]
    #    img   = imgs[i]

    #    # TODO : there was a bug in data_collector that corrupted all time-stamp data!
    #    # very unfortunate.
    #    #dt    = (stamps[i] - stamps[i-1])
    #    dt = 0.1

    #    # experimental : pass in scale as a parameter
    #    # TODO : estimate scale
    #    dps_gt = sub_p3d(odom[i], odom[i-1])
    #    s = np.linalg.norm(dps_gt[:2])

    #    prv = ukf.x[:3].copy()

    #    ukf.predict(dt=dt)
    #    suc, res = vo(img, s=s)
    #    if not suc:
    #        print('Visual Odometry Aborted!')
    #        break

    #    if res is None:
    #        # skip filter updates
    #        continue

    #    (img, dh, dt, pts, pts3) = res
    #    dps = np.float32([dt[0], dt[1], dh])
    #    print('(pred-gt) {} vs {}'.format(dps, dps_gt) )
    #    pos = add_p3d(prv, dps)
    #    ukf.update(pos)

    #    tx.append( float(ukf.x[0]) )
    #    ty.append( float(ukf.x[1]) )
    #    th.append( float(ukf.x[2]) )

    #    ax0.cla()
    #    ax1.cla()
    #    ax2.cla()

    #    ax0.imshow(img[...,::-1])
    #    ax0.axis('off')

    #    # TODO : plot err in dps
    #    #ax2.plot(dpss)
    #    #ax2.axis('off')

    #    #ax1.plot(tx, ty, 'r--')
    #    #ax1.quiver(tx, ty,
    #    #        np.cos(th), np.sin(th),
    #    #        scale_units='xy',
    #    #        angles='xy')

    #    ax1.plot(odom[:i+1,0], odom[:i+1,1], 'k--')

    #    # pts2 in the proper coordinate system
    #    pts_c = pts.dot(Rmat(odom[i,2]).T) + np.reshape(odom[i,:2], (1,2))

    #    ph = np.rad2deg(np.arctan2(pts[:,1], pts[:,0]))
    #    print 'fov', np.min(ph), np.max(ph)

    #    pt_map = np.concatenate([pt_map, pts_c], axis=0)
    #    ax1.plot(pts_c[:,0], pts_c[:,1], '.')

    #    ax2.scatter(pts3[:,0], pts3[:,1], pts3[:,2])
    #    ax2.set_xlabel('x')
    #    ax2.set_ylabel('y')
    #    ax2.set_zlabel('z')

    #    #ax1.quiver(tx, ty,
    #    #        np.cos(th), np.sin(th),
    #    #        scale_units='xy',
    #    #        angles='xy')

    #    fig.canvas.draw()

if __name__ == "__main__":
    main()
