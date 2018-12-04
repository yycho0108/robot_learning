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

def linear_LS_triangulation(P1, P2, u1, u2):
    """
    Linear Least Squares based triangulation.
    Relative speed: 0.1
    
    (u1, P1) is the reference pair containing normalized image coordinates (x, y) and the corresponding camera matrix.
    (u2, P2) is the second pair.
    
    u1 and u2 are matrices: amount of points equals #rows and should be equal for u1 and u2.
    
    The status-vector will be True for all points.

    from https://github.com/Eliasvan/Multiple-Quadrotor-SLAM/blob/master/Work/python_libs/triangulation.py
    """
    A = np.zeros((4, 3))
    b = np.zeros((4, 1))
    
    # Create array of triangulated points
    x = np.zeros((3, len(u1)))
    
    # Initialize C matrices
    C1 = -np.eye(2,3)
    C2 = -np.eye(2,3)
    
    for i in range(len(u1)):
        # Derivation of matrices A and b:
        # for each camera following equations hold in case of perfect point matches:
        #     u.x * (P[2,:] * x)     =     P[0,:] * x
        #     u.y * (P[2,:] * x)     =     P[1,:] * x
        # and imposing the constraint:
        #     x = [x.x, x.y, x.z, 1]^T
        # yields:
        #     (u.x * P[2, 0:3] - P[0, 0:3]) * [x.x, x.y, x.z]^T     +     (u.x * P[2, 3] - P[0, 3]) * 1     =     0
        #     (u.y * P[2, 0:3] - P[1, 0:3]) * [x.x, x.y, x.z]^T     +     (u.y * P[2, 3] - P[1, 3]) * 1     =     0
        # and since we have to do this for 2 cameras, and since we imposed the constraint,
        # we have to solve 4 equations in 3 unknowns (in LS sense).

        # Build C matrices, to construct A and b in a concise way
        C1[:, 2] = u1[i, :]
        C2[:, 2] = u2[i, :]
        
        # Build A matrix:
        # [
        #     [ u1.x * P1[2,0] - P1[0,0],    u1.x * P1[2,1] - P1[0,1],    u1.x * P1[2,2] - P1[0,2] ],
        #     [ u1.y * P1[2,0] - P1[1,0],    u1.y * P1[2,1] - P1[1,1],    u1.y * P1[2,2] - P1[1,2] ],
        #     [ u2.x * P2[2,0] - P2[0,0],    u2.x * P2[2,1] - P2[0,1],    u2.x * P2[2,2] - P2[0,2] ],
        #     [ u2.y * P2[2,0] - P2[1,0],    u2.y * P2[2,1] - P2[1,1],    u2.y * P2[2,2] - P2[1,2] ]
        # ]
        A[0:2, :] = C1.dot(P1[0:3, 0:3])    # C1 * R1
        A[2:4, :] = C2.dot(P2[0:3, 0:3])    # C2 * R2
        
        # Build b vector:
        # [
        #     [ -(u1.x * P1[2,3] - P1[0,3]) ],
        #     [ -(u1.y * P1[2,3] - P1[1,3]) ],
        #     [ -(u2.x * P2[2,3] - P2[0,3]) ],
        #     [ -(u2.y * P2[2,3] - P2[1,3]) ]
        # ]
        b[0:2, :] = C1.dot(P1[0:3, 3:4])    # C1 * t1
        b[2:4, :] = C2.dot(P2[0:3, 3:4])    # C2 * t2
        b *= -1
        
        # Solve for x vector
        cv2.solve(A, b, x[:, i:i+1], cv2.DECOMP_SVD)
    
    return x.T.astype(np.float32)#, np.ones(len(u1), dtype=bool)


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

    def match_v2(img1, img2, kp1):
        lk_params = dict( winSize  = (15,15),
                maxLevel = 20,
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # p0 = cv2.goodFeaturesToTrack(src_gray, mask = None, **feature_params)
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(img1_gray, img2_gray, p0, None, **lk_params)

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

        #kp2, des2 = self.orb_.detectAndCompute(img, None)
        kp = self.gftt_.detect(img)
        kp2, des2 = self.brisk_.compute(img, kp)

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
        #matches = self.match(des1, des2)
        matches = sorted(matches, key=lambda e:e.distance)

        if (matches is None) or (len(matches) <= 8):
            # no matches or insufficient # of matches
            return True, None

        i1, i2 = np.int32([(m.queryIdx, m.trainIdx) for m in matches]).T
        p1 = np.float32([e.pt for e in kp1[i1]])
        p2 = np.float32([e.pt for e in kp2[i2]])

        # TODO : expose these parameters
        Kmat_o = np.reshape([
            499.114583, 0.000000, 325.589216,
            0.000000, 498.996093, 238.001597,
            0.000000, 0.000000, 1.000000], (3,3)) # orig. scale
        Kmat = np.reshape([
            499.114583 / 2.0, 0.000000, 325.589216 / 2.0,
            0.000000, 498.996093 / 2.0, 238.001597 / 2.0,
            0.000000, 0.000000, 1.000000], (3,3)) # half scale
        print np.round(Kmat)

        distCoeffs = np.float32([0.158661, -0.249478, -0.000564, 0.000157, 0.000000])

        focal = Kmat[0,0]
        pp    = (Kmat[0,2], Kmat[1,2])

        # undistort
        p1p = p1
        p1 = cv2.undistortPoints(2.0*p1[None,...], Kmat_o, distCoeffs, P=Kmat_o)[0]/2.0
        p2 = cv2.undistortPoints(2.0*p2[None,...], Kmat_o, distCoeffs, P=Kmat_o)[0]/2.0
        #print('what?', np.max(np.abs(p1p - p1)))

        # correct matches
        Fmat, mask = cv2.findFundamentalMat(p2, p1, method=cv2.FM_LMEDS,
                param1=0.1, param2=0.999)
        mask = np.asarray(mask[:,0]).astype(np.bool)
        p2, p1 = cv2.correctMatches(Fmat, p2[None,mask], p1[None, mask])
        p1 = p1[0, ...]
        p2 = p2[0, ...]

        # filter NaN
        msk = np.logical_and(np.isfinite(p1), np.isfinite(p2))
        msk = np.logical_and(msk[:,0], msk[:,1])
        p1 = p1[msk].astype(np.float32)
        p2 = p2[msk].astype(np.float32)

        if len(p1) <= 8:
            # ?? TODO : sideways translation??
            return True, None

        #Emat0 = (Kmat.T).dot(Fmat).dot(Kmat)
        #print "emat", Emat0
        Emat, mask = cv2.findEssentialMat(p2, p1, Kmat,
                method=cv2.FM_LMEDS, prob=0.999, threshold=0.1)
        #Ki = np.linalg.inv(Kmat)
        #Fmat = Ki.T.dot(Emat).dot(Ki)
        #print 'emat2', Emat

        #print 'EE', Emat, (Kmat.T).dot(Fmat).dot(Kmat)
        #cmat = np.float32([focal,0,pp[0], 0, focal, pp[1], 0,0,1]).reshape(3,3)
        n_in, R, t, msk, pts_h = cv2.recoverPose(Emat,
                np.float32(p2),
                np.float32(p1),
                cameraMatrix=Kmat,
                distanceThresh=200.0) # TODO : or something like 10.0/s ??

        # TODO : do the correct scale estimation

        # validate triangulation
        #pts_h = cv2.triangulatePoints(Kmat.dot(np.eye(3,4)),
        #        Kmat.dot(np.concatenate([R, t], axis=1)),
        #        p2[None,...], p1[None,...])
        #print np.max(pts_h - pts_h2)

        # points: homogeneous --> 3d coordinates
        # msk = np.logical_and(msk, np.all(np.isfinite(pts_h),axis=0)[:,None])
        # pts_h = pts_h[:, (msk[:,0] > 0)]
        #pts3 = pts_h[:3] / pts_h[3]
        #pts3 = s * np.stack([pts3[2], -pts3[0], -pts3[1]], axis=-1)

        pts3 = linear_LS_triangulation(
                Kmat.dot(np.eye(3,4)),
                Kmat.dot(np.concatenate([R, t], axis=1)),
                p2, p1)
        pts3 = s * np.stack([pts3[:,2], -pts3[:,0], -pts3[:,1]], axis=-1)

        # opt2 : custom
        #pts3 = triangulatePoints(
        #        Kmat.dot(np.eye(3,4)),
        #        Kmat.dot(np.concatenate([R, t], axis=1)),
        #        p2, p1)
        #pts3 = pts3[msk[:,0]>0]
        #pts3 = s * np.stack([pts3[:,2], -pts3[:,0], -pts3[:,1]], axis=-1)

        # TODO : will this take care of scale?
        # convert to base_link coordinates

        #pts3 = pts3[:16] # select 16 points to draw
        #msk[np.where(msk)[0][16:]] = 0

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
        cv2.drawKeypoints(mim, kp1[i1][mskp[:,0]>0], mim, color=(0,0,255))
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
        self.ax0_ = fig.add_subplot(3,2,1)
        self.ax2_ = fig.add_subplot(3,2,3, projection='3d')
        self.ax3_ = fig.add_subplot(3,2,5)
        self.ax1_ = fig.add_subplot(1,2,2)

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
        Q = np.diag([1e-4, 1e-4, 1e-4, 1e-1, 1e-1]) #xyhvw
        R = np.diag([1e-2, 1e-2, 1e-4]) # xyh
        x0 = np.zeros(5)
        P0 = np.diag([1e-6,1e-6,1e-6, 1e-1, 1e-1])

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
        tx, ty, th = self.tx_, self.ty_, self.th_
        ax0,ax1,ax2,ax3 = self.ax0_, self.ax1_, self.ax2_, self.ax3_

        # index
        stamp = stamps[i]
        img   = imgs[i]

        # TODO : there was a bug in data_collector that corrupted all time-stamp data!
        # disable stamps dt for datasets with corrupted timestamps.
        # very unfortunate.
        dt    = (stamps[i] - stamps[i-1])
        #dt = 0.2

        # experimental : pass in scale as a parameter
        # TODO : estimate scale from points + camera height?
        dps_gt = sub_p3d(odom[i], odom[i-1])
        s = np.linalg.norm(dps_gt[:2])
        #s = 0.03

        prv = ukf.x[:3].copy()

        ukf.predict(dt=dt)
        suc, res = vo(img, s=s)
        if not suc:
            print('Visual Odometry Aborted!')
            return

        if res is None:
            # skip filter updates
            return

        (aimg, dh, dt, pts, pts3) = res
        dps = np.float32([dt[0], dt[1], dh])
        print('(pred-gt) {} vs {}'.format(dps, dps_gt) )
        pos = add_p3d(prv, dps)
        ukf.update(pos)

        tx.append( float(ukf.x[0]) )
        ty.append( float(ukf.x[1]) )
        th.append( float(ukf.x[2]) )

        # pts2 in the proper coordinate system
        pts_c = pts.dot(Rmat(odom[i,2]).T) + np.reshape(odom[i,:2], (1,2))
        self.map_ = np.concatenate([self.map_, pts_c], axis=0)
        #scan_c = self.scan_to_pt(scan[i]).dot(Rmat(odom[i,2]).T) + np.reshape(odom[i, :2], (1,2))

        ### EVERYTHING FROM HERE IS PLOTTING + VIZ ###
        ax0.cla()
        ax1.cla()
        ax2.cla()

        ax0.imshow(aimg[...,::-1])
        ax0.axis('off')

        ax3.imshow(img[...,::-1])
        ax3.axis('off')

        # TODO : plot err in dps
        #ax2.plot(dpss)
        #ax2.axis('off')

        ax1.plot(odom[:i+1,0], odom[:i+1,1], 'k--')


        ph = np.rad2deg(np.arctan2(pts[:,1], pts[:,0]))
        print 'fov', np.min(ph), np.max(ph)

        ax1.plot(pts_c[:,0], pts_c[:,1], 'r.', label='visual')
        #ax1.plot(scan_c[:,0], scan_c[:,1], 'b.', label='scan')
        ax1.plot([0],[0],'k+')

        lx = np.linspace(0, 5)
        lx = np.stack([lx,lx*0],axis=-1)

        fov_l = lx.dot(Rmat(odom[i,2]-np.deg2rad(73/2.)).T) + np.reshape(odom[i,:2], (1,2))
        fov_r = lx.dot(Rmat(odom[i,2]+np.deg2rad(73/2.)).T) + np.reshape(odom[i,:2], (1,2))
        h = np.linspace(-np.pi, np.pi)
        radius = 5.0 * np.stack([np.cos(h),np.sin(h)], axis=-1) + np.reshape(odom[i,:2], (1,2))

        ax1.plot(fov_l[:,0], fov_l[:,1], 'g--')
        ax1.plot(fov_r[:,0], fov_r[:,1], 'g--')
        ax1.plot(radius[:,0], radius[:,1], 'g--')

        cx, cy = odom[i,:2]
        ax1.set_xlim(cx-5.0, cx+5.0)
        ax1.set_ylim(cy-5.0, cy+5.0)

        ax2.plot(pts3[:,0], pts3[:,1], pts3[:,2], '.')
        mx = np.abs(pts3).max(axis=0)
        ax2.set_xlabel('x')
        #ax2.set_xlim(0.0, 5.0)
        ax2.set_ylabel('y')
        #ax2.set_ylim(-5.0, 5.0)
        ax2.set_zlabel('z')
        #ax2.set_zlim(-1.0, 5.0)
        plt.legend()

        ax1.plot(tx, ty, 'b--')
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
    #idx = np.random.choice(8)
    idx = 17
    print('idx', idx)

    # load data
    imgs   = np.load('../data/train/{}/img.npy'.format(idx))
    stamps = np.load('../data/train/{}/stamp.npy'.format(idx))
    odom   = np.load('../data/train/{}/odom.npy'.format(idx))
    #scan   = np.load('../data/train/{}/scan.npy'.format(idx))
    scan = None

    # set odom @ t0= (0,0,0)
    R0 = Rmat(odom[0,2])
    odom -= odom[0]
    odom[:,:2] = odom[:,:2].dot(R0)

    stamps -= stamps[0] # t0 = 0

    app = CVORunner(imgs, stamps, odom, scan)
    app.run(auto=False)

if __name__ == "__main__":
    main()
