import numpy as np
import cv2
from tf import transformations as tx
from matplotlib import pyplot as plt

def print_Rt(R, t):
    print '\tR', np.round(np.rad2deg(tx.euler_from_matrix(R)), 2)
    print '\tt', np.round(t.ravel() / np.linalg.norm(t), 2)

def stat(x):
    msg = '(min,mean,max,std), ({},{},{},{})'.format(
            np.min(x), np.mean(x), np.max(x), np.std(x))
    return msg

def generate_valid_points(
        n = 256,
        d_min=0.1, d_max=100.0,
        h = 240.0,
        w = 320.0,
        fov_v=0.895,
        fov_h=1.139,
        T_alt = None
        ):

    # margin
    fov_v = fov_v * 0.95
    fov_h = fov_h * 0.95

    res = np.empty(
            shape=(0,3),
            dtype=np.float32)

    while len(res) < n:
        # generate ...
        phi   = np.random.uniform(-fov_v/2, fov_v/2, n) # governs y coord
        theta = np.random.uniform(-fov_h/2, fov_h/2, n) # governs x coord
        d     = np.random.uniform(d_min, d_max, n)

        ct, st = np.cos(theta), np.sin(theta)
        cp, sp = np.cos(phi), np.sin(phi)

        # 3d coord
        x = st
        y = ct * sp
        z = ct * cp
        p = d[:,None] * np.stack([x,y,z], axis=-1).astype(np.float32)

        # w.r.t. view 2 @ (txn, rxn)
        if T_alt is not None:
            # T_alt is a transform
            # that takes points in C0 to C1.

            p_alt = p.dot(T_alt[:3,:3].T) + T_alt[:3,3:].T
            xa, ya, za = p_alt.T
            pa = np.arctan2(ya,za)
            cta = np.linalg.norm([y,z], axis=0)
            ta = np.arctan2(xa,cta)

            msk = np.logical_and.reduce([
                -fov_v/2 <= pa,
                pa <= fov_v/2,
                -fov_h/2 <= ta,
                ta <= fov_h/2,
                d_min <= za,
                za <= d_max
                ])

            p = p[msk]

        n_add = min(len(p), n-len(res))

        #print('adding {}/{}'.format(n_add, n))
        #print 'zamax', za.max()

        res = np.concatenate([res, p[:n_add]], axis=0)

    #assert np.all(res[:,2] > 0), 'must have positive z'
    return res

    # fov_v = 2 * atan(h / (2 * fy))
    # fov_h = 2 * atan(w / (2 * fx))

def recover_pose(E,K,
        pt1, pt2,
        #z_min = 1e-3,
        #z_max = 100.0,
        z_min = np.finfo(np.float32).eps,
        z_max = np.inf
        ):
    print 'entering recover_pose'

    P1 = np.eye(3,4)
    P2 = np.eye(3,4)

    R1, R2, t = cv2.decomposeEssentialMat(E)

    perm = [
            (R1, t),
            (R2, t),
            (R1, -t),
            (R2, -t)]

    perm_s = [
            '(R1,t)',
            '(R2,t)',
            '(R1,-t)',
            '(R2,-t)']

    msks = [None for _ in range(4)]
    pt3s = [None for _ in range(4)]

    sel   = 0
    ctest = -np.inf

    for i, (R, t) in enumerate(perm):
        #print '==== recover_pose validation ===='
        P2[:3,:3] = R
        P2[:3,3:] = t.reshape(3,1)
        #P2 = np.concatenate([R, t.reshape(3,1)], axis=1)

        KP1 = K.dot(P1)
        KP2 = K.dot(P2)

        pth_a = cv2.triangulatePoints(
                KP1, KP2,
                pt1[None,...],
                pt2[None,...]).astype(np.float32)
        pth_a /= pth_a[3:]

        pt3_a = P1.dot(pth_a)
        #print 'homo', pth_a[:,0]
        #print 'rec(pt0)', pt3_a[:,0]
        #print 'hmm', (P1.dot(pth_a)).std()
        #print 'hmm', (KP1.dot(pth_a)).std()
        pt3_b = P2.dot(pth_a)

        pt3s[i] = pt3_a

        # VALIDATION : reprojection unto image
        #pb_r, _ = cv2.projectPoints(
        #        pt3_a.T,
        #        cv2.Rodrigues(R)[0].ravel(), # or does it require inverse rvec/tvec?
        #        t.ravel(),
        #        cameraMatrix=K,
        #        distCoeffs=np.zeros(5)
        #        )
        #pb_r = np.squeeze(pb_r, axis=1)

        #msk_wtf = np.logical_and.reduce([
        #    0 <= pb_r[:,0],
        #    pb_r[:,0] < 320,
        #    0 <= pb_r[:,1],
        #    pb_r[:,1] < 240])
        #print 'wtf : {}/{}'.format(msk_wtf.sum(), msk_wtf.size)
        #print( stat(pb_r))

        #plt.plot(pt1[:,0], pt1[:,1], 'k.')
        #plt.plot(pt2[:,0], pt2[:,1], 'ko', 'gt', alpha=1.0)
        #plt.plot(pb_r[:,0], pb_r[:,1], 'r+', 'rec', alpha=0.5)
        #plt.xlim(0, 320)
        #plt.ylim(0, 240)
        #plt.legend()
        #plt.show()

        #pt3_a = pth_a[:3] / pth_a[3:] # w.r.t. first cam
        #pt3_b = pth_b[:3] / pth_b[3:] # w.r.t. second cam

        za, zb = pt3_a[2], pt3_b[2]

        #print 'rec-orig', (pb_r[0], pt2[0])
        #print 'error', np.sqrt(np.mean(np.square(pb_r - pt2)))
        #print_Rt(R, t)
        #c = (z > 0).sum()
        msk_i = np.logical_and.reduce([
            z_min < za,
            za < z_max,
            z_min < zb,
            zb < z_max
            ])
        msks[i] = msk_i
        c = msk_i.sum()
        #print '+z : {}/{}'.format(c, za.size )
        #print( stat(za) )
        if c > ctest:
            sel = i
            ctest = c

    R, t = perm[sel]
    msk = msks[sel]
    pt3 = pt3s[sel]
    n_in = msk.sum()

    return n_in, R, t, msk, pt3

def main():
    #np.random.seed(0)

    # parameters
    #method = cv2.FM_LMEDS
    method = cv2.FM_RANSAC
    
    H, W = 240.0, 320.0

    K = np.reshape([
        499.114583 / 2.0, 0.000000, 325.589216 / 2.0,
        0.000000, 498.996093 / 2.0, 238.001597 / 2.0,
        0.000000, 0.000000, 1.000000], (3,3))
    D = np.float32([0.158661, -0.249478, -0.000564, 0.000157, 0.000000])

    #K = np.eye(3,3)
    #D = np.zeros_like(D)

    p_min = [-5.0, -5.0, 0.0] # points distribution
    p_max = [5.0, 5.0, 5.0]   # points distribution

    t_max = 1.0
    r_max = np.deg2rad(10.0)#(np.pi / 4.0)
    n_pt  = 512 
    d_max = np.inf

    # camera relative pose definition
    t = np.random.uniform(-t_max, t_max, size=3)
    t /= np.linalg.norm(t) # convert to unit translation, for convenience

    #if(t[2]<0): t*=-1
    print('Real Translation', t)
    r = np.random.uniform(-r_max, r_max, size=3)
    R = tx.euler_matrix(*r)[:3,:3]
    tvec = t
    rvec = cv2.Rodrigues(R)[0]
    P1 = np.eye(3, 4)
    P2 = np.concatenate([R,t.reshape(3,1)], axis=1) # -> (3x4)

    # points distribution definision
    #pt = np.random.uniform(p_min, p_max, size=(n_pt, 3))
    #pt = pt.astype(np.float32)

    T_alt = np.eye(4,4)
    T_alt[:3,:3] = R
    T_alt[:3,3:] = t.reshape(3,1)
    #T_alt = np.linalg.inv(T_alt)

    pt = generate_valid_points(
        n = n_pt,
        d_min=0.1, d_max=10.0,
        h = 240.0, w = 320.0,
        fov_v=0.895, fov_h=1.139,
        T_alt = T_alt)

    print('pt0 = ', pt[0])

    print 'ground truth'
    R_gt = R
    t_gt = t
    print_Rt(R,t)

    # 2D projection
    pt1, _ = cv2.projectPoints(pt,
            np.zeros_like(rvec),
            np.zeros_like(tvec),
            cameraMatrix=K,
            distCoeffs=D)
    pt1 = np.round(pt1)
    pt1 = np.squeeze(pt1, axis=1)

    msk1 = np.logical_and.reduce([
        0 <= pt1[:,0],
        0 <= pt1[:,1],
        pt1[:,0] < W,
        pt1[:,1] < H])

    pt2, _ = cv2.projectPoints(pt,
            rvec,
            tvec,
            cameraMatrix=K,
            distCoeffs=D,
            )
    pt2 = np.round(pt2)
    pt2 = np.squeeze(pt2, axis=1)

    msk2 = np.logical_and.reduce([
        0 <= pt2[:,0],
        0 <= pt2[:,1],
        pt2[:,0] < W,
        pt2[:,1] < H])

    msk = (msk1 & msk2)

    print 'good : {}/{}'.format(msk.sum(), n_pt )

    # apply feasibility mask
    pt1 = pt1[msk]
    pt2 = pt2[msk]

    # undistort
    pt1 = cv2.undistortPoints(pt1[None,...],K,D,P=K)[0]
    pt2 = cv2.undistortPoints(pt2[None,...],K,D,P=K)[0]

    #plt.plot(pt1[:,0], pt1[:,1], 'r.')
    #plt.show()

    Fmat, msk = cv2.findFundamentalMat(pt1, pt2,
            method=method,
            param1=0.1, param2=0.999) # TODO : expose these thresholds
    msk = msk.astype(np.bool)
    #print('Fmat', Fmat)

    Emat, msk = cv2.findEssentialMat(pt1, pt2, K,
            method=method,
            prob=0.999, threshold=0.1)
    msk = msk.astype(np.bool)

    print('inl : {}/{}'.format(msk.sum(), msk.size))
    #print('Emat', Emat)

    Emat1 = np.linalg.multi_dot([K.T,Fmat,K])
    Emat2 = Emat

    print '---------------------'
    msk = msk[:,0]
    R, t = recover_pose(Emat2, K, pt1[msk], pt2[msk])
    print 'RP'
    print_Rt(R,t)
    print '---------------------'

    #n_in, R, t, msk, _ = cv2.recoverPose(Emat1,
    #        pt2,
    #        pt1,
    #        cameraMatrix=K,
    #        distanceThresh=d_max)

    #print 'FM'
    #print_Rt(R,t)

    n_in, R, t, msk, _ = cv2.recoverPose(Emat2, # pt2 w.r.t pt1
            pt2,
            pt1,
            cameraMatrix=K,
            distanceThresh=d_max)
    print('recoverPose Inliers {}/{}'.format(n_in, msk.size))

    print 'EM'
    print_Rt(R,t)

    print(t_gt.dot(t))

    # visualization


if __name__ == "__main__":
    main()
