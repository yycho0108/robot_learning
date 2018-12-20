import numpy as np
import cv2
from tf import transformations as tx

def print_Rt(R, t):
    print '\tR', np.round(np.rad2deg(tx.euler_from_matrix(R)), 2)
    print '\tt', np.round(t.ravel() / np.linalg.norm(t), 2)

def recover_pose(E,K,
        pt1, pt2,
        z_min = np.finfo(np.float32).eps,
        z_max = np.inf,
        log=False
        ):

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
    pt3 = pt3s[sel]
    n_in = msk.sum()

    return n_in, R, t, msk, pt3


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


