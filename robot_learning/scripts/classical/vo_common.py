import numpy as np
import cv2
from tf import transformations as tx
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt

def print_Rt(R, t):
    print '\tR', np.round(np.rad2deg(tx.euler_from_matrix(R)), 2)
    print '\tt', np.round(t.ravel() / np.linalg.norm(t), 2)

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
