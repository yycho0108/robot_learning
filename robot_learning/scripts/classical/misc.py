"""
Currently acting as archive.
"""

import numpy as np
def Rmat(x):
    c,s = np.cos(x), np.sin(x)
    R = np.float32([c,-s,s,c]).reshape(2,2)
    return R

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

def normalize_points(pts, cMat):
    pc = np.reshape([cMat[2,0], cMat[2,1]], [1,2])
    pf = np.reshape([cMat[0,0], cMat[1,1]], [1,2])
    return (pts - pc) / pf

