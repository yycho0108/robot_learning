import numpy as np
from tf import transformations as tx
import cv2

def to_h(x):
    return np.pad(x, [(0,0),(0,1)],
            mode='constant',
            constant_values=1.0
            )

def jac_h(x_h):
    # x = NxD
    # x_h = Nx(D+1) = [x0,x1,... s]
    # jac_h = d(x) / d(x_h) --> NxDxD+1
    # T = Nx4x4

    x_h = np.asarray(x_h)
    n, d_h = x_h.shape
    d = d_h - 1

    s = x_h[..., -1]
    s_i = (1.0 / s)
    s_i2 = np.square(s_i)

    J = np.zeros((n,d,d_h), dtype=np.float32)
    J[..., np.arange(d), np.arange(d)] = s_i[..., None] # x' = x/s, d(x')/dx = 1/s
    J[..., np.arange(d), -1] = - x_h[..., :d] * s_i2[..., None] # NxD

    return J

def generate_data(
        n=100,
        max_it=128,
        K=None,
        T_c2b=None,
        s_pos=None,
        s_pt3=None,
        seed=None,
        ):
    """
    Generate Multiview Data. (Currently 2 Views)
    """
    if seed is not None:
        np.random.seed(seed)

    if s_pos is None:
        s_pos = [0.5, 0.2, np.deg2rad(30.0)]
    if s_pt3 is None:
        s_pt3 = 5.0

    # camera intrinsic parameters
    if K is None:
        # default K
        K = np.reshape([
            499.114583, 0.000000, 325.589216,
            0.000000, 498.996093, 238.001597,
            0.000000, 0.000000, 1.000000], (3,3))
    Ki = tx.inverse_matrix(K)

    # camera extrinsic parameters
    if T_c2b is None:
        T_c2b = tx.compose_matrix(
                        angles=[-np.pi/2-np.deg2rad(10),0.0,-np.pi/2],
                        translate=[0.174,0,0.113])
    T_b2c = tx.inverse_matrix(T_c2b)

    # generate pose
    pose = np.random.normal(scale=s_pos, size=3)
    x,y,h = pose

    # convert base_link pose to camera pose
    R = tx.euler_matrix(0, 0, h)[:3,:3]
    T_b2o = tx.compose_matrix(
            translate= (x,y,0),
            angles = (0,0,h)
            )
    T_o2b = tx.inverse_matrix(T_b2o)

    Tcc = np.linalg.multi_dot([
        T_b2c,
        T_o2b,
        T_c2b
        ])
    Rcc = Tcc[:3,:3]
    tcc = Tcc[:3,3:]

    # convert camera pose to OpenCV format (Rodrigues parametrized)
    rvec = cv2.Rodrigues(Rcc)[0]
    tvec = tcc.ravel()

    # generate landmark points
    # ensure points are valid
    res = {
            'pt3' : [],
            'pt2a': [],
            'pt2b': []
            }
    cnt = 0 # keep track of how many points were added to the dataset

    for i in range(max_it):
        # generate more points than requested, considering masks
        pt3 = np.random.normal(scale=s_pt3, size=(n, 3))
        pt3[:,2] = np.abs(pt3[:,2]) # forgot! positive depth required.

        # view 1 (identity)
        pt2a = cv2.projectPoints(
                pt3, np.zeros(3), np.zeros(3),
                cameraMatrix=K,
                distCoeffs=np.zeros(5)
                )[0][:,0]

        # view 2 (w/h offset)
        pt2b = cv2.projectPoints(
                pt3, rvec, tvec,
                cameraMatrix=K,
                distCoeffs = np.zeros(5)
                )[0][:,0]

        # filter with mask
        msk = np.logical_and.reduce([
            0 <= pt2a[:,0],
            pt2a[:,0] < 640,
            0 <= pt2a[:,1],
            pt2a[:,1] < 480,
            0 <= pt2b[:,0],
            pt2b[:,0] < 640,
            0 <= pt2b[:,1],
            pt2b[:,1] < 480,
            ])

        # add data
        res['pt3'].append(pt3[msk])
        res['pt2a'].append(pt2a[msk])
        res['pt2b'].append(pt2b[msk])

        cnt += msk.sum()

        if cnt >= n:
            break

    pt3 = np.concatenate(res['pt3'], axis=0)[:n]
    pt2a = np.concatenate(res['pt2a'], axis=0)[:n]
    pt2b = np.concatenate(res['pt2b'], axis=0)[:n]

    return pt3, pt2a, pt2b, pose
