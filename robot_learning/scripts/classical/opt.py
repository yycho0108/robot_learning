import numpy as np
from tf import transformations as tx
from scipy.optimize import least_squares
from scipy.linalg import block_diag
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

def dTdp(p):
    # T = (4x4)
    # p = (3)
    J = np.zeros((4,4,3), dtype=np.float32)
    x,y,h = p.T
    c,s = np.cos(h), np.sin(h)


    # 1) Jac w.r.t x
    J[0,3,0] = -c
    J[1,3,0] = s
    
    # 2) Jac w.r.t y
    J[0,3,1] = -s
    J[1,3,1] = -c

    # 3) Jac w.r.t h
    J[0,0,2] = -s
    J[0,1,2] = c
    J[1,0,2] = -c
    J[1,1,2] = -s
    J[0,3,2] = -y*c+x*s
    J[1,3,2] = x*c+y*s

    return J

def proj_PNP(p, pt3, K, T_b2c, T_c2b,
        as_h=False
        ):
    # parse p
    x,y,h = p
    c = np.cos(h)
    s = np.sin(h)

    # directly construct T_o2b from p
    T_o2b = np.zeros((4,4), dtype=np.float32)
    T_o2b[0,0] = c
    T_o2b[0,1] = s
    T_o2b[1,0] = -s # note : transposed
    T_o2b[1,1] = c
    T_o2b[2,2] = 1
    T_o2b[0,3] = -y*s-x*c
    T_o2b[1,3] = x*s-y*c
    T_o2b[3,3] = 1

    y = reduce(np.matmul,[
        K, # 3x3
        T_b2c[:3], # 3x4
        T_o2b, # 4x4
        T_c2b, # 4x4
        to_h(pt3)[...,None] # Nx4x1
        ])[...,0] # Nx3

    if as_h:
        return y
    else:
        return y[...,:-1] / y[...,-1:]

def err_PNP(p, pt3, pt2, K, T_b2c, T_c2b):
    y = proj_PNP(p,pt3,K,T_b2c,T_c2b)
    return (y - pt2).ravel()

def jac_PNP(p, pt3, pt2, K, T_b2c, T_c2b):
    # d(e) / d(p), e = P(l,p) - o
    n = len(pt3)

    y_h = proj_PNP(p, pt3, K, T_b2c, T_c2b, as_h=True)

    J = reduce(np.matmul, [
        jac_h(y_h), # Nx2x3
        K, #3x3
        T_b2c[:3], # 3x4
        np.einsum(
            'ijk,jl,...l->...ik',
            dTdp( p ), # 4x4x3
            T_c2b, #4x4
            to_h(pt3) # Nx4
            )
        ]) # --> [N,2,3]

    return J.reshape(-1, 3) # (Nx2, 3) = #obs x #param

def solve_PNP(
        pt3, pt2, # << observations
        K, T_b2c, T_c2b, # << camera intrinsic/extrinsic parameters
        guess): # << independent variable to optimize
    res = least_squares(
            err_PNP, guess,
            jac=jac_PNP,
            x_scale='jac',
            args=(pt3, pt2, K, T_b2c, T_c2b),
            ftol=1e-8,
            xtol=np.finfo(float).eps,
            max_nfev=8192,
            bounds=[
                guess - [0.2, 0.2, np.deg2rad(60.0)], # << enforced bounds to prevent jumps
                guess + [0.2, 0.2, np.deg2rad(60.0)]
                ],
            loss='huber',
            method='trf',
            tr_solver='lsmr',
            verbose=1,
            f_scale=1.0
            )
    return res.x

def parse_CVPNP(rvec, tvec, T_c2b, T_b2c):
    T_m2c = np.eye(4, dtype=np.float64)
    T_m2c[:3,:3] = cv2.Rodrigues(rvec)[0]
    T_m2c[:3,3:] = tvec.reshape(3,1)
    T_c2m = tx.inverse_matrix(T_m2c)

    T_b2o = np.linalg.multi_dot([
            T_c2b,
            T_c2m,
            T_b2c
            ])
    t_b = tx.translation_from_matrix(T_b2o)
    r_b = tx.euler_from_matrix(T_b2o)

    return np.asarray([t_b[0], t_b[1], r_b[-1]])


def proj_TRI(
        params,
        K, T_b2c, T_c2b
        ):
    # parse params
    dp = params[:3] # -> [3], (dx,dy,dh)
    l  = params[3:].reshape(-1,3) # -> [N,3], (x,y,z)

    n = len(l)

    dx, dy, dh = dp.T
    c, s = np.cos(dh), np.sin(dh)

    l_h = to_h(l)

    # construct T from b2 coord -> b1 coord
    Tbb = np.zeros((4,4), dtype=np.float32)

    # rot
    Tbb[0,0] = c
    Tbb[0,1] = -s
    Tbb[1,0] = s
    Tbb[1,1] = c
    Tbb[2,2] = 1

    # trans
    Tbb[0,3] = dx
    Tbb[1,3] = dy

    # homogeneous
    Tbb[3,3] = 1

    # a-projection (homogeneous)
    y_h_a = np.einsum(
            'ij,jk,...k->...i',
            K, #3x3
            np.eye(3,4), #3x4
            to_h(l) # Nx4(x1)
            ) #-> Nx3

    # b-projection (homogeneous)
    y_h_b = np.einsum(
            'ij,jk,kl,lm,...m->...i',
            K, # 3x3
            T_b2c[:3], # 3x4
            Tbb, # 4x4
            T_c2b, # 4x4
            l_h # Nx4
            ) # -> Nx3

    return y_h_a, y_h_b, Tbb

def err_TRI(
        params,
        pt_a, pt_b,
        K, T_b2c, T_c2b
        ):
    # parse params
    dp = params[:3] # -> [3], (dx,dy,dh)
    l  = params[3:].reshape(-1,3) # -> [N,3], (x,y,z)
    n = len(l)

    y_h_a, y_h_b, _ = proj_TRI(params, K, T_b2c, T_c2b)

    # convert to non-homogeneous repr.
    y_a = y_h_a[...,:-1] / y_h_a[...,-1:]
    y_b = y_h_b[...,:-1] / y_h_b[...,-1:]

    e_a = y_a - pt_a
    e_b = y_b - pt_b

    e = np.concatenate([e_a.ravel(), e_b.ravel()], axis=0)
    return e

def jac_TRI(
        params,
        pt_a, pt_b,
        K, T_b2c, T_c2b
        ):
    # parse params
    dp = params[:3] # -> [3], (dx,dy,dh)
    l  = params[3:].reshape(-1,3) # -> [N,3], (x,y,z)

    n = len(l) # == len(pt_a)

    dx, dy, dh = dp.T
    c, s = np.cos(dh), np.sin(dh)

    l_h = to_h(l)

    # construct T from b2 coord -> b1 coord
    Tbb = np.zeros((4,4), dtype=np.float32)

    # rot
    Tbb[0,0] = c
    Tbb[0,1] = -s
    Tbb[1,0] = s
    Tbb[1,1] = c
    Tbb[2,2] = 1

    # trans
    Tbb[0,3] = dx
    Tbb[1,3] = dy

    # homogeneous
    Tbb[3,3] = 1

    # a-projection (homogeneous)
    y_h_a = np.einsum(
            'ij,jk,...k->...i',
            K, #3x3
            np.eye(3,4), #3x4
            to_h(l) # Nx4(x1)
            ) #-> Nx3

    # b-projection (homogeneous)
    y_h_b = np.einsum(
            'ij,jk,kl,lm,...m->...i',
            K, # 3x3
            T_b2c[:3], # 3x4
            Tbb, # 4x4
            T_c2b, # 4x4
            l_h # Nx4
            ) # -> Nx3

    y_h_a, y_h_b, Tbb = proj_TRI(params, K, T_b2c, T_c2b)

    J_h_a = jac_h(y_h_a)
    J_h_b = jac_h(y_h_b)

    J_ea_dp = np.zeros((n,2,3), dtype=np.float32) # -> Nx2x3 (dp doesn't influence ea)
    J_ea_dp = J_ea_dp.reshape(-1,3)

    J_eb_dp = np.einsum('...ij,jk,kl,lmn,mo,...o->...in',
            J_h_a, # Nx2x3
            K, # 3x3
            T_b2c[:3], # 3x4
            dTdp(dp), # 4x4x3
            T_c2b, # 4x4
            l_h, # Nx4
            optimize=True
            ) # -> Nx2x3
    J_eb_dp = J_eb_dp.reshape(-1,3)

    J_ea_dl = np.einsum('...ij,jk->...ik',
            J_h_b, # Nx2x3
            K, #3x3
            ) # -> Nx2x3 ( should be Nx2 x Nx3 )
    J_ea_dl = block_diag(*J_ea_dl)
    #J_ea_dl = bsr_matrix(
    #        (J_ea_dl, bix, bip),
    #        shape=(n*2,n*3),
    #        blocksize=(2,3)
    #        )
    J_eb_dl = np.einsum('...ij,jk,kl,lm,mn,no->...io',
            J_h_b, #Nx2x3
            K, #3x3
            T_b2c[:3], #3x4
            Tbb, #4x4
            T_c2b, #4x4
            np.eye(4,3), #4x3, d(x_h)/d(x)
            optimize=True
            ) # -> Nx2x3 ( should be Nx2 x Nx3 )
    J_eb_dl = block_diag(*J_eb_dl)
    #J_eb_dl = bsr_matrix(
    #        (J_eb_dl, bix, bip),
    #        shape=(n*2,n*3),
    #        blocksize=(2,3)
    #        )

    # objective : (Nx4)x3
    # (# target : (2xNx2) = ea(Nx2) + eb(Nx2) )
    # (# params : (1+N)x3 = p(1x3) + l(Nx3) )
    # ( -> objective : (Nx4 x 3+Nx3)
    J_dp = np.concatenate([J_ea_dp, J_eb_dp], axis=0) # 2x(Nx2) x 2x3
    J_dl = np.concatenate([J_ea_dl, J_eb_dl], axis=0) # 2x(Nx2) x Nx3
    J    = np.concatenate([J_dp, J_dl], axis=1) # 2x(Nx2) x (2+N)x3

    return J

def solve_TRI(
        pt_a, pt_b,
        K, T_b2c, T_c2b,
        guess):
    dp0 = guess.copy()

    T_b2o = tx.compose_matrix(
            translate=(guess[0],guess[1],0),
            angles=(0,0,guess[-1])
            )
    T_o2b = tx.inverse_matrix(T_b2o)
    Tcc = np.linalg.multi_dot([
            T_b2c,
            T_o2b,
            T_c2b
            ])

    # ensure unit translation
    Tcc[:3,3:] /= np.linalg.norm(Tcc[:3,3:])

    l0 = cv2.triangulatePoints(
            K.dot(np.eye(3,4)), K.dot(Tcc[:3]),
            pt_a[None,...],
            pt_b[None,...]).astype(np.float32)
    l0[:3] /= l0[3:]
    l0 = l0[:3].T
    print('tri-pt3 (orig, u)', tx.unit_vector(l0[0].ravel()))

    x0 = np.concatenate([dp0.ravel(), l0.ravel()], axis=0)

    bx_lo = np.concatenate([
            guess - [0.3, 0.3, np.deg2rad(60.0)],
            np.full((len(l0) *3), -np.inf)
            ])
    bx_hi = np.concatenate([
            guess + [0.3, 0.3, np.deg2rad(60.0)],
            np.full((len(l0) *3), np.inf)
            ])

    res = least_squares(
            err_TRI, x0,
            jac=jac_TRI,
            x_scale='jac',
            args=(pt_a, pt_b, K, T_b2c, T_c2b),
            ftol=1e-15,
            xtol=np.finfo(float).eps,
            max_nfev=128,
            # TODO : enforce bounds?
            bounds = [bx_lo, bx_hi],
            #bounds=[
            #    guess - [0.2, 0.2, np.deg2rad(60.0)], # << enforced bounds to prevent jumps
            #    guess + [0.2, 0.2, np.deg2rad(60.0)]
            #    ],
            loss='huber',
            method='trf',
            tr_solver='lsmr',
            verbose=2,
            f_scale=0.01
            )

    x1 = res.x

    dp1 = x1[:3]

    print('dp1 : {}'.format(dp1.ravel()))
    print('\tdp1(u) : {}'.format(tx.unit_vector(dp1[:2]), dp1[-1]))

    l1  = x1[3:].reshape(-1, 3)
    #print('tri-pt3 : {}'.format(l1[0].ravel() ))
    print('\ttri-pt3 (u) : {}'.format(tx.unit_vector(l1[0].ravel()[:2]), l1[-1]))
    return res.x


def main():
    from tests.test_fmat import generate_valid_points
    #np.random.seed(0)

    # define testing parameters
    n = 100
    s_noise = [0.2, 0.2, 0.05]

    # camera intrinsic/extrinsic parameters
    Ks = (1.0 / 1.0)
    K = np.reshape([
        499.114583 * Ks, 0.000000, 325.589216 * Ks,
        0.000000, 498.996093 * Ks, 238.001597 * Ks,
        0.000000, 0.000000, 1.000000], (3,3))
    T_c2b = tx.compose_matrix(
                    angles=[-np.pi/2-np.deg2rad(10),0.0,-np.pi/2],
                    translate=[0.174,0,0.113])
    T_b2c = tx.inverse_matrix(T_c2b)
    # generate pose
    p = 0.5 * np.random.uniform(-np.pi, np.pi, size=3)
    x,y,h = p

    print '\tground truth : {}'.format(p)
    print '\tground truth(u) : {}'.format(tx.unit_vector(p[:2]), p[-1])
    guess = np.random.normal(p, scale=s_noise)
    print '\tguess : {}'.format(guess)

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

    # generate landmark points + pose
    # ensure points are valid

    pt3 = np.random.uniform(-5.0, 5.0, size=(1024, 3))

    #pt3 = generate_valid_points(
    #    n=n,
    #    d_min=0.1, d_max=100.0,
    #    h=480.0,
    #    w=640.0,
    #    fov_v=0.895,
    #    fov_h=1.139,
    #    T_alt=Tcc
    #    ) # << NOTE : valid in camera coord

    rvec = cv2.Rodrigues(Rcc)[0]
    tvec = tcc.ravel()

    pt2 = cv2.projectPoints(
            pt3, rvec, tvec,
            cameraMatrix=K,
            distCoeffs = np.zeros(5)
            )[0][:,0]
    
    msk = np.logical_and.reduce([
        0 <= pt2[:,0],
        pt2[:,0] < 640,
        0 <= pt2[:,1],
        pt2[:,1] < 480,
        ])
    pt3 = pt3[msk]
    pt2 = pt2[msk]
    print 'input points validity', msk.sum(), msk.size

    print 'gt-pt3', pt3[0]
    print 'gt-pt3 (u)', tx.unit_vector(pt3[0])

    pnp = solve_PNP(pt3, pt2, K, T_b2c, T_c2b,
            guess)

    pt2_a = cv2.projectPoints(
            pt3, np.zeros(3), np.zeros(3),
            cameraMatrix=K, distCoeffs=np.zeros(5)
            )[0][:,0] # I projection
    pt2_b = pt2 # Tcc projection

    x1 = solve_TRI(pt2_a, pt2_b, K, T_b2c, T_c2b,
            guess
            #p
            )

    # comparison

    # construct extrinsic guess
    T_b2o_g = tx.compose_matrix(
            translate= (guess[0],guess[1],0),
            angles = (0,0,guess[2])
            )
    T_o2b_g = tx.inverse_matrix(T_b2o_g)
    Tcc_g = np.linalg.multi_dot([
        T_b2c,
        T_o2b_g,
        T_c2b
        ])
    rvec_g = cv2.Rodrigues(Tcc_g[:3,:3])[0]
    tvec_g = Tcc_g[:3,3].ravel()

    pnp_cv = cv2.solvePnPRansac(
            pt3[:,None], pt2[:,None],
            K, np.zeros(5),
            useExtrinsicGuess = True,
            rvec=rvec_g,
            tvec=tvec_g,
            iterationsCount=10000,
            reprojectionError=2.0,
            confidence=0.99,
            )
    _, rvec_cv, tvec_cv, _ = pnp_cv

    print 'pnp', pnp
    print 'pnp-cv', parse_CVPNP(rvec_cv, tvec_cv, T_c2b, T_b2c)

if __name__ == "__main__":
    main()
