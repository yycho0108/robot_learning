import numpy as np
from tf import transformations as tx
from scipy.optimize import least_squares
from scipy.sparse import bsr_matrix, csr_matrix
from scipy.linalg import block_diag
import cv2
from matplotlib import pyplot as plt

from common import jac_h, to_h, generate_data

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
            ftol=1e-4,
            xtol=np.finfo(float).eps,
            max_nfev=8192,
            bounds=[
                guess - [0.2, 0.2, np.deg2rad(60.0)], # << enforced bounds to prevent jumps
                guess + [0.2, 0.2, np.deg2rad(60.0)]
                ],
            #loss='cauchy', # << ??
            loss='huber',
            method='trf',
            tr_solver='lsmr',
            verbose=1,
            f_scale=2.0
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

def main():
    # default K
    K = np.reshape([
        499.114583, 0.000000, 325.589216,
        0.000000, 498.996093, 238.001597,
        0.000000, 0.000000, 1.000000], (3,3))
    Ki = tx.inverse_matrix(K)

    # camera extrinsic parameters
    T_c2b = tx.compose_matrix(
                    angles=[-np.pi/2-np.deg2rad(10),0.0,-np.pi/2],
                    translate=[0.174,0,0.113])
    T_b2c = tx.inverse_matrix(T_c2b)

    pt3, pt2a, pt2b, pose = generate_data(n=128, K=K, T_c2b=T_c2b)
    guess = np.random.normal(pose, scale=(0.5,0.2,np.deg2rad(30.0)))

    pnp = solve_PNP(pt3, pt2b, K, T_b2c, T_c2b,
            guess)

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
            pt3[:,None], pt2b[:,None],
            K, np.zeros(5),
            useExtrinsicGuess = True,
            rvec=rvec_g,
            tvec=tvec_g,
            iterationsCount=10000,
            reprojectionError=2.0,
            confidence=0.99,
            )
    _, rvec_cv, tvec_cv, _ = pnp_cv

    print 'ground truth', pose
    print 'guess', guess
    print 'pnp', pnp
    print 'pnp-cv', parse_CVPNP(rvec_cv, tvec_cv, T_c2b, T_b2c)

if __name__ == "__main__":
    main()
