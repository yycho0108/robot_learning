import numpy as np
from tf import transformations as tx
from scipy.optimize import least_squares
from scipy.sparse import bsr_matrix, csr_matrix
from scipy.linalg import block_diag
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from common import jac_h, to_h, generate_data

def proj_TRI(
        params,
        pt_a,
        K, Ki,
        T_b2c, T_c2b
        ):
    # parse params
    dm, dh = params[:2] # direction of motion + 2d rotation
    dd = params[2:] # landmark depth @ coord a

    # interpret params
    dx, dy = np.cos(dm), np.sin(dm)
    l  = np.einsum('ij,...j,...->...i', Ki, to_h(pt_a), dd)
    # --> l in coord A

    T_b2o = tx.compose_matrix(
            angles=(0,0,dh),
            translate=(dx,dy,0)
            )
    T_o2b = tx.inverse_matrix(T_b2o)

    l_h = to_h(l)

    # project to image
    y_h_b = np.einsum(
            'ij,jk,kl,lm,...m->...i',
            K, # 3x3
            T_b2c[:3], # 3x4
            T_o2b, # 4x4
            T_c2b, # 4x4
            l_h # Nx4
            ) # -> Nx3

    return y_h_b

def err_TRI(
        params,
        pt_a, pt_b,
        K, Ki, T_b2c, T_c2b
        ):
    y_h_b = proj_TRI(params, pt_a, K, Ki, T_b2c, T_c2b)
    y_b = y_h_b[...,:-1] / y_h_b[...,-1:] # Nx2
    e = y_b - pt_b
    return e.ravel()

def dTdmh(m, h):
    cm, sm = np.cos(m), np.sin(m)
    ch, sh = np.cos(h), np.sin(h)

    J = np.zeros((4,4,2), dtype=np.float32)

    # w.r.t m
    J[0,3,0] = -cm*sh + sm*ch
    J[1,3,0] = -sm*sh - cm*ch

    # w.r.t h
    J[0,0,1] = -sh
    J[0,1,1] = ch
    J[1,0,1] = -ch
    J[1,1,1] = -sh
    J[0,3,1] = -sm*ch+cm*sh
    J[1,3,1] = cm*ch+sm*sh

    return J

def jac_TRI(
        params,
        pt_a, pt_b,
        K, Ki, T_b2c, T_c2b
        ):
    # parse params
    dm, dh = params[:2] # direction of motion + 2d rotation
    dd = params[2:] # landmark depth @ coord a
    n = len(dd)

    # interpret params
    dp = np.asarray([np.cos(dm), np.sin(dm), dh])
    l  = np.einsum('ij,aj,a->ai', Ki, to_h(pt_a), dd)
    dx, dy, dh = dp
    s, c = np.sin(dh), np.cos(dh)

    T_b2o = tx.compose_matrix(
            angles=(0,0,dh),
            translate=(dx,dy,0)
            )
    T_o2b = tx.inverse_matrix(T_b2o)
    # == [c,s,0,-y*s-x*c
    #     -s,c,0,x*s-y*c
    #     0  0 1 ...

    l_h = to_h(l)

    y_h_b = proj_TRI(params, pt_a, K, Ki, T_b2c, T_c2b)

    J_h = jac_h(y_h_b)

    # objective : Nx2x2
    J_p = np.einsum('...ij,jk,kl,lmn,mo,...o->...in',
            J_h, # Nx2x3
            K, # 3x3
            T_b2c[:3], #3x4
            dTdmh(dm,dh), # 4x4x2
            # 4x4x2
            T_c2b, # 4x4
            l_h, # Nx4
            optimize=True
            )
    J_p = J_p.reshape(-1, 2)

    # somewhat confusing - construct directly
    J_l_rhs = np.zeros((n,3,n), dtype=np.float32)
    J_l_rhs[np.arange(n), :, np.arange(n)] = \
            np.einsum('ij,...j->...i', Ki, to_h(pt_a))

    # objective : Nx2xN
    J_l = np.einsum('...ij,jk,kl,lm,mn,no,...op->...ip',
            J_h, # Nx2x3
            K, # 3x3
            T_b2c[:3], # 3x4
            T_o2b, # 4x4
            T_c2b, # 4x4
            np.eye(4,3), # 4x3 d(x_h) / d(x)
            J_l_rhs, # Nx3xN
            optimize=True
            )
    J_l = J_l.reshape(-1, n)

    J = np.concatenate([J_p, J_l], axis=-1) # << pretty sparse
    #return J
    return csr_matrix(J)

def parse_guess(guess, T_b2c, T_c2b):
    Rc, tc = guess
    Tcc = np.eye(4)
    Tcc[:3,:3] = Rc
    Tcc[:3,3:] = tc.reshape(3,1)

    Tbb = np.linalg.multi_dot([
        T_c2b,
        Tcc,
        T_b2c
        ])

    dt = tx.translation_from_matrix(Tbb)
    dh = tx.euler_from_matrix(Tbb)[-1]

    return np.asarray([dt[0], dt[1], dh])

def form_guess(xyh, T_b2c, T_c2b):
    Tbb = tx.compose_matrix(
            translate=[xyh[0],xyh[1],0],
            angles=[0,0,xyh[2]]
            )
    Tcc = np.linalg.multi_dot([
        T_b2c,
        Tbb,
        T_c2b
        ])
    Rc = Tcc[:3,:3]
    tc = Tcc[:3,3:]
    tc /= np.linalg.norm(tc)
    return (Rc, tc)

def solve_TRI(
        pt_a, pt_b,
        K, Ki, T_b2c, T_c2b,
        guess,
        verbose=2,
        thresh=1.0
        ):

    if isinstance(guess, tuple):
        # input (R,t)
        dp0 = parse_guess(guess, T_b2c, T_c2b)
    else:
        # input (dx,dy,dh)
        dp0 = guess.copy()

    T_b2o = tx.compose_matrix(
            translate=(dp0[0],dp0[1],0),
            angles=(0,0,dp0[-1])
            )
    T_o2b = tx.inverse_matrix(T_b2o)

    Tcc = np.linalg.multi_dot([
            T_b2c,
            T_o2b,
            T_c2b
            ])

    # ensure unit translation
    # Tcc[:3,3:] /= np.linalg.norm(Tcc[:3,3:])

    # form initial guess
    l0 = cv2.triangulatePoints(
            K.dot(np.eye(3,4)),
            K.dot(Tcc[:3]),
            pt_a[None,...],
            pt_b[None,...]).astype(np.float32)
    dl0 = l0[2].T / l0[3].T

    m0 = np.arctan2(dp0[1], dp0[0])
    dh0 = dp0[2]
    dl0[dl0 < 0] = 0.0

    x0 = np.concatenate([ [m0, dh0], dl0])

    bx_lo = np.concatenate([
            [m0 - np.deg2rad(60.0)],
            [dh0 - np.deg2rad(60.0)],
            np.full_like(dl0, -np.inf)
            ])

    bx_hi = np.concatenate([
            [m0 + np.deg2rad(60.0)],
            [dh0 + np.deg2rad(60.0)],
            np.full_like(dl0, np.inf)
            ])

    # un-parametrize
    # tmp0 = np.asarray([np.cos(x0[0]), np.sin(x0[0]), x0[1]])
    # tmp1 = np.einsum('ij,...j,...->...i',
    #         Ki, to_h(pt_a), x0[2:]) #3x3 x Nx3 x N
    # print('\tdp0 : {}'.format(tmp0.ravel()))
    #print('tri-pt3 (u) : {}'.format(tx.unit_vector(tmp1[0].ravel())))

    res = least_squares(
            err_TRI, x0,
            jac=jac_TRI,
            x_scale='jac',
            args=(pt_a, pt_b, K, Ki, T_b2c, T_c2b),
            ftol=1e-6,
            xtol=1e-6,
            gtol=1e-13,
            max_nfev=1024,
            #bounds=[bx_lo, bx_hi],
            loss='huber',
            method='trf',
            tr_solver='lsmr',
            verbose=verbose,
            f_scale=2.0
            )

    x1 = res.x
    
    # un-parametrize
    dp1 = np.asarray([np.cos(x1[0]), np.sin(x1[0]), x1[1]])
    l1  = np.einsum('ij,...j,...->...i',
            Ki, to_h(pt_a), x1[2:]) #3x3 x Nx3 x N

    #inl = err_TRI(x1,...)

    #print('tri-pt3 (u) : {}'.format(tx.unit_vector(l1[0].ravel())))
    return form_guess(dp1, T_b2c, T_c2b), l1

def solve_TRI_fast(
        pt_a, pt_b,
        K, Ki, T_b2c, T_c2b, guess,
        n_it = 8, # max # of iterations
        n_sub = 32 # point subset to run optimization on
        ):
    best_err = np.inf
    best_guess = guess
    best_pt3 = None

    # == evaluate input == 
    # evaluate on whole set
    dp0 = parse_guess(guess, T_b2c, T_c2b)
    m0 = np.arctan2(dp0[1], dp0[0])
    dh0 = dp0[2]

    # form matrices
    T_b2o = tx.compose_matrix(
    translate=(dp0[0],dp0[1],0),
    angles=(0,0,dp0[-1])
    )
    T_o2b = tx.inverse_matrix(T_b2o)
    Tcc = np.linalg.multi_dot([
            T_b2c,
            T_o2b,
            T_c2b
            ]) 

    # triangulate all points
    l0 = cv2.triangulatePoints(
        K.dot(np.eye(3,4)), K.dot(Tcc[:3]),
        pt_a[None,...],
        pt_b[None,...]).astype(np.float32)
    dl0 = l0[2,:] / l0[3,:]
    x0 = np.concatenate([[m0, dh0], dl0])

    e = err_TRI(x0, pt_a, pt_b, K, Ki, T_b2c, T_c2b)
    e2 = e.reshape(-1,2)
    e = np.linalg.norm(e2, axis=-1)
    e = np.clip(e, 0.0, 64.0) # clip error
    err = e.mean()

    print('err : {:.2f}/{:.2f}={:.1f}%'.format(err, best_err, 100*err/best_err))

    if err < best_err:
        best_err = err
        best_guess = guess
        best_pt3 = l0[:3].T / l0[3:].T

    for i in range(n_it):
        # choose random subset
        idx = np.random.choice(
                len(pt_a), n_sub,
                replace=( len(pt_a) < n_sub )
                )

        # run least squares
        guess, l1 = solve_TRI(
                pt_a[idx], pt_b[idx],
                K, Ki, T_b2c, T_c2b, best_guess,
                verbose=0
                )

        # evaluate on whole set
        dp0 = parse_guess(guess, T_b2c, T_c2b)
        m0 = np.arctan2(dp0[1], dp0[0])
        dh0 = dp0[2]

        # form matrices
        T_b2o = tx.compose_matrix(
        translate=(dp0[0],dp0[1],0),
        angles=(0,0,dp0[-1])
        )
        T_o2b = tx.inverse_matrix(T_b2o)
        Tcc = np.linalg.multi_dot([
                T_b2c,
                T_o2b,
                T_c2b
                ]) 

        # triangulate all points
        l0 = cv2.triangulatePoints(
            K.dot(np.eye(3,4)), K.dot(Tcc[:3]),
            pt_a[None,...],
            pt_b[None,...]).astype(np.float32)
        dl0 = l0[2,:] / l0[3,:]
        x0 = np.concatenate([[m0, dh0], dl0])

        e = err_TRI(x0, pt_a, pt_b, K, Ki, T_b2c, T_c2b)
        e2 = e.reshape(-1,2)
        e = np.linalg.norm(e2, axis=-1)
        e = np.clip(e, 0.0, 64.0) # clip error
        err = e.mean()

        print('err : {:.2f}/{:.2f}={:.1f}%'.format(err, best_err, 100*err/best_err))

        if err < best_err:
            best_err = err
            best_guess = guess
            best_pt3 = l0[:3].T / l0[3:].T

    #return solve_TRI(
    #            pt_a, pt_b,
    #            K, Ki, T_b2c, T_c2b, best_guess,
    #            verbose=2
    #            )
    return best_guess, best_pt3

def Rctc2Rbtb(Rc, tc, T_c2b, T_b2c):
    Tcc = np.eye(4)
    Tcc[:3,:3] = Rc
    Tcc[:3,3:] = tc.reshape(3,1)
    Tbb = np.linalg.multi_dot([
        T_c2b,
        Tcc,
        T_b2c
        ])
    R = Tbb[:3,:3]
    t = Tbb[:3, 3]

    return (R, t)

def main():
    # default K
    K = np.reshape([
        499.114583, 0.000000, 325.589216,
        0.000000, 498.996093, 238.001597,
        0.000000, 0.000000, 1.000000], (3,3))#.astype(np.float32)
    Ki = tx.inverse_matrix(K)

    # camera extrinsic parameters
    T_c2b = tx.compose_matrix(
                    angles=[-np.pi/2-np.deg2rad(10),0.0,-np.pi/2],
                    translate=[0.174,0,0.113])
    T_b2c = tx.inverse_matrix(T_c2b)

    # generate data
    pt3, pt2_a, pt2_b, pose = generate_data(n=128, K=K, T_c2b=T_c2b,
            seed = 6
            )
    sc = np.linalg.norm(pose[:2])
    print('ground truth', pose)
    guess = np.random.normal(pose, scale=[0.1,0.1,np.deg2rad(10.0)])
    print('guess', guess)

    # cv2 validation
    E, msk_e = cv2.findEssentialMat(
            pt2_b, pt2_a,
            cameraMatrix=K,
            method=cv2.FM_RANSAC,
            prob=0.999,
            threshold=1.0)
    msk_e = msk_e[:,0].astype(np.bool)

    rv, Rc_e, tc_e, msk_r = cv2.recoverPose(E, pt2_b[msk_e], pt2_a[msk_e], K)
    msk_r = msk_r[:,0].astype(np.bool)
    msk_e[~msk_r] = False
    msk_r = msk_e

    Rb_e, tb_e = Rctc2Rbtb(Rc_e, tc_e, T_c2b, T_b2c)

    KPa = K.dot( np.eye(3,4) )
    KPb = K.dot( np.concatenate([Rc_e.T, -Rc_e.T.dot(tc_e.reshape(3,1))], axis=1) )

    pt3_e = cv2.triangulatePoints(
            KPa, KPb, # << for whatever reason requires swap
            pt2_a[None,...],
            pt2_b[None,...])
    pt3_e = pt3_e.T[msk_r]
    pt3_e /= pt3_e[:,3:]
    pt3_e = pt3_e[:,:3]
    guess_e = (Rc_e, tc_e) # Rc/tc

    print('E : {} {}'.format(sc * tb_e[:2], tx.euler_from_matrix(Rb_e)[-1]))

    # TRI validation
    (Rc_t, tc_t), pt3_t = solve_TRI(pt2_a, pt2_b, K, Ki, T_b2c, T_c2b,
            #guess
            guess_e
            #pose
            )
    Rb_t, tb_t = Rctc2Rbtb(Rc_t, tc_t, T_c2b, T_b2c)
    print('TRI : {} {}'.format(sc * tb_t[:2].ravel(), tx.euler_from_matrix(Rb_t)[-1]))

    idx_r = np.where(msk_r)[0]

    # viz-cloud
    #ax3 = plt.gca(projection='3d')
    #ax3.plot(
    #        pt3[:,0], pt3[:,1], pt3[:,2],
    #        'rx',
    #        label='depth-gt'
    #        )

    #ax3.plot(
    #        pt3_e[:,0]*sc, pt3_e[:,1]*sc, pt3_e[:,2]*sc,
    #        'b+',
    #        label='depth-em'
    #        )
    #ax3.plot(
    #        pt3_t[:,0]*sc, pt3_t[:,1]*sc, pt3_t[:,2]*sc,
    #        'c.',
    #        label='depth-tri'
    #        )

    # viz-depth
    #plt.plot(pt3[:,2], 'rx', label='depth-gt')
    #plt.plot(idx_r, pt3_e[:,2]*sc, 'c+', label='depth-em')
    #plt.plot(l1[:,2] * sc, 'b.', label='depth-tri')
    #plt.plot(sc*pt3_e[:,2], pt3[idx_r,2], '+')
    #tmp = (pt3[:,2])
    #plt.gca().set_ylim(tmp.min(), tmp.max())

    rvec = cv2.Rodrigues(Rc_e.T)[0]
    tvec = (-Rc_e.T.dot(tc_e.reshape(3,1))).ravel()
    print pt3_e.shape, pt3_e.dtype
    print rvec.shape, rvec.dtype
    print tvec.shape, tvec.dtype
    print K.shape, K.dtype
    print rvec.shape, rvec.dtype

    pt2_b_e = cv2.projectPoints(
            pt3_e[None,:], rvec, tvec,
            cameraMatrix=K,
            distCoeffs=np.zeros(5)
            )[0][:,0]
    rvec = cv2.Rodrigues(Rc_t.T)[0]
    tvec = (-Rc_t.T.dot(tc_t.reshape(3,1))).ravel()
    print pt3_t.shape, pt3_t.dtype
    print rvec.shape, rvec.dtype
    print tvec.shape, tvec.dtype
    print K.shape, K.dtype
    print rvec.shape, rvec.dtype
    pt2_b_t = cv2.projectPoints(
            pt3_t, rvec, tvec,
            cameraMatrix=K,
            distCoeffs=np.zeros(5)
            )[0][:,0]

    ax = plt.gca()
    ax.plot(pt2_b[:,0], pt2_b[:,1], 'r+', label='gt')
    ax.plot(pt2_b_t[:,0], pt2_b_t[:,1], 'bx', label='tri')
    ax.plot(pt2_b_e[:,0], pt2_b_e[:,1], 'c.', label='em')

    e  = (pt2_b - pt2_b_t).ravel()
    print 'x1-valid', [np.arctan2(tb_t[1], tb_t[0]), tx.euler_from_matrix(Rb_t)[-1]]

    params = np.concatenate([
        [np.arctan2(tb_t[1], tb_t[0]), tx.euler_from_matrix(Rb_t)[-1]],
        pt3_t[:,2]])

    e1 = err_TRI(
            params,
            pt2_a, pt2_b,
            K, Ki, T_b2c, T_c2b)

    tmp = proj_TRI(
        params,
        pt2_a,
        K, Ki,
        T_b2c, T_c2b
        )
    tmp /= tmp[:,2:]

    print np.abs(pt2_b - tmp[:,:2]).mean()

    fig = plt.figure()
    ax = fig.gca()
    ax.plot(tmp[:,0], tmp[:,1], 'rx')
    ax.plot(pt2_b[:,0], pt2_b[:,1], 'b+')
    ax.plot(pt2_b_t[:,0], pt2_b_t[:,1], 'c.')
    #fig.gca().plot(e1, label='err-tri')
    #fig.gca().plot(e, label='err-gt')
    fig.gca().legend()
    print 'cost-e1', np.sqrt(e1.dot(e1))
    print 'cost', np.sqrt(e.dot(e))

    plt.legend()
    plt.show()
    # == TRI END ==

if __name__ == "__main__":
    main()
