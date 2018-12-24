import numpy as np
from tf import transformations as tx

from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D

import time

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

    # validation
    #rotpart = np.matmul(R, np.reshape([0,0,1], (3,1)))[...,0]
    #print rotpart - u_z

    RT = np.transpose(R, [0, 2, 1])
    #print np.matmul(R,RT) # ==eye

    C = np.matmul(np.matmul(R, cov0[None,...]), RT)

    # apply depth scale for variance
    C = d[...,None] * C
    return C

def cov_sum(a, b, va, vb):
    """
    from https://math.stackexchange.com/a/2414813
    Doesn't really appear to be more efficient?
    """

    vc = va+vb
    L = np.linalg.cholesky(vc)

    # opt1 : batch processing
    rhs = np.concatenate([va, vb, a[:,None], b[:,None]], axis=1)
    lhs = np.linalg.solve(L, rhs)
    va_ = lhs[:, 0:3]
    vb_ = lhs[:, 3:6]
    a_  = lhs[:, 6:7]
    b_  = lhs[:, 7:8]

    # opt2 : solve each one
    #va_ = np.linalg.solve(L, va)
    #vb_ = np.linalg.solve(L, vb)
    #a_  = np.linalg.solve(L, a)
    #b_  = np.linalg.solve(L, b)

    c = vb_.T.dot(a_) + va_.T.dot(b_)
    vc = va_.T.dot(vb_)
    return c, vc

def cov_sum_kal(a, b, va, vb):
    y = (b-a)
    S = va+vb
    K = va.dot( np.linalg.inv(S) )
    x = a + K.dot(y)
    P = (np.eye(3) - K).dot(va)
    return x, P

def gen_pt(n, fov=1.15):
    phi   = np.random.uniform(-fov/2 ,fov/2, size=n)
    theta = np.random.uniform(-fov/2, fov/2, size=n)

    c, s = np.cos(theta), np.sin(theta)

    # 1. in general coord.
    x = c * np.cos(phi)
    y = s * np.cos(phi)
    z = np.sin(phi)

    p = np.stack([x, y, z], axis=-1)

    # 2. cvt to optical coord. (base->cam)
    R_opt = tx.euler_matrix(*[-np.pi/2,0.0,-np.pi/2])

    return p.dot(R_opt[:3,:3])

def get_ellipse(ax, p, c, col=None):
    x, y = p[0], p[1]
    c_2d = c[:2,:2] 
    l, v = np.linalg.eig(c_2d)
    l    = np.sqrt(l)
    h    = np.arctan2(v[1,0], v[0,0])
    ell = Ellipse(xy=(x, y),
                width=l[0]*2, height=l[1]*2,
                angle=np.rad2deg(h))
    ell.set_clip_box(ax.bbox)
    ell.set_alpha(0.5)
    if col is not None:
        ell.set_facecolor(col)
    return ell

def main():
    n_pt  = 1
    n_obs = 2

    # visualization
    n_p_v = n_pt
    n_o_v = n_obs

    p = gen_pt(n_pt, fov=1.15)
    d = np.random.uniform(0.1, 2.0, size=(n_pt,1))
    p = d * p

    # source point
    src = p

    # observations
    std_obs = 10.0 # observation (depth component noise)
    view    = np.random.normal(0.0, 1.0, size=(n_pt, n_obs, 3))
    obs_d   = tx.unit_vector(src[:,None] - view) # ==> [N_PT,N_VIEW,3]
    obs     = src[:,None] + np.random.normal(scale=std_obs, size=(n_pt,n_obs,1)) * obs_d

    # random perturbation
    noise_o = 0.0
    obs_n = np.random.normal(loc=obs, scale=noise_o)

    # covariance
    cov0 = np.diag(np.square([0.05, 0.05, 1.0])) # expected landmark variance @ ~ 1m
    cov = oriented_cov((obs_n - view).reshape(-1,3), cov0)
    cov = cov.reshape(n_pt, n_obs, 3, 3) 

    # prepare visualization
    src  = src[:n_p_v]
    view = view[:n_p_v, :n_o_v].reshape(-1, 3)
    obs  = obs[:n_p_v, :n_o_v].reshape(-1, 3)
    cov  = cov[:n_p_v, :n_o_v].reshape(-1, 3, 3)

    # cvt: camera --> base_link optical transform
    R_c2b = tx.euler_matrix(*[-np.pi/2,0.0,-np.pi/2])[:3,:3]

    # convert to x-y-z coordnates
    src  = src.dot(R_c2b.T)
    view = view.dot(R_c2b.T)
    obs  = obs.dot(R_c2b.T)
    cov  = np.matmul(np.matmul(R_c2b, cov), R_c2b.T)

    n_rep = 100
    t0 = time.time()
    for _ in range(n_rep):
        sp, sv = cov_sum(obs[0], obs[1], cov[0], cov[1])
    print sp, sv
    t1 = time.time()
    for _ in range(n_rep):
        sp, sv = cov_sum_kal(obs[0], obs[1], cov[0], cov[1])
    print sp, sv
    t2 = time.time()

    print 'chol', t1 - t0
    print 'kal',  t2 - t1

    ax = plt.gca()
    ax.plot([0],[0],'k+')
    ax.plot(src[:,0],  src[:,1],  'go', label='src')
    ax.plot(view[:,0], view[:,1], 'b*', label='view')
    ax.plot(obs[:,0],  obs[:,1], 'rx', label='obs')
    ax.plot([sp[0]],  [sp[1]], 'k.', label='sum')


    for v, o in zip(view, obs):
        x_part = [v[0], o[0]]
        y_part = [v[1], o[1]]
        ax.plot(x_part, y_part, 'y--')
    ax.legend()

    for p, c in zip(obs, cov):
        ell = get_ellipse(ax, p,c)
        ax.add_artist(ell)

    ell = get_ellipse(ax, sp, sv, 'r')
    ax.add_artist(ell)
    ax.set_aspect('equal', 'datalim')
    #ax.set_xlim(-5.0, 5.0)
    #ax.set_ylim(-5.0, 5.0)

    plt.show()

    #cov0 = np.diag(cov0) # 3x3
    #C = oriented_cov(p, cov0)

    #fig = plt.figure()
    #ax = fig.add_subplot(1,1,1, projection='3d')
    #ax.plot(p[:,0], p[:,1], p[:,2], '.')
    #ax.set_xlabel('x')
    #ax.set_ylabel('y')
    #ax.set_zlabel('z')
    #ax.set_xlim(-1,1)
    #ax.set_ylim(-1,1)
    #ax.set_zlim(-1,1)
    #plt.show()

    #p_max = np.deg2rad(60.0)
    #t_max = np.deg2rad(80.0)

    ##phi   = np.random.uniform(-p_max, p_max, size=n_pt)
    ##theta = np.random.uniform(-t_max, t_max, size=n_pt)

    ## in cam coord:
    ## phi   = rotation about x axis 
    ## theta = rotation about -y axis

    #phi   = np.float32([0.3, 0.3])
    #theta = np.float32([0.2, 0.2])
    #r1 = tx.euler_matrix(phi[0], -theta[0], 0)
    #p1 = r1[:3,:3].dot([0,0,1])
    #p1_r = np.stack([
    #    np.sin(theta),#*np.cos(-phi),
    #    np.sin(theta)*np.sin(-phi),
    #    np.cos(theta)], axis=-1)
    #print p1, p1_r[0]
    #r2 = tx.euler_matrix(phi[1], -theta[1], 0)
    #p2 = r2[:3,:3].dot([0,0,1])

if __name__ == "__main__":
    main()
