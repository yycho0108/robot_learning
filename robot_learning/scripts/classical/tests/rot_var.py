import numpy as np
from tf import transformations as tx
from test_fmat import generate_valid_points
import time

def print_r(x):
    print( np.round(x, 3) )

def v2vx(v):
    vx = np.reshape([
        0, -v[2], v[1],
        v[2], 0, -v[0],
        -v[1], v[0], 0], (3,3))
    return vx

def vv2R(a, b):
    print 'na', np.linalg.norm(a)
    print 'nb', np.linalg.norm(b)

    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = np.dot(a, b)

    vx1 = v2vx(v)
    #vx2 = v2vx(np.square(v))
    vx2 = np.square(vx1)


    R = np.eye(3) + vx1 + vx2*(1.0 / (1.0+c))
    return R


def zu2R(u):
    # finds an Rmat that aligns [0,0,1] to u.
    u_z = np.float32([0,0,1])
    #q = tx.quaternion_about_axis(
    #        np.arccos(np.dot(u_z, u)),
    #        np.cross(u_z, u)
    #        )
    q_xyz = np.cross(u_z, u)
    q_w   = [1.0 + np.dot(u_z, u)]
    q = np.concatenate([q_xyz,q_w])
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    #print q, q2
    R = tx.quaternion_matrix(q)[:3,:3]

    #print 'validation'
    #print 'u', u
    #print 'u_z', u_z
    #print 'R.u', R.dot(u.reshape(3,1)).ravel()
    #print 'R.u_z', R.dot(u_z.reshape(3,1)).ravel()

    return R

def zu2Rs(u):
    # specialization of zu2R for batch u.
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

    # cross product
    # x x y = z, y x z = x, z x x = y

    d = np.linalg.norm(pt3, axis=-1, keepdims=True)
    u_z = pt3 / d # Nx3
    R = zu2Rs(u_z)

    RT = np.transpose(R, [0,2,1])
    C = np.matmul(np.matmul(R, cov0[None,...]), RT)
    return C

def main():
    n = 2048
    sx = 0.5
    sy = 1.0
    sz = 0.75

    #data = np.random.normal(
    #        loc = (0,0,0),
    #        scale = (sx,sy,sz),
    #        size = (n, 3))

    data = generate_valid_points(
            n=n,
            h=480.0,
            w=640.0)

    #print data.std(axis=0)

    # affine transform
    r = np.random.uniform(-np.pi, np.pi, size=3)
    t = np.random.uniform(-5.0, 5.0, size=3)
    print 'transformation'
    print 'r(deg) : ', np.round(np.rad2deg(r), 2)
    print 't      : ', np.round(t, 3)
    print '=============='

    R = tx.euler_matrix(*r)[:3,:3]
    uvec = R.dot([1,0,0])
    print 'uvec', uvec, tx.vector_norm(uvec)

    data_t = data.dot(R.T) + t

    print 'transformation of covariance'
    C0 = np.cov(data.T)
    C1 = np.cov(data_t.T)
    print 'C0 (pre-transform)'
    print_r(C0)
    print 'C1 (post-transform)'
    print_r(C1)
    C1_r = R.dot(C0).dot(R.T)
    print 'C1-r (reconstruction based on transform)'
    print_r(C1_r)

    # == test 2 : oriented cov from directional vector ==
    print '== oriented cov =='
    print 'data', data.shape
    data_u = data / np.linalg.norm(data, axis=-1, keepdims=True)
    C1_r2 = oriented_cov(data_u, C0)
    R0 = zu2R(data_u[0])
    print 'du0', data_u[0]
    print 'du0-r', R0.dot([0,0,1])
    a = R0.dot(C0).dot(R0.T)
    b = C1_r2[0]

    print 'oriented covariance'
    print a
    print b

    print 'recovery'
    print 'CO', np.round(np.diag(np.square([sx,sy,sz])), 4)
    print 'Ca', np.round(R0.T.dot(a).dot(R0), 4)
    print 'Cb', np.round(R0.T.dot(b).dot(R0), 4)

    #print 'experimental'
    #print C0.T / (C1.dot(uvec.T)) # should theoretically give x-component cov


if __name__ == "__main__":
    main()
