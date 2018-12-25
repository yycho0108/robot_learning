import numpy as np
from tf import transformations as tx
import cv2

try:
  from pathlib import Path
except ImportError:
  from pathlib2 import Path  # python 2 backport

def anorm(x):
    """ angular value, converted to (-np.pi, np.pi) range """
    return (x + np.pi) % (2*np.pi) - np.pi

def no_op(*args, **kwargs):
    return

def mkdir(x):
    """ convenient mkdir wrapper"""
    return Path(x).mkdir(parents=True, exist_ok=True)

def main():
    Ks = 1.0
    K = np.reshape([
            499.114583 * Ks, 0.000000, 325.589216 * Ks,
            0.000000, 498.996093 * Ks, 238.001597 * Ks,
            0.000000, 0.000000, 1.000000], (3,3))
    D = np.float32([0.158661, -0.249478, -0.000564, 0.000157, 0.000000])

    T_c2b = tx.compose_matrix(
            angles=[-np.pi/2-np.deg2rad(10),0.0,-np.pi/2],
            #angles=[-np.pi/2,0.0,-np.pi/2],
            translate=[0.174,0,0.113])
    T_b2c = tx.inverse_matrix(T_c2b)

    # load data
    imgs    = np.load('../../data/train/27/img.npy')
    print imgs.shape
    lmk_pos = np.load('/tmp/lmk_pos.npy')
    lmk_col = np.load('/tmp/lmk_col.npy')
    lmk_var = np.load('/tmp/lmk_var.npy')
    cam_pos = np.load('/tmp/cam_pos.npy')

    # convert lmk_pos --> map[base_link] coordinate
    # TODO : is this necessary?

    lmk_pos_m = lmk_pos.dot(T_c2b[:3,:3].T) + T_c2b[:3,3:].T
    lmk_xyzrgb = np.concatenate([lmk_pos_m, lmk_col], axis=-1)

    np.savetxt('/tmp/lmk_xyzrgb.txt',
            lmk_xyzrgb,
            fmt='%f',
            #header='CONTOUR',
            delimiter=' ',
            #comments=''
            )
    return

    # build projection matrix ...
    n_c = len(cam_pos)
    T_b2o = np.zeros((n_c, 4, 4))

    x, y, h = cam_pos.T
    c = np.cos(h)
    s = np.sin(h)

    # z-rotation part
    T_b2o[:, 0, 0] = c
    T_b2o[:, 0, 1] = -s
    T_b2o[:, 1, 0] = s
    T_b2o[:, 1, 1] = c
    T_b2o[:, 2, 2] = 1
    # translation part
    T_b2o[:, 0, 3] = x
    T_b2o[:, 1, 3] = y
    # homogeneous part
    T_b2o[:, 3, 3] = 1

    # transform validation
    #i_test = 230
    #T_b2o_i = tx.compose_matrix(
    #        translate=[ x[i_test], y[i_test], 0],
    #        angles=[0, 0, h[i_test]]
    #        )
    #print T_b2o_i
    #print T_b2o[i_test]
    #print T_b2o_i - T_b2o[i_test]

    # convert from camera -> map[camera] coord
    T_c2m = reduce(np.matmul, [
        T_b2c,
        T_b2o,
        T_c2b,
        ])

    P_c2m = np.matmul(K, T_c2m[:, :3]) # Nx3x4

    # simple validation
    #p3h = lmk_pos[:64].dot(P_c2m[0,:3,:3].T) + P_c2m[0,:3,3:].T
    #print p3h / p3h[:,2:]

    mkdir('/tmp/pmvs')
    mkdir('/tmp/pmvs/txt')
    mkdir('/tmp/pmvs/visualize')

    WRITE_PATH = False
    WRITE_IMG  = False
    WRITE_OPT  = False

    if WRITE_PATH:
        for i in range(n_c):
            p = P_c2m[i]
            np.savetxt('/tmp/pmvs/txt/{:08d}.txt'.format(i),
                    p,
                    fmt='%f',
                    header='CONTOUR',
                    delimiter=' ',
                    comments=''
                    )
    if WRITE_IMG:
        for i in range(n_c):
            f = imgs[i]
            f_u = cv2.undistort(f, K, D, K)
            cv2.imwrite('/tmp/pmvs/visualize/{:08d}.ppm'.format(i),
                    f_u
                    )

    if WRITE_OPT:
        # get string of specifications based on "key-frame" interval

        def fs_sub(k):
            r = map(str, range(0, n_c, k))
            return ('%d ' % len(r)) + ' '.join(r)

        with open('/tmp/pmvs/opt.txt', 'w') as f:
            # strings for each of image subsets
            s_non = '0' # none
            s_sub = '{}'.format(fs_sub(8)) # sub-sample
            s_all = '-1 0 {}'.format(n_c) # all

            f.writelines([
                # OPT1 : nothing is enforced
                'timages {}\n'.format( s_sub ),
                'oimages {}\n'.format( s_all ),
                #'level 1\n',
                #'csize 9\n',
                'sequence 64\n'
                ])

if __name__ == "__main__":
    main()
