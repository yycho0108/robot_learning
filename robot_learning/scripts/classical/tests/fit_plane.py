import numpy as np
from tf import transformations as tx
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

def axisEqual3D(ax):
    """ from https://stackoverflow.com/a/19248731 """
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

def ransac_update_num_iters(p, ep, mpt, it,
        eps = np.finfo(np.float32).eps
        ):
    p  = np.clip(p, 0, 1)
    ep = np.clip(ep, 0, 1)

    nmr = max(1.0 - p, eps)
    dmr = 1.0 - np.power(1.0-ep, mpt)

    if dmr < eps:
        return 0

    nmr = np.log(nmr)
    dmr = np.log(dmr)

    res = it if (dmr >= 0 or -nmr >= it * -dmr) else np.round(nmr/dmr).astype(np.int32)
    return res

def estimate_plane_ransac(pts, n,
        conf = 0.99,
        thresh = 0.1):

    best_fit = None
    best_err = np.inf
    best_msk = None

    n_it = max(n, 1)
    i = 0

    while i < n_it:
        ## select three points that define a plane and go from there.
        #sel = np.random.randint(len(pts), size=3)
        sel = np.random.choice(len(pts), size=3, replace=False)

        pa, pb, pc = pts[sel]
        c = np.mean(pts[sel], axis=0, keepdims=True) # plane center

        ba = tx.unit_vector(pb-pa)
        ca = tx.unit_vector(pc-pa)

        n = tx.unit_vector(np.cross(ba, ca)) # plane normal

        err = (pts - c).dot(n.reshape(-1,1)) # Nx3 . 3x1
        err = np.abs(err)

        msk = (err < thresh)
        n_in = msk.sum()
        err = err.sum()

        n_it = ransac_update_num_iters(conf,
                float(msk.size - n_in) / msk.size, # idk what ep is
                3, # 3 points required to define a plane
                n_it)

        if err < best_err:
            best_err = err
            best_fit = (c, n)
            best_msk = msk

        i += 1

    print('completed in {} iterations'.format(i))

    return best_fit, best_err, best_msk

def main():
    n_pts = 128
    plane_pts = np.random.normal(
            scale=(5.0, 5.0, 1.0),
            size=(n_pts,3))

    r = np.random.uniform(-np.pi,np.pi, size=3)
    t = np.random.uniform(-5.0, 5.0, size=3)

    print 'r', r
    print 't', t

    T = tx.compose_matrix(translate=t, angles=r)

    plane_T = plane_pts.dot(T[:3,:3].T) + T[:3,3]

    fit, err, msk = estimate_plane_ransac(plane_T, n=1000000, thresh=0.1,
            conf=0.999
            )
    print fit
    center, nvec = fit
    Ti = tx.inverse_matrix(T)
    print 'should-be-z', Ti[:3,:3].dot(nvec)

    print '{}/{}'.format(msk.sum(), msk.size)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, projection='3d')
    ax.plot(plane_T[:,0], plane_T[:,1], plane_T[:,2], '.')

    l = np.linspace(0.0, 2.0)
    viz_l = l[:,None] * nvec.reshape(1,3)
    viz_l += np.reshape(center, [1,3])
    ax.plot(viz_l[:,0], viz_l[:,1], viz_l[:,2], '-')
    axisEqual3D(ax)
    #ax.set_aspect('equal', 'datalim')

    plt.show()

if __name__ == "__main__":
    main()
