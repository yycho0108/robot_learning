import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tf import transformations as tx
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import Delaunay

def estimate_normals(p, k=20):
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(p)
    _, indices = neigh.kneighbors()

    dp = (p[indices] - p[:,None])
    U, s, V = np.linalg.svd(dp.transpose(0,2,1))
    nv = U[:, :, -1]
    return nv / np.linalg.norm(nv, axis=-1, keepdims=True)

def non_max_suppression(pos, var, k=16, radius=0.1):
    v = np.linalg.norm(var[:,(0,1,2),(0,1,2)], axis=-1)
    idx = np.argsort(v) # handle small variance first
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(pos)
    d, i = neigh.kneighbors(return_distance=True)

    # filter by actual nearest distance (radius)
    msk_d = np.min(d, axis=1) < radius
    msk_v = np.all(v[i] > v[:,None], axis=1)

    msk = np.logical_or(
            np.logical_and(msk_d, msk_v),
            ~msk_d)
    return np.where(msk)[0]

from matplotlib import artist
import matplotlib.axes as maxes
import matplotlib.cbook as cbook
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import matplotlib.docstring as docstring
import matplotlib.scale as mscale
import matplotlib.transforms as mtransforms
from matplotlib.axes import Axes, rcParams
from matplotlib.colors import Normalize, LightSource
from matplotlib.transforms import Bbox
from matplotlib.tri.triangulation import Triangulation

from mpl_toolkits.mplot3d import art3d
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d import axis3d

def write_ply(pos, col):
    header = '\n'.join([
        'ply',
        'format ascii 1.0',
        'element vertex {}'.format(len(pos)),
        'property float x',
        'property float y',
        'property float z',
        'property uchar red',
        'property uchar green',
        'property uchar blue',
        'end_header'
        ])
    arr = np.concatenate([pos, col*255], axis=-1)
    np.savetxt(
            '/tmp/test.ply',
            arr,
            fmt='%f %f %f %d %d %d',
            header=header,
            comments='')


def plot_trisurf(ax, args, facecolors,
        color=None, norm=None, vmin=None, vmax=None,
        lightsource=None, **kwargs
        ):
    """
    adapted from matplotlib plot_trisurf source code.

    ============= ================================================
    Argument      Description
    ============= ================================================
    *X*, *Y*, *Z* Data values as 1D arrays
    *color*       Color of the surface patches
    *cmap*        A colormap for the surface patches.
    *norm*        An instance of Normalize to map values to colors
    *vmin*        Minimum value to map
    *vmax*        Maximum value to map
    *shade*       Whether to shade the facecolors
    ============= ================================================

    The (optional) triangulation can be specified in one of two ways;
    either::

        plot_trisurf(triangulation, ...)

    where triangulation is a :class:`~matplotlib.tri.Triangulation`
    object, or::

        plot_trisurf(X, Y, ...)
        plot_trisurf(X, Y, triangles, ...)
        plot_trisurf(X, Y, triangles=triangles, ...)

    in which case a Triangulation object will be created.  See
    :class:`~matplotlib.tri.Triangulation` for a explanation of
    these possibilities.

    The remaining arguments are::

        plot_trisurf(..., Z)

    where *Z* is the array of values to contour, one per point
    in the triangulation.

    Other arguments are passed on to
    :class:`~mpl_toolkits.mplot3d.art3d.Poly3DCollection`

    **Examples:**

    .. plot:: gallery/mplot3d/trisurf3d.py
    .. plot:: gallery/mplot3d/trisurf3d_2.py

    .. versionadded:: 1.2.0
        This plotting function was added for the v1.2.0 release.
    """

    self = ax
    had_data = self.has_data()

    # TODO: Support custom face colours
    if color is None:
        color = self._get_lines.get_next_color()
    color = np.array(mcolors.to_rgba(color))

    cmap = kwargs.get('cmap', None)
    shade = kwargs.pop('shade', cmap is None)

    tri, args, kwargs = Triangulation.get_from_args_and_kwargs(*args, **kwargs)
    if 'Z' in kwargs:
        z = np.asarray(kwargs.pop('Z'))
    else:
        z = np.asarray(args[0])
        # We do this so Z doesn't get passed as an arg to PolyCollection
        args = args[1:]

    triangles = tri.get_masked_triangles()
    xt = tri.x[triangles]
    yt = tri.y[triangles]
    zt = z[triangles]

    # filter by tri size
    pt = np.stack([xt,yt,zt], axis=1) # Nx3x3
    dv1 = (pt[:,0] - pt[:,1])
    dv2 = (pt[:,0] - pt[:,2])
    ar = np.linalg.norm(np.cross(dv1, dv2), axis=-1)
    c = np.percentile(ar, 75)

    # apply filtered try
    triangles = triangles[ar<=c]
    xt = tri.x[triangles]
    yt = tri.y[triangles]
    zt = z[triangles]

    verts = np.stack((xt, yt, zt), axis=-1)

    polyc = art3d.Poly3DCollection(verts, *args, **kwargs)

    if cmap:
        # average over the three points of each triangle
        avg_z = verts[:, :, 2].mean(axis=1)
        polyc.set_array(avg_z)
        if vmin is not None or vmax is not None:
            polyc.set_clim(vmin, vmax)
        if norm is not None:
            polyc.set_norm(norm)
    else:
        if facecolors is not None:
            col = facecolors[triangles.ravel()].reshape([-1,3,3])
            col = np.sqrt(np.mean(np.square(col), axis=1))
            polyc.set_facecolors(col)
        else:
            if shade:
                v1 = verts[:, 0, :] - verts[:, 1, :]
                v2 = verts[:, 1, :] - verts[:, 2, :]
                normals = np.cross(v1, v2)
                colset = self._shade_colors(color, normals)
            else:
                colset = color
            polyc.set_facecolors(colset)

    self.add_collection(polyc)
    self.auto_scale_xyz(tri.x, tri.y, z, had_data)

    return polyc

def main():
    T_c2b_ = tx.compose_matrix(
            angles=[-np.pi/2-np.deg2rad(10),0.0,-np.pi/2],
            #angles=[-np.pi/2,0.0,-np.pi/2],
            translate=[0.174,0,0.113])

    lmk_pos = np.load('/tmp/lmk_pos.npy')
    lmk_pos = lmk_pos.dot(T_c2b_[:3,:3].T) + T_c2b_[:3,3:].T
    lmk_col = np.load('/tmp/lmk_col.npy')
    lmk_var = np.load('/tmp/lmk_var.npy')
    cam_pos = np.load('/tmp/cam_pos.npy')

    ## basic thresholding filter
    #v = np.linalg.norm(lmk_var[:,(0,1,2),(0,1,2)], axis=-1)
    ##lo = np.percentile(v, 20, axis=0)
    ##hi = np.percentile(v, 80, axis=0)
    ##plt.hist(v[np.logical_and(lo<v,v<hi)])
    ##plt.show()
    #v_idx = np.where(v < 0.2)[0]
    #print('plotting {}/{}'.format(len(v_idx), len(v)))
    #lmk_pos = lmk_pos[v_idx]
    #lmk_col = lmk_col[v_idx]
    #lmk_var = lmk_var[v_idx]
    ##cam_pos = cam_pos[v_idx]

    idx = np.where(lmk_pos[:,2] > -0.01) # plot above ground-plane
    lmk_pos, lmk_col, lmk_var = [e[idx] for e in (lmk_pos, lmk_col, lmk_var)]

    idx = non_max_suppression(lmk_pos, lmk_var, k=16, radius=0.025)
    lmk_pos, lmk_col, lmk_var = [e[idx] for e in (lmk_pos, lmk_col, lmk_var)]

    lo = np.percentile(lmk_pos, 5, axis=0)
    hi = np.percentile(lmk_pos, 95, axis=0)
    clip_msk = np.logical_and.reduce([
        np.all(lo[None,:] <= lmk_pos, axis=1),
        np.all(lmk_pos <= hi[None,:], axis=1)
        ])
    idx = np.where(clip_msk)[0]
    lmk_pos, lmk_col, lmk_var = [e[idx] for e in (lmk_pos, lmk_col, lmk_var)]

    lo = np.percentile(lmk_pos, 20, axis=0)
    hi = np.percentile(lmk_pos, 80, axis=0)
    sc = np.max(np.abs(hi - lo))
    md = (hi + lo) / 2.0

    lo = md - sc / 2.0
    hi = md + sc / 2.0

    xlim, ylim, zlim = zip(*[lo,hi])

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, projection='3d')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)

    col = lmk_col[...,::-1]/255.
    col = col.astype(np.float32)

    plot_trisurf(ax, [lmk_pos[:,0], lmk_pos[:,1], lmk_pos[:,2]],
            facecolors = col,
            zorder=0
            )

    write_ply(lmk_pos, col)

    # normal vector color
    lmk_nvc = estimate_normals(lmk_pos, k=10)
    lmk_wtf = np.concatenate([lmk_pos, lmk_nvc], axis=-1)
    np.savetxt('/tmp/lmk_wtf.txt', lmk_wtf)
    #print lmk_nvc.min(axis=0), lmk_nvc.max(axis=0)
    #col = (0.5 + lmk_nvc * 0.5)
    #print col.min(axis=0), col.max(axis=0)

    ax.scatter(lmk_pos[:,0], lmk_pos[:,1], lmk_pos[:,2], 
            marker='D',
            c=col,
            zorder=1
            )

    ax.plot(cam_pos[:,0], cam_pos[:,1], 0*cam_pos[:,2], # index 2 is heading, not Z
            'b--',
            zorder=2
            )

    # show origin
    ax.plot([0,1],[0,0],[0,0], 'r-',
            zorder=3
            )
    ax.plot([0,0],[0,1],[0,0], 'g-',
            zorder=3
            )
    ax.plot([0,0],[0,0],[0,1], 'b-',
            zorder=3
            )

    plt.show()

if __name__ == "__main__":
    main()
