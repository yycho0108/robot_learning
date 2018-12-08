import numpy as np
from matplotlib import pyplot as plt
import sys
from misc import Rmat

def axisEqual3D(ax):
    """ from https://stackoverflow.com/a/19248731 """
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

class VoGUI(object):
    def __init__(self, parent):
        self.parent_ = parent

    @staticmethod
    def draw_top(ax,
            path0, pts,
            path1=None, scan=None
            ):
        ax.cla()

        # origin
        ax.plot([0],[0],'k+')

        ax.plot(path0[:,0], path0[:,1], 'b--')

        # trajectory (ground truth)
        ax.plot(path1[:,0], path1[:,1], 'k--')

        # reconstruction
        ax.plot(pts[:,0], pts[:,1], 'r.', label='visual')

        # in a way, reconstruction "ground truth"
        if scan is not None:
            ax.plot(scan[:,0], scan[:,1], 'b.', label='scan')

        # hud : visual field + sensing radius
        lx = np.linspace(0, 5)
        lx = np.stack([lx,lx*0],axis=-1)

        h0  = path1[-1, 2]
        pos = np.reshape(path1[-1,:2], (1,2))
        fov = np.deg2rad(73.0)

        fov_l = lx.dot(Rmat(h0 - fov/2).T) + pos
        fov_r = lx.dot(Rmat(h0 + fov/2).T) + pos
        h = np.linspace(-np.pi, np.pi)
        radius = 5.0 * np.stack([np.cos(h),np.sin(h)], axis=-1) + pos
        ax.plot(fov_l[:,0], fov_l[:,1], 'g--')
        ax.plot(fov_r[:,0], fov_r[:,1], 'g--')
        ax.plot(radius[:,0], radius[:,1], 'g--')

        cx, cy = pos[0]
        ax.set_xlim(cx-5.0, cx+5.0)
        ax.set_ylim(cy-5.0, cy+5.0)

    @staticmethod
    def draw_3d(ax, pts):
        ax.cla()
        ax.plot(pts[:,0], pts[:,1], pts[:,2], '.')
        axisEqual3D(ax)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

    @staticmethod
    def draw_2d_proj(ax, img, pts, color=None):
        # TODO : support color
        ax.cla()
        VoGUI.draw_img(ax, img)
        n, m = np.shape(img)[:2]
        ax.plot(pts[:,0], pts[:,1], '.')
        ax.set_xlim(0, m)
        ax.set_ylim(0, n)
        ax.set_aspect('equal')
        if not ax.yaxis_inverted():
            ax.invert_yaxis()

    @staticmethod
    def draw_img(ax, img):
        ax.cla()
        ax.imshow(img)
        ax.axis('off')

    def handle_key(self, event):
        k = event.key
        if k in ['n', ' ', 'enter']:
            self.index_ += 1
            if self.index_ < self.n_:
                self.step()
        if k in ['q', 'escape']:
            self.parent_.quit()
            sys.exit(0)
