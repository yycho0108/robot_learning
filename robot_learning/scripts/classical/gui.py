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
    def draw_hud(ax, origin,
            fov, rad, style):
        h  = origin[-1]
        pos = np.reshape(origin[:2], (1,2))

        lx = np.linspace(0, rad)
        lx = np.stack([lx,lx*0],axis=-1)
        lh = np.linspace(h-fov/2, h+fov/2)

        fov_l = lx.dot(Rmat(h - fov/2).T) + pos
        fov_r = lx.dot(Rmat(h + fov/2).T) + pos

        s_range = rad * np.stack([np.cos(lh),np.sin(lh)], axis=-1) + pos
        ax.plot(fov_l[:,0], fov_l[:,1], style)
        ax.plot(fov_r[:,0], fov_r[:,1], style)
        ax.plot(s_range[:,0], s_range[:,1], style)

    @staticmethod
    def draw_top(ax,
            path0, pts,
            path1=None, scan=None
            ):
        ax.cla()

        # origin
        ax.plot([0],[0],'k+')

        ax.plot(path0[:,0], path0[:,1], 'b:')
        ax.plot(path0[-1:,0], path0[-1:,1], 'bo', markersize=5, label='visual odometry')

        # trajectory (ground truth)
        ax.plot(path1[:,0], path1[:,1], 'k--')
        ax.plot(path1[-1:,0], path1[-1:,1], 'ko', markersize=5, label='ground truth')

        # reconstruction
        ax.plot(pts[:,0], pts[:,1], 'r.', label='points')

        ax.set_title('VO Status Overview')
        ax.legend()

        # in a way, reconstruction "ground truth"
        if scan is not None:
            ax.plot(scan[:,0], scan[:,1], 'b.', label='scan')

        # ground-truth HUD
        fov = 1.14 # ~ 65.3'
        VoGUI.draw_hud(ax, path0[-1], fov, 5.0, 'b:')
        VoGUI.draw_hud(ax, path1[-1], fov, 5.0, 'k--')

        cx, cy = path1[-1, :2]
        ax.set_xlim(cx-5.0, cx+5.0)
        ax.set_ylim(cy-5.0, cy+5.0)
        ax.grid()
        ax.set_axisbelow(True)
        ax.set_aspect('equal', 'datalim')

    @staticmethod
    def draw_3d(ax, pts):
        ax.cla()
        ax.plot(pts[:,0], pts[:,1], pts[:,2], '.')
        axisEqual3D(ax)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title('Point Cloud Reconstruction')

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
        ax.set_title('Current Landmark Projection')

    @staticmethod
    def draw_img(ax, img):
        ax.cla()
        ax.imshow(img)
        ax.axis('off')

    @staticmethod
    def draw_err(ax, x, y):
        ax.cla()

        d = y - x 
        # normalize angle component
        d[:,2] = (d[:,2] + np.pi) % (2*np.pi) - np.pi

        # use abs val as error
        d = np.abs(d)

        ax.plot(d[:,0], label='dx')
        ax.plot(d[:,1], label='dy')
        ax.plot(d[:,2], label='dh')
        ax.legend()
        ax.grid()
        ax.set_axisbelow(True)
        ax.set_title('VO Error')

    def handle_key(self, event):
        k = event.key
        if k in ['n', ' ', 'enter']:
            self.index_ += 1
            if self.index_ < self.n_:
                self.step()
        if k in ['q', 'escape']:
            self.parent_.quit()
            sys.exit(0)
