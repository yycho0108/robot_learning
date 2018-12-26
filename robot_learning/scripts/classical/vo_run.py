#!/usr/bin/env python2
import time
import numpy as np
from numpy import s_
import sys
import cv2
from scipy.optimize import linear_sum_assignment
from matplotlib import pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D

from collections import deque

try:
  from pathlib import Path
except ImportError:
  from pathlib2 import Path  # python 2 backport

from vo3 import ClassicalVO
from gui import VoGUI

sys.path.append('../')
from utils.vo_utils import add_p3d, sub_p3d
from utils.kitti_utils import KittiLoader
from misc import Rmat

class CVORunner(object):
    def __init__(self, imgs, stamps, odom, scan=None):
        self.index_ = 0
        self.n_ = len(imgs)
        self.imgs_ = imgs
        self.stamps_ = stamps
        self.odom_ = odom
        self.scan_ = scan

        self.fig_ = fig = plt.figure(figsize=(16,12), dpi=60)
        gridspec.GridSpec(3,3)

        self.ax0_ = fig.add_subplot(3,3,1)
        self.ax2_ = fig.add_subplot(3,3,4, projection='3d')
        self.ax3_ = fig.add_subplot(3,3,7)
        self.ax1_ = plt.subplot2grid((3,3), (0,1), colspan=2, rowspan=2)
        self.ax4_ = plt.subplot2grid((3,3), (2,1), colspan=2)

        self.map_ = np.empty((0, 2), dtype=np.float32)
        self.vo_ = ClassicalVO()

        self.vo_(imgs[0], 0.0) # initialize GUI

        self.tx_ = []
        self.ty_ = []
        self.th_ = []

        #self.gui_ = VoGUI()
        self.quit_ = False

    def handle_key(self, event):
        k = event.key
        if k in ['n', ' ', 'enter']:
            self.index_ += 1
            if self.index_ < self.n_:
                self.step()
        if k in ['q', 'escape']:
            self.quit_ = True
            sys.exit(0)

    def scan_to_pt(self, scan):
        r, mask = scan[:,0], scan[:,1]
        a = np.linspace(0, 2*np.pi, 361)
        mask = mask.astype(np.bool)
        r, a = r[mask], a[mask]
        #mask = ( np.abs(a) < np.deg2rad(45) )
        #r, a = r[mask], a[mask]
        c, s = np.cos(a), np.sin(a)
        p = r[:,None] * np.stack([c,s], axis=-1)
        return p

    def show(self,
            aimg,
            pts3,
            pts2,
            scan_c,
            pts_r,
            cov,
            pts_col,
            msg='title'
            ):
        # unroll
        i = self.index_
        n = self.n_
        ax0,ax1,ax2,ax3,ax4 = \
            self.ax0_, self.ax1_, self.ax2_, self.ax3_, self.ax4_

        odom   = self.odom_
        scan   = self.scan_
        stamps = self.stamps_
        imgs  = self.imgs_
        vo    = self.vo_
        tx, ty, th = self.tx_, self.ty_, self.th_

        rec_path = np.stack([tx,ty,th], axis=-1)

        ax0.set_title('Tracking Visualization')
        VoGUI.draw_img(ax0, aimg[..., ::-1])
        VoGUI.draw_top(ax1, rec_path, pts2, odom[:i+1], scan_c, cov, pts_col)
        VoGUI.draw_3d(ax2, pts3, pts_col)
        VoGUI.draw_2d_proj(ax3, imgs[i, ..., ::-1], pts_r)
        VoGUI.draw_err(ax4, rec_path, odom[:i])

        if self.vo_.pnp_p_ is not None:
            self.ax1_.plot(
                    [self.vo_.pnp_p_[0]],
                    [self.vo_.pnp_p_[1]],
                    'go',
                    label='pnp',
                    alpha=0.5
                    )
            self.ax1_.quiver(
                    [self.vo_.pnp_p_[0]],
                    [self.vo_.pnp_p_[1]],
                    [np.cos(self.vo_.pnp_h_)],
                    [np.sin(self.vo_.pnp_h_)],
                    angles='xy',
                    #scale=1,
                    color='g',
                    alpha=0.5
                    )


        self.fig_.canvas.draw()
        self.fig_.suptitle(msg)

    def step(self):
        i = self.index_
        n = self.n_
        print('i : {}/{}'.format(i,n))

        if i >= n:
            return

        # unroll properties
        odom   = self.odom_
        scan   = self.scan_
        stamps = self.stamps_
        imgs  = self.imgs_
        vo    = self.vo_
        tx, ty, th = self.tx_, self.ty_, self.th_

        # index
        stamp = stamps[i]
        img   = imgs[i]

        # TODO : there was a bug in data_collector that corrupted all time-stamp data!
        # disable stamps dt for datasets with corrupted timestamps.
        # very unfortunate.
        dt  = (stamps[i] - stamps[i-1])
        #dt = 0.2
        print('dt', dt)

        # experimental : pass in scale as a parameter
        # TODO : estimate scale from points + camera height?
        dps_gt = sub_p3d(odom[i], odom[i-1])
        s = np.linalg.norm(dps_gt[:2])
        print('Ground Truth Scale : [ {} ]'.format(s))
        print('Ground Truth Motion : {}'.format( dps_gt ))

        scale = None
        if i <= 2:
            # TODO : maybe prefer vo.initialize_scale()
            # rather than checking for index
            # scale initialization is required
            scale = s
        else:
            # implicit : scale = None
            pass
        res = vo(img, dt, scale=scale)

        if res is None:
            # vo aborted for some unknown reason
            print('Visual Odometry Aborted!')
            return

        (aimg, vo_pos, pts_r, pts3, col_p, msg) = res
        #print('(pred-gt) {} vs {}'.format(dps, dps_gt) )
        pos = vo_pos

        tx.append(pos[0])
        ty.append(pos[1])
        th.append(pos[2])

        pts2 = pts3[:,:2]
        self.map_ = np.concatenate([self.map_, pts2], axis=0)
        if scan is not None:
            scan_c = self.scan_to_pt(scan[i]).dot(Rmat(odom[i,2]).T) + np.reshape(odom[i, :2], (1,2))
        else:
            scan_c = None

        ### EVERYTHING FROM HERE IS PLOTTING + VIZ ###
        if (i % 1) == 0:
            P = self.vo_.ukf_l_.P
            self.show(aimg, pts3, pts2, scan_c, pts_r, P, col_p, ('[%d/%d] '%(i,n)) + msg)

    def quit(self):
        self.quit_ = True

    def run(self, auto=False):
        #self.gui_.run()
        self.fig_.canvas.mpl_connect('key_press_event', self.handle_key)
        self.fig_.canvas.mpl_connect('close_event', sys.exit)
        if auto:
            while not self.quit_:
                if self.index_ < self.n_:
                    self.index_ += 1
                    self.step()
                plt.pause(0.001)
                #plt.savefig('/tmp/{:04d}.png'.format(self.index_))
        else:
            plt.show()

        # save ...
        cam_pos = np.stack([self.tx_, self.ty_, self.th_], axis=-1)
        lmk_pos = self.vo_.landmarks_.pos
        lmk_var = self.vo_.landmarks_.var
        lmk_col = self.vo_.landmarks_.col

        np.save('/tmp/cam_pos.npy', cam_pos)
        np.save('/tmp/lmk_pos.npy', lmk_pos)
        np.save('/tmp/lmk_var.npy', lmk_var)
        np.save('/tmp/lmk_col.npy', lmk_col)

def main():
    np.set_printoptions(precision=4)
    #idx = np.random.choice(8)
    idx = 34
    print('idx', idx)

    # load data
    i0 = 0
    di = 1

    imgs   = np.load('../../data/train/{}/img.npy'.format(idx))[i0::di]
    #imgs = np.asarray([cv2.resize(e, None, fx=2, fy=2) for e in imgs])
    stamps = np.load('../../data/train/{}/stamp.npy'.format(idx))[i0::di]
    odom   = np.load('../../data/train/{}/odom.npy'.format(idx))[i0::di]
    try:
        scan   = np.load('../../data/train/{}/scan.npy'.format(idx))
    except Exception as e:
        print('scan file does not exist')
        scan = None

    # set odom @ t0= (0,0,0)
    R0 = Rmat(odom[0,2])
    odom -= odom[0]
    odom[:,:2] = odom[:,:2].dot(R0)

    stamps -= stamps[0] # t0 = 0

    app = CVORunner(imgs, stamps, odom, scan)
    app.run(auto=True)

if __name__ == "__main__":
    main()
