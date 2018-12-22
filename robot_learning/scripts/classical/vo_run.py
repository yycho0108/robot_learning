#!/usr/bin/env python2

import numpy as np
from numpy import s_
import sys
import cv2
from scipy.optimize import linear_sum_assignment
from matplotlib import pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D

from tf import transformations as tx


from collections import deque

try:
  from pathlib import Path
except ImportError:
  from pathlib2 import Path  # python 2 backport

from vo3 import ClassicalVO
from ukf import build_ukf
from gui import VoGUI

sys.path.append('../')
from utils.vo_utils import add_p3d, sub_p3d
from utils.kitti_utils import KittiLoader
from misc import Rmat


# ukf = (x,y,h,v,w)

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
        self.ukf_ = build_ukf()
        self.vo_(imgs[0], [0,0,0]) # initialize GUI

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
        ukf   = self.ukf_
        vo    = self.vo_
        tx, ty, th = self.tx_, self.ty_, self.th_

        rec_path = np.stack([tx,ty,th], axis=-1)

        VoGUI.draw_img(ax0, aimg[..., ::-1])
        ax0.set_title('Tracking Visualization')
        VoGUI.draw_top(ax1, rec_path, pts2, odom[:i+1], scan_c, cov)
        #VoGUI.draw_top(ax1, rec_path, pts2, np.stack([tx,ty,th], axis=-1), scan_c)
        VoGUI.draw_3d(ax2, pts3)
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

    def get_QR(self, pose, dt):
        # Get appropriate Q/R Matrices from current pose.
        # Mostly just deals with getting the right orientation.
        Q0 = np.diag(np.square([3e-2, 5e-3, 1e-1, 3e-3, 1e-3, 2e-2]))
        R0 = np.diag(np.square([5e-2, 7e-2, 4e-2]))
        T = Rmat(pose[-1])

        Q = Q0.copy()

        # constant acceleration model?
        # NOTE : currently results in non-positive-definite matrix
        #g = [dt**2/2, dt]
        #G = np.outer(g,g)
        # x-part
        # Q[np.ix_([0,3],[0,3])] = G * (0.3)**2 # 10 cm/s^2
        # Q[np.ix_([1,4],[1,4])] = G * (0.1)**2 # 2 cm/s^2
        # Q[np.ix_([2,5],[2,5])] = G * (0.5)**2 # ~ 6 deg/s^2

        # apply rotation to translational parts
        Q[:2,:2] = T.dot(Q[:2,:2]).dot(T.T)
        #Q[3:5,3:5] = T.dot(Q[3:5,3:5]).dot(T.T)
        #NOTE: vx-vy components are invariant to current pose

        #Q = (Q+Q.T)/2.0
        #if np.any(Q<0):
        #    Q -= np.min(Q)

        # R has nothing to do with time
        R = R0.copy()
        R[:2,:2] = T.dot(R[:2,:2]).dot(T.T)

        return Q, R

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
        ukf   = self.ukf_
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
        #s = 0.2
        Q,R = self.get_QR(ukf.x[:3], dt)
        ukf.Q=Q
        ukf.R=R
        #ukf.P=np.abs(ukf.P)

        prv = ukf.x[:3].copy()
        try:
            ukf.predict(dt=dt)
        except Exception as e:
            print 'Wat? {}'.format(e)
            print 'Q', ukf.Q
            print 'R', ukf.R
            print 'x', ukf.x
            print 'P', ukf.P
            raise e
        # TODO : currently passing 'ground-truth' position
        #suc, res = vo(img, odom[i], s=s)
        suc, res = vo(img, ukf.x[:3].copy(), scale=s)
        if not suc:
            print('Visual Odometry Aborted!')
            return

        if res is None:
            # skip filter updates
            return

        (aimg, vo_h, vo_t, pts_r, pts3, msg) = res
        #dps = np.float32([dt[0], dt[1], dh])
        #print('dh', np.rad2deg(dh))
        #print('(pred-gt) {} vs {}'.format(dps, dps_gt) )
        #pos = add_p3d(prv, dps)
        pos = [vo_t[0], vo_t[1], vo_h]
        ukf.update(pos)

        tx.append( float(ukf.x[0]) )
        ty.append( float(ukf.x[1]) )
        th.append( float(ukf.x[2]) )
        #tx.append( vo_t[0] )
        #ty.append( vo_t[1] )
        #th.append( vo_h    )

        pts2 = pts3[:,:2]
        self.map_ = np.concatenate([self.map_, pts2], axis=0)
        if scan is not None:
            scan_c = self.scan_to_pt(scan[i]).dot(Rmat(odom[i,2]).T) + np.reshape(odom[i, :2], (1,2))
        else:
            scan_c = None

        ### EVERYTHING FROM HERE IS PLOTTING + VIZ ###
        self.show(aimg, pts3, pts2, scan_c, pts_r, ukf.P, ('[%d/%d] '%(i,n)) + msg)

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
                #plt.savefig('/tmp/{:03d}.png'.format(self.index_))
        else:
            plt.show()

def main():
    #idx = np.random.choice(8)
    idx = 23
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
