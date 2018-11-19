#!/usr/bin/env python2
from __future__ import print_function

import config as cfg
import rospkg
import numpy as np
import os
import cv2
from matplotlib import pyplot as plt
from functools import partial
import sys
from utils import anorm, no_op

def sub_p3d(b,a):
    x0,y0,h0 = a
    x1,y1,h1 = b

    dh = anorm(h1-h0)
    dx = x1-x0
    dy = y1-y0

def add_p3d(a,b):
    # final p3d composition -- mostly for verification
    x0,y0,h0 = a
    dx,dy,dh = b
    c, s = np.cos(h0), np.sin(h0)
    R = np.reshape([c,-s,s,c], [2,2]) # [2,2,N]
    dp = R.dot([dx,dy])
    x1 = x0 + dp[0]
    y1 = y0 + dp[1]
    h1 = anorm(h0 + dh)
    return [x1,y1,h1]

class DataManager(object):
    def __init__(self, dirs=None, log=no_op):
        if dirs is None:
            # automatically resolve directory
            rospack   = rospkg.RosPack() 
            pkg_root  = rospack.get_path('robot_learning') # Gets the package
            data_root = os.path.join(pkg_root, 'data')
            subdir = os.listdir(data_root)
            dirs = [os.path.join(data_root, d) for d in subdir]

        dirs = np.sort(dirs)

        self.data_ = [self.load(d) for d in dirs]
        self.data_ = [self.format(*d) for d in self.data_]

        deltas = [d[1] for d in self.data_]
        log('- dataset stats -')
        log('max : {}'.format(np.concatenate(deltas, axis=0).max(axis=0)))
        log('min : {}'.format(np.concatenate(deltas, axis=0).min(axis=0)))

        # get dataset statistics
        dlen = [len(d[0])-cfg.TIME_STEPS for d in self.data_]
        basename = [os.path.basename(os.path.normpath(d)) for d in dirs]
        log('Dataset Source : {}'.format(basename))
        log('Dataset Length : {}'.format(dlen))
        log('Total : {}'.format(sum(dlen)))
        log('-----------------')

        # dataset selection probability
        self.prob_ = np.float32(dlen)
        self.prob_ /= self.prob_.sum()

    def load(self, path):
        img   = np.load(os.path.join(path, 'img.npy'))
        odom  = np.load(os.path.join(path, 'odom.npy'))
        return img, odom

    def format(self, img, odom):
        # format the data to be compatible with the training network
        o = odom
        prv = o[:-1]
        nxt = o[1:]
        #x1,y1,h1 w.r.t x0,y0,h0
        delta = nxt - prv #[N,2]
        h0 = prv[:,2] 
        c, s = np.cos(h0), np.sin(h0)
        R = np.reshape([c,-s,s,c], [2,2,-1]) # [2,2,N]
        dp = np.einsum('ijk,ki->kj', R, delta[:,:2])
        dh = anorm(delta[:,2:])
        
        delta = np.concatenate([dp,dh], axis=-1)
        delta = np.concatenate([np.zeros_like(delta[0:1]), delta], axis=0)
        return img, delta

    def get_1(self, data, time_steps):
        img, lab = data
        i0 = np.random.randint(0, high=len(img)-time_steps)
        return img[i0:i0+time_steps], lab[i0:i0+time_steps]

    def get(self, batch_size, time_steps):
        set_idx = np.random.choice(len(self.data_),
                batch_size, replace=True, p=self.prob_)
        data = [self.get_1(self.data_[i], time_steps) for i in set_idx]
        img, lab = zip(*data)

        img = np.stack(img, axis=0) # [NxHxWxC]
        lab = np.stack(lab, axis=0) # [Nx3]
        return img, lab 

    def get_null(self, batch_size, time_steps):
        x = np.zeros([cfg.BATCH_SIZE,cfg.TIME_STEPS,cfg.IMG_HEIGHT,cfg.IMG_WIDTH,cfg.IMG_DEPTH])
        y = np.zeros([cfg.BATCH_SIZE,cfg.TIME_STEPS,3])
        return x, y

    def inspect(self,n=100):
        global index
        fig, (ax0, ax1) = plt.subplots(2,1)
        bt_imgs, bt_labs = self.get(batch_size=n, time_steps=4) # batch-time
        print(np.max(np.abs(bt_labs), axis=(0,1)))
        index = 0

        def show(i):
            t_imgs = bt_imgs[i]
            t_labs = bt_labs[i]

            cat = np.concatenate(t_imgs, axis=1)
            next = False

            # construct path
            p0 = np.zeros_like(t_labs[0])
            ps = [p0]
            for dp in t_labs[1:]:
                p = add_p3d(ps[-1], dp)
                ps.append(p)
            ps = np.float32(ps)

            # show path
            ax0.cla()
            ax1.cla()

            ax0.set_title('Data {}/{}'.format(i+1,n))
            ax0.plot(ps[:,0], ps[:,1], '--') # starting point
            ax0.quiver(
                    ps[:,0], ps[:,1],
                    np.cos(ps[:,2]), np.sin(ps[:,2]),
                    scale_units='xy',
                    angles='xy'
                    )
            ax1.imshow(cat[...,::-1])
            fig.canvas.draw()

        def handle_key(event):
            global index
            sys.stdout.flush()
            if event.key == 'q':
                sys.exit()
            else:
                index += 1
                if (index >= n): sys.exit(0)
                show(index)

        # register event handlers
        fig.canvas.mpl_connect('close_event', sys.exit)
        fig.canvas.mpl_connect('key_press_event', handle_key)

        # show
        show(0)
        plt.show()

def main():
    rospack   = rospkg.RosPack() 
    pkg_root  = rospack.get_path('robot_learning') # Gets the package
    data_root = os.path.join(pkg_root, 'data')
    subdir = os.listdir(data_root)
    #subdir = ['7']
    dirs = [os.path.join(data_root, d) for d in subdir]
    dm = DataManager(dirs=dirs)
    dm.inspect()

if __name__ == "__main__":
    main()
