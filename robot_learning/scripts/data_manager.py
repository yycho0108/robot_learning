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

from utils.vo_utils import sub_p3d, add_p3d, add_p3d_batch, dps2pos, batch_augment, VoShow

class DataManager(object):
    def __init__(self, dirs=None, mode='train', log=no_op):
        if dirs is None:
            # automatically resolve directory
            rospack   = rospkg.RosPack() 
            pkg_root  = rospack.get_path('robot_learning') # Gets the package
            data_root = os.path.join(pkg_root, 'data', mode)
            subdir = os.listdir(data_root)
            dirs = [os.path.join(data_root, d) for d in subdir]

        dirs = np.sort(dirs)

        self.data_ = [self.load(d) for d in dirs]

        # hack to split data
        #img, odom =  self.data_[0]
        #n = len(img)
        #np.save('/tmp/img0.npy', img[:n/2])
        #np.save('/tmp/odom0.npy', odom[:n/2])
        #np.save('/tmp/img1.npy', img[n/2:])
        #np.save('/tmp/odom1.npy', odom[n/2:])

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

        # augmentation functor

    def load(self, path):
        img   = np.load(os.path.join(path, 'img.npy'))
        img   = img[..., ::-1] # stored as BGR : convert to RGB
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

    def get_1(self, data, time_steps, flip=False, target_size=None):
        img, lab = data
        i0 = np.random.randint(0, high=len(img)-time_steps)
        img, lab = img[i0:i0+time_steps], lab[i0:i0+time_steps]

        if flip:
            # flip left-right
            img = np.copy(img[:,:,::-1])
            # flip sign for dy-dh
            lab = np.copy(lab)
            lab[:,1:] = lab[:,1:] * -1.0

        if target_size is not None:
            img = [cv2.resize(e, target_size) for e in img]
            img = np.stack(img, axis=0)

        return [img, lab]

    def get(self, batch_size, time_steps, aug=True,
            as_path=False, target_size=None
            ):
        set_idx = np.random.choice(len(self.data_),
                batch_size, replace=True, p=self.prob_)
        lr_flip = np.random.choice(2, batch_size, replace=True).astype(np.bool)
        if not aug:
            lr_flip = np.zeros_like(lr_flip)
        data = [self.get_1(self.data_[i], time_steps, f, target_size) for (i,f) in zip(set_idx, lr_flip)]
        img, lab = zip(*data)
        if aug:
            # TODO : consider using augmentation from utils/img_utils.py
            img = np.stack([batch_augment(timg) for timg in img], axis=0)
            #img  = np.stack(img, axis=0)
        else:
            img = np.stack(img, axis=0) # [NxTxHxWxC]
        lab = np.stack(lab, axis=0) # [Nx3]
        if as_path:
            lab = dps2pos(lab)
        return [img, lab]

    def get_null(self, batch_size, time_steps):
        x = np.zeros([cfg.BATCH_SIZE,cfg.TIME_STEPS,cfg.IMG_HEIGHT,cfg.IMG_WIDTH,cfg.IMG_DEPTH])
        y = np.zeros([cfg.BATCH_SIZE,cfg.TIME_STEPS,3])
        return [x, y]

    @staticmethod
    def show(t_imgs, t_labs, fig, ax0, ax1, clear=True, draw=True, label='path', color=None):
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
        if clear:
            ax0.cla()
            ax1.cla()

        ax0.plot(ps[:,0], ps[:,1], '--', label=label,color=color) # starting point
        ax0.quiver(
                ps[:,0], ps[:,1],
                np.cos(ps[:,2]), np.sin(ps[:,2]),
                scale_units='xy',
                angles='xy',
                color=color
                )
        #ax0.set_xlim([-0.2,1.0])
        #ax0.set_ylim([-0.4,0.4])
        ax1.imshow(cat)
        ax0.legend()
        if draw:
            fig.canvas.draw()

    def inspect(self, n=100):
        global index
        fig, (ax0, ax1) = plt.subplots(2,1)
        bt_imgs, bt_labs = self.get(batch_size=n, time_steps=4) # batch-time
        index = 0
        print('- instructions -')
        print('q to quit; any other key to inspect next sample')
        print('----------------')

        def handle_key(event):
            global index
            sys.stdout.flush()
            if event.key == 'q':
                sys.exit()
            else:
                index += 1
                print('{}/{}'.format(index,n))
                if (index >= n): sys.exit(0)
                t_imgs = bt_imgs[index]
                t_labs = bt_labs[index]
                DataManager.show(t_imgs, t_labs, fig, ax0, ax1)

        # register event handlers
        fig.canvas.mpl_connect('close_event', sys.exit)
        fig.canvas.mpl_connect('key_press_event', handle_key)

        # show
        DataManager.show(bt_imgs[0], bt_labs[0], fig, ax0, ax1)
        plt.show()

def main():
    as_path = False
    target_size = (256, 192)

    # opt 1.0 : all training data
    # dirs = None

    # opt 1.1 : specify subdir/dirs/ ...
    rospack   = rospkg.RosPack() 
    pkg_root  = rospack.get_path('robot_learning') # Gets the package
    #data_root = os.path.join(pkg_root, 'data', 'valid')
    data_root = os.path.join(pkg_root, 'data', 'valid')
    #subdir = os.listdir(data_root)
    subdir = ['0']
    dirs = [os.path.join(data_root, d) for d in subdir]

    dm = DataManager(dirs=dirs, log=print)

    img, dps = dm.get(batch_size=32, time_steps=16,
            as_path=as_path,
            target_size=target_size)

    data = zip(img, dps)
    disp = VoShow(data, as_path=as_path)
    disp.show()

    # opt 2.0 : data augmentation observation
    #s = np.random.randint(65536)
    #np.random.seed(s)
    #img1, lab1 = dm.get_1(dm.data_[0],4,flip=False)
    #np.random.seed(s)
    #img2, lab2 = dm.get_1(dm.data_[0],4,flip=True)

    #fig, ((ax0, ax2), (ax1, ax3)) = plt.subplots(2,2)
    #dm.show(img1, lab1, fig, ax0, ax1, draw=False,  label='orig', color='k')
    #dm.show(img2, lab2, fig, ax2, ax3, clear=False, label='flip', color='k')
    #plt.show()

    # opt 2.1 : overall inspection
    #dm.inspect()

if __name__ == "__main__":
    main()
