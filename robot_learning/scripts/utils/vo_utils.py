#!/usr/bin/env python2
from __future__ import print_function

import numpy as np
from matplotlib import pyplot as plt
import sys
from utils import anorm, no_op
from tf import transformations as tx
import imgaug as ia
from imgaug import augmenters as iaa

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

def add_p3d_batch(a, b):
    x0,y0,h0 = a.T
    dx,dy,dh = b.T
    c, s = np.cos(h0), np.sin(h0)
    dx_R = (c*dx - s*dy)
    dy_R = (s*dx + c*dy)

    x = x0+dx_R
    y = y0+dy_R
    h = anorm(h0+dh)
    return np.stack([x,y,h], axis=-1)

def dps2pos(dps):
    # construct path
    p0 = np.zeros_like(dps[0])
    ps = [p0]
    for dp in dps[1:]:
        #p = add_p3d(ps[-1], dp)
        p = add_p3d_batch(ps[-1], dp)
        ps.append(p)
    ps = np.float32(ps)
    return ps

def to_pose2d(pos):
    # pos = N?x3x4 transformation matrix
    # 2d rotation requires (-y) component
    # 2d translation requires (z, -x) component
    
    # == test ==
    # -- opt 1 : einsum --
    T_pre = tx.euler_matrix(-np.pi/2,0,-np.pi/2) #4x4
    T_0i  = tx.inverse_matrix(np.concatenate((pos[0], [[0,0,0,1]]), axis=0)) #4x4
    T_cam  = np.stack([np.concatenate((p, [[0,0,0,1]]), axis=0) for p in pos], axis=0)
    pos_n = np.einsum('ij,jk,nkl,lm->nim', T_pre, T_0i, T_cam, T_pre.T) # normalized position
    res = []
    for p in pos_n:
        t = tx.translation_from_matrix(p)
        q = tx.euler_from_matrix(p)
        res.append([ t[0], t[1], q[-1] ])
    res = np.stack(res, axis=0)
    return res
    # --------------------

    # -- opt 2 : no einsum --
    #T_pre = tx.euler_matrix(-np.pi/2,0,-np.pi/2) #4x4
    #T_0i  = tx.inverse_matrix(np.concatenate((pos[0], [[0,0,0,1]]), axis=0)) #4x4
    #T_cam  = np.stack([np.concatenate((p, [[0,0,0,1]]), axis=0) for p in pos], axis=0)
    #res = []
    #for p in pos:
    #    T_cam  = np.concatenate((p, [[0,0,0,1]]), axis=0)
    #    T_base = T_pre.dot(T_0i).dot(T_cam).dot(T_pre.T)
    #    t = tx.translation_from_matrix(T_base)
    #    q = tx.euler_from_matrix(T_base)
    #    res.append([t[0], t[1], q[-1]])
    #res = np.stack(res, axis=0)
    return res
    # -----------------------
    # ==========

    # -- opt 3 : WRONG - do not use --
    #R, T = np.split(pos, (3,), axis=-1)
    #T = np.squeeze(T, axis=-1) # N?x3

    ## normalize
    #T = T - T[:1]
    #R = (R[0].T).dot(R.transpose(1,2,0)).transpose(2,0,1) #3x3 . Nx3x3

    #ry = np.arctan2(-R[..., 2, 0],
    #        np.sqrt(np.square(R[...,2,1]) + np.square(R[...,2,2])))

    #h = -ry

    #x,y,z = T.T
    ##t = np.stack([z,-x], axis=-1)
    ##return t, h

    #p = np.stack([z,-x,h], axis=-1)
    #p = p - p[0,:]
    #p[:,2] = anorm(p[:,2])
    ##p = np.stack([x,y,h], axis=-1)
    # --------------------------------

class VOAug(object):
    def __init__(self):
        self.seq_ = iaa.Sequential(
            [
                # execute 0 to 5 of the following (less important) augmenters per image
                # don't execute all of them, as that would often be way too strong
                iaa.SomeOf((0, 5),
                    [
                        iaa.OneOf([
                            iaa.GaussianBlur((0, 2.0)), # blur images with a sigma between 0 and 3.0
                            iaa.AverageBlur(k=(3, 5)), # blur image using local means with kernel sizes between 2 and 7
                            iaa.MedianBlur(k=(3, 7)), # blur image using local medians with kernel sizes between 2 and 7
                        ]),
                        #iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                        # search either for all edges or for directed edges,
                        # blend the result with the original image using a blobby mask
                        #iaa.SimplexNoiseAlpha(iaa.OneOf([
                        #    iaa.EdgeDetect(alpha=(0.5, 1.0)),
                        #    iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                        #])),
                        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images

                        iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                        iaa.AddToHueAndSaturation((-10, 10)), # change hue and saturation
                        # either change the brightness of the whole image (sometimes
                        # per channel) or change the brightness of subareas
                        iaa.OneOf([
                            iaa.Multiply((0.75, 1.25), per_channel=0.5),
                            #iaa.FrequencyNoiseAlpha(
                            #    exponent=(-4, 0),
                            #    first=iaa.Multiply((0.5, 1.5), per_channel=True),
                            #    second=iaa.ContrastNormalization((0.5, 2.0))
                            #)
                        ]),
                        iaa.ContrastNormalization((0.75, 1.33), per_channel=0.5), # improve or worsen the contrast
                    ],
                    random_order=True
                )
            ],
            random_order=True
        )
    def __call__(self, imgs):
        n,h,w,c = imgs.shape
        return self.seq_.augment_image(imgs.reshape(n*h,w,c)).reshape(n,h,w,c)

batch_augment = VOAug()

class VoShow(object):
    def __init__(self, data, as_path=True):
        self.data_ = data
        self.as_path_ = as_path

    def _draw(self, ax, ps, label='path', color=None):
        ax.plot(ps[:,0], ps[:,1], '--', label=label,color=color) # starting point
        ax.quiver(
                ps[:,0], ps[:,1],
                np.cos(ps[:,2]), np.sin(ps[:,2]),
                scale_units='xy',
                angles='xy',
                color=color
                )

    def draw(self, clear=True, draw=True, label='path'):
        # unroll data
        i = self.index_

        t_pred = None
        if len(self.data_[i]) == 2:
            t_imgs, t_labs = self.data_[i]
        elif len(self.data_[i]) == 3:
            t_imgs, t_labs, t_pred = self.data_[i]

        fig = self.fig_
        ax0, ax1 = self.ax_

        cat = np.concatenate(t_imgs, axis=1)
        next = False

        # show path
        if clear:
            ax0.cla()
            ax1.cla()


        # construct path
        if self.as_path_:
            # data came in as path
            ps = t_labs
        else:
            ps = dps2pos(t_labs)
        self._draw(ax0, ps, 'path')

        if t_pred is not None:
            if self.as_path_:
                # data came in as path
                ps = t_pred
            else:
                ps = dps2pos(t_pred)
            self._draw(ax0, ps, 'pred', color='r')

        ax0.set_aspect('equal', 'datalim')
        #ax0.set_xlim([-0.2,1.0])
        #ax0.set_ylim([-0.4,0.4])
        ax1.imshow(cat)
        fig.suptitle('{}/{}'.format(self.index_+1, len(self.data_)))
        ax0.legend()
        if draw:
            fig.canvas.draw()

    def show(self):
        fig, ax = plt.subplots(2,1)
        
        self.fig_ = fig
        self.ax_ = ax

        n = len(self.data_)
        self.index_ = 0

        print('- instructions -')
        print('q to quit; any other key to inspect next sample')
        print('----------------')

        def handle_key(event):
            sys.stdout.flush()
            index = self.index_
            if event.key in ['q', 'escape']:
                sys.exit()
            elif event.key in [' ', 'n', 'right']:
                index += 1
            elif event.key in ['p', 'left', 'backspace']:
                index -= 1
            print('{}/{}'.format(index,n))
            index = index % n
            self.index_ = index
            self.draw()

        # register event handlers
        fig.canvas.mpl_connect('close_event', sys.exit)
        fig.canvas.mpl_connect('key_press_event', handle_key)

        # show
        self.draw()
        plt.show()
