#!/usr/bin/env python2

import numpy as np
from scipy.misc import imread
import cv2
import tensorflow as tf
import os
import sys
import time

from utils.fchair_utils import load_chair, load_ilsvrc
from utils.opt_utils import apply_opt, flow_to_image, FlowShow
from utils import mkdir, proc_img
from flow_net_bb import FlowNetBB

#root = '/home/jamiecho/Repos/tf_flownet2'
#sys.path.append(root)
#from FlowNet2_src import flow_to_image

# Visualization
from matplotlib import pyplot as plt

import config as cfg

class GetOptFlowNet(object):
    def __init__(self):
        pass

    def _build(self):
        # Graph construction
        #im1_pl = tf.placeholder(tf.float32, [None, None, None, 3])
        #im2_pl = tf.placeholder(tf.float32, [None, None, None, 3])
        #im1_in = tf.image.resize_images(im1_pl, (384,512))
        #im2_in = tf.image.resize_images(im2_pl, (384,512))

        im1_in = im1_pl = tf.placeholder(tf.float32, [None, cfg.IMG_HEIGHT, cfg.IMG_WIDTH, cfg.IMG_DEPTH])
        im2_in = im2_pl = tf.placeholder(tf.float32, [None, cfg.IMG_HEIGHT, cfg.IMG_WIDTH, cfg.IMG_DEPTH])
        net = FlowNetBB(step=None, img=tf.stack([im1_in, im2_in],axis=1),
                lab=None, train=False, eval=False)

        self.im1_ = im1_pl
        self.im2_ = im2_pl
        self.flow_ = net.pred_

    def load(self, sess, ckpt):
        tf.train.Saver().restore(sess, ckpt)

    def __call__(self, sess,
            im1, im2):
        return sess.run(self.flow_, {self.im1_:im1,self.im2_:im2})

def normalize(x, mn=0.0, mx=1.0):
    xmn = np.min(x)
    xmx = np.max(x)
    return (x-xmn)*((mx-mn)/(xmx-xmn)) + mn

def main():
    graph = tf.Graph()
    with graph.as_default():
        net = GetOptFlowNet()
        net._build()

    ckpt_file = os.path.expanduser('~/fn/37/ckpt/model.ckpt-12700')
    #gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.95)
    #config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
    config=None

    n_test  = 32
    n_batch = 8
    n_split = int(np.round(n_test / float(n_batch)))

    ilsvrc_root = os.path.expanduser('~/dispset/data')
    chair_root = os.path.expanduser('~/Downloads/FlyingChairs/data')

    with tf.Session(graph=graph, config=config) as sess:
        net.load(sess, ckpt_file)

        #img1, img2, gt_flow = load_chair(chair_root, n=n_test, size=(cfg.IMG_WIDTH, cfg.IMG_HEIGHT))
        img1, img2, gt_flow = load_ilsvrc(ilsvrc_root, n=n_test, size=(cfg.IMG_WIDTH, cfg.IMG_HEIGHT))
        imgs = np.stack([img1,img2], axis=1)
        p_imgs = proc_img(imgs)

        flow = []
        sb_imgs = np.array_split(p_imgs, n_split, axis=0)
        for i, b_imgs in enumerate(sb_imgs):
            flow_tmp = net(sess, b_imgs[:,0], b_imgs[:,1])
            print('{}/{} : {}'.format(i, n_split, len(b_imgs)))
            flow.append(flow_tmp)
        flow = np.concatenate(flow, axis=0)
        print('complete?')

    disp = FlowShow(n=3, m=2,
            code_path='utils/middlebury_flow_code.png'
            )
    disp.configure([
        [FlowShow.AX_IMG1, FlowShow.AX_IMG2],
        [FlowShow.AX_I2I1, FlowShow.AX_OVLY],
        [FlowShow.AX_FLOW, FlowShow.AX_CODE]])
    disp.add(img1, img2, flow)
    disp.show()

if __name__ == '__main__':
    main()
    # Feed forward
