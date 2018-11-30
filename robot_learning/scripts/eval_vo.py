#!/usr/bin/env python2
from __future__ import print_function

import config as cfg
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim

import os
import sys
from vo_net import VONet
from data_manager import DataManager
from utils import anorm, proc_img
from utils.tf_utils import latest_checkpoint
from utils.kitti_utils import KittiLoader
from utils.vo_utils import VoShow

from matplotlib import pyplot as plt

def main():
    global index
    # restore/train flags
    # checkpoint file to restore from
    restore_ckpt = latest_checkpoint('~/vo')
    print('Restoring from : {}'.format(restore_ckpt))

    is_training = False

    # override cfg params
    cfg.BATCH_SIZE = 1
    cfg.TIME_STEPS = 1

    n_test = 16
    n_step = 64

    dm = KittiLoader(root='~/datasets/kitti')
    #dm = DataManager(mode='valid', log=print)
    #dm = DataManager(mode='train', log=print)

    graph = tf.get_default_graph()
    with graph.as_default():
        net = VONet(step=None, train=is_training, cfg=cfg, log=print)
        saver = tf.train.Saver()

    config = None
    with tf.Session(graph=graph, config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, restore_ckpt)

        img, lab = dm.get(batch_size=n_test, time_steps=n_step,
                target_size=(cfg.IMG_WIDTH, cfg.IMG_HEIGHT),
                aug=False
                )
        pimg = proc_img(img)
        nax = np.newaxis

        pos = []
        for t_img, t_lab in zip(pimg, lab):
            t_pos = []
            rnn_s = None
            for img1, lab1 in zip(t_img, t_lab):
                if rnn_s is None:
                    dps1, rnn_s = sess.run([net.dps_, net.rnn_s1_], 
                            {net.img_ : img1[nax,nax,...]})
                else:
                    dps1, rnn_s = sess.run([net.dps_, net.rnn_s1_], 
                            {net.img_ : img1[nax,nax,...], net.rnn_s0_ : rnn_s})
                t_pos.append(dps1[0,0])
            pos.append(t_pos)
        pos = np.float32(pos)
        print(np.shape(pos), np.shape(lab))

    data = zip(img, lab, pos)
    disp = VoShow(data, as_path=False)
    disp.show()

if __name__ == "__main__":
    main()
