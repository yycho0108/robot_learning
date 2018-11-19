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

from matplotlib import pyplot as plt

def main():
    global index
    # restore/train flags
    # checkpoint file to restore from
    # restore_ckpt = '/tmp/vo/20/ckpt/model.ckpt-4'
    restore_ckpt = '/tmp/vo/1/ckpt/model.ckpt-1000'
    is_training = False

    dm = DataManager(mode='valid', log=print)

    graph = tf.get_default_graph()
    with graph.as_default():
        net = VONet(step=None, train=is_training, log=print)
        saver = tf.train.Saver()

    config = None
    with tf.Session(graph=graph, config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, restore_ckpt)

        img, lab = dm.get(batch_size=cfg.BATCH_SIZE, time_steps=cfg.TIME_STEPS)
        pimg = proc_img(img)

        # TODO : try to also feed rnn_s0_ and stuff
        pos, rnn_s1 = sess.run([net.pos_, net.rnn_s1_], 
                {net.img_ : pimg})

    # plotting results
    index = 0
    fig, (ax0, ax1) = plt.subplots(2,1)
    def handle_key(event):
        global index
        sys.stdout.flush()
        if event.key == 'q':
            sys.exit()
        else:
            if (index >= cfg.BATCH_SIZE): sys.exit(0)
            print('{}/{}'.format(index, cfg.BATCH_SIZE))
            img1 = img[index]
            lab1 = lab[index]
            pos1 = pos[index]
            dm.show(img1, lab1, fig, ax0, ax1, draw=False, label='true')
            dm.show(img1, pos1, fig, ax0, ax1, clear=False, label='pred')
            index += 1

    #print('pos, lab', pos, lab)
    fig.canvas.mpl_connect('close_event', sys.exit)
    fig.canvas.mpl_connect('key_press_event', handle_key)
    plt.show()

if __name__ == "__main__":
    main()
