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
    restore_ckpt = os.path.expanduser('~/vo/0/ckpt/model.ckpt-1000')
    is_training = False

    # override cfg params
    cfg.BATCH_SIZE = 1
    cfg.TIME_STEPS = 1

    n_test = 8
    n_step = 64

    dm = DataManager(mode='valid', log=print)
    #dm = DataManager(mode='train', log=print)

    graph = tf.get_default_graph()
    with graph.as_default():
        net = VONet(step=None, train=is_training, cfg=cfg, log=print)
        saver = tf.train.Saver()

    config = None
    with tf.Session(graph=graph, config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, restore_ckpt)

        img, lab = dm.get(batch_size=n_test, time_steps=n_step)
        pimg = proc_img(img)
        nax = np.newaxis

        # TODO : try to also feed rnn_s0_ and stuff
        pos = []
        for t_img, t_lab in zip(pimg, lab):
            t_pos = []
            rnn_s = None
            for img1, lab1 in zip(t_img, t_lab):
                if rnn_s is None:
                    pos1, rnn_s = sess.run([net.pos_, net.rnn_s1_], 
                            {net.img_ : img1[nax,nax,...]})
                else:
                    pos1, rnn_s = sess.run([net.pos_, net.rnn_s1_], 
                            {net.img_ : img1[nax,nax,...], net.rnn_s0_ : rnn_s})
                t_pos.append(pos1[0,0])
            pos.append(t_pos)
        pos = np.float32(pos)
        print(np.shape(pos), np.shape(lab))

    # plotting results
    index = 0
    fig, (ax0, ax1) = plt.subplots(2,1)
    def handle_key(event):
        global index
        sys.stdout.flush()
        if event.key in ['q', 'escape']:
            sys.exit()
        else:
            if event.key in ['p',  'left']:
                index -= 1
            if event.key in ['n', 'right']:
                index += 1
            index = np.clip(index, 0, n_test-1)
            print('{}/{}'.format(index, n_test))
            img1 = img[index]
            lab1 = lab[index]
            pos1 = pos[index]
            dm.show(img1, lab1, fig, ax0, ax1, draw=False, label='true', color='k')
            dm.show(img1, pos1, fig, ax0, ax1, clear=False, label='pred', color='r')

    #print('pos, lab', pos, lab)
    fig.canvas.mpl_connect('close_event', sys.exit)
    fig.canvas.mpl_connect('key_press_event', handle_key)

    img1 = img[index]
    lab1 = lab[index]
    pos1 = pos[index]
    dm.show(img1, lab1, fig, ax0, ax1, draw=False, label='true', color='k')
    dm.show(img1, pos1, fig, ax0, ax1, clear=False, label='pred', color='r')

    plt.show()

if __name__ == "__main__":
    main()
