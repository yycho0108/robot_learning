#!/usr/bin/env python2
from __future__ import print_function

import config as cfg
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim

import os
from vo_net import VONet
from data_manager import DataManager
from utils import anorm, mkdir, proc_img

def main():
    # restore/train flags
    # checkpoint file to restore from
    # restore_ckpt = '/tmp/vo/20/ckpt/model.ckpt-4'
    restore_ckpt = None
    is_training = True

    # directory
    save_root = '/tmp/vo'
    mkdir(save_root)

    # resolve current run id + directory
    try:
        run_id = len(os.listdir(save_root))
    except Exception as e:
        run_id = 0
    run_id = str(run_id)
    #run_root = os.path.join(save_root, run_id)
    #mkdir(run_root)

    # resolve log + ckpt sub-directories
    log_root  = os.path.join(save_root, run_id)
    mkdir(log_root)
    ckpt_root = os.path.join(log_root, 'ckpt')
    mkdir(ckpt_root)
    ckpt_file = os.path.join(ckpt_root, 'model.ckpt')


    dm = DataManager(mode='train',log=print)

    graph = tf.get_default_graph()
    with graph.as_default():
        global_step = slim.get_or_create_global_step()
        net = VONet(global_step, train=is_training, log=print)

        learning_rate = tf.train.exponential_decay(cfg.LEARNING_RATE,
                global_step, cfg.STEPS_PER_DECAY, cfg.DECAY_FACTOR, staircase=True)
        learning_rate = tf.where(global_step < 50, 1e-6, learning_rate) # employ slow initial learning rate
        tf.summary.scalar('learning_rate',learning_rate)

        summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter(log_root, graph)
        saver = tf.train.Saver()

    config = None
    with tf.Session(graph=graph, config=config) as sess:
        sess.run(tf.global_variables_initializer())
        if restore_ckpt is not None:
            saver.restore(sess, restore_ckpt)
        i = i0 = sess.run(global_step)

        for i in range(i0, cfg.TRAIN_STEPS):
            img, lab = dm.get(batch_size=cfg.BATCH_SIZE, time_steps=cfg.TIME_STEPS)
            img = proc_img(img)
            s, err, _ = sess.run([summary, net.err_, net.opt_],
                    {net.img_ : img, net.lab_ : lab})
            writer.add_summary(s, i)
            #err, _, rnn_s1 = sess.run([net.err_, net.opt_, net.rnn_s1_], 
            #        {net.img_ : img, net.lab_ : lab})
            ##print('rs1',np.shape(rnn_s1))
            #err, _ = sess.run([net.err_, net.opt_], 
            #        {net.img_ : img, net.lab_ : lab, net.rnn_s0_ : rnn_s1})
            #print('err', err)
            if (i>0) and (i%cfg.SAVE_STEPS)==0:
                saver.save(sess, ckpt_file, global_step=i)

        saver.save(sess, ckpt_file, global_step=global_step)
        #tf.saved_model.simple_save(session,
        #        '/tmp',
        #        inputs={'img':net.img_},
        #        outputs={'pos':net.pos_})

if __name__ == "__main__":
    main()
