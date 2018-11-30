#!/usr/bin/env python2
from __future__ import print_function

import config as cfg
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim

import os
import threading

from vo_net import VONet
from data_manager import DataManager
from utils import anorm, mkdir, proc_img, no_op
from utils.kitti_utils import KittiLoader

import signal

from tensorflow.python import debug as tf_debug

class StopRequest(object):
    def __init__(self):
        self._start = False
        self._stop = False
        signal.signal(signal.SIGINT, self.sig_cb)
    def start(self):
        self._start = True
    def sig_cb(self, signal, frame):
        self._stop = True
        if not self._start:
            sys.exit(0)

def main():
    sig = StopRequest()

    # restore/train flags
    # checkpoint file to restore from
    # restore_ckpt = '/tmp/vo/20/ckpt/model.ckpt-4'
    restore_ckpt = None
    #restore_ckpt = '/home/jamiecho/fn/15/ckpt/model.ckpt-1000'

    #fn_ckpt = None
    #fn_ckpt = os.path.expanduser('~/fn/4/ckpt/model.ckpt-80000')
    fn_ckpt = os.path.expanduser('~/fnckpt/model.ckpt-281200')

    is_training = True

    # directory
    save_root = os.path.expanduser('~/vo')
    #save_root = '~/vo'
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
    log_root_t  = os.path.join(log_root, 'train')
    log_root_v  = os.path.join(log_root, 'valid')
    mkdir(log_root)
    mkdir(log_root_t)
    mkdir(log_root_v)
    ckpt_root = os.path.join(log_root, 'ckpt')
    mkdir(ckpt_root)
    ckpt_file = os.path.join(ckpt_root, 'model.ckpt')

    dm = KittiLoader(root='~/datasets/kitti')
    #dm = DataManager(mode='train',log=print)
    dm_v = DataManager(mode='valid',log=no_op)

    graph = tf.get_default_graph()
    with graph.as_default():
        global_step = tf.train.get_or_create_global_step()

        with tf.name_scope('queue'):
            q_img = tf.placeholder(tf.float32, 
                        [None, cfg.TIME_STEPS, cfg.IMG_HEIGHT, cfg.IMG_WIDTH, cfg.IMG_DEPTH], name='q_img')
            q_lab = tf.placeholder(tf.float32, [None, cfg.TIME_STEPS, 3], name='q_lab') # label

            q_i = [q_img, q_lab]
            q_t = [e.dtype for e in q_i]
            q_s = [e.shape[1:] for e in q_i]
            Q = tf.FIFOQueue(capacity=128, dtypes=q_t, shapes=q_s)
            enqueue_op = Q.enqueue_many(q_i)
            
            img, lab = Q.dequeue_many(cfg.BATCH_SIZE)

        # ============================
        # option 1 : ramp-up -> standard
        # initial ramp-up 1e-6 -> 1e-4
        lr0 = tf.train.exponential_decay(cfg.LR_RAMP_0,
                global_step, cfg.LR_RAMP_STEPS, cfg.LEARNING_RATE/cfg.LR_RAMP_0, staircase=False)
        
        # standard decay 1e-4 -> 1e-3
        lr1 = tf.train.exponential_decay(cfg.LEARNING_RATE,
                global_step, cfg.STEPS_PER_DECAY, cfg.DECAY_FACTOR, staircase=False)
        learning_rate = tf.where(global_step < cfg.LR_RAMP_STEPS, lr0, lr1) # employ slow initial learning rate
        
        # option 2 : standard
        #learning_rate = tf.train.exponential_decay(cfg.LEARNING_RATE,
        #        global_step, cfg.STEPS_PER_DECAY, cfg.DECAY_FACTOR, staircase=True)
        # ============================

        with tf.name_scope('net_t'):
            net = VONet(global_step,
                    learning_rate=learning_rate, img=img, lab=lab,
                    batch_size=cfg.BATCH_SIZE,
                    train=is_training, log=print)
        with tf.name_scope('net_v'):
            net_v = VONet(global_step,
                    batch_size=cfg.VAL_BATCH_SIZE,
                    train=False, log=no_op, reuse=True)

        tf.summary.scalar('qsize', Q.size(), collections=['train'])
        tf.summary.scalar('learning_rate',learning_rate, collections=['train'])

        summary_t = tf.summary.merge([tf.summary.merge_all(), tf.summary.merge_all('train')])
        summary_v = tf.summary.merge_all('valid')

        writer_t = tf.summary.FileWriter(log_root_t, graph)
        writer_v = tf.summary.FileWriter(log_root_v, graph)
        saver = tf.train.Saver()

    #gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.95)
    #config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
    config = None

    with tf.Session(graph=graph, config=config) as sess:
        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        coord = tf.train.Coordinator()
        def enqueue():
            q_ts = [q_img, q_lab] # input tensors
            while not coord.should_stop():
                q_vs = dm.get(batch_size=8, time_steps=cfg.TIME_STEPS, aug=True,
                        as_path=False, target_size=(cfg.IMG_WIDTH,cfg.IMG_HEIGHT)
                        )
                q_vs[0] = proc_img(q_vs[0])
                sess.run(enqueue_op, feed_dict={t:v for (t,v) in zip(q_ts, q_vs)})

        # initialization
        sess.run(tf.global_variables_initializer())
        if fn_ckpt is not None:
            fn_vars = [v for v in slim.get_model_variables() if ('vo/sconv' in v.name) or ('vo/conv' in v.name)]
            print('flownet vars', fn_vars)
            fn_saver = tf.train.Saver(var_list=fn_vars)
            fn_saver.restore(sess, fn_ckpt)

        if restore_ckpt is not None:
            saver.restore(sess, restore_ckpt)

        i = i0 = sess.run(global_step)
        q_threads = [threading.Thread(target=enqueue) for _ in range(cfg.Q_THREADS)]
        for t in q_threads:
            t.daemon = True
            t.start()

        sig.start()

        for i in range(i0, cfg.TRAIN_STEPS):
            if sig._stop:
                break

            # ====================
            # option 1: usual training
            # usual training
            # img, lab = dm.get(batch_size=cfg.BATCH_SIZE, time_steps=cfg.TIME_STEPS, aug=True,
            #         as_path=False, target_size=(cfg.IMG_WIDTH, cfg.IMG_HEIGHT)
            #         )
            # img = proc_img(img)
            # s, err, _ = sess.run([summary_t, net.err_, net.opt_], {net.img_ : img, net.lab_ : lab})

            # option 2 : queue-based
            s, err, _ = sess.run([summary_t, net.err_, net.opt_])
            writer_t.add_summary(s, i)
            # =====================

            if (i>0) and (i%cfg.VAL_STEPS)==0:
                # validation
                img, lab = dm_v.get(batch_size=cfg.VAL_BATCH_SIZE, time_steps=cfg.TIME_STEPS, aug=True,
                        as_path=False, target_size=(cfg.IMG_WIDTH, cfg.IMG_HEIGHT)
                        )
                img = proc_img(img)
                s_v, err_v = sess.run([summary_v, net_v.err_],
                        {net_v.img_ : img, net_v.lab_ : lab})
                writer_v.add_summary(s_v, i)

            if (i>0) and (i%cfg.SAVE_STEPS)==0:
                # save
                saver.save(sess, ckpt_file, global_step=i)

        coord.request_stop()
        sess.run([Q.close(cancel_pending_enqueues=True)])
        coord.join(q_threads)

        saver.save(sess, ckpt_file, global_step=global_step)
        #tf.saved_model.simple_save(session,
        #        '/tmp',
        #        inputs={'img':net.img_},
        #        outputs={'pos':net.pos_})

if __name__ == "__main__":
    main()
