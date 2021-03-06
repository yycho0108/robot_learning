#!/usr/bin/env python2
from __future__ import print_function

import config as cfg
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim

import os
import threading

from flow_net_bb import FlowNetBB
from data_manager import DataManager
from utils import anorm, mkdir, proc_img, no_op, proc_img_tf
from utils.ilsvrc_utils import ILSVRCLoader
from utils.fchair_utils import load_chair, load_ilsvrc
from utils.img_utils import augment_image_color, augment_image_affine

import sys
import signal

class StopRequest(object):
    """
    Thin wrapper around custom handling for SIGINT.
    """
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

def load_data(
        data_root,
        sample_ratio=0.35):
    """
    (Deprecated due to large memory footprint)
    Load Image from ILSVRC Indexing Cache.
    """
    data_img1 = []
    data_img2 = []
    data_pred = []
    #for i in sel:
    for i in range(1,31):
        print('loading {}/{}'.format(i, 30))
        data_subdir = os.path.join(data_root, str(i))
        img1 = (np.load(os.path.join(data_subdir, 'img1.npy')))
        img2 = (np.load(os.path.join(data_subdir, 'img2.npy')))
        pred = (np.load(os.path.join(data_subdir, 'pred.npy')))

        n = len(img1)
        idx = np.random.choice(n, int(n * sample_ratio), replace=False)
        data_img1.append(img1[idx])
        data_img2.append(img2[idx])
        data_pred.append(pred[idx])

    data_img1 = np.concatenate(data_img1, axis=0)
    data_img2 = np.concatenate(data_img2, axis=0)
    data_pred = np.concatenate(data_pred, axis=0)
    dlen = len(data_pred)
    print(np.shape(data_img1))
    print(np.shape(data_img2))
    print(np.shape(data_pred))
    return data_img1, data_img2, data_pred, dlen

def main():
    """
    Primary training routine;
    currently, the dynamic parameters apart from the configuration include:
        restore_ckpt : the file path of the checkpoint from which to restore the model weights
        is_training : essentially, the flag to enable batch normalization updates.
        aug : flag to enable data augmentation in the input data.
    """
    sig = StopRequest()

    # restore/train flags
    # checkpoint file to restore from

    restore_ckpt = None
    #restore_ckpt = os.path.expanduser('~/fn/18/ckpt/model.ckpt-40000')
    is_training = True
    aug = True

    # directory
    save_root = os.path.expanduser('~/fn')
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

    mkdir(log_root)
    mkdir(log_root_t)
    ckpt_root = os.path.join(log_root, 'ckpt')
    mkdir(ckpt_root)
    ckpt_file = os.path.join(ckpt_root, 'model.ckpt')

    #loaders = [ILSVRCLoader(os.getenv('ILSVRC_ROOT'), data_type=('train_%d' % i)) for i in range(1,31)]
    #data_img1, data_img2, data_pred, dlen = load_data(
    #        data_root, sample_ratio=sample_ratio)
    #sample_ratio = 0.5

    ilsvrc_root = os.path.expanduser('~/datasets/ilsvrc_opt')
    chair_root = os.path.expanduser('~/datasets/fchair/data')

    #sel = np.random.choice(np.arange(1,31), size=10, replace=False)
    #print('selected : {}'.format(sel))

    #train_cnt = 0

    graph = tf.get_default_graph()
    with graph.as_default():
        global_step = tf.train.get_or_create_global_step()
        with tf.name_scope('queue'):
            q_img = tf.placeholder(tf.uint8, 
                    [None, 2, cfg.IMG_HEIGHT, cfg.IMG_WIDTH, cfg.IMG_DEPTH], name='img')
            q_lab = tf.placeholder(tf.float32,
                    [None, cfg.IMG_HEIGHT, cfg.IMG_WIDTH, 3], name='lab') # flow label, (di,dj,mask)
            q_i = [q_img, q_lab]
            q_t = [e.dtype for e in q_i]
            q_s = [e.shape[1:] for e in q_i]
            Q = tf.FIFOQueue(capacity=128, dtypes=q_t, shapes=q_s)
            enqueue_op = Q.enqueue_many(q_i)
            img, lab = Q.dequeue_many(cfg.FN_BATCH_SIZE)

            # opt1 : augmentation
            if aug:
                aimg, plab = augment_image_affine(img, lab, [cfg.IMG_HEIGHT, cfg.IMG_WIDTH, cfg.IMG_DEPTH])
                pimg = augment_image_color(aimg) # NOTE: this augmentation ALSO scales image.
            else:
                # opt2 : no augmentation
                pimg = proc_img_tf(img)
                plab = lab

        # =========================
        # initial ramp-up 1e-6 -> 1e-4
        #lr0 = tf.train.exponential_decay(cfg.FN_LR_RAMP_0,
        #        global_step, cfg.FN_LR_RAMP_STEPS, cfg.FN_LEARNING_RATE/cfg.FN_RAMP_0, staircase=False)
        
        # standard decay 1e-4 -> 1e-3
        lr1 = tf.train.exponential_decay(cfg.FN_LR0,
                global_step, cfg.FN_LR_STEPS_PER_DECAY, cfg.FN_LR_DECAY_FACTOR, staircase=False)

        # opt1 : ramp
        #learning_rate = tf.where(global_step < cfg.FN_RAMP_STEPS, lr0, lr1) # employ slow initial learning rate

        # opt2 : standard
        #learning_rate = lr1

        # opt3 : decay + constant
        learning_rate = tf.where(
                global_step < cfg.FN_LR_DECAY_STEPS, lr1, cfg.FN_LR1)
        # =========================
        
        net = FlowNetBB(global_step,
                learning_rate=learning_rate, img=pimg, lab=plab,
                train=is_training, log=print)

        tf.summary.scalar('qsize', Q.size(), collections=['train'])
        tf.summary.scalar('learning_rate',learning_rate, collections=['train'])

        summary_t = tf.summary.merge([tf.summary.merge_all(), tf.summary.merge_all('train')])
        writer_t = tf.summary.FileWriter(log_root_t, graph)
        saver = tf.train.Saver()

    #gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.95)
    #config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
    config = None

    with tf.Session(graph=graph, config=config) as sess:
        coord = tf.train.Coordinator()
        def enqueue():
            msk = None
            while not coord.should_stop():
                #img, lab =  np.random.choice(loaders).grab_opt(8)

                # ilsvrc-mode
                ## get label
                #idx = np.random.choice(dlen, size=8)
                ## stack + proc
                #img = np.stack([data_img1[idx], data_img2[idx]], axis=1)[...,::-1] # RGB->BGR
                #flow = data_pred[idx]

                if (np.random.random() < 0.5):
                    # fchair-mode
                    img1, img2, flow = load_chair(chair_root, 8, size=(cfg.IMG_WIDTH, cfg.IMG_HEIGHT))
                else:
                    # ilsvrc-mode
                    img1, img2, flow = load_ilsvrc(ilsvrc_root, 8, size=(cfg.IMG_WIDTH, cfg.IMG_HEIGHT))

                img = np.stack([img1, img2], axis=1)# rgb-u8
                # (cpu) process + label
                # pimg = proc_img(img)

                if msk is None:
                    msk = np.ones_like(flow[...,:1])

                lab = np.concatenate([flow, msk], axis=-1) # PRED = [x,y,mask]
                sess.run(enqueue_op, feed_dict={q_img:img, q_lab:lab})

        # initialization
        sess.run(tf.global_variables_initializer())
        if restore_ckpt is not None:
            saver.restore(sess, restore_ckpt)
        i = i0 = sess.run(global_step)

        q_threads = [threading.Thread(target=enqueue) for _ in range(cfg.Q_THREADS)]
        for t in q_threads:
            t.daemon = True
            t.start()

        sig.start()
        errs = []
        for i in range(i0, cfg.FN_TRAIN_STEPS):
            # dataset management
            #train_cnt += cfg.FN_BATCH_SIZE
            #if train_cnt > dlen * cnt_per_data:
            #    data_img1, data_img2, data_pred, dlen = load_data(
            #            data_root, sample_ratio=0.35)
            #    train_cnt = 0

            if sig._stop:
                break
            # usual training
            # img, lab = dm.get(batch_size=cfg.BATCH_SIZE, time_steps=cfg.TIME_STEPS, aug=True)
            # img = proc_img(img)
            s, err, _ = sess.run([summary_t, net.err_, net.opt_]) #{net.img_ : img, net.lab_ : lab})
            errs.append(err)

            if (i>0) and (i%cfg.LOG_STEPS)==0:
                err_mean = np.mean(errs)
                print('{} : {}'.format(i, err_mean))
                errs = []
                s_emean = tf.Summary(value=[
                    tf.Summary.Value(tag="err_smooth", simple_value=err_mean)
                    ])
                writer_t.add_summary(s_emean, i)
            writer_t.add_summary(s, i)

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
