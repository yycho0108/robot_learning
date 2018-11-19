#!/usr/bin/env python2

import config as cfg
import numpy as np
import tensorflow as tf
from vo_net import VONet

def get_data():
    x = np.zeros([cfg.BATCH_SIZE,cfg.TIME_STEPS,cfg.IMG_HEIGHT,cfg.IMG_WIDTH,cfg.IMG_DEPTH])
    y = np.zeros([cfg.BATCH_SIZE,cfg.TIME_STEPS,3])
    return x,y

def main():
    net = VONet()
    #config = tf.ConfigProto(
    #        device_count = {'GPU': 0}
    #    )
    config=None
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        with tf.control_dependencies([net.rnn_reset_]):
            img, lab = get_data()
            err, _ = sess.run([net.err_, net.opt_], 
                    {net.img_ : img, net.lab_ : lab})

if __name__ == "__main__":
    main()
