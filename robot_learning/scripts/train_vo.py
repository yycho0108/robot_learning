#!/usr/bin/env python2
from __future__ import print_function

import config as cfg
import numpy as np
import tensorflow as tf
import os
from vo_net import VONet
from data_manager import DataManager
from utils import anorm

def main():
    dm = DataManager(log=print)
    #img, lab = get_data()
    img, lab = dm.get(batch_size=cfg.BATCH_SIZE, time_steps=cfg.TIME_STEPS)
    net = VONet(log=print)
    ##config = tf.ConfigProto(
    ##        device_count = {'GPU': 0}
    ##    )
    config=None
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        with tf.control_dependencies([net.rnn_reset_]):
            err, _ = sess.run([net.err_, net.opt_], 
                    {net.img_ : img, net.lab_ : lab})
            print('err', err)

if __name__ == "__main__":
    main()
