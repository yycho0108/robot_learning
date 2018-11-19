#!/usr/bin/env python2

import config as cfg
import numpy as np
import tensorflow as tf
import os
from vo_net import VONet
from data_managert import DataManager

def anorm(x):
    return (x + np.pi) % (2*np.pi) - np.pi

#def add_p3d(a,b):
#    # final p3d composition -- mostly for verification
#    x0,y0,h0 = a
#    dx,dy,dh = b
#    c, s = np.cos(h0), np.sin(h0)
#    R = np.reshape([c,-s,s,c], [2,2]) # [2,2,N]
#    dp = R.dot([dx,dy])
#    x1 = x0 + dp[0]
#    y1 = y0 + dp[1]
#    h1 = anorm(h0 + dh)
#    return [x1,y1,h1]

def main():
    dm = DataManager(dirs=['/tmp/data/0', '/tmp/data/1'])
    #img, lab = get_data()
    img, lab = dm.get(batch_size=cfg.BATCH_SIZE, time_steps=cfg.TIME_STEPS)
    net = VONet()
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
