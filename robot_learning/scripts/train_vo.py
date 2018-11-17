import config as cfg
import numpy as np
import tensorflow as tf
import os
from vo_net import VONet

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

class DataManager(object):
    def __init__(self, dirs):
        self.data_ = [self.load(d) for d in dirs]
        self.data_ = [self.format(*d) for d in self.data_]

        # dataset selection probability
        self.prob_ = np.float32([len(d[0])-cfg.TIME_STEPS for d in self.data_])
        print('Dataset Stats : {}'.format(self.prob_))
        self.prob_ /= self.prob_.sum()

    def load(self, path):
        img   = np.load(os.path.join(path, 'img.npy'))
        odom  = np.load(os.path.join(path, 'odom.npy'))
        return img, odom

    def format(self, img, odom):
        # format the data to be compatible with the training network
        o = odom
        prv = o[:-1]
        nxt = o[1:]
        #x1,y1,h1 w.r.t x0,y0,h0
        delta = nxt - prv #[N,2]
        h0 = prv[:,2] 
        c, s = np.cos(h0), np.sin(h0)
        R = np.reshape([c,-s,s,c], [2,2,-1]) # [2,2,N]
        dp = np.einsum('ijk,ki->kj', R, delta[:,:2])
        dh = anorm(delta[:,2:])
        
        delta = np.concatenate([dp,dh], axis=-1)
        delta = np.concatenate([np.zeros_like(delta[0:1]), delta], axis=0)
        return img, delta

    def get_1(self, data, time_steps):
        img, lab = data
        i0 = np.random.randint(0, high=len(img)-time_steps)
        return img[i0:i0+time_steps], lab[i0:i0+time_steps]

    def get(self, batch_size, time_steps):
        set_idx = np.random.choice(len(self.data_),
                batch_size, replace=True, p=self.prob_)
        data = [self.get_1(self.data_[i], time_steps) for i in set_idx]
        img, lab = zip(*data)

        img = np.stack(img, axis=0)  # [NxHxWxC]
        lab = np.stack(lab, axis=0) # [Nx3]
        return img, lab 

    def get_null(self, batch_size, time_steps):
        x = np.zeros([cfg.BATCH_SIZE,cfg.TIME_STEPS,cfg.IMG_HEIGHT,cfg.IMG_WIDTH,cfg.IMG_DEPTH])
        y = np.zeros([cfg.BATCH_SIZE,cfg.TIME_STEPS,3])
        return x, y

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
