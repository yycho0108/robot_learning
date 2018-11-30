import tensorflow as tf
import numpy as np
from tensorflow.contrib.framework import nest
import os
slim = tf.contrib.slim
from matplotlib import pyplot as plt

def all_subdirs_of(b='.'):
    """ from https://stackoverflow.com/a/2014704 """
    result = []
    for d in os.listdir(b):
        bd = os.path.join(b, d)
        if os.path.isdir(bd):
            result.append(bd)
    return result

def latest_subdir(root):
    result = max(all_subdirs_of(root), key=os.path.getmtime)
    return result

def latest_checkpoint(root):
    root = os.path.expanduser(root)
    d = latest_subdir(root)
    ckpt_dir = os.path.join(d, 'ckpt')
    return tf.train.latest_checkpoint(ckpt_dir)

def normalizer_no_op(x, *a, **k):
    """ passthrough normalization """
    return x

def tf_shape(x):
    s_s = x.get_shape().as_list()
    if None in s_s:
        # cannot be fully resolved statically
        s_d = tf.unstack(tf.shape(x))
        n = len(s_s)
        for i in range(n):
            if (s_s[i] is None):
                s_s[i] = s_d[i]
    return s_s

def split_reshape(x, i, n):
    with tf.name_scope('split_reshape', [x,i,n]):
        s = tf_shape(x)
        if np.issubdtype(type(s[i]), np.integer):
            assert (s[i]%n)==0, 'x.shape[i] must be divisible by n! {}/{}'.format(s[i], n)
        s = nest.flatten([s[:i], (s[i]/n), n, s[i+1:]])
        x = tf.reshape(x, s)
    return x

def isint(x):
    return np.issubdtype(type(x), np.integer)

def merge_dim(a,b):
    if isint(a) and isint(b):
        if a>=0 and b>=0:
            return a*b
        else:
            return -1
    else:
        return -1

def axial_reshape(x, ix):
    with tf.name_scope('axial_reshape', [x,ix]):
        # rectify ix
        ix = [([e] if np.isscalar(e) else e) for e in ix]

        # resolve input shape
        s = tf_shape(x)
        s = x.get_shape().as_list()

        ix_f = nest.flatten(ix)
        assert (len(s) - 1) == np.max(ix_f) # assert input correctness

        # transpose if necessary
        if not np.all(np.diff(ix_f) == 1):
            x = tf.transpose(x, ix_f)

        # reshape
        tm = nest.pack_sequence_as(ix, [s[i] for i in ix_f])
        s_out = [reduce(merge_dim, e, 1) for e in tm]
        x = tf.reshape(x, s_out)
    return x

def net_size(scope=None, return_all=False):
    ws = slim.get_model_variables(scope=scope)
    ss = [reduce(lambda a,b:a*b, w.get_shape().as_list()) for w in ws]
    if return_all:
        return (ws, ss)
    total = sum(ss)
    return total

def decay_np(learning_rate, step, steps_per_decay, decay_factor):
    return learning_rate * np.power(decay_factor, step / steps_per_decay)

def cyclic_decay(fin_lr, step, period, decay_steps,
        min_lr, max_lr
        ):
    with tf.name_scope('cyclic_decay',
            [fin_lr,step,period,decay_steps,min_lr,max_lr]):
        # upper bound
        lr_u = tf.train.exponential_decay(max_lr, step,
                decay_steps, (fin_lr / max_lr), staircase=False)
        # lower bound
        lr_l = tf.train.exponential_decay(min_lr, step,
                decay_steps, (fin_lr / min_lr), staircase=False)

        # define a cyclic term that starts at the minimum
        w = float(2*3.1415926535897932 / period)
        oc = (-tf.cos(tf.to_float(step) * w) + 1.0)/2.0
        lr_m = lr_l + ((lr_u-lr_l) * oc)

        fin_lr = tf.broadcast_to(fin_lr, lr_m.shape)
        lr = tf.where(step<decay_steps, lr_m, fin_lr)

    return lr

def main():
    #t1 = tf.placeholder(dtype=tf.float32, shape=[5,2,None,4])
    #print tf_shape(t1)
    #print(split_reshape(t1, 2, 2))

    #t2 = axial_reshape(t1, [0,2,(1,3)])
    #print(t2.shape)

    #steps = np.linspace(0, 20000)
    #lr0 = decay_np(1e-6, steps, 10000 / 6., 100 ** (1.0 / 6))
    #lr1 = decay_np(1e-3, steps, 10000 / 6., 0.1 ** (1.0 / 6))
    #p   = 2000.
    #lr  = lr0 + ((lr1-lr0)/2. * (np.sin(steps*2*np.pi/p)+1.0))
    #plt.plot(steps, lr0, label='lr0')
    #plt.plot(steps, lr1, label='lr1')
    #plt.plot(steps, lr, label='lr_m')
    #plt.legend()
    #plt.show()

    lr = cyclic_decay(1e-4, tf.linspace(0.0, 20000.0, 100), 2000, 10000,
            1e-6, 1e-3)
    with tf.Session() as sess:
        lr_ = sess.run(lr)
    steps = np.linspace(0,20000,100)
    plt.plot(steps, lr_)
    plt.show()


if __name__ == "__main__":
    main()
