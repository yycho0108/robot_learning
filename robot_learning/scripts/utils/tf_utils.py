import tensorflow as tf
import numpy as np
from tensorflow.contrib.framework import nest
import os
slim = tf.contrib.slim

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

def main():
    t1 = tf.placeholder(dtype=tf.float32, shape=[5,2,None,4])
    print tf_shape(t1)
    print(split_reshape(t1, 2, 2))

    #t2 = axial_reshape(t1, [0,2,(1,3)])
    #print(t2.shape)

if __name__ == "__main__":
    main()
