import tensorflow as tf
import numpy as np
from tensorflow.contrib.framework import nest

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

def main():
    t1 = tf.placeholder(dtype=tf.float32, shape=[5,2,None,4])
    print tf_shape(t1)
    print(split_reshape(t1, 2, 2))

    #t2 = axial_reshape(t1, [0,2,(1,3)])
    #print(t2.shape)

if __name__ == "__main__":
    main()
