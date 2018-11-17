from __future__ import print_function

import config as cfg
import tensorflow as tf
from tensorflow.contrib import slim

no_op = (lambda *a, **k: None)

class VONet(object):
    def __init__(self, log=print):
        self._build(log=log)

    def _build(self, log=no_op):
        log('- configuration -')
        log(open(cfg.__file__.replace('.pyc','.py'), 'r').read())
        log('-----------------')
        with tf.name_scope('input'):
            # NTCHW
            img = tf.placeholder(tf.float32, 
                    [cfg.BATCH_SIZE, cfg.TIME_STEPS, cfg.IMG_HEIGHT, cfg.IMG_WIDTH, cfg.IMG_DEPTH], name='img')
            lab = tf.placeholder(tf.float32, [cfg.BATCH_SIZE, cfg.TIME_STEPS, 3], name='lab') # label

        cnn = self._build_cnn(img, log)
        rnn, rnn_s, rnn_keep, rnn_reset = self._build_rnn(cnn, log)
        pos = self._build_pos(rnn, log)
        err = self._build_err(pos, lab, log=log)
        opt = self._build_opt(err, log=log)

        # cache tensors
        self.img_ = img
        self.pos_ = pos
        self.lab_ = lab
        self.err_ = err
        self.opt_ = opt
        self.rnn_keep_ = rnn_keep
        self.rnn_reset_ = rnn_reset

        # also return them
        return [img,pos,lab,err,opt]

    def _build_cnn(self, x, log=no_op):
        log('- build-cnn -')
        with tf.name_scope('build_cnn', [x]):
            with tf.name_scope('format_in'):
                s = x.get_shape().as_list()
                log('cnn-input', s)
                x = tf.reshape(x, [-1, s[2], s[3], s[4]])
            with tf.name_scope('cnn'):
                with self._arg_scope():
                    x = slim.stack(x,
                            slim.conv2d,
                            [(64,7,2),(128,5,2), (256,5,2)],
                            scope='conv')
                    log('post-conv', x.shape) #NTx30x40
                    x = slim.stack(x,
                            slim.separable_conv2d,
                            [(256,3,1,2), (512,3,1,2), (1024,3,1,2)], scope='sconv')
                    log('post-sconv', x.shape) #NTx4x5
                    x = tf.reduce_mean(x, axis=[-2,-1]) # avg pooling
                    log('post-cnn', x.shape)
            with tf.name_scope('format_out'):
                x = tf.reshape(x, [s[0], s[1], -1])
                log('cnn-output', x.shape)
        log('-------------')
        return x

    def _build_rnn(self, x, log=no_op):
        log('- build-rnn -')
        with tf.name_scope('build_rnn', [x]):
            log('rnn-input', x)
            with self._arg_scope():
                lstms = [tf.nn.rnn_cell.LSTMCell(cfg.LSTM_SIZE) for _ in range(cfg.NUM_LSTM)]
                cell = tf.nn.rnn_cell.MultiRNNCell(lstms)
                with tf.variable_scope('rnn_state'):
                    state_variables = []
                    for state_c, state_h in cell.zero_state(cfg.BATCH_SIZE, tf.float32):
                        state_variables.append(tf.nn.rnn_cell.LSTMStateTuple(
                            tf.Variable(state_c, trainable=False),
                            tf.Variable(state_h, trainable=False)))
                    state0 = tuple(state_variables)
                output, state1 = tf.nn.dynamic_rnn(
                        cell=cell,
                        inputs=x,
                        initial_state=state0,
                        time_major=False,
                        scope='rnn')
                log('rnn-output', output.shape)
                with tf.name_scope('rnn_keep'):
                    # for stateful LSTM (during runtime)
                    keep_ops = []
                    for (s0c,s0h), (s1c,s1h) in zip(state0, state1):
                        # Assign the new state to the state variables on this layer
                        # for both (c,h)
                        keep_ops.extend([s0c.assign(s1c), s0h.assign(s1h)])
                    rnn_keep_op = tf.group(keep_ops)
                with tf.name_scope('rnn_reset'):
                    # for stateless LSTM (during training)
                    # Define an op to reset the hidden state to zeros
                    reset_ops = []
                    for (s0c,s0h) in state0:
                        # Assign the new state to the state variables on this layer
                        # for both (c,h)
                        reset_ops.extend([
                            s0c.assign(tf.zeros_like(s0c)),
                            s0h.assign(tf.zeros_like(s0h))])
                    rnn_reset_op = tf.group(reset_ops)
        log('-------------')
        return output, state1, rnn_keep_op, rnn_reset_op

    def _build_pos(self, x, log=no_op):
        log('- build-pos -')
        with tf.name_scope('build-pos'):
            with tf.name_scope('pos'):
                x = slim.fully_connected(x, 128, activation_fn=tf.nn.elu)
                x = slim.fully_connected(x, 3, activation_fn=None) # operate in 2d : (dx,dy,dh)
                # NOTE: don't bother composing motion here
        log('pos-output', x.shape)
        log('-------------')
        return x

    def _build_err(self, x, y, k=cfg.W_COST, log=no_op):
        log('- build-err -')
        err = tf.square(x-y)
        log('raw-err', err.shape)
        err = tf.reduce_mean(err * k) # scale orientation error!
        log('fin-err', err.shape)
        log('-------------')
        return err

    def _build_opt(self, c, log=no_op):
        log('- build-opt -')
        opt = tf.train.AdamOptimizer(learning_rate=cfg.LEARNING_RATE)
        op = opt.minimize(c)
        log('-------------')
        return op

    def _arg_scope(self):
        return slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                padding='SAME',
                data_format='NHWC',
                activation_fn=tf.nn.elu
                )

def main():
    net = VONet()

if __name__ == "__main__":
    main()
