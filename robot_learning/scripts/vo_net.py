from __future__ import print_function

import config as cfg
import tensorflow as tf
from tensorflow.contrib import slim
from utils import no_op
from tensorflow.contrib.framework import nest

class VONet(object):
    def __init__(self, step, learning_rate=None,
            img=None, lab=None,
            batch_size=None, 
            train=True, reuse=tf.AUTO_REUSE,
            cfg=cfg, log=print):

        self.img_ = img
        self.lab_ = lab
        self.col_ = [('train' if train else 'valid')]
        self.learning_rate_ = (cfg.LEARNING_RATE if (learning_rate is None) else learning_rate)

        self.step_ = step
        self.train_ = train
        self.reuse_ = reuse
        self.log_ = log

        # override some parameters
        self.batch_size_ = (cfg.BATCH_SIZE if (batch_size is None) else batch_size)
        self._build(cfg=cfg, log=log)

    def _build(self, cfg=cfg, log=no_op):
        log('- configuration -')
        log(open(cfg.__file__.replace('.pyc','.py'), 'r').read())
        log('-----------------')

        with tf.name_scope('input'):
            # NTCHW
            if self.img_ is None:
                img = tf.placeholder(tf.float32, 
                        [None, None, cfg.IMG_HEIGHT, cfg.IMG_WIDTH, cfg.IMG_DEPTH], name='img')
            else:
                img = self.img_
            if self.lab_ is None:
                lab = tf.placeholder(tf.float32, [None, None, 3], name='lab') # label
            else:
                lab = self.lab_

        cnn = self._build_cnn(img, log)
        rnn, rnn_s1, rnn_s0 = self._build_rnn(cnn, log)
        pos = self._build_pos(rnn, log)
        err = self._build_err(pos, lab, log=log)

        if self.train_:
            opt       = self._build_opt(err, log=log)
            self.opt_ = opt
        _   = self._build_log(log=log, err=err, lab=lab, pos=pos)

        # cache tensors
        self.img_ = img
        self.pos_ = pos
        self.lab_ = lab
        self.err_ = err
        self.rnn_s1_ = rnn_s1
        self.rnn_s0_ = rnn_s0
        return

    def _build_cnn(self, x, log=no_op):
        log('- build-cnn -')
        with tf.name_scope('build_cnn', [x]):
            with tf.name_scope('format_in'):
                s_s = x.get_shape().as_list()
                s_d = tf.unstack(tf.shape(x))
                log('s-static',  s_s)
                log('s-dynamic', s_d)
                log('cnn-input', x.shape)
                x = tf.reshape(x, [s_d[0]*s_d[1], s_s[2], s_s[3], s_s[4]])
                log('cnn-format', x.shape)
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
                    x = tf.reduce_mean(x, axis=[1,2]) # avg pooling
                    log('post-cnn', x.shape)
            with tf.name_scope('format_out'):
                x = tf.reshape(x, [s_d[0], s_d[1], 1024])
                log('cnn-output', x.shape)
        log('-------------')
        return x

    def _build_rnn(self, x, log=no_op):
        log('- build-rnn -')
        with tf.name_scope('build_rnn', [x]):
            log('rnn-input', x)
            bs = tf.unstack(tf.shape(x))[0] # figure out dynamic batch size
            with self._arg_scope():
                lstms = [tf.nn.rnn_cell.LSTMCell(cfg.LSTM_SIZE) for _ in range(cfg.NUM_LSTM)]
                cell = tf.nn.rnn_cell.MultiRNNCell(lstms)
                #print('c0',cell.zero_state(cfg.BATCH_SIZE, tf.float32))
                state0 = nest.map_structure(
                        lambda x : tf.placeholder_with_default(x, [None] + list(x.shape)[1:], x.op.name),
                        cell.zero_state(self.batch_size_, tf.float32))
                #with tf.variable_scope('rnn_state'):
                #    state_variables = []
                #    for state_c, state_h in cell.zero_state(bs, tf.float32):
                #        state_variables.append(tf.nn.rnn_cell.LSTMStateTuple(
                #            tf.Variable(state_c, trainable=False, validate_shape=False),
                #            tf.Variable(state_h, trainable=False, validate_shape=False)))
                #    state0 = tuple(state_variables)
                with tf.variable_scope('rnn', reuse=self.reuse_):
                    output, state1 = tf.nn.dynamic_rnn(
                            cell=cell,
                            inputs=x,
                            initial_state=state0,
                            #initial_state=state0,
                            time_major=False,
                            dtype=tf.float32)
                #log('rnn-output', output.shape)
                #with tf.name_scope('rnn_keep'):
                #    # for stateful LSTM (during runtime)
                #    keep_ops = []
                #    for (s0c,s0h), (s1c,s1h) in zip(state0, state1):
                #        # Assign the new state to the state variables on this layer
                #        # for both (c,h)
                #        keep_ops.extend([s0c.assign(s1c), s0h.assign(s1h)])
                #    rnn_keep_op = tf.group(keep_ops)
                #with tf.name_scope('rnn_reset'):
                #    # for stateless LSTM (during training)
                #    # Define an op to reset the hidden state to zeros
                #    reset_ops = []
                #    for (s0c,s0h) in state0:
                #        # Assign the new state to the state variables on this layer
                #        # for both (c,h)
                #        reset_ops.extend([
                #            s0c.assign(tf.zeros_like(s0c)),
                #            s0h.assign(tf.zeros_like(s0h))])
                #    rnn_reset_op = tf.group(reset_ops)

        log('-------------')
        return output, state1, state0

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
        with tf.name_scope('build-err', [x,y]):
            err = tf.square(x-y)
            log('raw-err', err.shape)
            err = tf.reduce_mean(err * k) # scale orientation error!
            log('fin-err', err.shape)
        log('-------------')
        return err

    def _build_opt(self, c, log=no_op):
        log('- build-opt -')
        opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate_)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            #grads, vars = zip(*opt.compute_gradients(c))
            #grads_c, _ = tf.clip_by_global_norm(grads, 1.0)
            #train_op = opt.apply_gradients(zip(grads_c, vars), global_step=self.step_)
            ##train_op = opt.minimize(c, global_step=self.step_)
            train_op = tf.contrib.layers.optimize_loss(c, self.step_,
                    learning_rate=self.learning_rate_,
                    optimizer='Adam',
                    clip_gradients=1.0,
                    summaries=['loss', 'learning_rate', 'global_gradient_norm', 'gradients'])

        log('-------------')
        return train_op

    def _build_log(self, log=no_op, **tensors):
        log('- build-log -')
        tf.summary.scalar('err', tensors['err'], collections=self.col_)
        tf.summary.histogram('pos', tensors['pos'], collections=self.col_)
        tf.summary.histogram('lab', tensors['lab'], collections=self.col_)
        #if self.train_:
        #    tf.summary.histogram('grad', tensors['grad'], collections=self.col_)
        log('-------------')
        return None

    def _arg_scope(self):
        bn_params = {
                'is_training' : self.train_,
                'decay' : 0.92,
                'fused' : True,
                'scale' : True,
                'reuse' : self.reuse_,
                'data_format' : 'NHWC',
                'scope' : 'batch_norm',
                }
        return slim.arg_scope(
                [slim.conv2d, slim.separable_conv2d],
                padding='SAME',
                data_format='NHWC',
                activation_fn=tf.nn.elu,
                normalizer_fn=slim.batch_norm,
                normalizer_params=bn_params,
                reuse=self.reuse_
                )

def main():
    net = VONet(step=None)

if __name__ == "__main__":
    main()
