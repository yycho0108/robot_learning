from __future__ import print_function

import config as cfg
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from utils import no_op
from tensorflow.contrib.framework import nest
from se2 import SE2CompositeLayer
from utils.tf_utils import tf_shape, axial_reshape, split_reshape, normalizer_no_op

def ang_err(h0, h1):
    c0, s0 = tf.cos(h0), tf.sin(h0)
    c1, s1 = tf.cos(h1), tf.sin(h1)
    return tf.square(c1-c0) + tf.square(s1-s0)

class VONet(object):
    def __init__(self, step, learning_rate=None,
            img=None, lab=None,
            batch_size=None, 
            train=True, reuse=tf.AUTO_REUSE,
            cfg=cfg, log=print,
            ):

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

        with tf.variable_scope('vo', reuse=self.reuse_):
            cnn = self._build_cnn(img, log)
            rnn, rnn_s1, rnn_s0 = self._build_rnn(cnn, log)
            dps = self._build_dps(rnn, log)
            x_pos, x_poss, y_pos, y_poss = self._build_pos(dps, lab, log)

        err_c, (err_x, err_y, err_h) = self._build_err(x_poss, y_poss, log=log)
        #err_c, (err_x, err_y, err_h) = self._build_err(dps, lab, log=log)

        if self.train_:
            reg_c = tf.add_n(tf.losses.get_regularization_losses())
            tf.summary.scalar('err_loss', err_c)
            tf.summary.scalar('reg_loss', reg_c)
            cost = (err_c + reg_c)
            opt       = self._build_opt(cost, log=log)
            self.opt_ = opt

        _   = self._build_log(log=log,
                lab=lab, dps=dps,
                err_c=err_c,
                err_x=err_x,
                err_y=err_y,
                err_h=err_h
                )

        # cache tensors
        self.img_ = img
        self.dps_ = dps
        self.lab_ = lab
        self.err_ = err_c
        self.rnn_s1_ = rnn_s1
        self.rnn_s0_ = rnn_s0

        ws = slim.get_model_variables()
        ss = [reduce(lambda a,b:a*b, w.get_shape().as_list()) for w in ws]
        log('- net size - ')
        for (w,s) in zip(ws,ss):
            log(w.name, s)
        log('total : {}'.format(sum(ss)))
        log('-------------')
        return

    def _build_cnn(self, x, log=no_op):
        log('- build-cnn -')
        with tf.name_scope('build_cnn', [x]):
            with tf.name_scope('format_in'):
                log('cnn-input', x.shape)
                dim_t = tf_shape(x)[1]
                x = axial_reshape(x, [(0,1), 2, 3, 4]) # (merge "time" with batch)
                log('cnn-format', x.shape)
            with tf.name_scope('cnn'):
                with slim.arg_scope(self._arg_scope()):
                    x = slim.conv2d(x, 64, 7, 2, scope='conv', padding='SAME')
                    x = slim.stack(x,
                            slim.separable_conv2d,
                            [(128,3,1,2),(256,3,1,2),(196,1,1,1),(384,3,1,2),(256,1,1,1),(512,3,1,2),(512,1,1,1)],
                            scope='sconv',
                            padding='SAME',
                            )
                    log('post-sconv', x.shape) #NTx4x5
                    #x = tf.reduce_mean(x, axis=[1,2])

                    #x = axial_reshape(x, [0,(1,2,3)])
                    x = slim.separable_conv2d(x, 512, (6,8), 1, 1, scope='reduction', padding='VALID')
                    #x = tf.expand_dims(x, 1)
                    #x = tf.expand_dims(x, 1)
                    #x = slim.separable_conv2d(x, 1024, 1, 1, 1, scope='reduction')
                    x = slim.dropout(x, keep_prob=0.2, is_training=self.train_, scope='dropout')
                    x = tf.squeeze(x, [1,2])
                    log('post-cnn', x.shape)
            with tf.name_scope('format_out'):
                x = split_reshape(x, 0, dim_t) # ==> [N,T,...]
                log('cnn-output', x.shape)
        log('-------------')
        return x

    def _build_rnn(self, x, log=no_op):
        log('- build-rnn -')
        with tf.name_scope('build_rnn', [x]):
            log('rnn-input', x)
            bs = tf.unstack(tf.shape(x))[0] # figure out dynamic batch size
            with slim.arg_scope(self._arg_scope()):
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

    def _build_dps(self, x, log=no_op):
        log('- build-dps-')
        with tf.name_scope('build-dps'):
            with tf.name_scope('dps'):
                x = slim.fully_connected(x, 128, activation_fn=tf.nn.elu, scope='fc1')
                x = slim.fully_connected(x, 64, activation_fn=tf.nn.elu, scope='fc2')
                xyh = slim.fully_connected(x, 3, activation_fn=None,
                        normalizer_fn=normalizer_no_op,
                        scope='xyh')
                xy, h = tf.split(xyh, [2,1], axis=-1)

                # explicitly model scale 
                s = slim.fully_connected(x, 1, activation_fn=tf.exp,
                        normalizer_fn=normalizer_no_op,
                        scope='s')

                x = tf.concat([xy*s, h], axis=-1)
                # NOTE: don't bother composing motion here
        log('dps-output', x.shape)
        log('-------------')
        return x

    def _build_pos(self, x, y, log=no_op):
        log('-build-pos-')
        with tf.name_scope('build-pos'):
            cell = SE2CompositeLayer()

            x_poss, x_pos=tf.nn.dynamic_rnn(
                    cell=cell,
                    inputs=x[:,1:], # output @ t=0 shouldn't count
                    dtype=tf.float32)

            y_poss, y_pos=tf.nn.dynamic_rnn(
                    cell=cell,
                    inputs=y[:,1:], # output @ t=0 shouldn't count
                    dtype=tf.float32)
        log('pos-output', x_poss.shape, x.shape)
        log('-----------')
        return x_pos, x_poss, y_pos, y_poss

    def _build_err(self, x, y, ws=cfg.W_COST, log=no_op):
        log('- build-err -')
        with tf.name_scope('build-err', [x,y]):
            prd_x,prd_y,prd_h = tf.unstack(x, axis=-1)
            lab_x,lab_y,lab_h = tf.unstack(y, axis=-1)

            err_x = tf.reduce_mean(tf.square(prd_x - lab_x))
            err_y = tf.reduce_mean(tf.square(prd_y - lab_y))
            #err_h = tf.reduce_mean(tf.square(prd_h - lab_h))
            err_h = tf.reduce_mean(ang_err(prd_h, lab_h))

            err = tf.losses.compute_weighted_loss(
                    losses=[err_x, err_y, err_h],
                    weights=(np.float32(ws) / np.sum(ws))
                    )
            #err = tf.square(x-y)
            #log('raw-err', err.shape)
            #scale orientation error!
            #err = tf.reduce_mean(err * k)

            log('fin-err', err.shape)
        log('-------------')
        return err, [err_x, err_y, err_h]

    def _build_opt(self, c, freeze_cnn=cfg.FREEZE_CNN, log=no_op):
        log('- build-opt -')
        opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate_)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        if freeze_cnn:
            log('freezing cnn')
            train_vars = [v for v in slim.get_trainable_variables() if ('vo/sconv' not in v.name) and ('vo/conv' not in v.name)]
            log('train variables:')
            for v in train_vars:
                log('\t {} : {}'.format(v.name, v.shape))
            log('---------')
        else:
            # train everything
            train_vars = None

        with tf.control_dependencies(update_ops):
            #grads, vars = zip(*opt.compute_gradients(c))
            #grads_c, _ = tf.clip_by_global_norm(grads, 1.0)
            #train_op = opt.apply_gradients(zip(grads_c, vars), global_step=self.step_)
            ##train_op = opt.minimize(c, global_step=self.step_)

            train_op = tf.contrib.layers.optimize_loss(c, self.step_,
                    learning_rate=self.learning_rate_,
                    optimizer='Adam',
                    clip_gradients=3.0,
                    summaries=['loss', 'learning_rate', 'global_gradient_norm', 'gradients'],
                    variables=train_vars
                    )

        log('-------------')
        return train_op

    def _build_log(self, log=no_op, **tensors):
        log('- build-log -')
        tf.summary.scalar('err_c', tensors['err_c'], collections=self.col_)

        tf.summary.scalar('err_x', tensors['err_x'], collections=self.col_)
        tf.summary.scalar('err_y', tensors['err_y'], collections=self.col_)
        tf.summary.scalar('err_h', tensors['err_h'], collections=self.col_)

        tf.summary.histogram('dps', tensors['dps'], collections=self.col_)
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
        with slim.arg_scope(
                [slim.conv2d, slim.separable_conv2d],
                padding='SAME',
                data_format='NHWC',
                activation_fn=tf.nn.elu,
                weights_regularizer=(slim.l2_regularizer(1e-6) if self.train_ else None),
                normalizer_fn=slim.batch_norm,
                normalizer_params=bn_params,
                reuse=self.reuse_
                ):
            with slim.arg_scope(
                    [slim.fully_connected],
                    weights_regularizer=(slim.l2_regularizer(1e-6) if self.train_ else None),
                    reuse=self.reuse_
                    ) as sc:
                return sc

def main():
    graph = tf.Graph()

    with graph.as_default():
        global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.train.exponential_decay(cfg.LEARNING_RATE,
                global_step, cfg.STEPS_PER_DECAY, cfg.DECAY_FACTOR, staircase=True)

        net = VONet(step=global_step,
                learning_rate=learning_rate,
                batch_size=cfg.BATCH_SIZE,
                train=True,
                log=print
                )
        #print([v.name for v in slim.get_trainable_variables()])

    #with tf.Session(graph=graph) as sess:
    #    sess.run(tf.global_variables_initializer())
    #    img = np.zeros(dtype=np.float32,
    #            shape=[cfg.BATCH_SIZE, cfg.TIME_STEPS, cfg.IMG_HEIGHT, cfg.IMG_WIDTH, cfg.IMG_DEPTH])
    #    lab = np.zeros(dtype=np.float32,
    #            shape=[cfg.BATCH_SIZE, cfg.TIME_STEPS, 3])
    #    print(lab.shape)
    #    dps, err, _ = sess.run([net.dps_, net.err_, net.opt_],
    #            feed_dict={net.img_:img, net.lab_:lab})
    #    print(dps.shape)

if __name__ == "__main__":
    main()
