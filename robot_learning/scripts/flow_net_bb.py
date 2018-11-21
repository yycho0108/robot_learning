from __future__ import print_function

import config as cfg
import tensorflow as tf
from utils import no_op
slim = tf.contrib.slim

class FlowNetBB(object):
    def __init__(self, step, learning_rate=None,
            img=None, lab=None,
            train=True, reuse=None, log=print):
        self.img_ = img
        self.lab_ = lab

        self.col_ = [('train' if train else 'valid')]
        self.learning_rate_ = (cfg.LEARNING_RATE if (learning_rate is None) else learning_rate)
        self.step_ = step
        self.train_ = train
        self.reuse_ = reuse
        self.log_ = log

        self.build(log=print)

    def _build_input(self, log=no_op):
        with tf.name_scope('input'):
            img = tf.placeholder(tf.float32, 
                    [None, 2, cfg.IMG_HEIGHT, cfg.IMG_WIDTH, cfg.IMG_DEPTH], name='img')
            lab = tf.placeholder(tf.float32,
                    [None, cfg.IMG_HEIGHT, cfg.IMG_WIDTH, 3], name='lab') # flow label, (di,dj,mask)
        self.img_ = img
        self.lab_ = lab

    def _build_cnn(self, x, log=no_op):
        log('- build-cnn -')
        # see vo_net.VONet for reference here

        def reshape_feat(f):
            l = f.get_shape().as_list()
            s1 = [-1, 2]
            s1.extend(l[1:])
            f = tf.reshape(f, s1)
            f = tf.transpose(f, [0,2,3,4,1])
            f = tf.reshape(f, [-1,l[1],l[2],l[3]*2])
            return f

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
                with slim.arg_scope(self._arg_scope()):
                    with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                            outputs_collections=['cnn-feats']):

                        x = slim.conv2d(x, 64, 7, 2, scope='conv', padding='SAME')

                        x = slim.stack(x,
                                slim.separable_conv2d,
                                [(128,3,1,2),(128,3,1,2),(256,3,1,2)],
                                scope='sconv_pre',
                                padding='SAME',
                                )
                        log('post-conv', x.shape) #NTx30x40

                        x = slim.stack(x,
                                slim.separable_conv2d,
                                [(256,3,1,1), (256,3,1,1), (256,3,1,1), (512,3,1,1), (512,3,1,1)],
                                scope='sconv_post',
                                padding='VALID'
                                )
                        log('post-sconv', x.shape) #NTx4x5

                        #x = slim.stack(x,
                        #        slim.conv2d,
                        #        [(64,7,2),(128,3,2),(128,3,2),(256,3,2)],
                        #        scope='conv',
                        #        padding='SAME',
                        #        )
                        #log('post-conv', x.shape) #NTx30x40
                        #x = slim.stack(x,
                        #        slim.separable_conv2d,
                        #        [(256,3,1,1), (256,3,1,1), (256,3,1,1), (512,3,1,1)],
                        #        scope='sconv',
                        #        )
                        #log('post-sconv', x.shape) #NTx4x5
            #        x = tf.reduce_mean(x, axis=[1,2]) # avg pooling
            #        log('post-cnn', x.shape)
            with tf.name_scope('format_out'):
                s = x.get_shape().as_list()
                #x = tf.reshape(x, [s_d[0], s[1], s[2], s_s[1]*512]) # Nx1024
                x = reshape_feat(x)
                log('cnn-output', x.shape)
        log('-------------')
        feats = slim.utils.convert_collection_to_dict('cnn-feats')
        print(feats)
        feats = [feats[s] for s in ['vo/sconv_pre/sconv_pre_2', 'vo/sconv_post/sconv_post_1','vo/sconv_post/sconv_post_4']]
        feats = [reshape_feat(f) for f in feats]
        print('feats', feats)
        return x, feats

    def _build_tcnn(self, x, feats, log=no_op):
        # Nx1024 --> Nx240x320
        log('- build-tcnn -')
        with tf.name_scope('build_tcnn', [feats]):
            with slim.arg_scope(self._arg_scope()):
                log('tcnn-input', x.shape)
                # objective : (15x20), (30x40), (60x80), (120x160), (240x320)
                x = slim.separable_conv2d(x, 256, 1, stride=1, padding='SAME') # feat reduction
                #log('tcnn-rdc', x.shape)
                x = slim.conv2d_transpose(x, 128, 3, stride=1, padding='VALID')
                feats2_rdc = slim.separable_conv2d(feats[2], 128, 1, stride=1, padding='SAME')
                x = tf.concat([x, feats2_rdc],axis=-1)
                log('cat-1', x.shape)
                x = slim.conv2d_transpose(x, 128, 5, stride=1, padding='VALID')
                x = slim.conv2d_transpose(x, 128, 3, stride=1, padding='VALID')
                x = tf.concat([x, feats[1]],axis=-1)
                log('cat-2', x.shape)
                x = slim.conv2d_transpose(x, 128, 3, stride=1, padding='VALID')
                x = slim.conv2d_transpose(x, 128, 3, stride=2, padding='SAME') #14x18
                x = tf.concat([x, feats[0]],axis=-1)
                log('cat-3', x.shape)
                x = slim.conv2d_transpose(x, 64, 3, stride=2, padding='SAME') #15x20
                x = slim.conv2d_transpose(x, 64, 3, stride=2, padding='SAME')
                x = slim.conv2d_transpose(x, 32, 3, stride=2, padding='SAME')
                x = slim.conv2d(x, 2, 1, 1, padding='SAME',
                        activation_fn=None, normalizer_fn=None
                        )
                log('post-tcnn', x.shape)
        return x

    def _build_err(self, x, y, log=no_op):
        log('- build-err -')
        with tf.name_scope('build-err', [x,y]):
            opt, msk = tf.split(y, [2,1], axis=-1)

            err = tf.sqrt(tf.reduce_sum(tf.square(x-opt),axis=-1,keepdims=True))
            err = tf.reduce_mean(err)

            #err = tf.losses.absolute_difference(labels=opt, predictions=x)
            #err = tf.abs(x-opt)
            #err = tf.reduce_mean(err)

            #err = tf.losses.huber_loss(labels=opt, predictions=x,
            #        reduction=tf.losses.Reduction.MEAN
            #        )

            #err = tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(x-opt),axis=-1,keepdims=True))*msk) / tf.reduce_sum(msk)
            log('err', err.shape)
        log('-------------')
        return err

    def _build_opt(self, c, log=no_op):
        log('- build-opt -')
        opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate_)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = tf.contrib.layers.optimize_loss(c, self.step_,
                    learning_rate=self.learning_rate_,
                    optimizer='Adam',
                    clip_gradients=None,
                    summaries=['loss', 'learning_rate', 'global_gradient_norm', 'gradients'])
        log('-------------')
        return train_op

    def _build_log(self, log=no_op, **tensors):
        log('- build-log -')
        tf.summary.scalar('err', tensors['err'], collections=self.col_)
        log('-------------')
        return None

    def build(self, log=no_op):
        # NTCHW
        if self.img_ is None:
            self._build_input()

        img = self.img_
        lab = self.lab_

        with tf.variable_scope('vo', reuse=self.reuse_):
            cnn, feats = self._build_cnn(img, log=log)
            tcnn = self._build_tcnn(cnn, feats, log=log)
        err_c = self._build_err(tcnn, lab, log=log)

        if self.train_:
            reg_c = tf.add_n(tf.losses.get_regularization_losses())
            tf.summary.scalar('err_loss', err_c)
            tf.summary.scalar('reg_loss', reg_c)
            cost = (err_c + reg_c)
            opt  = self._build_opt(cost, log=log)
            self.opt_ = opt
        _   = self._build_log(log=log, err=err_c)

        self.img_ = img
        self.lab_ = lab
        self.err_ = err_c

        ws = slim.get_model_variables()
        ss = [reduce(lambda a,b:a*b, w.get_shape().as_list()) for w in ws]
        log('- net size - ')
        for (w,s) in zip(ws,ss):
            log(w.name, s)
        log('total : {}'.format(sum(ss)))
        log('-------------')
        return

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
                normalizer_fn=slim.batch_norm,
                normalizer_params=bn_params,
                ):
            with slim.arg_scope(
                    [slim.conv2d, slim.separable_conv2d, slim.conv2d_transpose],
                    padding='VALID',
                    data_format='NHWC',
                    activation_fn=tf.nn.elu,
                    weights_regularizer=(slim.l2_regularizer(1e-6) if self.train_ else None),
                    reuse=self.reuse_
                    ):
                with slim.arg_scope(
                        [slim.fully_connected],
                        weights_regularizer=(slim.l2_regularizer(1e-6) if self.train_ else None),
                        reuse=self.reuse_
                        ) as sc:
                    return sc
def main():
    net = FlowNetBB(step=None)

if __name__ == "__main__":
    main()
