from __future__ import print_function

import config as cfg
import numpy as np
import tensorflow as tf
from utils import no_op, nest_log, normalize
from utils.tf_utils import axial_reshape, split_reshape, tf_shape, net_size
from tensorflow.contrib.framework import nest
import sys

slim = tf.contrib.slim

import os

fn_repo = os.getenv('FN_REPO')
sys.path.append(fn_repo)
from correlation import correlation as xcor

def normalizer_no_op(x, *a, **k):
    """ passthrough normalization """
    return x

def upsample(x):
    """ upsample source to 2x """
    h, w = x.get_shape().as_list()[1:3]
    return tf.image.resize_images(x, (2*h,2*w), align_corners=True)

def upconv(x, f, *a, **k):
    """ simple up-convolution """
    with tf.name_scope('upconv', [x]):
        x = upsample(x)
        x = f(x, *a, **k)
    return x

def epe(x, y):
    """ endpoint error """
    with tf.name_scope('epe', [x,y]):
        e = tf.square(y-x)
        e = tf.reduce_sum(e, axis=-1, keepdims=True)
        e = tf.sqrt(e)
    return e

def xcor_feat(f,
        d=10,
        s_h=1,
        s_w=2,
        log=no_op
        ):
    """ compute feature-field cross-correlation """
    with tf.name_scope('xcor_feat', [f]):
        h, w = tf_shape(f)[1:3]
        f_ab = split_reshape(f, 0, 2)
        fa, fb = tf.unstack(f_ab, axis=1)
        # a, b, kernel_size, max_disp, stride-1, stride_2, pad
        fxf = xcor(fa, fb, 1, d*2, s_h, s_w, d*2)
        fxf = tf.nn.elu(fxf)
        log('coverage : ({:.0%}x{:.0%})'.format(d/float(h), d/float(w)))
        log('shape : {}'.format(fxf.shape))
    return fxf

def merge_feat(f, log=no_op):
    """ merge 2x-stacked feature streams into channel axis """
    f = split_reshape(f, 0, 2) # Nx2/2, 2, ...
    f = axial_reshape(f, [0,2,3,(4,1)])
    return f

class FlowNetBB(object):
    def __init__(self, step, learning_rate=None,
            img=None, lab=None,
            train=True, eval=True,
            reuse=None, log=print):
        self.img_ = img
        self.lab_ = lab

        self.col_ = [('train' if train else 'valid')]
        self.learning_rate_ = (cfg.LEARNING_RATE if (learning_rate is None) else learning_rate)
        self.step_ = step
        self.train_ = train
        self.eval_  = eval
        self.reuse_ = reuse
        self.log_ = log

        self.build(log=print)

    def _build_input(self, log=no_op):
        """ build input part """
        with tf.name_scope('input'):
            img = tf.placeholder(tf.float32, 
                    [None, 2, cfg.IMG_HEIGHT, cfg.IMG_WIDTH, cfg.IMG_DEPTH], name='img')
            if self.eval_:
                lab = tf.placeholder(tf.float32,
                        [None, cfg.IMG_HEIGHT, cfg.IMG_WIDTH, 3], name='lab') # flow label, (di,dj,mask)
            else:
                lab = None
        self.img_ = img
        self.lab_ = lab

    def _build_cnn(self, x, log=no_op):
        """ build contraction part """
        log('- build-cnn -')
        # see vo_net.VONet for reference here

        with tf.name_scope('build_cnn', [x]):
            with tf.name_scope('format_in'):
                log('cnn-input', x.shape)
                x = axial_reshape(x, [(0,1), 2, 3, 4]) # (merge "time" with batch)
                log('cnn-format', x.shape)
            with tf.name_scope('cnn'):
                with slim.arg_scope(self._arg_scope()):
                    with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                            outputs_collections=['cnn-feats']):
                        x = slim.conv2d(x, 64, 7, 2, scope='conv', padding='SAME')
                        x = slim.stack(x,
                                slim.separable_conv2d,
                                [(128,3,1,2),(256,3,1,2),(196,1,1,1),(384,3,1,2),(256,1,1,1),(512,3,1,2),(512,1,1,1)],
                                scope='sconv',
                                padding='SAME',
                                )
                        log('post-sconv', x.shape) #NTx4x5

        log('-------------')
        xs = slim.utils.convert_collection_to_dict('cnn-feats')

        log('- feats -')
        for (k,v) in xs.items():
            log(k,v)
        log('---------')

        xs = [xs[s] for s in [
            'vo/sconv/sconv_1', #48x64
            'vo/sconv/sconv_3', #24x32
            'vo/sconv/sconv_5', #12x16
            'vo/sconv/sconv_7', #6x8
            ]]

        if cfg.FN_USE_XCOR:
            log('- xcor -')
            # d, s_h, s_w
            xcor_args =  [
                    (7, 1, 1), #48x64x ((2*7+1)**2 == 361), win ~ (0.1875  x 0.140625)
                    (5, 1, 1), #24x32x ((2*5+1)**2 == 225), win ~ (0.2916' x 0.21875)
                    (3, 1, 1), #12x16x ((2*3+1)**2 == 121), win ~ (0.416'  x 0.3125)
                    (3, 1, 1), #6x8x   ((2*3+1)**2 == 49),  win ~ (0.5x0.375)
                    ]
            xs = [xcor_feat(x,*a,log=log) for (x,a) in zip(xs,xcor_args)]
            log('--------')
        else:
            log('- merge -')
            xs = [merge_feat(x) for x in xs]
            log('---------')

        log('- xs -')
        for x in xs:
            log(x.name, x.shape)
        log('------')
        return xs

    def _build_tcnn(self, xs, img, log=no_op):
        """ build expansion part """
        # Nx1024 --> Nx240x320
        def to_flow(fx_in, pad='SAME', scope='flow'):
            return slim.conv2d(fx_in, 2, 3, padding=pad,
                    activation_fn=None,
                    normalizer_fn=normalizer_no_op,
                    scope=scope,
                    outputs_collections='flow'
                    )

        log('- build-tcnn -')
        with tf.name_scope('build_tcnn', [xs]):
            with slim.arg_scope(self._arg_scope()):
                with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                        outputs_collections=['tcnn-feats']):
                    # objective : (15x20), (30x40), (60x80), (120x160), (240x320)
                    #x = slim.separable_conv2d(x, 256, 1, stride=1, padding='SAME') # feat reduction
                    #log('tcnn-rdc', x.shape)
                    x = slim.separable_conv2d(xs[3], 1024, 3, stride=1, padding='SAME')
                    f0 = to_flow(x, scope='flow_0')
                    log('xs-0', x.shape)
                    log('flow-0', f0.shape)

                    #x = slim.conv2d_transpose(x, 128, 3, stride=2, padding='SAME')
                    c, s = slim.conv2d, slim.separable_conv2d
                    x   = upconv(x, s, 512, 3, stride=1, padding='SAME')
                    f0u = upconv(f0, c, 2, 3, stride=1, padding='SAME',
                            activation_fn=None,
                            normalizer_fn=normalizer_no_op
                            )
                    x = tf.concat([x, xs[2], f0u],axis=-1)
                    f1 = to_flow(x, scope='flow_1')
                    log('xs-1', x.shape)
                    log('flow-1', f1.shape)

                    x   = upconv(x, s, 256, 3, stride=1, padding='SAME')
                    f1u = upconv(f1, c, 2, 3, stride=1, padding='SAME',
                            activation_fn=None,
                            normalizer_fn=normalizer_no_op
                            )
                    x = tf.concat([x, xs[1], f1u], axis=-1)
                    f2 = to_flow(x, scope='flow_2')
                    log('xs-1', x.shape)
                    log('flow-1', f2.shape)

                    x   = upconv(x, s, 128, 3, stride=1, padding='SAME')
                    f2u = upconv(f2, c, 2, 3, stride=1, padding='SAME',
                            activation_fn=None,
                            normalizer_fn=normalizer_no_op
                            )
                    x = tf.concat([x, xs[0], f2u], axis=-1)
                    f3 = to_flow(x, scope='flow_3')
                    log('xs-2', x.shape)
                    log('flow-2', f3.shape)

                    x   = upconv(x, s, 64, 3, stride=1, padding='SAME')
                    f3u = upconv(f3, c, 2, 3, stride=1, padding='SAME',
                            activation_fn=None,
                            normalizer_fn=normalizer_no_op
                            )
                    x = tf.concat([x, f3u], axis=-1)
                    f4 = to_flow(x, scope='flow_4')
                    log('xs-3', x.shape)
                    log('flow-3', f4.shape)

                    x   = upconv(x, s, 32, 3, stride=1, padding='SAME')
                    f4u = upconv(f4, c, 2, 3, stride=1, padding='SAME',
                            activation_fn=None,
                            normalizer_fn=normalizer_no_op
                            )
                    img2 = axial_reshape(img, [0,2,3,(4,1)])
                    x = tf.concat([x, f4u, img2], axis=-1)
                    x = to_flow(x, pad='SAME', scope='flow_5')
                    log('xs-4', x.shape)
                    log('flow-4', x.shape)

        fs = slim.utils.convert_collection_to_dict('flow').values()
        log('--------------')
        return x, fs

    def _build_err(self, xs, y, log=no_op):
        """ build error part """
        def err_x(f, y):
            with tf.name_scope('err_x', [f, y]):
                h0, w0 = y.get_shape().as_list()[1:3]
                h, w = f.get_shape().as_list()[1:3]
                y_rsz = tf.image.resize_images(y, (h,w), align_corners=True)
                flo, msk = tf.split(y_rsz, [2,1], axis=-1)

                # scale flow appropriately
                rsz_s = [tf.cast(w,tf.float32)/w0, tf.cast(h, tf.float32)/h0]
                rsz_s = tf.reshape(rsz_s, [1,1,1,2])
                flo_s = flo * rsz_s # scaled flow

                err = epe(f, flo_s)
                #err = tf.sqrt(tf.reduce_sum(tf.square(f-y_rsz),
                #    axis=-1,keepdims=True))
                err = tf.reduce_sum(err * msk) / tf.reduce_sum(msk)
            return err

        log('- build-err -')
        errs = {}
        with tf.name_scope('build-err', [xs,y]):
            for f in xs:
                h, w = f.get_shape().as_list()[1:3]
                key = 'err_{}x{}'.format(w,h)
                errs[key] = err_x(f, y)
            ks = ['err_8x6', 'err_16x12','err_32x24', 'err_64x48', 'err_128x96', 'err_256x192']

            # decay weight per size of flow-image

            # opt1 : dynamic error weight scaling (over step)
            err_scale = tf.train.exponential_decay(cfg.FN_ERR_SCALE,
                self.step_, cfg.FN_ERR_SCALE_DECAY_STEPS, cfg.FN_ERR_SCALE_DECAY_FACTOR, staircase=False)
            ws = tf.pow(err_scale, -tf.to_float(tf.range(len(ks))) )
            ws = ws / tf.reduce_sum(ws) # normalize weights

            # opt2 : static error weight scaling
            # ws = np.float32([cfg.FN_ERR_DECAY**-e for e in range(len(ks))])
            # ws /= ws.sum() # normalize weights

            #err = tf.reduce_mean(errs.values())
            err = tf.losses.compute_weighted_loss(
                    losses=[errs[k] for k in ks],
                    weights=ws
                    )
            log('err', err.shape)
        log('-------------')
        return err, errs, err_scale

    def _build_opt(self, c, log=no_op):
        """ build optimizer part """
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
        """ build log summaries part """
        log('- build-log -')
        tf.summary.scalar('err', tensors['err'], collections=self.col_)
        tf.summary.scalar('err_scale', tensors['err_scale'], collections=self.col_)
        for (k,v) in tensors['errs'].items():
            tf.summary.scalar(k, v, collections=self.col_)
        log('-------------')
        return None

    def build(self, log=no_op):
        """ build everything """
        # NTCHW
        if self.img_ is None:
            self._build_input()

        img = self.img_
        lab = self.lab_

        with tf.variable_scope('vo', reuse=self.reuse_):
            xs  = self._build_cnn(img, log=log)
            tcnn, prds = self._build_tcnn(xs, img, log=log)

        if self.eval_:
            err_c, errs, err_scale = self._build_err(prds, lab, log=log)

        if self.train_:
            reg_c = tf.add_n(tf.losses.get_regularization_losses())
            tf.summary.scalar('err_loss', err_c)
            tf.summary.scalar('reg_loss', reg_c)
            cost = (err_c + reg_c)
            opt  = self._build_opt(cost, log=log)
            self.opt_ = opt

        if self.eval_:
            _   = self._build_log(log=log, err=err_c, errs=errs,
                    err_scale=err_scale
                    )

        self.img_ = img
        self.pred_ = tcnn
        self.lab_ = lab

        if self.eval_:
            self.err_ = err_c
        else:
            self.err_ = None

        ns = net_size()
        log('- net size - ')
        log('total : {}'.format(ns))
        log('-------------')
        return

    def _arg_scope(self):
        """ function default arguments """
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
                    padding='SAME',
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
    net = FlowNetBB(step=tf.train.get_or_create_global_step())

if __name__ == "__main__":
    main()
