from __future__ import print_function

import config as cfg
import tensorflow as tf
from utils import no_op, nest_log
from utils.tf_utils import axial_reshape, split_reshape, tf_shape, net_size
from tensorflow.contrib.framework import nest
import sys

slim = tf.contrib.slim

root = '/home/jamiecho/Repos/tf_flownet2'
sys.path.append(root)
from FlowNet2_src.correlation import correlation as xcor

def normalizer_no_op(x, *a, **k):
    return x

def upsample(x):
    h, w = x.get_shape().as_list()[1:3]
    return tf.image.resize_images(x, (2*h,2*w), align_corners=True)

def upconvolution(x, *a, **k):
    with tf.name_scope('upconv', [x]):
        x = upsample(x)
        x = slim.separable_conv2d(x, *a, **k)
    return x

def epe(x, y):
    with tf.name_scope('epe', [x,y]):
        e = tf.square(y-x)
        e = tf.reduce_sum(e, axis=-1, keepdims=True)
        e = tf.sqrt(e)
    return e

#def xcor_feat_(f, d=3, s_h=1, s_w=2):
#    # for FlownetC-flavored implementations
#    # WARNING: this is wrong - do not use
#    with tf.name_scope('xcor', [f]):
#        with tf.name_scope('format_f'):
#            f = split_reshape(f, 0, 2)
#            fa, fb = tf.unstack(f, axis=1)
#            c_in = tf_shape(fa)[3]
#
#        # "source" = NHWC
#        # placing batch in channel dimension
#        # to produce correct correspondence in correlation results
#        with tf.name_scope('format_fa'):
#            fa = axial_reshape(fa, [1,2,[0,3]])
#            fa = tf.expand_dims(fa, axis=0)
#            fa = tf.pad(fa, [[0,0],[d,d],[d,d],[0,0]], mode='SYMMETRIC')
#
#        # "filter" = HWC1
#        with tf.name_scope('format_fb'):
#            fb = axial_reshape(fb, [1,2,[0,3]])
#            fb = tf.expand_dims(fb, axis=-1)
#
#        # xcor + shape
#        fxf = tf.nn.depthwise_conv2d(fa, fb,
#                strides=[1,s_h,s_w,1],
#                padding='VALID', name='correlate',
#                data_format='NHWC')
#
#        with tf.name_scope('format_fxf'):
#            fxf = tf.squeeze(fxf, axis=0) # --> HW(NC)
#            fxf = split_reshape(fxf, 2, c_in) # HxWxNxC
#            fxf = tf.reduce_sum(fxf, axis=3, keepdims=True) # HWN
#            fxf = axial_reshape(fxf, [2,0,1,3])
#        print('fxf', fxf.shape) # --> NxDxD, D=2*d+1
#    # out = 
#    print('xcor', f[0].shape, f[1].shape)

def xcor_feat(f,
        d=10,
        s_h=1,
        s_w=2,
        log=no_op
        ):
    with tf.name_scope('xcor_feat', [f]):
        h, w = tf_shape(f)[1:3]
        f_ab = split_reshape(f, 0, 2)
        fa, fb = tf.unstack(f_ab, axis=1)
        # a, b, kernel_size, max_disp, stride-1, stride_2, pad
        fxf = xcor(fa, fb, 1, d*2, s_h, s_w, d*2)
        log('coverage : {:.0%}x{:.0%}'.format(d/float(h), d/float(w)))
        log('shape : {}'.format(fxf.shape))
    return fxf

def reshape_feat(f):
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
        log('- build-cnn -')
        # see vo_net.VONet for reference here

        with tf.name_scope('build_cnn', [x]):
            with tf.name_scope('format_in'):
                s_s = x.get_shape().as_list()
                s_d = tf.unstack(tf.shape(x))
                log('s-static',  s_s)
                log('s-dynamic', s_d)
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
            with tf.name_scope('format_out'):
                x = reshape_feat(x)
                #x = xcor_feat(x, d=3, s_h=1, s_w=1, log=log) # 6x8x ((2*3+1)**2 == 49), win ~ (0.5x0.375)
                log('cnn-output', x.shape)

        log('-------------')
        feats = slim.utils.convert_collection_to_dict('cnn-feats')
        for (k,v) in feats.items():
            log(k,v)
        #feats = [feats[s] for s in ['vo/sconv_pre/sconv_pre_2', 'vo/sconv_post/sconv_post_1','vo/sconv_post/sconv_post_4']]

        feats = [feats[s] for s in [
            'vo/sconv/sconv_1', #48x64
            'vo/sconv/sconv_3', #24x32
            'vo/sconv/sconv_5', #12x16
            ]]

        # d, s_h, s_w
        xargs =  [
                (7, 1, 1), #48x64x ((2*9+1)**2 == 361), win ~ (0.1875  x 0.140625)
                (5, 1, 1), #24x32x ((2*7+1)**2 == 225), win ~ (0.2916' x 0.21875)
                (3, 1, 1)  #12x16x ((2*5+1)**2 == 121), win ~ (0.416'  x 0.3125)
                ]
        feats = [reshape_feat(f) for f in feats]
        #feats = [xcor_feat(f,*a,log=log) for (f,a) in zip(feats,xargs)]
        for t in feats:
            log(t.name, t.shape)
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
                x = slim.conv2d_transpose(x, 256, 3, stride=1, padding='SAME')
                feats2_rdc = slim.separable_conv2d(feats[2], 128, 1, stride=1, padding='SAME')
                x = tf.concat([x, feats2_rdc],axis=-1)
                log('cat-1', x.shape)
                x = slim.conv2d_transpose(x, 256, 5, stride=1, padding='SAME')
                x = slim.conv2d_transpose(x, 128, 3, stride=1, padding='SAME')
                x = tf.concat([x, feats[1]],axis=-1)
                log('cat-2', x.shape)
                x = slim.conv2d_transpose(x, 128, 3, stride=1, padding='SAME')

                # resize-conv
                log('xs-1', x.shape)
                x = upsample(x)
                x = slim.separable_conv2d(x, 128, 3, stride=1, padding='SAME')
                log('xs-2', x.shape)
                x = tf.concat([x, feats[0]],axis=-1)
                log('cat-3', x.shape)
                x = upsample(x)
                x = slim.separable_conv2d(x, 128, 3, stride=1, padding='SAME')
                log('xs-3', x.shape)
                x = upsample(x)
                x = slim.separable_conv2d(x, 64, 3, stride=1, padding='SAME')
                x = upsample(x)
                x = slim.conv2d(x, 2, 3, stride=1, padding='SAME',
                        activation_fn=None, normalizer_fn=normalizer_no_op
                        )

                #x = slim.conv2d_transpose(x, 128, 3, stride=2, padding='SAME') #14x18
                #x = tf.concat([x, feats[0]],axis=-1)
                #log('cat-3', x.shape)
                #x = slim.conv2d_transpose(x, 64, 3, stride=2, padding='SAME') #15x20
                #x = slim.conv2d_transpose(x, 64, 3, stride=2, padding='SAME')
                #x = slim.conv2d_transpose(x, 32, 3, stride=2, padding='SAME')
                #x = slim.conv2d(x, 2, 1, 1, padding='SAME',
                #        activation_fn=None, normalizer_fn=None
                #        )
                log('post-tcnn', x.shape)
        return x

    def _build_tcnn_multi(self, inputs, feats, img, log=no_op):
        # Nx1024 --> Nx240x320
        log('- build-tcnn -')

        def to_flow(fx_in, pad='SAME', scope='flow'):
            return slim.conv2d(fx_in, 2, 3, padding=pad,
                    activation_fn=None,
                    normalizer_fn=normalizer_no_op,
                    scope=scope,
                    outputs_collections='flow'
                    )

        with tf.name_scope('build_tcnn', [inputs, feats]):
            x = inputs
            with slim.arg_scope(self._arg_scope()):
                log('tcnn-input', x.shape)
                # objective : (15x20), (30x40), (60x80), (120x160), (240x320)
                #x = slim.separable_conv2d(x, 256, 1, stride=1, padding='SAME') # feat reduction
                #log('tcnn-rdc', x.shape)
                x = slim.separable_conv2d(x, 1024, 3, stride=1, padding='SAME')
                #feats2_rdc = slim.separable_conv2d(feats[2], 128, 1, stride=1, padding='SAME')
                #x = tf.concat([x, feats2_rdc],axis=-1)
                f0 = to_flow(x, scope='flow_0')
                log('xs-0', x.shape)
                log('flow-0', f0.shape)

                #x = slim.conv2d_transpose(x, 128, 3, stride=2, padding='SAME')
                x   = upconvolution(x, 512, 3, stride=1, padding='SAME')
                f0u = upconvolution(f0, 2, 3, stride=1, padding='SAME')
                x = tf.concat([x, feats[2], f0u],axis=-1)
                f1 = to_flow(x, scope='flow_1')
                log('xs-1', x.shape)
                log('flow-1', f1.shape)

                x   = upconvolution(x, 256, 3, stride=1, padding='SAME')
                f1u = upconvolution(f1, 2, 3, stride=1, padding='SAME')
                x = tf.concat([x, feats[1], f1u], axis=-1)
                f2 = to_flow(x, scope='flow_2')
                log('xs-1', x.shape)
                log('flow-1', f2.shape)

                x = upconvolution(x, 128, 3, stride=1, padding='SAME')
                f2u = upconvolution(f2, 2, 3, stride=1, padding='SAME')
                x = tf.concat([x, feats[0], f2u], axis=-1)
                f3 = to_flow(x, scope='flow_3')
                log('xs-2', x.shape)
                log('flow-2', f3.shape)

                x   = upconvolution(x, 64, 3, stride=1, padding='SAME')
                f3u = upconvolution(f3, 2, 3, stride=1, padding='SAME')
                x = tf.concat([x, f3u], axis=-1)
                f4 = to_flow(x, scope='flow_4')
                log('xs-3', x.shape)
                log('flow-3', f4.shape)

                x = upconvolution(x, 32, 3, stride=1, padding='SAME')
                f4u = upconvolution(f4, 2, 3, stride=1, padding='SAME')
                img2 = axial_reshape(img, [0,2,3,(4,1)])
                x = tf.concat([x, f4u, img2], axis=-1)
                x = to_flow(x, pad='SAME', scope='flow_5')
                log('xs-4', x.shape)
                log('flow-4', x.shape)

                #x = slim.conv2d_transpose(x, 128, 3, stride=2, padding='SAME') #14x18
                #x = tf.concat([x, feats[0]],axis=-1)
                #log('cat-3', x.shape)
                #x = slim.conv2d_transpose(x, 64, 3, stride=2, padding='SAME') #15x20
                #x = slim.conv2d_transpose(x, 64, 3, stride=2, padding='SAME')
                #x = slim.conv2d_transpose(x, 32, 3, stride=2, padding='SAME')
                #x = slim.conv2d(x, 2, 1, 1, padding='SAME',
                #        activation_fn=None, normalizer_fn=None
                #        )
                log('post-tcnn', x.shape)

        fs = slim.utils.convert_collection_to_dict('flow').values()
        return x, fs

    def _build_err_multi(self, x, xs, y, log=no_op):
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
        log('- build-err-multi -')

        errs = {}
        with tf.name_scope('build-err-multi', [x,y,xs]):
            for f in nest.flatten([x, xs]):
                h, w = f.get_shape().as_list()[1:3]
                key = 'err_{}x{}'.format(w,h)
                errs[key] = err_x(f, y)
            err = tf.reduce_mean(errs.values())
            log('err', err.shape)
        log('-------------------')
        return err, errs

    def _build_err(self, x, y, log=no_op):
        log('- build-err -')
        with tf.name_scope('build-err', [x,y]):
            opt, msk = tf.split(y, [2,1], axis=-1)
            #err = tf.sqrt(tf.reduce_sum(tf.square(x-opt),axis=-1,keepdims=True))
            err = epe(opt, x)
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
        for (k,v) in tensors['errs'].items():
            tf.summary.scalar(k, v, collections=self.col_)
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
            #tcnn = self._build_tcnn(cnn, feats, log=log)
            tcnn, prds = self._build_tcnn_multi(cnn, feats, img, log=log)

        if self.eval_:
            #err_c = self._build_err(tcnn, lab, log=log)
            err_c, errs = self._build_err_multi(tcnn, prds, lab, log=log)

        if self.train_:
            reg_c = tf.add_n(tf.losses.get_regularization_losses())
            tf.summary.scalar('err_loss', err_c)
            tf.summary.scalar('reg_loss', reg_c)
            cost = (err_c + reg_c)
            opt  = self._build_opt(cost, log=log)
            self.opt_ = opt

        if self.eval_:
            _   = self._build_log(log=log, err=err_c, errs=errs)

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
