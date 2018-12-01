import os
import sys
import tensorflow as tf
from tf_utils import tf_shape, axial_reshape
from functools import partial
import cv2
import numpy as np
from matplotlib import pyplot as plt
from opt_utils import flow_to_image, apply_opt, FlowShow
from fchair_utils import load_chair, load_ilsvrc

def augment_image_affine_1(img, opt, size=None):
    # size formatted (h,w,c)

    with tf.name_scope('augment_image_affine_single', [img, opt]):
        box = [[[0.0,0.0,1.0,1.0]]] # full box

        b0, bs, _ = tf.image.sample_distorted_bounding_box(
                size, box,
                min_object_covered=0.25,
                area_range=[0.25, 1],
                use_image_if_no_bounding_boxes=True,
                )

        imga, imgb = tf.unstack(img, axis=0)

        # 1: affine transform
        img1a = tf.slice(imga, b0, bs)
        s_d = img1a.get_shape().as_list()
        img1a.set_shape(s_d[:-1] + [3])
        img1b = tf.slice(imgb, b0, bs)
        img1b.set_shape(s_d[:-1] + [3])
        img1  = tf.stack([img1a, img1b], axis=0)

        opt1 = tf.slice(opt, b0, bs)

        # 2: rectify flow map scale based on box
        xs = (size[1] / tf.to_float(bs[1]))
        ys = (size[0] / tf.to_float(bs[0]))
        dx  = opt1[...,0] * xs
        dy  = opt1[...,1] * ys

        img2 = tf.image.resize_images(img1,
                [size[0], size[1]],
                align_corners=True)
        opt2 = tf.stack([dx,dy], axis=-1)
        opt2 = tf.image.resize_images(opt2,
                [size[0], size[1]],
                align_corners=True)
        img2 = tf.cast(img2, tf.uint8)

        #msk = opt1[...,2]
        # TODO : currently ignores input mask (which doesn't exit)
        # for robustness, consider using input mask.
        irange = tf.range( size[0] )
        jrange = tf.range( size[1] )
        gy, gx = tf.meshgrid(irange, jrange, indexing='ij')
        g      = tf.stack([gx,gy], axis=-1)

        dst = (tf.cast(g, tf.float32) + opt2) # == where the pixel would end up, in x-y orientation

        l_lim = tf.cast(tf.reshape([0,0], (1,1,2)), tf.float32)
        u_lim = tf.cast(tf.reshape([size[1], size[0]], (1,1,2)), tf.float32)
        msk = tf.cast(tf.logical_and(
                tf.math.greater_equal(dst, l_lim),
                tf.math.less(dst, u_lim)),tf.float32)
        msk = tf.reduce_min(msk, axis=-1, keepdims=True) # ~= logical_and
        #msk = tf.logical_and(msk, opt1[...,2]
        opt2 = tf.concat([opt2, msk], axis=-1)

    return (img2, opt2)

def augment_image_affine(img, opt, size):
    with tf.name_scope('augment_image_affine', [img, opt]):
        #augfun = partial(augment_image_affine_1, size=size)
        #augfun(img[0], opt[0])
        augfun = (lambda x: augment_image_affine_1(x[0],x[1],size=size))
        img2, opt2 = tf.map_fn(augfun, (img, opt), dtype=(tf.uint8, tf.float32))
    return img2, opt2

def augment_image_color(img):
    shape = tf_shape(img)
    img = tf.cast(img, tf.float32) / 255.0 # u8 -> f32
    img = img + tf.random_normal(shape=shape, mean=0.0, stddev=0.03, dtype=tf.float32)
    img = tf.image.random_contrast(img, 0.8, 1.2)
    img = tf.image.random_saturation(img, 0.9, 1.1)
    img = tf.image.random_hue(img, 0.1)

    shape_g = [shape[0]] + [1 for _ in shape[1:]]

    gamma = tf.random_uniform(shape=shape_g, minval=0.7, maxval=1.5)
    # TODO : handle image rank better ^^^

    img = tf.image.random_brightness(img, 0.1)
    img = tf.clip_by_value(img, 0.0, 1.0)
    img = tf.pow(img, gamma)

    # clip to region (-1.0, 1.0)
    img = tf.clip_by_value(img, 0.0, 1.0)
    img = (img - 0.5) * 2.0 #  ~~ (u8/128-1.0)
    return img

def unproc(x):
    return (x / 2.0) + 0.5

def main():
    chair_root = os.path.expanduser('~/datasets/fchair/data')
    ilsvrc_root = os.path.expanduser('~/datasets/ilsvrc_opt')

    h, w, c = 192, 256, 3

    img1_t = tf.placeholder(tf.uint8, [None,h,w,c])
    img2_t = tf.placeholder(tf.uint8, [None,h,w,c])
    flow_t = tf.placeholder(tf.float32, [None,h,w,c])

    img_t = tf.stack([img1_t, img2_t], axis=1)

    aimg_t, aflo_t  = augment_image_affine(img_t, flow_t, [h,w,c])
    aimg_t = augment_image_color(aimg_t)

    n_test = 16
    img1_gt, img2_gt, flow_gt = load_chair(chair_root, n_test, size=(w,h))
    #img1_gt, img2_gt, flow_gt = load_ilsvrc(ilsvrc_root, n_test, size=(w,h))

    flow_gt = np.concatenate([flow_gt, np.ones_like(flow_gt[..., :1])], axis=-1) # msk dim

    config = tf.ConfigProto(
            device_count = {'GPU': 0})

    with tf.Session(config=config) as sess:
        aimg, aflo = sess.run([aimg_t, aflo_t], {img1_t:img1_gt, img2_t:img2_gt, flow_t:flow_gt})

    # simple validation
    #print('simple validation')
    #print(flow_gt[0].max(axis=(0,1,2)))
    #print(aflo[0].max(axis=(0,1,2)))

    disp = FlowShow(code_path='middlebury_flow_code.png')
    disp.configure([
        [FlowShow.AX_NULL, FlowShow.AX_NULL, FlowShow.AX_FLOW],
        [FlowShow.AX_IMG1, FlowShow.AX_IMG2, FlowShow.AX_OVLY],
        [FlowShow.AX_I2I1, FlowShow.AX_I1I2, FlowShow.AX_I2OV],
        ])

    disp.add_user_cb(
            lambda d, i, a, f: d._draw_ax(
                a[0,0],
                [img1_gt[i],img2_gt[i],flow_gt[i]],
                FlowShow.AX_IMG1
            ))
    disp.add_user_cb(
            lambda d, i, a, f: d._draw_ax(
                a[0,1],
                [img1_gt[i],img2_gt[i],flow_gt[i]],
                FlowShow.AX_FLOW
            ))

    img1 = unproc(aimg[:,0])
    img2 = unproc(aimg[:,1])
    flow = aflo
    disp.add(img1, img2, flow)
    disp.show()

if __name__ == "__main__":
    main()
