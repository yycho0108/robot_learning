from __future__ import absolute_import

import os
import tensorflow as tf
from fchair_utils import load_chair
from tf_utils import tf_shape
from functools import partial

def augment_image_affine_1(img, opt, size=None):
    # size formatted (h,w,c)

    with tf.name_scope('augment_image_affine_single', [img, opt]):
        box = [[[0.0,0.0,1.0,1.0]]] # full box

        b0, bs, _ = tf.image.sample_distorted_bounding_box(
                size, box,
                min_object_covered=0.5,
                area_range=[0.5, 1],
                use_image_if_no_bounding_boxes=True
                )

        imga, imgb = tf.unstack(img, axis=0)

        # 1: affine transform
        img1a = tf.slice(imga, b0, bs)
        img1b = tf.slice(imgb, b0, bs)
        img1  = tf.stack([img1a, img1b], axis=0)

        opt1 = tf.slice(opt, b0, bs)

        # 2: rectify flow map scale based on box
        dx  = opt1[...,0] * (size[1] / tf.cast(bs[1], tf.float32))
        dy  = opt1[...,1] * (size[0] / tf.cast(bs[0], tf.float32))
        msk = opt1[...,2]

        img2 = tf.image.resize_images(img1,
                [size[0], size[1]],
                align_corners=True)
        opt2 = tf.stack([dx,dy,msk], axis=-1)
    return (img2, opt2)

def augment_image_affine(img, opt, size):
    with tf.name_scope('augment_image_affine', [img, opt]):
        #augfun = partial(augment_image_affine_1, size=size)
        #augfun(img[0], opt[0])
        augfun = (lambda x: augment_image_affine_1(x[0],x[1],size=size))
        img2, opt2 = tf.map_fn(augfun, [img, opt], dtype=(tf.uint8, tf.float32))
    return img2, opt2

def augment_image_color(img):
    shape = tf_shape(img)
    img = tf.cast(img, tf.float32) / 255.0 # u8 -> f32
    img = img + tf.random.normal(shape=shape, mean=0.0, stddev=0.03, dtype=tf.float32)
    img = tf.image.random_contrast(img, 0.8, 1.2)
    img = tf.image.random_saturation(img, 0.9, 1.1)
    img = tf.image.random_hue(img, 0.2)

    gamma = tf.random.uniform(shape=[shape[0],1,1,1], minval=0.7, maxval=1.5)
    # TODO : handle image rank better ^^^

    img = tf.pow(img, gamma)
    img = tf.image.random_brightness(img, 0.2)
    # clip to region (-1.0, 1.0)
    img = tf.clip_by_value(img, 0.0, 1.0)
    img = (img - 0.5) * 2.0 #  ~~ (u8/128-1.0)
    return img

def main():
    chair_root = os.path.expanduser('~/Downloads/FlyingChairs/data')

    h, w, c = 192, 256, 3

    img1_t = tf.placeholder(tf.uint8, [None,h,w,c])
    img2_t = tf.placeholder(tf.uint8, [None,h,w,c])
    flow_t = tf.placeholder(tf.float32, [None,h,w,c])

    img_t = tf.stack([img1_t, img2_t], axis=1)

    aimg_t, aflo_t  = augment_image_affine(img_t, flow_t, [h,w,c])
    print aimg_t.shape
    print aflo_t.shape
    aimg_t = augment_image_color(aimg_t)
    print aimg_t.shape

    #img1, img2, flow = load_chair(chair_root, 8, size=(w,h))

    #with tf.Session() as sess:
    #    sess.run(

if __name__ == "__main__":
    main()
