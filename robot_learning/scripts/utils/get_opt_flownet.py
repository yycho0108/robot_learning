import numpy as np
from scipy.misc import imread
import cv2
import tensorflow as tf
import os
import sys

import time

# WARNING : external dependency.
# build tf_flownet2 from https://github.com/vt-vl-lab/tf_flownet2.git !
root = '/home/jamiecho/Repos/tf_flownet2'
sys.path.append(root)

from FlowNet2_src import FlowNet2, LONG_SCHEDULE

from ilsvrc_utils import ILSVRCLoader
from utils import mkdir

class GetOptFlowNet(object):
    def __init__(self):
        pass

    def _build(self):
        # Graph construction
        #im1_pl = tf.placeholder(tf.float32, [None, None, None, 3])
        #im2_pl = tf.placeholder(tf.float32, [None, None, None, 3])
        #im1_in = tf.image.resize_images(im1_pl, (384,512))
        #im2_in = tf.image.resize_images(im2_pl, (384,512))

        im1_in = im1_pl = tf.placeholder(tf.float32, [None, 384, 512, 3])
        im2_in = im2_pl = tf.placeholder(tf.float32, [None, 384, 512, 3])

        flownet2 = FlowNet2()
        inputs = {'input_a': im1_in, 'input_b': im2_in}
        flow_dict = flownet2.model(inputs, LONG_SCHEDULE, trainable=False)
        pred_flow = flow_dict['flow']

        self.im1_ = im1_pl
        self.im2_ = im2_pl
        self.flow_ = pred_flow

    def load(self, sess, ckpt):
        tf.train.Saver().restore(sess, ckpt)

    def __call__(self, sess,
            im1, im2):
        return sess.run(self.flow_, {self.im1_:im1,self.im2_:im2})

def normalize(x, mn=0.0, mx=1.0):
    xmn = np.min(x)
    xmx = np.max(x)
    return (x-xmn)*((mx-mn)/(xmx-xmn)) + mn

def main():
    graph = tf.Graph()
    with graph.as_default():
        net = GetOptFlowNet()
        net._build()

    #im1 = os.path.join(root, 'FlowNet2_src/example/0img0.ppm')
    #im2 = os.path.join(root, 'FlowNet2_src/example/0img1.ppm')
    #im1 = imread(im1)/255.
    #im2 = imread(im2)/255.
    #im1 = cv2.resize(im1, (320,240))
    #im2 = cv2.resize(im2, (320,240))

    ckpt_file = os.path.join(root,
            'FlowNet2_src/checkpoints/FlowNet2/flownet-2.ckpt-0')
    #saver = tf.train.Saver()

    #gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.95)
    #config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
    config=None

    source_size = (512,384)
    target_size = (320,240)

    # flow scale rectification
    flow_s_w = float(target_size[0]) / source_size[0]
    flow_s_h = float(target_size[1]) / source_size[1]
    flow_s   = np.reshape([flow_s_w, flow_s_h], [1,1,1,2])

    target_dir  = os.path.expanduser('~/datasets/ilsvrc_opt')

    with tf.Session(graph=graph, config=config) as sess:
        #saver.restore(sess, ckpt_file)
        net.load(sess, ckpt_file)

        cnt = 0
        for data_i in range(1,31):
            print('Current Set Index : {}/{}'.format(data_i, 31)
            #out_dir = os.path.join(target_dir, str(data_i))
            out_dir = target_dir # flattened!
            mkdir(out_dir)

            data_type = ('train_%d' % data_i) 
            loader = ILSVRCLoader(os.getenv('ILSVRC_ROOT'), data_type=('train_%d'%data_i), T=8,
                    size=(512,384) # expected network image size
                    )

            print('data length-0 : {}'.format(len(loader.keys)))
            imgs = loader.grab_pair(batch_size=-1, per_seq=4)
            print('data length-1 : {}'.format(len(imgs)))
            split_n = np.round(len(imgs)/8.).astype(np.int32) # end up with batch_size of about 8
            print('split into {} sections'.format(split_n))
            simgs = np.array_split(imgs, split_n, axis=0)
            for si, img in enumerate(simgs):
                print('\t{}/{}'.format(si, len(simgs)))
                im1, im2 = np.split(img, 2, axis=1)
                im1 = np.squeeze(im1, axis=1)
                im2 = np.squeeze(im2, axis=1) # u8-bgr

                im1f_rgb = np.float32(im1[...,::-1]) / 255. # f32-rgb
                im2f_rgb = np.float32(im2[...,::-1]) / 255.
                flow_val = net(sess, im1f_rgb, im2f_rgb)

                # important: rectify flow scale
                flow_val *= flow_s

                # resize everything
                img1_rsz = [cv2.resize(e, target_size) for e in im1]
                img2_rsz = [cv2.resize(e, target_size) for e in im2]
                flow_rsz = [cv2.resize(e, target_size) for e in flow_val]

                #np.save(os.path.join(out_dir, '%05d_img1.npy' % cnt), img1_rsz)
                #np.save(os.path.join(out_dir, '%05d_img2.npy' % cnt), img2_rsz)

                for i, (img1_1,img2_1,flow_1) in enumerate(zip(img1_rsz, img2_rsz, flow_rsz)):
                    np.save(os.path.join(out_dir, '%05d_img1.npy') % (cnt+i), img1_1)
                    np.save(os.path.join(out_dir, '%05d_img2.npy') % (cnt+i), img2_1)
                    np.save(os.path.join(out_dir, '%05d_flow.npy') % (cnt+i), flow_1)
                cnt += len(img1_rsz)

        #pred_flow_val = net(sess, im1, im2)

        ## warm-up
        #for i in range(10):
        #    pred_flow_val = net(sess, im1, im2)

        #start = time.time()
        #for i in range(100):
        #    pred_flow_val = net(sess, im1, im2)
        #end = time.time()
        #print('Took {} sec'.format(end-start)) # ~60ms per one!!

        # Double check loading is correct
        #for var in tf.all_variables():
        #  print(var.name, var.eval(session=sess).mean())
        #feed_dict = {im1_pl: im1, im2_pl: im2}
        #pred_flow_val = sess.run(pred_flow, feed_dict=feed_dict)

    # Visualization
    #import matplotlib.pyplot as plt
    #flow_im = flow_to_image(flow_val[0])
    #overlay = cv2.addWeighted(im1[0], 0.5, im2[0], 0.5, 0.0)
    #fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2,2)
    #ax0.imshow(overlay)
    #ax1.imshow(flow_im)
    opt#ax2.imshow(normalize(flow_val[0,...,0]), cmap='gray') # u-channel
    #ax3.imshow(normalize(flow_val[0,...,1]), cmap='gray') # v-channel
    #plt.show()

if __name__ == '__main__':
    main()
    # Feed forward
