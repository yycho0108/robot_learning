import numpy as np
from scipy.misc import imread
import cv2
import tensorflow as tf
import os
import sys
import time
from utils.ilsvrc_utils import ILSVRCLoader
from utils import mkdir, proc_img

from flow_net_bb import FlowNetBB

root = '/home/jamiecho/Repos/tf_flownet2'
sys.path.append(root)
from FlowNet2_src import flow_to_image

# Visualization
from matplotlib import pyplot as plt

class GetOptFlowNet(object):
    def __init__(self):
        pass

    def _build(self):
        # Graph construction
        #im1_pl = tf.placeholder(tf.float32, [None, None, None, 3])
        #im2_pl = tf.placeholder(tf.float32, [None, None, None, 3])
        #im1_in = tf.image.resize_images(im1_pl, (384,512))
        #im2_in = tf.image.resize_images(im2_pl, (384,512))

        im1_in = im1_pl = tf.placeholder(tf.float32, [None, 240, 320, 3])
        im2_in = im2_pl = tf.placeholder(tf.float32, [None, 240, 320, 3])
        net = FlowNetBB(step=None, img=tf.stack([im1_in, im2_in],axis=1),
                lab=None, train=False, eval=False)

        self.im1_ = im1_pl
        self.im2_ = im2_pl
        self.flow_ = net.pred_

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

    ckpt_file = os.path.expanduser('~/fn/69/ckpt/model.ckpt-20000')
    #gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.95)
    #config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
    config=None


    with tf.Session(graph=graph, config=config) as sess:
        #saver.restore(sess, ckpt_file)
        net.load(sess, ckpt_file)

        def handle_key(event):
            global index
            sys.stdout.flush()
            if event.key in ['q', 'escape']:
                sys.exit()
            else:
                data_i = np.random.randint(1,31)
                loader = ILSVRCLoader(os.getenv('ILSVRC_ROOT'), data_type=('train_%d'%data_i), T=8,
                        size=(320,240)
                        )
                imgs = loader.grab_pair(batch_size=1)[...,::-1] # RGB->BGR
                p_imgs = proc_img(imgs)
                pred_flow_val = net(sess, p_imgs[:,0], p_imgs[:,1])

                im1 = imgs[:,0]
                im2 = imgs[:,1]

                flow_im = flow_to_image(pred_flow_val[0])
                overlay = cv2.addWeighted(im1[0], 0.5, np.roll(im2[0],1,axis=-1), 0.5, 0.0)
                ax0.imshow(overlay)
                ax1.imshow(flow_im)
                ax2.imshow(normalize(pred_flow_val[0,...,0]), cmap='gray') # u-channel
                ax3.imshow(normalize(pred_flow_val[0,...,1]), cmap='gray') # v-channel
                fig.canvas.draw()

        fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2,2)
        fig.canvas.mpl_connect('close_event', sys.exit)
        fig.canvas.mpl_connect('key_press_event', handle_key)
        plt.show()

        #while True:
        #    data_i = np.random.randint(1,31)
        #    loader = ILSVRCLoader(os.getenv('ILSVRC_ROOT'), data_type=('train_%d'%data_i), T=8,
        #            size=(320,240)
        #            )
        #    imgs = loader.grab_pair(batch_size=1)
        #    p_imgs = proc_img(imgs)
        #    pred_flow_val = net(sess, p_imgs[:,0], p_imgs[:,1])

        #    im1 = imgs[:,0]
        #    im2 = imgs[:,1]

        #    #pred_flow_val = net(sess, im1, im2)

        #    ## warm-up
        #    #for i in range(10):
        #    #    pred_flow_val = net(sess, im1, im2)

        #    #start = time.time()
        #    #for i in range(100):
        #    #    pred_flow_val = net(sess, im1, im2)
        #    #end = time.time()
        #    #print('Took {} sec'.format(end-start)) # ~60ms per one!!

        #    # Double check loading is correct
        #    #for var in tf.all_variables():
        #    #  print(var.name, var.eval(session=sess).mean())
        #    #feed_dict = {im1_pl: im1, im2_pl: im2}
        #    #pred_flow_val = sess.run(pred_flow, feed_dict=feed_dict)

        #    flow_im = flow_to_image(pred_flow_val[0])
        #    overlay = cv2.addWeighted(im1[0], 0.5, im2[0], 0.5, 0.0)
        #    ax0.imshow(overlay)
        #    ax1.imshow(flow_im)
        #    ax2.imshow(normalize(pred_flow_val[0,...,0]), cmap='gray') # u-channel
        #    ax3.imshow(normalize(pred_flow_val[0,...,1]), cmap='gray') # v-channel

if __name__ == '__main__':
    main()
    # Feed forward
