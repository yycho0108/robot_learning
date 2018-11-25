import numpy as np
import os
import sys
import cv2

from opt_utils import flow_to_image, apply_opt
from matplotlib import pyplot as plt
from utils import proc_img

def load_flo(f):
    header = f.read(4) # == 'PIEH'
    w, h = [np.fromfile(f, np.int32, 1).squeeze() for _ in range(2)]
    flow = np.fromfile(f, np.float32, w*h*2).reshape( (h,w,2) )
    return flow

def load_chair1(data_root, index, size=None):
    fname = os.path.join(data_root, ('%05d_flow.flo' % index))
    with open(fname, 'rb') as f:
        flow = load_flo(f)
    if size is not None:
        flow = cv2.resize(flow, size)
        # rectify flow magnitude
        w1, h1 = size
        h0, w0 = flow.shape[:2]
        flow[...,0] *= (w1 / w0)
        flow[...,1] *= (h1 / h0)
    img1  = cv2.imread(os.path.join(data_root, ('%05d_img1.ppm' % index)))
    if size is not None:
        img1  = cv2.resize(img1, size) # 320x240 u8
    img2  = cv2.imread(os.path.join(data_root, ('%05d_img2.ppm' % index)))
    if size is not None:
        img2  = cv2.resize(img2, size) # 320x240 u8
    return (img1, img2, flow)

def load_chair(data_root, n, size=None):
    idx = 1 + np.random.choice(22872, size=n, replace=False)
    data = zip(*[load_chair1(data_root, i, size=size) for i in idx])
    img1, img2, flow = [np.stack(e, axis=0) for e in data]
    return (img1, img2, flow)

def load_ilsvrc1(data_root, index, size=None):
    img1 = np.load(os.path.join(data_root, '%05d_img1.npy' % index))[...,::-1]
    if size is not None:
        img1 = cv2.resize(img1, size)
    img2 = np.load(os.path.join(data_root, '%05d_img2.npy' % index))[...,::-1]
    if size is not None:
        img2 = cv2.resize(img2, size)
    flow = np.load(os.path.join(data_root, '%05d_flow.npy' % index))
    if size is not None:
        flow = cv2.resize(flow, size)
        # rectify flow magnitude
        w1, h1 = size
        h0, w0 = flow.shape[:2]
        flow[...,0] *= (w1 / w0)
        flow[...,1] *= (h1 / h0)
    return (img1, img2, flow)

def load_ilsvrc(data_root, n, size=None):
    #idx = np.random.choice(22732, size=n, replace=False)
    idx = np.random.choice(95, size=n, replace=False)
    data = zip(*[load_ilsvrc1(data_root, i, size=size) for i in idx])
    img1, img2, flow = [np.stack(e, axis=0) for e in data]
    return (img1, img2, flow)

def main():
    n_test = 32
    #data_root = os.path.expanduser('~/Downloads/FlyingChairs/data')
    #img1, img2, flow = load_chair(data_root, n_test)

    #data_root = os.path.expanduser('~/dispset/data/')
    #img1, img2, flow = load_ilsvrc(data_root, n_test)

    data_root = os.path.expanduser('~/dispset/data2/')
    img1, img2, flow = load_ilsvrc(data_root, n_test)

    #print img1.std(), img1.mean()
    #pimg = proc_img(img1)
    #print pimg.std(), pimg.mean()
    #print img2.std()

    print flow.std()
    print flow.shape
    print flow.dtype, flow.max(), flow.min()
    
    cache = {'index':0}

    def show(index):
        ax0.imshow(img1[index])
        ax1.imshow(img2[index])
        ax2.imshow(flow_to_image(flow[index]))

        img1_re = apply_opt(img2[index], flow[index], inv=True)
        ax3.imshow(img1_re)

        d1 = np.clip(img2[index].astype(np.int32) - img1[index], 0, 255).astype(np.uint8)
        d1 = np.mean(np.abs(d1), axis=-1).astype(np.uint8)
        print(d1.sum())

        d2 = np.clip(img1_re.astype(np.int32) - img1[index].astype(np.int32), 0, 255).astype(np.uint8)
        d2 = np.mean(np.abs(d2), axis=-1).astype(np.uint8)
        print(d2.sum())

        ax4.imshow(d1)
        ax5.imshow(d2)

        fig.canvas.draw()

    def press(event):
        index = cache['index']
        if event.key in ['x','q','escape']:
            sys.exit()
        if event.key in ['right', 'n']:
            index += 1
        if event.key in ['left', 'p']:
            index -= 1
        index = (index % n_test)
        cache['index'] = index
        show(index)

    fig, ((ax0,ax1),(ax2,ax3), (ax4,ax5)) = plt.subplots(3,2)
    fig.canvas.mpl_connect('key_press_event', press)
    show(cache['index'])
    plt.show()

if __name__ == "__main__":
    main()
