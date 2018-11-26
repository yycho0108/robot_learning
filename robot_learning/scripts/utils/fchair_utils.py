import numpy as np
import os
import sys
import cv2

from opt_utils import flow_to_image, apply_opt, FlowShow
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
    idx = np.random.choice(22732, size=n, replace=False)
    data = zip(*[load_ilsvrc1(data_root, i, size=size) for i in idx])
    img1, img2, flow = [np.stack(e, axis=0) for e in data]
    return (img1, img2, flow)

def main():
    n_test = 32

    #data_root = os.path.expanduser('~/Downloads/FlyingChairs/data')
    #img1, img2, flow = load_chair(data_root, n_test)
    data_root = os.path.expanduser('~/dispset/data/')
    img1, img2, flow = load_ilsvrc(data_root, n_test)

    disp = FlowShow(code_path='middlebury_flow_code.png')
    disp.configure([
        [FlowShow.AX_IMG1, FlowShow.AX_IMG2, FlowShow.AX_I2ER, FlowShow.AX_DIFF],
        [FlowShow.AX_I1I2, FlowShow.AX_FLOW, FlowShow.AX_I2I1, FlowShow.AX_OVLY],
        [FlowShow.AX_FLOG, FlowShow.AX_FLOF, FlowShow.AX_CODE, FlowShow.AX_NULL],
        ])
    disp.add(img1, img2, flow)
    disp.show()

if __name__ == "__main__":
    main()
