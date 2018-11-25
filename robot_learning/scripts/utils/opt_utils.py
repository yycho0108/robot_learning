import numpy as np
import cv2
import os
import sys

from utils import normalize
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt

def apply_opt(img, opt, scale=1.0, inv=True):
    n,m = np.shape(img)[:2]
    g = np.mgrid[0:n,0:m]
    g = np.stack([g[1], g[0]], axis=-1) # u-v

    mp = (g+opt*scale).astype(np.float32) # mp(x_a,y_b) -> (x_b,y_b)
    #print(mp.dtype, mp.shape)

    if inv: # img2 -> img1
        # cv2.remap ... dst(x,y) = src(m(x), m(y))
        # img1_r[y,x] = img2[ m[y,x][1], m[y,x][0] ]
        #mp_m = np.clip(mp,[0,0],[m-1,n-1]).astype(np.int32)
        #msk = np.zeros_like(img[...,:1])
        #msk[mp_m[...,1], mp_m[...,0], :] = 1
        #img = img * msk
        return cv2.remap(img, mp, None,
                interpolation=cv2.INTER_LINEAR
                )
    else:
        #opt1 : simple
        res = np.full_like(img, 255)

        fmag = np.linalg.norm(mp, axis=-1)
        print fmag.shape, mp.shape

        idx = np.argsort(fmag.ravel())
        #idx = np.arange(fmag.size).reshape(fmag.shape)

        #idx = np.unravel_index(idx, fmag.shape)

        #mp, uidx = np.unique(mp[idx], axis=2)
        #g = g[uidx]
         
        mp_j = mp[...,0].ravel()[idx].reshape( (n,m) )#[idx]
        mp_i = mp[...,1].ravel()[idx].reshape( (n,m) )#[idx]
        g_j  = g[...,0].ravel()[idx].reshape( (n,m) )#[idx]
        g_i  = g[...,1].ravel()[idx].reshape( (n,m) )#[idx]

        mp_j  = np.clip(mp_j, 0, m-1).astype(np.int32)
        mp_i  = np.clip(mp_i, 0, n-1).astype(np.int32)
        g_j  = np.clip(g_j, 0, m-1).astype(np.int32)
        g_i  = np.clip(g_i, 0, n-1).astype(np.int32)
        res[mp_i, mp_j] = img[g_i, g_j]
        return res

        #opt2 : invert map
        #mp_f = mp.reshape(-1,2) # flatten
        #g_f  = g.reshape(-1,2)
        #neigh = NearestNeighbors(4)
        #neigh.fit(mp_f) # samples from x_b
        #idx = neigh.kneighbors(g_f, return_distance=False) # x_b -> x_a
        #mp_i = np.mean(mp_f[idx], axis=1).reshape(n,m,2)
        #return cv2.remap(img, mp_i, None,
        #        interpolation=cv2.INTER_NEAREST
        #        )

def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255

    return colorwheel

def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u**2+v**2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a+1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))

    return img

def flow_to_image(flow, display=False, thresh=1e7):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    # from https://github.com/vt-vl-lab/tf_flownet2.git
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.

    idxUnknow = (abs(u) > thresh) | (abs(v) > thresh)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    if display:
        print("max flow: %.4f\nflow range:\nu = %.3f .. %.3f\nv = %.3f .. %.3f" % (maxrad, minu,maxu, minv, maxv))

    u = u/(maxrad + np.finfo(float).eps)
    v = v/(maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)


class FlowShow(object):
    # configurations
    AX_NULL=0
    AX_IMG1=1
    AX_IMG2=2
    AX_FLOW=3
    AX_OVLY=4
    AX_I1I2=5 # apply flow i1->i2
    AX_I2I1=6 # apply flow i2->i1 (inverse)
    AX_FLOX=7 # flow-x component
    AX_FLOY=8 # flow-y component
    AX_CODE=9 # show middlebury color code
    AX_GRAY=10 # gray image

    def __init__(self, n, m, cfg=None):
        self.n_ = n
        self.m_ = m
        self.index_ = 0

        # axis configuration
        cfg = np.zeros([n,m], dtype=np.int32) if (cfg is None) else cfg
        self.cfg_ = cfg

        #assert cfg.shape[0] == n, 'Invalid Shape! '

        # gui
        self.start_ = False
        self.fig_ = None
        self.ax_  = None

        self.data_ = []
        self.code_ = cv2.imread('middlebury_flow_code.png')[...,::-1]

    def start(self):
        self.fig_, self.ax_ = plt.subplots(self.n_, self.m_)
        self.start_ = True

    def configure(self, cfg):
        cfg = np.int32(cfg) # -> cvt to np array
        self.cfg_ = cfg
        n, m = np.shape(cfg)
        self.n_ = n
        self.m_ = m

    def config_axis(self, idx, t):
        self.cfg_[idx] = t

    def add(self, img1, img2, flow):
        if np.ndim(img1) == 4:
            # batch-add
            for (a,b,c) in zip(img1,img2,flow):
                self.add(a,b,c)
        else:
            # single-add
            self.data_.append( [img1, img2, flow] )

    def _draw_ax(self, i, j):
        ax = self.ax_[i,j]
        cfg = self.cfg_[i,j]
        img1, img2, flow = self.data_[self.index_]

        ax.cla()
        ax.axis('off')

        if cfg == FlowShow.AX_NULL:
            pass
        elif cfg == FlowShow.AX_IMG1:
            ax.set_title('img1')
            ax.imshow(img1)
        elif cfg == FlowShow.AX_IMG2:
            ax.set_title('img2')
            ax.imshow(img2)
        elif cfg == FlowShow.AX_FLOW:
            ax.set_title('flow')
            print(flow.max(), flow.min())
            ax.imshow( flow_to_image(flow) )
        elif cfg == FlowShow.AX_OVLY:
            ax.set_title('overlay')
            ovly = cv2.addWeighted(img1, 0.5, img2, 0.5, 0.0)
            ax.imshow(ovly)
        elif cfg == FlowShow.AX_I1I2:
            ax.set_title('i1>i2')
            img = apply_opt(img1, flow[...,:2], inv=False)
            ax.imshow(img)
        elif cfg == FlowShow.AX_I2I1:
            ax.set_title('i2>i1')
            img = apply_opt(img2, flow[...,:2], inv=True)
            ax.imshow(img)
        elif cfg == FlowShow.AX_FLOX:
            ax.set_title('flow_x')
            ax.imshow( normalize(flow[...,0]) )
        elif cfg == FlowShow.AX_FLOY:
            ax.set_title('flow_y')
            ax.imshow( normalize(flow[...,1]) )
        elif cfg == FlowShow.AX_CODE:
            ax.set_title('code')
            ax.imshow(self.code_)
        elif cfg == FlowShow.AX_GRAY:
            ax.set_title('gray')
            ax.imshow(self.code_, cmap='gray')

    def draw(self):
        for i in range(self.n_):
            for j in range(self.m_):
                self._draw_ax(i,j)
        self.fig_.suptitle('Optical Flow Display {}/{}'.format(1+self.index_, len(self.data_)))
        self.fig_.canvas.draw()

    def key_cb(self, event):
        index = self.index_
        print('key', event.key)
        if event.key in ['x','q','escape']:
            sys.exit()
        if event.key in ['right', 'n']:
            index += 1
        if event.key in ['left', 'p']:
            index -= 1
        index = (index % len(self.data_))

        if (index != self.index_):
            # redraw only if necessary
            self.index_ = index
            self.draw()

    def show(self):
        if not self.start_:
            self.start()
        self.fig_.canvas.mpl_connect('close_event', sys.exit)
        self.fig_.canvas.mpl_connect('key_press_event', self.key_cb)
        self.draw()
        plt.show()

def main():
    pass

    #root = os.path.expanduser('~/dispset')
    #didx = str(np.random.randint(1,31))
    #ddir = os.path.join(root, didx)

    #pred = np.load(os.path.join(ddir, 'pred.npy'))
    #img1 = np.load(os.path.join(ddir, 'img1.npy'))
    #img2 = np.load(os.path.join(ddir, 'img2.npy'))

    #print pred.shape
    #print img1.shape

    #n = len(pred)
    #for i in range(n):
    #    pu = normalize(pred[i,...,0])
    #    pv = normalize(pred[i,...,1])
    #    cv2.imshow('i1',img1[i,...,::-1])
    #    cv2.imshow('i2',img2[i,...,::-1])
    #    #img2_re = apply_opt(img2[i], pred[i])
    #    img1_re = apply_opt(img1[i], pred[i], inv=False)
    #    cv2.imshow('re',img1_re[...,::-1])
    #    cv2.imshow('pu',pu)
    #    cv2.imshow('pv',pv)
    #    if cv2.waitKey(0) == 27:
    #        break

if __name__ == "__main__":
    main()
