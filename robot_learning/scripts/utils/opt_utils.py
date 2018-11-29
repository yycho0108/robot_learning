import numpy as np
import cv2
import os
import sys

from utils import normalize
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt

def map_inv(src, delta, window=9, eps=1e-9):
    shape = np.shape(src)

    src   = np.float32(src.reshape([-1, 2]))
    delta = np.float32(delta.reshape([-1, 2]))
    dst   = src + delta

    neigh = NearestNeighbors(window)
    neigh.fit(dst) # samples from x_b

    dist, idx = neigh.kneighbors(src) # x_b -> x_a
    # == dst[idx] is a neighbor of src

    idx = idx.reshape(shape[0], shape[1], -1)

    # inverse-delta
    #delta_i = src[idx] - dst[idx]
    delta_i = -delta[idx]
    weight  = (1.0 / (dist + eps)).reshape(shape[0], shape[1], -1, 1)

    # below is probably only valid for opt-flow related applications
    # weigh by flow magnitude
    # weight *= np.linalg.norm(delta_i)[...,np.newaxis,np.newaxis]

    # weighted mean
    delta_i = np.sum(delta_i*weight, axis=2) / (np.sum(weight, axis=2))
    map_i = np.float32(src.reshape(shape) + delta_i)

    return map_i
    # mp_fw[src1=(src+delta)] = src0
    # mp_fw[ 

    # mp_bw[src+delta] = src

    # src+delta -> src
    #return cv2.remap(-src_map, src_map, None, interpolation=cv2.INTER_LINEAR)

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
                interpolation=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255,0,0)
                )
    else:
        ##opt1 : simple
        #res = np.full_like(img, 255)

        #fmag = np.linalg.norm(mp, axis=-1)
        #print fmag.shape, mp.shape

        #idx = np.argsort(fmag.ravel())
        ##idx = np.arange(fmag.size).reshape(fmag.shape)

        ##idx = np.unravel_index(idx, fmag.shape)

        ##mp, uidx = np.unique(mp[idx], axis=2)
        ##g = g[uidx]
        # 
        #mp_j = mp[...,0].ravel()[idx].reshape( (n,m) )#[idx]
        #mp_i = mp[...,1].ravel()[idx].reshape( (n,m) )#[idx]
        #g_j  = g[...,0].ravel()[idx].reshape( (n,m) )#[idx]
        #g_i  = g[...,1].ravel()[idx].reshape( (n,m) )#[idx]

        #mp_j  = np.clip(mp_j, 0, m-1).astype(np.int32)
        #mp_i  = np.clip(mp_i, 0, n-1).astype(np.int32)
        #g_j  = np.clip(g_j, 0, m-1).astype(np.int32)
        #g_i  = np.clip(g_i, 0, n-1).astype(np.int32)
        #res[mp_i, mp_j] = img[g_i, g_j]
        #return res

        #opt2 : invert map
        #mp_f = mp.reshape(-1,2) # flatten
        #g_f  = g.reshape(-1,2)
        #neigh = NearestNeighbors(81)
        #neigh.fit(mp_f) # samples from x_b
        #idx = neigh.kneighbors(g_f, return_distance=False) # x_b -> x_a
        #mp_i = np.max(mp_f[idx], axis=1).reshape(n,m,2)

        mp_i = map_inv(g, opt)

        return cv2.remap(img, mp_i, None,
                interpolation=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255,0,0)
                )

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

#def flow_to_image_tf(flow, thresh=1e7, eps=1e-9):
#    """
#    Convert flow into middlebury color code image
#    :param flow: optical flow map
#    :return: optical flow image in middlebury color
#    """
#    # from https://github.com/vt-vl-lab/tf_flownet2.git
#
#
#    # first, rectify by flow magnitude
#    rad  = tf.norm(flow, axis=-1)
#    flow = tf.where(rad > thresh,
#            tf.zeros_like(flow),
#            flow)
#
#
#    rad  = tf.norm(flow, axis=-1)
#    flow = flow / (eps + tf.reduce_max(rad))
#
#    ang  = 
#
#    u,v = tf.unstack(flow, axis=-1)[:2]
#
#    u = flow[:, :, 0]
#    v = flow[:, :, 1]
#
#    maxu = -999.
#    maxv = -999.
#    minu = 999.
#    minv = 999.
#
#    msk_unk = tf.logical_or(
#            tf.greater(tf.abs(u), thresh),
#            tf.greater(tf.abs(v), thresh)
#            ) # 'unknown'
#    u = tf.where(msk_unk, tf.zeros_like(u), u)
#    v = tf.where(msk_unk, tf.zeros_like(v), v)
#
#    #rad = np.sqrt(u ** 2 + v ** 2)
#    rad = tf.norm(flow, axis=-1)
#    rad_max = tf.maximum(-1.0, tf.reduce_max(rad))
#
#    u = u/(rad_max + eps)
#    v = v/(rad_max + eps)
#
#    img = compute_color(u, v)
#
#    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
#    img[idx] = 0
#
#    return np.uint8(img)



class FlowShow(object):
    # indexing
    I_IMG1 = 0
    I_IMG2 = 1
    I_FLOW = 2

    # configurations
    AX_NULL=0
    AX_IMG1=1
    AX_IMG2=2
    AX_FLOW=3
    AX_OVLY=4
    AX_DIFF=5
    AX_I1I2=6 # apply flow i1->i2
    AX_I2I1=7 # apply flow i2->i1 (inverse)
    AX_FLOX=8 # flow-x component
    AX_FLOY=9 # flow-y component
    AX_CODE=10 # show middlebury color code
    AX_GRAY=11 # gray image
    AX_FLOG=12 # apply flow on grid
    AX_FLOF=13 # flow field
    AX_I2ER=14 # i2-err, (i1->i2) - i2
    AX_I2OV=15 # i2-overlay, (i1->i2), i2

    AX_USER=100 # apply user-defined drawing fn from here

    def __init__(self, layout=None, cfg=None,
            code_path=''
            ):

        self.n_ = None
        self.m_ = None
        self.cfg_ = None
        self.configured_ = False

        if (layout is not None):
            self.configure_layout(layout)
        if (cfg is not None):
            self.configure(cfg)
        if not self.configured_:
            print('No configuration initialized during __init__')

        # gui
        self.start_ = False
        self.fig_ = None
        self.ax_  = None

        self.data_ = []
        self.udata_ = {}
        self.index_ = 0
        self.draw_cb_ = []

        if len(code_path) > 0:
            try:
                self.code_ = cv2.imread(code_path)[...,::-1]
            except Exception as e:
                print('Exception : {}'.format(e))
                print('Could not read code image from supplied path : {}'.format(code_path))
        else:
            # self.code_ is fine as "None" as long as AX_CODE is not requested.
            self.code_ = None

    def start(self):
        self.fig_, self.ax_ = plt.subplots(self.n_, self.m_)
        self.start_ = True

    def configure_layout(self, layout):
        n,m = layout
        cfg = np.zeros([n,m], dtype=np.int32)
        self.n_ = n
        self.m_ = m
        self.cfg_ = cfg
        self.configured_ = True

    def configure(self, cfg):
        cfg = np.int32(cfg) # -> cvt to np array
        n, m = np.shape(cfg)
        self.n_ = n
        self.m_ = m
        self.cfg_ = cfg
        self.configured_ = True

    def config_axis(self, idx, t):
        if not self.configured_:
            print('Unable to configure axis before specifying layout!')
        else:
            self.cfg_[idx] = t

    def add(self, img1, img2, flow):
        if np.ndim(img1) == 4:
            # batch-add
            for (a,b,c) in zip(img1,img2,flow):
                self.add(a,b,c)
        else:
            # single-add
            self.data_.append( [img1, img2, flow] )

    @staticmethod
    def _grid(ref_img):
        h,w = ref_img.shape[:2]
        grid = np.full([h,w], 255, dtype=np.uint8)
        di = int(max(np.round(h / 32.), 2))
        dj = int(max(np.round(w / 32.), 2))
        grid[0:h:di,:] = 0
        grid[1:h:di,:] = 0
        grid[:,0:w:dj] = 0
        grid[:,1:w:dj] = 0
        return grid

    def _draw_ax(self, ax, data, ax_type):
        # prep
        ax.cla()
        ax.axis('off')

        # unroll data
        img1, img2, flow = data

        # TODO : restructure with opcode-based cfg indexing?
        # i.e. cfg = [STYLE={DIFF,OVLY,FLOW,PASSTHROUGH,...}]

        # draw
        if ax_type == FlowShow.AX_NULL:
            ax.set_title('null')
        elif ax_type == FlowShow.AX_IMG1:
            ax.set_title('img1')
            ax.imshow(img1)
        elif ax_type == FlowShow.AX_IMG2:
            ax.set_title('img2')
            ax.imshow(img2)
        elif ax_type == FlowShow.AX_FLOW:
            ax.set_title('flow')
            print(flow.max(), flow.min())
            ax.imshow( flow_to_image(flow) )
        elif ax_type == FlowShow.AX_OVLY:
            ax.set_title('i1-i2 overlay')
            ovly = cv2.addWeighted(img1, 0.5, img2, 0.5, 0.0)
            ax.imshow(ovly)
        elif ax_type == FlowShow.AX_DIFF:
            ax.set_title('i1-i2 difference')
            diff = np.clip(np.abs(np.float32(img2) - img1), 0, 255).astype(img2.dtype)
            ax.imshow(diff)
        elif ax_type == FlowShow.AX_I1I2:
            ax.set_title('i1>i2')
            img = apply_opt(img1, flow[...,:2], inv=False)
            ax.imshow(img)
        elif ax_type == FlowShow.AX_I2I1:
            ax.set_title('i2>i1')
            img = apply_opt(img2, flow[...,:2], inv=True)
            ax.imshow(img)
        elif ax_type == FlowShow.AX_FLOX:
            ax.set_title('flow_x')
            ax.imshow( normalize(flow[...,0]) )
        elif ax_type == FlowShow.AX_FLOY:
            ax.set_title('flow_y')
            ax.imshow( normalize(flow[...,1]) )
        elif ax_type == FlowShow.AX_CODE:
            ax.set_title('code')
            ax.imshow(self.code_)
        elif ax_type == FlowShow.AX_GRAY:
            ax.set_title('gray')
            ax.imshow(self.code_, cmap='gray')
        elif ax_type == FlowShow.AX_FLOG:
            ax.set_title('flow_grid')
            grid = self._grid(img1)
            img = apply_opt(grid, flow[...,:2], inv=False)
            ax.imshow(img, cmap='gray')
        elif ax_type == FlowShow.AX_FLOF:
            ax.set_title('flow_field')
            n, m = np.shape(flow)[:2]

            srcy, srcx = np.meshgrid(
                    np.arange(0,n),
                    np.arange(0,m),
                    indexing='ij')
            flox, floy = flow[...,0], flow[...,1]

            n_sample = int(1024 / 1.618)
            s = np.maximum(np.round( (n*m) / n_sample), 1)
            idx = np.arange(0, n*m, s)
            print len(idx)
            #idx = np.random.choice(n*m, size=n_sample, replace=True)
            [srcx,srcy,flox,floy] = [e.ravel()[idx] for e in 
                    (srcx,srcy,flox,floy)]

            ax.set_ylim(0, n)
            ax.set_xlim(0, m)
            ax.set_aspect('equal')
            ax.quiver(
                    srcx,
                    srcy,
                    flox, #x-y
                    floy, # flip y
                    angles = 'xy',
                    scale_units = 'xy',
                    scale = 1,
                    headwidth = 9.0,
                    headlength = 9.0
                    )
            if not ax.yaxis_inverted():
                ax.invert_yaxis()
        elif ax_type == FlowShow.AX_I2ER:
            ax.set_title('i2-(i1>i2) diff')
            img2_r = apply_opt(img1, flow[...,:2], inv=False)
            diff = np.clip(np.abs(np.float32(img2)-img2_r), 0, 255).astype(img2.dtype)
            ax.imshow(diff)
        elif ax_type == FlowShow.AX_I2OV:
            ax.set_title('i2,(i1>i2) overlay')
            img2_r = apply_opt(img1, flow[...,:2], inv=False)
            ovly = cv2.addWeighted(img2, 0.5, img2_r, 0.5, 0.0)
            ax.imshow(ovly)

    def _draw_ax_at(self, i, j):
        ax = self.ax_[i,j]
        cfg = self.cfg_[i,j]
        if cfg >= FlowShow.AX_USER:
            u_idx, ax_type = self.decode_user(cfg)
            ax.set_title('user')
            print 'dbg', len(self.udata_[u_idx][self.index_])
            self._draw_ax(ax, self.udata_[u_idx][self.index_], ax_type)
        else:
            self._draw_ax(ax, self.data_[self.index_], cfg)

    @staticmethod
    def encode_user(index, ax_type):
        return FlowShow.AX_USER + (index * 16) + (ax_type)

    @staticmethod
    def decode_user(code):
        code = (code - FlowShow.AX_USER)
        index = code // 16
        ax_type = code % 16
        return index, ax_type

    def set_user_data(self, k, v, t):
        if k in self.udata_:
            n = len(v)
            for i in range(n):
                self.udata_[k][i][t] = v[i]
        else:
            # initialize self.udata_ --> (n,3)
            n = len(v)
            self.udata_[k] = [[None,None,None] for _ in range(n)]
            self.set_user_data(k, v, t)

    def add_user_cb(self, fn):
        self.draw_cb_.append(fn)

    @staticmethod
    def full_config():
        cfg = np.int32([[FlowShow.AX_IMG1, FlowShow.AX_IMG2, FlowShow.AX_I2ER, FlowShow.AX_DIFF],
                [FlowShow.AX_I2I1, FlowShow.AX_I1I2, FlowShow.AX_FLOW, FlowShow.AX_OVLY],
                [FlowShow.AX_FLOG, FlowShow.AX_FLOF, FlowShow.AX_CODE, FlowShow.AX_I2OV]])
        return cfg

    def draw(self):
        for i in range(self.n_):
            for j in range(self.m_):
                self._draw_ax_at(i,j)

        for fn in self.draw_cb_:
            fn(self, self.index_, self.ax_, self.fig_)

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
