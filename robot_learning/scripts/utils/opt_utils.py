import numpy as np
import cv2
import os
import sys

from utils import normalize

from sklearn.neighbors import NearestNeighbors

def apply_opt(img, opt, scale=1.0, inv=True):
    n,m = np.shape(img)[:2]
    g = np.mgrid[0:n,0:m]
    g = np.stack([g[1], g[0]], axis=-1) # u-v

    mp = (g+opt*scale).astype(np.float32) # mp(x_a,y_b) -> (x_b,y_b)

    print(mp.dtype, mp.shape)

    if inv: # img2 -> img1
        # cv2.remap ... dst(x,y) = src(m(x), m(y))
        return cv2.remap(img, mp, None,
                interpolation=cv2.INTER_LINEAR
                )
    else:
        #opt1 : simple
        #res = np.full_like(img, 255)
        #mp_j  = np.clip(mp[...,0], 0, m-1).astype(np.int32)
        #mp_i  = np.clip(mp[...,1], 0, n-1).astype(np.int32)
        #g_j  = np.clip(g[...,0], 0, m-1).astype(np.int32)
        #g_i  = np.clip(g[...,1], 0, n-1).astype(np.int32)
        #res[mp_i, mp_j] = img[g_i, g_j]
        #return res
        mp_f = mp.reshape(-1,2) # flatten
        g_f  = g.reshape(-1,2)
        neigh = NearestNeighbors(4)
        neigh.fit(mp_f) # samples from x_b
        idx = neigh.kneighbors(g_f, return_distance=False) # x_b -> x_a
        mp_i = np.mean(mp_f[idx], axis=1).reshape(n,m,2)
        return cv2.remap(img, mp_i, None,
                interpolation=cv2.INTER_NEAREST
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

def show_flow(img1, img2, flow):
    cache = {'index':0}
    def show(index):
        ax0.imshow(unproc(aimg[index,0]))
        ax1.imshow(unproc(aimg[index,1]))
        ax2.imshow(flow_to_image(aflo[index]))
        ax3.imshow(apply_opt(unproc(aimg[index,1]), aflo[index,...,:2]))
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

    fig, ((ax0,ax1),(ax2,ax3)) = plt.subplots(2,2)
    fig.canvas.mpl_connect('close_event', sys.exit)
    fig.canvas.mpl_connect('key_press_event', press)
    show(cache['index'])
    plt.show()

def main():
    root = os.path.expanduser('~/dispset')
    didx = str(np.random.randint(1,31))
    ddir = os.path.join(root, didx)

    pred = np.load(os.path.join(ddir, 'pred.npy'))
    img1 = np.load(os.path.join(ddir, 'img1.npy'))
    img2 = np.load(os.path.join(ddir, 'img2.npy'))

    print pred.shape
    print img1.shape

    n = len(pred)
    for i in range(n):
        pu = normalize(pred[i,...,0])
        pv = normalize(pred[i,...,1])
        cv2.imshow('i1',img1[i,...,::-1])
        cv2.imshow('i2',img2[i,...,::-1])
        #img2_re = apply_opt(img2[i], pred[i])
        img1_re = apply_opt(img1[i], pred[i], inv=False)
        cv2.imshow('re',img1_re[...,::-1])
        cv2.imshow('pu',pu)
        cv2.imshow('pv',pv)
        if cv2.waitKey(0) == 27:
            break

if __name__ == "__main__":
    main()
