import numpy as np
import cv2
import os

from utils import normalize

def apply_opt(img, opt, scale=1.0):
    n,m = np.shape(img)[:2]
    g = np.mgrid[0:n,0:m]
    g = np.stack([g[1], g[0]], axis=-1) # u-v
    mp = (g.astype(np.float32)+opt*scale)
    return cv2.remap(img, mp, None,
            interpolation=cv2.INTER_LINEAR
            )

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
        img2_re = apply_opt(img2[i], pred[i])
        cv2.imshow('re',img2_re[...,::-1])
        cv2.imshow('pu',pu)
        cv2.imshow('pv',pv)
        if cv2.waitKey(0) == 27:
            break

if __name__ == "__main__":
    main()
