import numpy as np
import cv2
from matplotlib import pyplot as plt

def drawlines(img1,img2,lines,pts1,pts2,cols):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape[:2]
    for r,pt1,pt2,color in zip(lines,pts1,pts2,cols):
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

def print_ratio(msg, a, b):
    as_int = np.issubdtype(type(a), np.integer)
    if b == 0:
        q = np.nan
    else:
        q = float(a) / b

    if as_int:
        print '{} : {}/{} = {:.2f}%'.format(
                msg, a, b, 100 * q)
    else:
        print '{} : {:.4f}/{:.4f} = {:.2f}%'.format(
                msg, a, b, 100 * q)

def drawMatches(img1, img2, pt1, pt2, msk,
        radius = 3
        ):
    h,w = np.shape(img1)[:2]
    pt1 = np.round(pt1).astype(np.int32)
    pt2 = np.round(pt2 + [[w,0]]).astype(np.int32)

    mim = np.concatenate([img1, img2], axis=1)
    mim0 = mim.copy()

    for (p1, p2) in zip(pt1[msk], pt2[msk]):
        p1 = tuple(p1)
        p2 = tuple(p2)
        col = tuple(np.random.randint(255, size=4))
        cv2.line(mim, p1, p2, col, 2)
        cv2.circle(mim, tuple(p1), radius, col, 2)
        cv2.circle(mim, tuple(p2), radius, col, 2)

    for p in pt1[~msk]:
        cv2.circle(mim, tuple(p), radius, (255,0,0), 1)

    for p in pt2[~msk]:
        cv2.circle(mim, tuple(p), radius, (255,0,0), 1)

    mim = cv2.addWeighted(mim0, 0.5, mim, 0.5, 0.0)

    return mim

lmk_fig = None
def show_landmark_2d(pos, cov, clear=True, draw=True,
        style='k+',
        colors=None,
        label=''
        ):
    """ from https://stackoverflow.com/a/20127387 """

    global lmk_fig
    if lmk_fig is None:
        lmk_fig = plt.figure()
    ax = lmk_fig.gca()
    if clear:
        ax.cla()

    # subsample
    if len(pos) <= 0:
        return

    n = min(256, len(pos))
    idx = np.random.randint(0, len(pos), size=n)
    pos = pos[idx]
    cov = cov[idx]

    x = pos[:,2]
    y = -pos[:,1]


    if colors is None:
        colors = np.random.uniform(size=(n,3))
    
    ax.plot(x, y, style, alpha=0.75,label=label
            )

    for p,c,col in zip(pos, cov, colors):
        # re-orient to the things we care about ...
        x,y = p[2], -p[1]
        c_2d = np.reshape([
                c[2,2], c[2,1],
                c[1,2], c[1,1]], (2,2))

        l, v = np.linalg.eig(c_2d)
        l    = np.sqrt(l)
        ell = Ellipse(xy=(x, y),
                    width=l[0]*2, height=l[1]*2,
                    angle=np.rad2deg(np.arccos(v[0, 0])))
        #ell.set_facecolor('none') #??
        ax.add_artist(ell)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.25)
        ell.set_facecolor(col)

    ax.set_xlim(-1.0, 10.0)
    ax.set_ylim(-5.0, 5.0)
    #ax.set_aspect('equal', 'datalim')
    #ax.autoscale(True)
    #print 'should have added {} ellipses'.format(len(pos))
    if draw:
        ax.legend()
        ax.plot([0],[0],'k+')
        lmk_fig.canvas.draw()

def axisEqual3D(ax):
    """ from https://stackoverflow.com/a/19248731 """
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

def get_points_color(img, pts, w=3):
    n, m = img.shape[:2]
    pis, pjs = np.round(pts[:,::-1]).T.reshape(2,-1).astype(np.int32)
    oi, oj = np.mgrid[-w:w+1,-w:w+1]
    iw, jw = pis[:,None,None] + oi, pjs[:,None,None] + oj
    iw = np.clip(iw, 0, n-1)
    jw = np.clip(jw, 0, m-1)

    cols_w = img[iw, jw] # n,2*w+1,2*w+1,3

    # opt 1 : naive mean
    # cols = np.mean(cols_w, axis=(1,2))
    cols = cols_w.astype(np.float32)
    # opt 2 : rms
    cols = np.sqrt(np.mean(np.square(cols),axis=(1,2)))
    return np.asarray(cols, dtype=img.dtype)


