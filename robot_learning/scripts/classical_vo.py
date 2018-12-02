import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
from matplotlib import pyplot as plt

from tf import transformations as tx
from utils.vo_utils import add_p3d

class ClassicalVO(object):
    def __init__(self):
        self.prv_ = None # previous image

        # build detector
        self.orb_ = cv2.ORB_create(nfeatures=256, scaleFactor=1.2, WTA_K=2)

        # build flann matcher
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        self.flann_ = flann

    def match_v1(self, des1, des2, thresh=150.0):
        # apply flann
        flann = self.flann_
        if len(des2) < 2:
            return None

        try:
            matches = flann.knnMatch(des1, des2, k=2)
        except Exception as e:
            print 'Exception during match : {}'.format(e)
            print np.shape(des1)
            print np.shape(des2)
            return None

        # lowe filter
        good = []
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.75*n.distance and \
                    m.distance < thresh:
                good.append(m)
        good = np.asarray(good, dtype=cv2.DMatch)
        return good

    def match(self, des1, des2, thresh=150.0):
        # (inefficient & slow right now)
        # hungarian algorithm based matching
        nax = np.newaxis
        c = np.linalg.norm(des1[:,nax] - des2[nax,:], axis=-1)

        # cost mask with outliers removed
        i_good = np.where(np.min(c, axis=1) < 150.0)[0]
        j_good = np.where(np.min(c, axis=0) < 150.0)[0]

        c = c[i_good[:,nax], j_good]

        mi, mj = linear_sum_assignment(c)

        # matching cost
        c      = c[mi,mj]

        # real match index
        mi, mj = i_good[mi], j_good[mj]

        matches = [cv2.DMatch(
            _queryIdx = e1,
            _trainIdx = e2,
            _distance = e3)
            for (e1,e2,e3) in zip(mi, mj, c)]
        return matches
        #return mi, mj


        ## homography
        #M, mask = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
        #mask = np.cast(mask, np.bool)
        #return good[mask]

    #def E(self, kp1, kp2, matches, focal=530.0):
    #    return cv2.findEssentialMat(p1, p2)[0]

    def __call__(self, img):
        kp2, des2 = self.orb_.detectAndCompute(img, None)
        if des2 is None:
            # something is invalid about the image?
            return True, None

        kp2 = np.array(kp2)
        img2 = img

        des2 = np.float32(des2)
        if self.prv_ is None:
            self.prv_ = (kp2, des2, img2)
            return True, None

        kp1, des1, img1 = self.prv_
        matches = self.match_v1(des1, des2)
        if matches is None:
            return True, None
        #matches = self.match(des1, des2)

        if len(matches) <= 5:
            # skip frame without updating prv
            return True, None

        draw_params = dict(matchColor = (0,255,0),
                singlePointColor = (255,0,0),
                flags = 0)

        i1, i2 = np.int32([(m.queryIdx, m.trainIdx) for m in matches]).T
        p1 = np.int32([e.pt for e in kp1[i1]])
        p2 = np.int32([e.pt for e in kp2[i2]])

        focal = 530.0 / 2.0
        pp    = (0,0)#(160,120)

        Emat = cv2.findEssentialMat(p2, p1, focal=focal, pp=pp)
        Emat = Emat[0] # p2 w.r.t. p1
        _, R, t, _ = cv2.recoverPose(Emat, p2, p1, focal=focal, pp=pp)

        h = tx.euler_from_matrix(R)
        #h = np.round(np.rad2deg(tx.euler_from_matrix(R)), 2)
        #t = np.round(t, 2)
        #print ('h-z', -h[1])
        #print ('t-xy', t[2,0], -t[0,0])

        h = -h[1]
        t = [t[2,0], -t[0,0]]

        ##kim1 = cv2.drawKeypoints(img1, kp1[m1], img1.copy() )
        ##kim2 = cv2.drawKeypoints(img2, kp2[m2], img2.copy() )
        #h, w = img1.shape[:2]
        #mim  = np.concatenate([img1, img2], axis=1)
        #color = np.random.randint(0,255,(len(m1),3))
        #kp1_m = kp1[m1]
        #kp2_m = kp2[m2]

        #for i,(new,old) in enumerate(zip(kp1_m,kp2_m)):
        #    a,b = np.int32(old.pt)
        #    c,d = np.int32(new.pt)
        #    cv2.line(mim, (a,b),(c+w,d), color[i].tolist(), 2)
        #    cv2.circle(mim,(a,b),5,color[i].tolist(),-1)
        #    cv2.circle(mim,(c+w,d),5,color[i].tolist(),-1)


        self.prv_ = (kp2, des2, img2)

        mim = cv2.drawMatches(
                img1,kp1,img2,kp2,
                matches,None,**draw_params)

        return True, (mim, h, t)

        #print 'm', m

def main():
    imgs = np.load('../data/train/9/img.npy')
    vo = ClassicalVO()

    fig, (ax0,ax1) = plt.subplots(1,2)

    pos = np.zeros(shape=3, dtype=np.float32)

    tx = []
    ty = []

    for img in imgs:
        suc, res = vo(img)
        if suc:
            if res is not None:
                (img, dh, dt) = res
                dps = np.float32([dt[0], dt[1], dh])
                pos = add_p3d(pos, dps)
                tx.append(pos[0])
                ty.append(pos[1])
                ax0.imshow(img[...,::-1])
                ax1.plot(tx, ty)
                fig.canvas.draw()
                plt.pause(0.001)
        else:
            return

if __name__ == "__main__":
    main()
