import os
from bs4 import BeautifulSoup
import numpy as np
import cv2
import multiprocessing
from collections import defaultdict
from vid_utils import VIDLoaderBase
from matplotlib import pyplot as plt

def rint(x):
    return np.int32(np.round(x))

def get(ann, tag):
    return ann.findChild(tag).contents[0]

def ann2bbox(ann):
    with open(ann) as f:
        xml = f.readlines()
        xml = ''.join([line.strip('\t') for line in xml])
        ann = BeautifulSoup(xml)

    width = int(get(ann,'width'))
    height = int(get(ann,'height'))
    obj = ann.findChild('object')
    if obj is None:
        return None, None, None
    trackid = str(get(ann,'trackid'))
    box = obj.findChild('bndbox')
    cls = get(obj, 'name')
    ymin = float(get(box,'ymin')) / height
    xmin = float(get(box,'xmin')) / width 
    ymax = float(get(box,'ymax')) / height
    xmax = float(get(box,'xmax')) / width
    return np.asarray([ymin,xmin,ymax,xmax], dtype=np.float32), trackid, cls

def create_index_(data_type, index_dir, ann_dirs, data_dirs, set_dirs):
    V_FLAG = False
    if data_type == 'val':
        V_FLAG = True

    index_file = os.path.join(index_dir, data_type + '.npy')
    print index_file
    if os.path.exists(index_file):
        return

    data = defaultdict(lambda:{
        'boxs' : [],
        'imgs' : [],
        'lbls' : []
        })

    def add_ann_(data, ann_dir, ann_file, seq):
        if ann_file.endswith('.xml'):
            base = ann_file.split('.xml')[0]
            box, trackid, cls = ann2bbox(os.path.join(ann_dir, ann_file))
            if box is not None and trackid is not None:
                seq_key = ('%s_%s' % (seq, trackid))
                img = base+'.JPEG'
                data[seq_key]['boxs'].append(box)
                data[seq_key]['imgs'].append(img)
                data[seq_key]['lbls'].append(cls)

    with open(os.path.join(set_dirs, data_type+'.txt')) as f:
        data_folder = data_type.split('_')[0]
        l = [e.strip().split()[0] for e in f.readlines()]
        print '%s num seq : %d' % (data_type, len(l))
        if V_FLAG:
            ann_dir = ann_dirs[data_folder]
            data_dir = data_dirs[data_folder]
            print len(os.listdir(ann_dir))
            for seq in os.listdir(ann_dir):
                seq_dir = os.path.join(ann_dir, seq)
                for ann_file in sorted(os.listdir(seq_dir)):
                    add_ann_(data, seq_dir, ann_file, seq)
        else:
            for (i, seq) in enumerate(l):
                if(i%10 == 0):
                    print('%s : %d/%d' % (data_type, i, len(l)))
                ann_dir = os.path.join(ann_dirs[data_folder], seq)
                data_dir = os.path.join(data_dirs[data_folder], seq)
                for ann_file in sorted(os.listdir(ann_dir)):
                    add_ann_(data, ann_dir, ann_file, seq)

    for seq_key in data.iterkeys():
        data[seq_key]['boxs'] = np.stack(data[seq_key]['boxs'], axis=0)
        data[seq_key]['imgs'] = np.stack(data[seq_key]['imgs'], axis=0)
        data[seq_key]['lbls'] = np.stack(data[seq_key]['lbls'], axis=0)

    print '%s finished!' % data_type
    np.save(index_file, dict(data))
    return True

def parmap(X, nprocs=multiprocessing.cpu_count()):
    n = len(X)
    proc = [multiprocessing.Process(target=create_index_, args=x) for x in X]
    for p in proc:
        p.daemon = True
        p.start()
    [p.join() for p in proc]

def box_iou(bbox_1, bbox_2):
    lr = np.minimum(bbox_1[3], bbox_2[3]) - np.maximum(bbox_1[1], bbox_2[1])
    tb = np.minimum(bbox_1[2], bbox_2[2]) - np.maximum(bbox_1[0], bbox_2[0])
    lr = np.maximum(lr, lr * 0)
    tb = np.maximum(tb, tb * 0)
    ixn = np.multiply(tb, lr)
    uxn = np.subtract(
      np.multiply((bbox_1[3] - bbox_1[1]), (bbox_1[2] - bbox_1[0])) +
      np.multiply((bbox_2[3] - bbox_2[1]), (bbox_2[2] - bbox_2[0])),
      ixn
    )
    iou = ixn / uxn
    return iou

class ILSVRCLoader(VIDLoaderBase):
    def __init__(self, root_dir, data_type, T=128, size=None):
        self.size_ = (size if size is not None else (320,240))

        self.root_dir = root_dir
        ann_dir = os.path.join(root_dir, 'Annotations', 'VID')
        self.ann_dir = {
                s : os.path.join(ann_dir, s) for s in ['train','val','test']
                }
        data_dir = os.path.join(root_dir, 'Data', 'VID')
        self.data_dir = {
                s: os.path.join(data_dir, s) for s in ['train','val','test']
                }
        self.set_dir = os.path.join(root_dir, 'ImageSets','VID')

        self.index_dir = os.path.join(root_dir, 'Index', 'VID')

        self.data_type = data_type.split('_')[0]

        if not os.path.exists(self.index_dir):
            os.makedirs(self.index_dir)
            self.create_index(self.index_dir)
        self.index = np.load(os.path.join(self.index_dir, data_type+'.npy')).item()

        self.T = T
        self.keys = self.list_all()

    def create_index(self, index_dir):
        #data_types = ['train_%d' % i for i in range(1,31)]# + ['val']
        #data_types = ['val']
        index_split  = np.array_split(np.arange(1,32), 8)
        for indices in index_split:
            data_types = ['train_%d' % i for i in indices]# + ['val']
            args = map(lambda d : (d, index_dir, self.ann_dir, self.data_dir, self.set_dir), data_types)
            parmap(args)

    def list_all(self):
        return list(self.index.keys())

    def grab_delta(self, batch_size, min_T=1, max_T=16):
        seqs = np.random.choice(self.keys, batch_size, replace=True)

        imgs = []
        lbls = []

        for seq in seqs:
            data = self.index[seq]
            seq_dir = '_'.join(seq.split('_')[:-1])
            n = len(data['boxs'])

            max_dT = np.random.randint(min_T, min(n-1, max_T))
            i = np.random.randint(n-max_dT)

            img0 = cv2.imread(
                    os.path.join(
                        self.data_dir[self.data_type],
                        seq_dir,
                        data['imgs'][i]))
            img0 = cv2.resize(img0, self.size_)

            box0 = data['boxs'][i]

            img1 = cv2.imread(
                    os.path.join(
                        self.data_dir[self.data_type],
                        seq_dir, data['imgs'][i+dT]))
            img1 = cv2.resize(img1, self.size_)
            box1 = data['boxs'][i+dT]

            # clear "irrelevant" flow data outside of box
            h,w = img1.shape[:2]
            i00, j00, i01, j01 = box0
            i10, j10, i11, j11 = box1

            # calculate "delta"
            ic0, jc0 = (i00+i01)/2.0, (j00+j01)/2.0
            h0,  w0  = (i01-i00), (j01-j00)
            ic1, jc1 = (i10+i11)/2.0, (j10+j11)/2.0
            h1,  w1  = (i11-i10), (j11-j10)

            di = (ic1 - ic0)
            print('di', di)
            dj = (jc1 - jc0)
            dh = np.log(h1 / h0)
            dw = np.log(w1 / w0)
            lbl = np.float32([di,dj,dh,dw])

            imgs.append(np.concatenate([img0,img1], axis=-1))
            lbls.append(lbl)

        return [np.stack(a, axis=0) for a in [imgs, lbls]]

    def grab_pair(self, batch_size=-1, min_T=2, max_T=8,
            per_seq=1
            ):
        imgs = []

        all_flag = (batch_size == -1)
        
        if all_flag:
            batch_size = len(self.keys) * per_seq

        cnt = 0

        while cnt < batch_size:
            if all_flag:
                seq = self.keys[cnt % len(self.keys)]
            else:
                seq = np.random.choice(self.keys)
            data = self.index[seq]
            seq_dir = '_'.join(seq.split('_')[:-1])
            n = len(data['boxs'])

            if(n <= 1):
                cnt += 1
                continue

            try:
                max_dT = np.random.randint(min_T, min(n, max_T))
                i = np.random.randint(n-max_dT)
            except Exception as e:
                print(e, 'n', n)
                max_dT = 1
                i  = 0

            img0 = cv2.imread(
                    os.path.join(
                        self.data_dir[self.data_type],
                        seq_dir,
                        data['imgs'][i]))
            img0 = cv2.resize(img0, self.size_)
            box0 = data['boxs'][i]

            # "good" sampling
            for dT in range(max_dT, 0, -1):
                box1 = data['boxs'][i+dT]
                iou = box_iou(box0, box1)
                if iou > 0.5: # "Good"
                    break
            # if no "good" dT was found, dT will default to 1 (which is probably good enough)

            img1 = cv2.imread(
                    os.path.join(
                        self.data_dir[self.data_type],
                        seq_dir, data['imgs'][i+dT]))
            img1 = cv2.resize(img1, self.size_)
            box1 = data['boxs'][i+dT]

            imgs.append(np.stack([img0,img1], axis=0))
            cnt += 1

        return np.stack(imgs, axis=0)

    def grab_opt(self, batch_size=-1, min_T=1, max_T=4):
        imgs = []
        lbls = []

        if batch_size == -1:
            batch_size = len(self.keys)

        while len(lbls) < batch_size:
            if batch_size == -1:
                seq = np.random.choice(self.keys)
            else:
                seq = self.keys[len(lbls)]
            data = self.index[seq]
            seq_dir = '_'.join(seq.split('_')[:-1])
            n = len(data['boxs'])

            if(n <= 1): continue

            try:
                dT = np.random.randint(min_T, min(n, max_T))
                i = np.random.randint(n-dT)
            except Exception as e:
                print(e, 'n', n)
                dT = 1
                i  = 0

            img0 = cv2.imread(
                    os.path.join(
                        self.data_dir[self.data_type],
                        seq_dir,
                        data['imgs'][i]))
            img0 = cv2.resize(img0, self.size_)

            box0 = data['boxs'][i]

            img1 = cv2.imread(
                    os.path.join(
                        self.data_dir[self.data_type],
                        seq_dir, data['imgs'][i+dT]))
            img1 = cv2.resize(img1, self.size_)
            box1 = data['boxs'][i+dT]

            # clear "irrelevant" flow data outside of box
            h,w = img1.shape[:2]
            i00, j00, i01, j01 = rint(box0 * [h,w,h,w])
            i10, j10, i11, j11 = rint(box1 * [h,w,h,w])

            i0,j0,i1,j1 = [min(i00,i10), min(j00,j10), max(i01,i11), max(j01,j11)]

            img0g = cv2.cvtColor(img0[i0:i1,j0:j1], cv2.COLOR_BGR2GRAY)
            img1g = cv2.cvtColor(img1[i0:i1,j0:j1], cv2.COLOR_BGR2GRAY)
            opt = np.zeros(dtype=np.float32, shape=[h,w,2])

            # construct flow guess
            flow0 = np.zeros(dtype=np.float32, shape=[i1-i0,j1-j0,2])
            ex_i0, ex_j0 = i00-i0, j00-j0 # "offset" index

            ic0, jc0 = (i00+i01)/2.0, (j00+j01)/2.0
            h0,  w0  = (i01-i00), (j01-j00)
            ic1, jc1 = (i10+i11)/2.0, (j10+j11)/2.0
            h1,  w1  = (i11-i10), (j11-j10)

            s = np.float32([h1/float(h0), w1/float(w0)])
            M0 = np.transpose(np.mgrid[i00:i01,j00:j01], [1,2,0])
            M = [ic1, jc1] + s*(M0-[ic0,jc0]) - M0
            flow0[ex_i0:ex_i0+(i01-i00), ex_j0:ex_j0+(j01-j00)] = M[...,::-1] # x-y flip

            #M1p = (M0+M)
            #print ('M0', M0[0,0], M0[-1,-1])
            #print ('M', M[0,0], M[-1,-1])
            #print ('M1(calc)', M1p[0,0], M1p[-1,-1])
            #print ('M0-coord', i00, j00, i01, j01)
            #print ('M1(true)', i10, j10, i11, j11)

            #opt[i00:i01,j00:j01] = M

            #flow = flow0
            flow = cv2.calcOpticalFlowFarneback(img0g, img1g, np.copy(flow0),
                    0.5, 3, max(max(i1-i0,j1-j0)/8,1),
                    3, 5, 1.2,
                    flags=cv2.OPTFLOW_USE_INITIAL_FLOW
                    )

            ex_i0, ex_j0 = i00-i0, j00-j0 # "offset" index

            opt[i00:i01,j00:j01] = flow[ex_i0:ex_i0+(i01-i00), ex_j0:ex_j0+(j01-j00),::-1] # flip back
            #msk = np.any(np.greater(opt, 1.0), axis=-1)
            #msk_mn = np.percentile(flow, 20, axis=(0,1))
            #msk_mx = np.percentile(flow, 80, axis=(0,1))
            #print msk_mn, msk_mx, np.mean(opt, axis=(0,1))
            msk = np.any(np.abs(opt)>0.0,axis=-1).astype(np.float32)

            lbl = np.concatenate([opt, msk[...,np.newaxis]], axis=-1)
            #msk = np.any(
            #        np.logical_and(
            #            np.logical_and(msk_mn <= opt, opt < msk_mx),
            #            np.abs(opt) >= 0.5
            #            ), axis=-1)

            #flow = flow[...,::-1] #fix to i-j ordering

            #flow[0:i00,:] = 0
            #flow[:,0:j00] = 0
            #flow[i01+1:,:] = 0
            #flow[:,j01+1:] = 0
            #opt = flow

            # non opt-flow (purely bbox based)
            #h,w = img1.shape[:2]

            #opt  = np.zeros(dtype=np.float32, shape=[img0.shape[0], img0.shape[1], 2])

            #i00, j00, i01, j01 = rint(box0 * [h,w,h,w])
            #ic0, jc0 = (i00+i01)/2.0, (j00+j01)/2.0
            #h0,  w0  = (i01-i00), (j01-j00)

            #i10, j10, i11, j11 = rint(box1 * [h,w,h,w])
            #ic1, jc1 = (i10+i11)/2.0, (j10+j11)/2.0
            #h1,  w1  = (i11-i10), (j11-j10)

            #s = np.float32([h1/float(h0), w1/float(w0)])

            #M0 = np.transpose(np.mgrid[i00:i01,j00:j01], [1,2,0])
            #M = [ic1, jc1] + s*(M0-[ic0,jc0]) - M0

            #M1p = (M0+M)
            #print ('M0', M0[0,0], M0[-1,-1])
            #print ('M', M[0,0], M[-1,-1])
            #print ('M1(calc)', M1p[0,0], M1p[-1,-1])
            #print ('M0-coord', i00, j00, i01, j01)
            #print ('M1(true)', i10, j10, i11, j11)

            #opt[i00:i01,j00:j01] = M

            imgs.append(np.stack([img0,img1], axis=0))
            lbls.append(lbl)

        return [np.stack(a, axis=0) for a in [imgs, lbls]]#, (flow, flow0)


    def grab(self, batch_size, same=True):
        T = self.T
        if same:
            seq = np.random.choice(self.keys)
            seq_dir = '_'.join(seq.split('_')[:-1])
            index = self.index[seq]
            boxs = index['boxs']
            imgs = np.array(index['imgs'])
            lbl = index['lbls'][0]

            n = len(boxs)
            if (n > T * batch_size):
                offset = np.random.randint(n - T * batch_size)
                inds = offset + np.random.choice(T*batch_size, batch_size, replace=False)
            elif (n > batch_size):
                inds = np.random.choice(n, size=batch_size, replace=False)
            else:
                inds = np.random.randint(n, size=batch_size)
            inds = sorted(inds)

            imgs = [os.path.join(self.data_dir[self.data_type], seq_dir, img) for img in imgs[inds]]
            
            boxs = boxs[inds]
        else:
            #assert(batch_size == 2)
            #assert(len(self.keys) >= batch_size)
            seqs = np.random.choice(self.keys, batch_size, replace=False)

            imgs = []
            boxs = []
            lbl = None

            for seq in seqs:
                data = self.index[seq]
                seq_dir = '_'.join(seq.split('_')[:-1])
                n = len(data['boxs'])
                i = np.random.randint(n)
                img = data['imgs'][i]
                img = os.path.join(self.data_dir[self.data_type], seq_dir, img)
                box = data['boxs'][i]
                imgs.append(img)
                boxs.append(box)
                if lbl is None:
                    lbl = data['lbls'][i]

        return np.stack(imgs,0), boxs, lbl

def normalize(x, mn=0.0, mx=1.0):
    xmn = np.min(x)
    xmx = np.max(x)
    return (x-xmn)*((mx-mn)/(xmx-xmn)) + mn

def main():
    #loaders = [ILSVRCLoader(os.getenv('ILSVRC_ROOT'), data_type='train_%d'%i, T=8) for i in range(1,31)]
    #for i in range(100):
    #    np.random.choice(loaders).grab_opt(8)
    #    print('ok')

    #loader = ILSVRCLoader(os.getenv('ILSVRC_ROOT'), data_type='val')
    loader = ILSVRCLoader(os.getenv('ILSVRC_ROOT'), data_type='train_1', T=8)

    # grab_delta test
    #while True:
    #    imgs, lbls = loader.grab_delta(1)
    #    cv2.imshow('prv', imgs[0,...,:3])
    #    cv2.imshow('nxt', imgs[0,...,3:])
    #    print lbls[0]
    #    k = cv2.waitKey(0)
    #    if k == 27:
    #        break

    #cv2.imshow('img0', im0)
    #cv2.imshow('img1', im1)
    #cv2.imshow('di', di)
    #cv2.imshow('dj', dj)

    #flow = cv2.calcOpticalFlowFarneback(im0g, im1g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    #flow = flow[...,::-1] #fix to i-j ordering
    #print np.max(flow, axis=(0,1))
    ##mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    ##hsv = np.zeros_like(im0)
    ##hsv[...,1] = 255
    ##hsv[...,0] = ang*180/np.pi/2
    ##hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    ##bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    ##cv2.imshow('flow', bgr)

    # testing opt
    #fig, (ax0, ax1, ax2) = plt.subplots(3,1)
    fig, ax0 = plt.subplots(1,1)
    while True:
        #(imgs, opts, msks), dbg = loader.grab_opt(1)
        (imgs, lbls) = loader.grab_opt(1)
        im0, im1 = np.float32(imgs[0]) / 255
        opts = lbls[...,:2]
        msks = lbls[...,2]

        msk = cv2.cvtColor(np.float32(msks[0]), cv2.COLOR_GRAY2BGR).astype(np.float32)
        im_overlay = cv2.addWeighted(im0, 0.5, im1, 0.5, 0.0)
        msk_overlay = cv2.addWeighted(im0, 0.5, np.float32(msk), 0.5, 0.0)
        di = cv2.cvtColor(normalize(opts[0,...,:1]), cv2.COLOR_GRAY2BGR)
        dj = cv2.cvtColor(normalize(opts[0,...,1:]), cv2.COLOR_GRAY2BGR)
        viz = np.concatenate(
                [im_overlay,di,dj,msk_overlay], axis=1)
        #cv2.imshow('viz', viz)

        ax0.cla()
        ax0.imshow(viz[...,::-1])

        #ax1.cla()
        #ax1.hist(dbg[0][...,0].flatten(), color=[1.0,0.0,0.0,0.5], label='guess')
        #ax1.hist(dbg[1][...,0].flatten(), color=[0.0,1.0,0.0,0.5], label='calc')
        #ax1.set_title('di')
        #ax1.set_ylabel('count')
        #ax1.set_xlabel('di')
        #ax1.legend()

        #ax2.cla()
        #ax2.hist(dbg[0][...,1].flatten(), color=[1.0,0.0,0.0,0.5], label='guess')
        #ax2.hist(dbg[1][...,1].flatten(), color=[0.0,1.0,0.0,0.5], label='calc')
        #ax2.set_title('dj')
        #ax2.set_ylabel('count')
        #ax2.set_xlabel('di')
        #ax2.legend()

        fig.canvas.draw()
        plt.pause(0.001)
        if plt.waitforbuttonpress() is False:
            break

        #k = cv2.waitKey(0)
        #if k == 27:
        #    break

    #loader.test(100)

    #loader.create_index(loader.index_dir)

    #lens = []
    #for i in range(1, 31):
    #    loader = ILSVRCLoader(os.getenv('ILSVRC_ROOT'), data_type='train_%d' % i)
    #    lens += [len(loader.index[seq]['imgs']) for seq in loader.list_all()]
    #print np.mean(lens)
    #return

if __name__ == "__main__":
    main()
