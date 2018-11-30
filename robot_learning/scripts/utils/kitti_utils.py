import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from vo_utils import VoShow, to_pose2d, batch_augment
from utils import anorm, mkdir
import time

class KittiLoader(object):
    def __init__(self, root, mode='train',
            from_dump=True,
            as_path=False
            ):
        root = os.path.expanduser(root)
        self.as_path_ = as_path
        self.from_dump_ = from_dump
        self.root_ = root
        print('root : {}'.format(self.root_))
        self.pos_root = os.path.join(root, 'poses')
        self.seq_root = os.path.join(root, 'sequences')
        if mode is 'train':
            self.drange_ = range(0, 11)
        else:
            # valid
            self.drange_ = range(11, 22)
        self.seq_dir_ = [os.path.join(self.seq_root, '%02d' % i)
                for i in self.drange_]

        if from_dump:
            self.load_dump()
        else:
            self.pos_ = [np.loadtxt(os.path.join(self.pos_root, '%02d.txt' % i)).reshape(-1,3,4)
                    for i in self.drange_]
            self.seq_len_ = np.float32([e.shape[0] for e in self.pos_])

    def format(self, pos):
        # format the data to be compatible with the training network
        prv = pos[:-1]
        nxt = pos[1:]
        #x1,y1,h1 w.r.t x0,y0,h0
        delta = nxt - prv #[N,2]
        h0 = prv[:,2] 
        c, s = np.cos(h0), np.sin(h0)
        R = np.reshape([c,-s,s,c], [2,2,-1]) # [2,2,N]
        dp = np.einsum('ijk,ki->kj', R, delta[:,:2])
        dh = anorm(delta[:,2:])
        
        dps = np.concatenate([dp,dh], axis=-1)
        dps = np.concatenate([0*delta[0:1], delta], axis=0)

        return dps

    def get_1(self, time_steps,
            target_size=None,
            seq_idx=None
            ):

        # parse arguments
        sel_p = self.seq_len_ / self.seq_len_.sum()

        seq_idx = (np.random.choice(len(self.drange_), p=sel_p)
                if (seq_idx is None)
                else seq_idx)
        #img_dir = (np.random.choice(['image_2','image_3'])
        #        if (img_dir is None)
        #        else img_dir)
        img_dir = 'image_2'

        # account for stereo camera baseline
        # takes the pose into coordinate frame of the right camera
        # if the reference image is from the right camera.
        # TODO : is this important at all??
        # offset = (np.reshape([0,0.54,0], (-1,3))
        #     if (img_dir == 'image_3')
        #     else 0)

        n = self.seq_len_[seq_idx]
        i0 = np.random.randint(0, n-time_steps)

        if self.from_dump_:
            img = self.img_[seq_idx][i0:i0+time_steps]
            pos = self.pos_[seq_idx][i0:i0+time_steps]
        else:
            img = [cv2.imread(os.path.join(self.seq_dir_[seq_idx], img_dir,
                '%06d.png' % i)) for i in range(i0, i0+time_steps)]
            pos = self.pos_[seq_idx][i0:i0+time_steps]

        if not self.as_path_:
            pos2d = to_pose2d(pos)
            dps = self.format(pos2d)

        if not self.from_dump_:
            # all formatting should have been done in the dump
            # TODO : make it more robust?
            if target_size is not None:
                img = [cv2.resize(e, target_size) for e in img]
            img = [e[...,::-1] for e in img] # bgr -> rgb
            img = np.stack(img, axis=0)

        if self.as_path_:
            pos = np.stack(pos, axis=0)
            return [img, pos]
        else:
            dps = np.stack(dps , axis=0)
            return [img, dps]

    def get(self, batch_size, time_steps,
            aug=True,
            as_path=None, # TODO: WARNING: ignored!
            target_size=None):
        img, ps = zip(*[self.get_1(time_steps, target_size) for _ in range(batch_size)])
        if aug:
            img = [batch_augment(e) for e in img]
        img = np.stack(img, axis=0)
        ps  = np.stack(ps, axis=0)
        return [img, ps]

    def save_dump(self, dump_dir=None,
            target_size=(256,192)
            ):
        dump_dir = os.path.join(self.root_, 'seq_dump') if (dump_dir is None) else dump_dir
        mkdir(dump_dir)

        for (seq_idx, p, d) in enumerate(zip(self.drange_, self.pos_, self.seq_dir_)):
            n = len(p)
            print('n', n)
            imgs = [cv2.resize(cv2.imread(os.path.join(d, 'image_2',
                '%06d.png' % i)), target_size)[...,::-1] for i in range(0, n)]
            imgs = np.stack(imgs, axis=0)

            print os.path.join(dump_dir, 'img_%02d.npy' % seq_idx)

            #np.save(os.path.join(dump_dir, 'img_%02d.npy' % seq_idx), imgs)
            #np.save(os.path.join(dump_dir, 'pos_%02d.npy' % seq_idx), p)

    def load_dump(self, dump_dir=None):
        dump_dir = os.path.join(self.root_, 'seq_dump') if (dump_dir is None) else dump_dir
        try:
            self.img_ = [np.load(os.path.join(dump_dir, 'img_%02d.npy' % i)) for i in self.drange_]
            self.pos_ = [np.load(os.path.join(dump_dir, 'pos_%02d.npy' % i)) for i in self.drange_]
            self.seq_len_ = np.float32([e.shape[0] for e in self.pos_])
            print('seq_len : {}'.format(self.seq_len_))
        except Exception as e:
            print('dump index does not exist yet : {}'.format(e))

def main():
    as_path = False
    #target_size = (256,192)
    target_size = None

    loader = KittiLoader(root='~/datasets/kitti',
            from_dump=True,
            mode='train',
            as_path=as_path
            )
    #loader.save_dump()
    img, dps = loader.get(batch_size=32, time_steps=4,
        target_size=target_size)
    end   = time.time()

    data = zip(img, dps)
    disp = VoShow(data, as_path=as_path)
    disp.show()

if __name__ == "__main__":
    main()
