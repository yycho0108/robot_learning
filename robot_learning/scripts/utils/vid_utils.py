from abc import ABCMeta, abstractmethod
from vis_utils import draw_bbox
import cv2

class VIDLoaderBase(object):
    def __init__(self):
        pass
    @abstractmethod
    def create_index(self):
        pass
    @abstractmethod
    def list_all(self):
        pass
    @abstractmethod
    def grab(self, batch_size):
        pass
    def test(self, n=1):
        for _ in range(n):
            imgs, boxs, lbl = self.grab(2)
            for img, box in zip(imgs,boxs):
                img = cv2.imread(img)
                draw_bbox(img, box, str(lbl), (255,0,0))
                cv2.imshow('img', img)
                if cv2.waitKey(400) == 27:
                    return
