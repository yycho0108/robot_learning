from __future__ import print_function
import numpy as np
import tensorflow as tf

# path stuff
try:
  from pathlib import Path
except ImportError:
  from pathlib2 import Path  # python 2 backport

def anorm(x):
    return (x + np.pi) % (2*np.pi) - np.pi

def no_op(*args, **kwargs):
    return

def mkdir(x):
    return Path(x).mkdir(parents=True, exist_ok=True)

def proc_img(x):
    #return np.float32(x) / 255. - 0.5
    #return (np.float32(x)/128.) - 1.0
    return (np.float32(x) - 128.) / 64.

def proc_img_tf(x):
    #return (tf.cast(x,tf.float32)/128.) - 1.0
    return (tf.cast(x,tf.float32) - 128.) / 64.

def normalize(x, mn=0.0, mx=1.0):
    xmn = np.min(x)
    xmx = np.max(x)
    return (x-xmn)*((mx-mn)/(xmx-xmn)) + mn

class nest_log(object):
    ilvl=0
    def __init__(self, msg, log_fn=print, sw=2):
        self.s_  = str(msg)
        self.s0_ = ('- %s -' % self.s_)
        self.s1_ = ('-' * len(self.s0_))
        self.i_ = (' ' * (sw * nest_log.ilvl))
        self.log_ = log_fn
    def __enter__(self):
        nest_log.ilvl += 1
        self.log_(self.i_ + self.s0_)
    def __exit__(self, type, value, traceback):
        self.log_(self.i_ + self.s1_)
        nest_log.ilvl -= 1
    def __call__(self):
        self.log_(self.i_ + self.s_)

def main():
    with nest_log('hello'):
        with nest_log('x2'):
            nest_log('y')()
            with nest_log('hmm'):
                nest_log('x')()

if __name__ == "__main__":
    main()
