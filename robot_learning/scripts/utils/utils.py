import numpy as np

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
    return np.float32(x) / 255. - 0.5
