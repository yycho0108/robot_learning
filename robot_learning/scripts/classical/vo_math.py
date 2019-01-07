import numpy as np

def intersect2d(a, b):
    """
    from https://stackoverflow.com/a/8317403
    """
    nrows, ncols = a.shape
    dtype={'names':['f{}'.format(i) for i in range(ncols)],
        'formats':ncols * [a.dtype]}
    c = np.intersect1d(a.view(dtype), b.view(dtype))
    c = c.view(a.dtype).reshape(-1, ncols)
    return c

def robust_mean(x, margin=10.0, weight=None):
    if len(x) <= 0:
        return np.nan
    x = np.asarray(x, dtype=np.float32)
    s_lo = np.percentile(x, 50.0 - margin)
    s_hi = np.percentile(x, 50.0 + margin)
    msk = np.logical_and.reduce([
        s_lo <= x, x <= s_hi
        ])

    if weight is None:
        return x[msk].mean()
    else:
        # compute normalized weight
        #print 'weighting'
        #print weight.shape
        #print msk.shape
        w = weight[msk[...,0]]
        w = w / w.sum()

        return np.sum(x[msk] * w, axis=-1)


