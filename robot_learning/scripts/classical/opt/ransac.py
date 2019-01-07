import numpy as np

class RANSACModel(object):
    def __init__(self,
            n_model,
            model_fn,
            err_fn,
            thresh,
            prob
            ):
        self.n_model_ = n_model
        self.model_fn_ = model_fn
        self.err_fn_ = err_fn
        self.thresh_ = thresh
        self.prob_ = prob # desired confidence

    def n_it(self, w, n0, eps=np.finfo(np.float32).eps):
        # protect against invalid input
        p  = np.clip(self.prob_, 0, 1)
        ep = np.clip(w, 0, 1)

        # prep numerator + denominator, k= log(1-p)/log(1-w**m)
        nmr = max(1.0 - p, eps)
        dmr = 1.0 - np.power(1.0-ep, self.n_model_)

        if dmr < eps:
            return 0

        nmr = np.log(nmr)
        dmr = np.log(dmr)

        res = n0 if (dmr >= 0 or -nmr >= n0*-dmr) else np.round(nmr/dmr).astype(np.int32)
        return res

    def step(self, n_data):
        sidx = np.random.choice(
                n_data, self.n_model_,
                replace=(n_data > self.n_model_)
                )
        model = self.model_fn_(sidx)
        err   = self.err_fn_(model)
        inl = (err < self.thresh_)
        w = float( inl.sum() ) / inl.size

        return model, err, inl, w

    def __call__(self, n_data, max_it):
        # prepare result
        best = {
                'model' : None,
                'err' : np.inf,
                'inl' : None,
                'w'   : 0.0
                }
        # prep loop
        it = 0
        n = max(1, max_it)
        while it < n:
            model, err, inl, w = self.step(n_data)
            n = self.n_it(w, n) # update num iterations
            if w > best['w']:
                best['w'] = w
                best['model'] = model
                best['err'] = err
                best['inl'] = inl
            it += 1

        return it, best
