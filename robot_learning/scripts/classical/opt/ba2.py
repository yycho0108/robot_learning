import numpy as np
from tf import transformations as tx
from scipy.optimize import least_squares
from scipy.sparse import bsr_matrix, csr_matrix
from scipy.linalg import block_diag
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from common import to_h, from_h, jac_h, generate_data

def rec(src, pt_s,
        inv_d, Ki, T_c2b):
    n = len(src)

    d = (1.0 / inv_d)
    v_s = np.einsum('ij,...j->...i',
            Ki, to_h(pt_s))
    lmk_s = d * v_s # Nx3

    T_b2o = np.zeros((n, 4, 4),
            dtype=np.float32)

    x = src[..., 0]
    y = src[..., 1]
    h = src[..., 2]
    c, s = np.cos(h), np.sin(h)

    # rotation part
    T_b2o[..., 0,0] = c
    T_b2o[..., 0,1] = -s
    T_b2o[..., 1,0] = s
    T_b2o[..., 1,1] = c
    T_b2o[..., 2,2] = 1

    # translation part
    T_b2o[..., 0,3] = x
    T_b2o[..., 1,3] = y

    # homogeneous part
    T_b2o[..., 3,3] = 1

    return from_h(np.einsum('...ab,bc,...c->...a',
            T_b2o, T_c2b, to_h(lmk_s)))

class BASolver(object):
    def __init__(self, K, Ki, T_c_b, T_b_c, pBA):
        # camera intrinsic/extrinsic parameters
        self.K_ = K
        self.Ki_ = Ki
        self.T_c_b_ = T_c_b
        self.T_b_c_ = T_b_c

        # scipy least_squares BA parameters
        self.pBA_ = pBA

    def Tbb(self, src, dst):
        """
        Nx4x4 Homogeneous Transform from
        src coord to dst coord.
        """
        dx = src[..., 0] - dst[..., 0]
        dy = src[..., 1] - dst[..., 1]
        dh = src[..., 2] - dst[..., 2]

        csd, ssd = np.cos(dh), np.sin(dh)
        cd, sd = np.cos(dst[...,2]), np.sin(dst[...,2])

        T = np.zeros((len(dx), 4, 4), dtype=np.float32)

        # rotation part
        # Rd^{T}.Rs == R(h_s - h_d)
        T[..., 0,0] = csd
        T[..., 0,1] = -ssd
        T[..., 1,0] = ssd
        T[..., 1,1] = csd
        T[..., 2,2] = 1

        # translation part
        # Rd^{T}.{ts - td}
        T[..., 0,3] = cd*dx + sd*dy
        T[..., 1,3] = -sd*dx + cd*dy

        # homogeneous part
        T[..., 3,3] = 1

        return T

    def dTds(self, src, dst, deltas):
        [dx,dy,dh,csd,ssd,cd,sd] = deltas

        J = np.zeros( (len(src), 4, 4, 3),
                dtype=np.float32 )
        # w.r.t. x
        J[:,0,3,0] = cd
        J[:,1,3,0] = -sd

        # w.r.t. y
        J[:,0,3,1] = sd
        J[:,1,3,1] = cd

        # w.r.t. h
        J[:,0,0,2] = -ssd
        J[:,0,1,2] = -csd
        J[:,1,0,2] =  csd
        J[:,1,1,2] = -ssd

        return J

    def dTdd(self, src, dst, deltas):
        [dx,dy,dh,csd,ssd,cd,sd] = deltas

        J = np.zeros( (len(src), 4, 4, 3),
                dtype=np.float32 )

        # w.r.t. x
        J[:,0,3,0] = -cd
        J[:,1,3,0] = sd

        # w.r.t. y
        J[:,0,3,1] = -sd
        J[:,1,3,1] = -cd

        # w.r.t. h
        J[:,0,0,2] =  ssd
        J[:,0,1,2] =  csd
        J[:,1,0,2] = -csd
        J[:,1,1,2] =  ssd

        J[:,0,3,2] = -sd*dx + cd*dy
        J[:,1,3,2] = -cd*dx - sd*dy

        return J

    def project(self, src, dst, inv_d, pt_s,
            collect=False,
            eps=np.finfo(np.float32).eps):
        """
        src = Nx3 base_link pose (x,y,h)
        dst = Nx3 base_link pose (x,y,h)

        Note:
            if collect:=True, All intermediate results will also be returned.
            (Useful for jacobian computation)
        """
        # unroll constants
        K, Ki = self.K_, self.Ki_
        T_b_c, T_c_b = self.T_b_c_, self.T_c_b_

        # parse landmark depth from source
        d = (1.0 / (inv_d + eps) )

        v_s = np.einsum('ij,...j->...i',
                Ki, to_h(pt_s))

        # convert to 3d landmark location
        lmk_s = d * v_s # Nx3
        lmk_s_h = to_h(lmk_s) # Nx4

        # obtain transform from pose difference
        T_bs_bd = self.Tbb(src, dst)

        # project to dst coordinates
        lmk_d_h = np.einsum('ij,...jk,kl,...l->...i',
                T_b_c, T_bs_bd, T_c_b, lmk_s_h,
                optimize=True
                ) # Nx4
        lmk_d = from_h(lmk_d_h) # Nx3

        pt_d_h = np.einsum('ij,...j->...i',
                K, lmk_d) # Nx3
        pt_d = from_h(pt_d_h)

        if collect:
            vals = {
                    'v_s' : v_s,
                    'lmk_s_h' : lmk_s_h,
                    'lmk_d_h' : lmk_d_h,
                    'pt_d_h' : pt_d_h,
                    'T_bs_bd' : T_bs_bd
                    }
            return pt_d, vals
        else:
            return pt_d

    def err(self,
            src, dst,
            inv_d,
            pt_s, pt_d,
            ravel=True
            ):
        pt_d_r = self.project(src, dst, inv_d, pt_s)

        res = (pt_d_r - pt_d)
        if ravel:
            res = res.ravel()
        return res

    def jac(self, src, dst, inv_d, pt_s, pt_d):
        pt_d_r, vals = self.project(
                src, dst, inv_d, pt_s,
                collect = True)
        # unroll constants
        K, Ki = self.K_, self.Ki_
        T_b_c, T_c_b = self.T_b_c_, self.T_c_b_

        # temporary value (-d**-2 where d = inverse depth)
        ndi2 = ((-1.0 / np.square(inv_d)).reshape(-1,1)) # Nx1

        dx = src[..., 0] - dst[..., 0]
        dy = src[..., 1] - dst[..., 1]
        dh = src[..., 2] - dst[..., 2]
        csd, ssd = np.cos(dh), np.sin(dh)
        cd, sd = np.cos(dst[...,2]), np.sin(dst[...,2])
        deltas = [dx,dy,dh,csd,ssd,cd,sd]

        # cache intermediate jacobians
        j_y = jac_h(vals['pt_d_h'])
        j_l = jac_h(vals['lmk_d_h'])
        j_i = vals['v_s'] * ndi2 # Nx3
        j_s = self.dTds(src, dst, deltas) # Nx4x4x3
        j_d = self.dTdd(src, dst, deltas) # Nx4x4x3

        j_lhs = np.einsum(
                '...ab,bc,...cd,de->...ae',
                j_y, K, j_l, T_b_c,
                optimize=True
                ) # Nx2x4

        # J_i objective : Nx2 x 1
        J_i = np.einsum('...ab,...bc,cd,de,...e->...a',
                j_lhs, # Nx2x4
                vals['T_bs_bd'], # Nx4x4
                T_c_b, # 4x4
                np.eye(4,3), # 4x3
                j_i,
                optimize=True
                )[..., None] #Nx3

        # J_s objective : Nx2 x 3
        J_s = np.einsum('...ab,...bcd,ce,...e->...ad',
                j_lhs, # Nx2x4
                j_s, # Nx4x4x3
                T_c_b, # 4x4
                vals['lmk_s_h'], # Nx4
                optimize=True
                )

        # J_d objective : Nx2 x 3
        J_d = np.einsum('...ab,...bcd,ce,...e->...ad',
                j_lhs, # Nx2x4
                j_d, # Nx4x4x3
                T_c_b, # 4x4
                vals['lmk_s_h'], # Nx4
                optimize=True
                )

        return J_i, J_s, J_d

    def err_i_p(self, pos, lid,
            n_c, n_l,
            i_s, i_d, i_i,
            pt_s, pt_d,
            s_c=3, s_l=1
            ):
        # rectify, just in case
        pos = pos.reshape(-1, s_c)
        lid = lid.reshape(-1, s_l)

        src   = pos[i_s] # should be n_o x 3
        dst   = pos[i_d] # should be n_o x 3
        inv_d = lid[i_i] # should be n_o x 1

        e = self.err(src, dst, inv_d,
                pt_s, pt_d, ravel=True)
        return e

    def err_i(self, params,
            n_c, n_l,
            i_s, i_d, i_i,
            pt_s, pt_d,
            s_c=3, s_l=1,
            ):
        """ err() from sparse indices """
        pos = params[:n_c*s_c].reshape(-1, s_c) # should be n_c x 3
        lid = params[n_c*s_c:].reshape(-1, s_l) # should be n_l x 1
        return self.err_i_p(pos, lid,
                n_c, n_l,
                i_s, i_d, i_i,
                pt_s, pt_d,
                s_c, s_l)

    def jac_i_p(self, pos, lid,
            n_c, n_l,
            i_s, i_d, i_i,
            pt_s, pt_d,
            s_o=2, s_c=3, s_l=1):
        # rectify, just in case
        pos = pos.reshape(-1, s_c)
        lid = lid.reshape(-1, s_l)

        n_o = len(i_s) # == len(i_d) == len(i_i)

        src   = pos[i_s] # should be n_o x 3
        dst   = pos[i_d] # should be n_o x 3
        inv_d = lid[i_i] # should be n_o x 1

        _, J_s, J_d = self.jac(src, dst, inv_d, pt_s, pt_d)
        # result ^ : N_ox2 x N_o, N_ox2 x 3, N_ox2 x 3

        # objective : Nx2 x (3+3+N)
        # err@o_i = err (lmk[i_i], pt[i_s], pt[i_d])

        # allocate space for jacobian
        J = np.zeros(
                (n_o*s_o, n_c*s_c),
                dtype=np.float32)

        # slice, for more intuitive indexing
        J_c = J[:, :n_c*s_c].reshape(n_o, s_o, n_c, s_c) # Nx2xPx3
        J_c[np.arange(n_o), :, i_s, :] = J_s 
        J_c[np.arange(n_o), :, i_d, :] = J_d
        J = csr_matrix(J)
        return J

    def sparsity_i(self):
        pass

    def jac_i(self, params,
            n_c, n_l,
            i_s, i_d, i_i,
            pt_s, pt_d,
            s_o=2, s_c=3, s_l=1
            ):
        """
        Call jac() from sparse indices.

        i_s = N_o pose indices corresponding to *source* viewpoints.
        i_d = N_o pose indices corresponding to *target* viewpoints.

        Num params = n_c*s_c + n_l*s_l
        Num errors = n_o*2
        """
        pos = params[:n_c*s_c].reshape(-1, s_c) # should be n_c x 3
        lid = params[n_c*s_c:].reshape(-1, s_l) # should be n_l x 1

        n_o = len(i_s) # == len(i_d) == len(i_i)

        src   = pos[i_s] # should be n_o x 3
        dst   = pos[i_d] # should be n_o x 3
        inv_d = lid[i_i] # should be n_o x 1

        J_i, J_s, J_d = self.jac(src, dst, inv_d, pt_s, pt_d)
        # result ^ : N_ox2 x N_o, N_ox2 x 3, N_ox2 x 3

        # objective : Nx2 x (3+3+N)
        # err@o_i = err (lmk[i_i], pt[i_s], pt[i_d])

        # allocate space for jacobian
        J = np.zeros(
                (n_o*s_o, n_c*s_c + n_l*s_l),
                dtype=np.float32)
        # print 'Js', J.shape

        # slice, for more intuitive indexing
        J_c = J[:, :n_c*s_c].reshape(n_o, s_o, n_c, s_c) # Nx2xPx3
        J_l = J[:, n_c*s_c:].reshape(n_o, s_o, n_l, s_l) # Nx2xLx1

        J_c[np.arange(n_o), :, i_s, :] = J_s 
        J_c[np.arange(n_o), :, i_d, :] = J_d
        J_l[np.arange(n_o), :, i_i, :] = J_i

        J = csr_matrix(J)
        return J

    def __call__(self,
            pos, lmk,
            i_s, i_d, i_i,
            pt_s, pt_d,
            s_o=2, s_c=3, s_l=1
            ):
        n_c = len( pos )
        n_l = len( lmk )

        ## == primary 'FAST' BA ==
        x0 = np.concatenate([pos.ravel(), lmk.ravel()], axis=0)
        res = least_squares(
                self.err_i, x0,
                jac=self.jac_i,
                args=(n_c, n_l, i_s, i_d, i_i, pt_s, pt_d),
                **self.pBA_
                )
        pos_r = res.x[:n_c*s_c].reshape(-1, s_c)
        lmk_r = res.x[n_c*s_c:].reshape(-1, s_l)

        ## == secondary 'SLOW' BA ==

        # x scale
        # xs = np.concatenate([
        #             np.repeat([0.01, 0.01, np.deg2rad(1.0)], n_c),
        #             np.full(n_l, 0.2)], axis=0)
        # pBA['x_scale'] = xs
        # pBA['ftol'] = 1e-6
        # pBA['loss'] = 'linear'
        # x0 = np.concatenate([pos_r.ravel(), lmk_r.ravel()], axis=0)
        # res = least_squares(
        #         self.err_i, x0,
        #         jac=self.jac_i,
        #         args=(n_c, n_l, i_s, i_d, i_i, pt_s, pt_d),
        #         **pBA
        #         )

        ## == pose-only BA BEG ==
        #x0 = pos_r.ravel()
        #self.pBA_['ftol'] = 1e-9

        ## fix lmk and run pose-only BA
        #res = least_squares(
        #        self.err_i_p, x0,
        #        #jac_sparsity=A,
        #        jac=self.jac_i_p,
        #        #x_scale='jac',
        #        #x_scale=xs,
        #        #x_scale=1.0,
        #        args=(lmk_r, n_c, n_l, i_s, i_d, i_i, pt_s, pt_d),
        #        **self.pBA_
        #        )
        #pos_r = res.x.reshape( pos.shape )
        ## == pose-only BA END ==

        return pos_r, lmk_r, res

def R2(x):
    c,s = np.cos(x), np.sin(x)
    return np.reshape([c,-s,s,c], (2,2))

def gen_pos(n=128, dt=0.1):
    res = []
    p  = np.zeros(3)
    v  = np.zeros(3)

    for i in range(n):
        a = np.random.normal(scale=(0.1,0.1,0.1), size=3)
        vdt = v * dt
        dp = np.concatenate([
                R2(p[-1]).dot(vdt[:2]),
                [vdt[-1]]
                ])
        p += dp
        v += a*dt
        res.append( p.copy() )

    return np.stack(res, axis=0)

def gen_lmk(n=1024, lim=10.0):
    return np.random.uniform(
            low=(-lim, -lim, 0.0),
            high=(lim, lim, lim),
            size=(n,3)
            )

def gen_obs(
        pos, lmk,
        K, Ki, T_c2b, T_b2c,
        n=512
        ):
    n_p = len(pos)

    i_s_ = np.random.choice(len(pos), len(lmk)) # record of first sightings
    i_i = np.random.choice(len(lmk), n) # which landmark was observed
    i_s = i_s_[i_i] # pose index corresponding to the landmark
    i_d = np.random.choice(len(pos), n) # pose index corresponding to secondary observations

    # do not allow i_s == i_d
    while True:
        i_r = np.where(i_s == i_d)[0]
        n_r = len(i_r)
        if n_r <= 0:
            break
        i_d[i_r] = np.random.choice(len(pos), n_r)

    T_b2o = np.zeros((n_p, 4, 4),
            dtype=np.float32)

    x = pos[..., 0]
    y = pos[..., 1]
    h = pos[..., 2]
    c, s = np.cos(h), np.sin(h)

    # rotation part
    T_b2o[..., 0,0] = c
    T_b2o[..., 0,1] = -s
    T_b2o[..., 1,0] = s
    T_b2o[..., 1,1] = c
    T_b2o[..., 2,2] = 1

    # translation part
    T_b2o[..., 0,3] = x
    T_b2o[..., 1,3] = y

    # homogeneous part
    T_b2o[..., 3,3] = 1

    # invert to get the reverse transformation
    T_o2b = np.linalg.inv(T_b2o)

    lmk_s = from_h(
            np.einsum('ab,...bc,...c->...a',
                T_b2c, # 4x4
                T_o2b[i_s], # Nx4x4
                to_h(lmk[i_i]), # Nx4
                )
            )
    lmk_d = from_h(
            np.einsum('ab,...bc,...c->...a',
                T_b2c, # 4x4
                T_o2b[i_d], # Nx4x4
                to_h(lmk[i_i]), # Nx4
                )
            )

    d_s = lmk_s[..., 2]
    d_d = lmk_d[..., 2]

    pt_s = from_h(np.einsum('ab,...b->...a',
        K, lmk_s))

    pt_d = from_h(np.einsum('ab,...b->...a',
        K, lmk_d))

    # apply filter (positive depth + visibility)
    i_v = np.logical_and.reduce([
            d_s > 0.0,
            d_d > 0.0,
            0.0 <= pt_s[:,0],
            pt_s[:,0] < 640,
            0.0 <= pt_s[:,1],
            pt_s[:,1] < 480,
            0.0 <= pt_d[:,0],
            pt_d[:,0] < 640,
            0.0 <= pt_d[:,1],
            pt_d[:,1] < 480,
            ])

    print 'survived filter : {}'.format(i_v.sum())

    # post-filter results
    pt_s = pt_s[i_v]
    pt_d = pt_d[i_v]
    d_s  = d_s[i_v]
    i_s  = i_s[i_v]
    i_d  = i_d[i_v]
    i_i  = i_i[i_v]

    # rectify indices
    i_i_u, i_i_idx, i_i_idx_inv  = np.unique(
            i_i, return_index=True, return_inverse=True
            )

    # comparison
    # print 'comparison'
    # print d_s
    # print d_s[i_i_idx][i_i_idx_inv]

    i_i = i_i_idx_inv # index into new list of unique observed landmarks
    d_s = d_s[i_i_idx] # depths observed at each landmark

    return pt_s, pt_d, d_s, i_s, i_d, i_i, i_i_u, i_i_idx

def main():
    np.random.seed( 2 )

    # default K
    K = np.reshape([
        499.114583, 0.000000, 325.589216,
        0.000000, 498.996093, 238.001597,
        0.000000, 0.000000, 1.000000], (3,3))
    Ki = tx.inverse_matrix(K)

    # camera extrinsic parameters
    T_c2b = tx.compose_matrix(
                    angles=[-np.pi/2-np.deg2rad(10),0.0,-np.pi/2],
                    translate=[0.174,0,0.113])
    T_b2c = tx.inverse_matrix(T_c2b)

    # BA parameters
    pBA = dict(
            ftol=1e-4,
            xtol=np.finfo(np.float32).eps,
            loss='soft_l1',
            max_nfev=8192,
            method='trf',
            verbose=2,
            tr_solver='lsmr',
            f_scale=1.0
            )

    # instantiate solver handle
    solver = BASolver(K, Ki, T_c2b, T_b2c, pBA)

    # generate data
    pos = gen_pos(n=256, dt=0.2)
    lmk = gen_lmk(n=1024, lim=10.0)

    # generate observations
    pt_s, pt_d, d_s, i_s, i_d, i_i, i_i_u, i_i_i = gen_obs(pos, lmk,
        K, Ki, T_c2b, T_b2c, n=8192 * 4)

    print '# lmk', len(d_s)
    print '# obs', len(pt_s)

    # remember! inverse depth parametrization.
    di_s = 1.0 / (d_s)

    # perturb data
    pos_g = np.random.normal(
            pos,
            scale=(0.2, 0.2, np.deg2rad(10.0))
            )

    # err validation

    # VIZ : projection
    # pt_d_r = solver.project(
    #         pos[i_s], pos[i_d], di_s[:,None],
    #         pt_s,
    #         )
    # plt.figure()
    # plt.plot(pt_d[:,0], pt_d[:,1], 'rx', label='pt_d-gt')
    # plt.plot(pt_d_r[:,0], pt_d_r[:,1], 'b+', label='pt_d-r')

    e0 = solver.err(
            pos_g[i_s], pos_g[i_d], di_s[i_i,None],
            pt_s, pt_d,
            ravel=False
            )
    e0r = e0.ravel()
    
    e0 = np.linalg.norm(e0, axis=-1)

    pos_r, di_s_r , res = solver(
            pos_g, di_s,
            i_s, i_d, i_i,
            pt_s, pt_d,
            #s_o=2, s_c=3, s_l=1
            )

    # convert to world coord
    e1 = solver.err(
            pos_r[i_s], pos_r[i_d], di_s_r[i_i],
            pt_s, pt_d,
            ravel=False
            )
    e1r = e1.ravel()

    # recreate in s coordinate
    lmk2  = rec(pos[i_s[i_i_i]], pt_s[i_i_i], di_s[:,None], Ki, T_c2b)
    lmk_g = rec(pos_g[i_s[i_i_i]], pt_s[i_i_i], di_s[:,None], Ki, T_c2b)
    lmk_r = rec(pos_r[i_s[i_i_i]], pt_s[i_i_i], di_s_r[:], Ki, T_c2b)

    e1 = np.linalg.norm(e1, axis=-1)

    plt.figure()
    plt.plot(e0, 'b:', label='e0')
    plt.plot(e1, 'r--', label='e1')
    plt.legend()

    plt.figure()
    plt.plot(pos[:,0], pos[:,1], 'k', label='pos-gt')
    plt.plot(lmk[i_i_u,0], lmk[i_i_u,1], 'ko', label='lmk-gt', alpha=0.5)
    plt.plot(lmk2[:,0], lmk2[:,1], 'g^', label='lmk-gt(rec)', alpha=0.5)

    plt.plot(pos_g[:,0], pos_g[:,1], 'b:', label='pos-guess')
    plt.plot(lmk_g[:,0], lmk_g[:,1], 'bx', label='lmk-guess')

    plt.plot(lmk_r[:,0], lmk_r[:,1], 'r+', label='lmk-ba')
    plt.plot(pos_r[:,0], pos_r[:,1], 'r--', label='pos-ba')

    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
