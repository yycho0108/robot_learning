from scipy.sparse.linalg import inv as sinv
from scipy.linalg import cho_solve
from scipy.sparse import bsr_matrix, lil_matrix
import numpy as np
from tf import transformations as tx
import sparse
import time

def to_homo(x):
    return np.pad(x, [(0,0),(0,1)],
            mode='constant',
            constant_values=1.0
            )

def ba_J(p, l,
        K, T_b2c, T_c2b):
    #print 'shape validation'
    #print p.shape
    #print l.shape
    #print K.shape
    #print T_b2c.shape
    #print T_c2b.shape

    # convert to sparse versions
    coords = [[0,0,1,1], [0,2,1,2]]
    K = sparse.COO(
            coords=coords,
            data=K[coords],
            shape=(2,3)
            )
    coords = [
            [0,0,0,0,1,1,1,1,2,2,2,2,3],
            [0,1,2,3,0,1,2,3,0,1,2,3,3]]
    T_b2c = sparse.COO(
            coords=coords,
            data=T_b2c[coords],
            shape=(4,4)
            )
    T_c2b = sparse.COO(
            coords=coords,
            data=T_c2b[coords],
            shape=(4,4)
            )

    n = len(p)

    # compute T_o2b
    x = p[...,0]
    y = p[...,1]
    h = p[...,2]
    c = np.cos(h)
    s = np.sin(h)

    # directly construct batchwise T_o2b
    c0 = np.tile(np.arange(n), 8)
    c1 = np.tile([0,0,0,1,1,1,2,3], n).reshape(n,-1).T.ravel()
    c2 = np.tile([0,1,3,0,1,3,2,3], n).reshape(n,-1).T.ravel()

    one = np.ones_like(c)
    T_o2b = sparse.COO(
            coords=[c0,c1,c2],
            data=np.ravel([c,s,-y*s-x*c,-s,c,x*s-y*c,one,one]),
            shape=(n,4,4),
            fill_value=0.0
            )

    #print T_o2b.todense()[0]

    # validation with dense equivalent
    #T_o2b_np = np.zeros((n,4,4), dtype=np.float32)
    #T_o2b_np[:,0,0] = c
    #T_o2b_np[:,0,1] = s
    #T_o2b_np[:,0,3] = -y*s - x*c
    #T_o2b_np[:,1,0] = -s
    #T_o2b_np[:,1,1] = c
    #T_o2b_np[:,1,3] = x*s - y*c
    #T_o2b_np[:,2,2] = 1
    #T_o2b_np[:,3,3] = 1
    #print 'delta', np.abs(T_o2b.todense() - T_o2b_np).sum()

    # jacobian : landmark part
    j_l = reduce(sparse.matmul,[
        K, T_b2c[:3], T_o2b, T_c2b])
    # 2x3, 3x4, Nx4x4, 4x4 --> Nx2x4
    j_l = j_l[...,:3] # remove homogeneous part
    # --> Nx2x3
    #print 'j_l', j_l.shape

    c0 = np.tile([0,0,1,1,2,2,2,2,2,2], n).reshape(n,-1).T.ravel()
    c1 = np.tile(np.arange(n), 10)
    c2 = np.tile([0,1,0,1,0,0,0,1,1,1], n).reshape(n,-1).T.ravel()
    c3 = np.tile([3,3,3,3,0,1,3,0,1,3], n).reshape(n,-1).T.ravel()

    j_data = np.ravel(
            [-c,s,-s,-c,-s,c,-y*c+x*s,-c,-s,x*c+y*s])

    # data format = [All, n]
    j_BP = sparse.COO(
            coords=[c0,c1,c2,c3],
            data = j_data,
            shape=(3,n,3,4),
            fill_value=0.0
            )

    ## dense version
    #j_BP_np = np.zeros((3,n,3,4), dtype=np.float32)
    #j_BP_np[0,:,0,3] = -c
    #j_BP_np[0,:,1,3] = s
    #j_BP_np[1,:,0,3] = -s
    #j_BP_np[1,:,1,3] = -c
    #j_BP_np[2,:,0,0] = -s
    #j_BP_np[2,:,0,1] = c
    #j_BP_np[2,:,0,3] = -y*c+x*s
    #j_BP_np[2,:,1,0] = -c
    #j_BP_np[2,:,1,1] = -s
    #j_BP_np[2,:,1,3] = x*c+y*s

    ##print j_BP_np
    ##print j_BP.todense()
    #print 'delta', np.abs(j_BP_np - j_BP.todense()).sum()
    ##j_BP = sparse.COO.from_numpy(j_BP)

    #print 'J_BP', j_BP.nnz, j_BP.size

    # print T_c2b.nnz, T_c2b.size

    # test
    # j_p = reduce(sparse.matmul,[
    #     K, j_BP, T_c2b]) # consider tensordot?
    # print 'prv', j_p.nnz, j_p.size

    l_h = to_homo(l)[..., None]
    #l_h = sparse.COO(l_h)

    # jacobian : camera pose part

    #print '????????????????????????'
    #print K.shape
    #print j_BP.shape
    #print T_c2b.shape
    #print l_h.shape

    j_p = reduce(sparse.matmul,[
        K, j_BP, T_c2b, l_h]) # consider tensordot?

    # 2x3, (3xN)x3x4, 4x4, 4x1
    # --> (3xNx2x1)
    j_p = j_p[...,0]
    j_p = np.transpose(j_p, (1,2,0)) # --> (Nx2x3)
    #print 'j_p', j_l.shape

    # f(nx3 p, nx3 l, nx2 y) -> nx2 e (residual)
    # j_l, j_p represent jac e_i/x_i, 0<=i<N
    # need to convert jac, which has "eye"-like structure
    # jac --> (NxNx2x3)
    # then 

    c0 = np.tile(np.arange(n), 2*3).reshape(-1,n).T.ravel()
    c1 = c0
    c2 = np.tile([0,0,0,1,1,1], n).ravel()
    c3 = np.tile([0,1,2,0,1,2], n).ravel()

    j_l = sparse.COO(
            coords=[c0,c1,c2,c3],
            data=j_l.todense().ravel(),
            shape=(n,n,2,3)
            )
    j_p = sparse.COO(
            coords=[c0,c1,c2,c3],
            data=j_p.ravel(),
            shape=(n,n,2,3)
            )
    j = sparse.concatenate([j_p,j_l], axis=-1)
    # result [y_i, x_i] = 
    # result [i_n, y_i, i_n, x_i] = j[i_n, y,_i, x_i]
    # currently result[i_n,i_n,y_i,x_i] = j[i_n,i_n,y_i,x_i]
    # required format : (Nx2, Nx6)
    j = j.transpose([0,2,1,3])
    j = j.reshape([n*2, n*6])
    return j.todense()
    #return j.to_scipy_sparse()

def ba_J_v2(p, l, K, R_b2c, t_b2c, pt_h):
    P_b2c = np.concatenate([R_b2c, t_b2c], axis=-1) # 3x4

    n = len(p)

    # unroll input parameters
    x,y,h = p[...,0], p[...,1], p[...,2]
    c,s = np.cos(h), np.sin(h)
    lx,ly,lz = l[...,0], l[...,1], l[...,2]
    hx,hy,hs = pt_h[...,0], pt_h[...,1], pt_h[...,2]

    R00,R01,R02,R10,R11,R12,R20,R21,R22 = R_b2c.ravel()
    t00,t10,t20 = t_b2c.ravel()

    # left jacobian ( homogeneous part )
    hs_i = (1.0 / hs)
    hs_i2 = hs_i * hs_i
    J_l = np.zeros((n,2,3), dtype=np.float32)
    J_l[:,0,0] = hs_i
    J_l[:,0,2] = - hx * hs_i2
    J_l[:,1,1] = hs_i
    J_l[:,1,2] = - hy * hs_i2

    # right jacobian ( all the other parts )
    J_r = np.zeros((n,4,6), dtype=np.float32)
    J_r[:,0,0] = -c
    J_r[:,0,1] = -s
    J_r[:,0,2] = -lx*(s*R00 - c*R01) - ly*(s*R10 - c*R11) - lz*(s*R20 - c*R21) + x*s - y*c - (R00*t00 + R10*t10 + R20*t20)*s + (R01*t00 + R11*t10 + R21*t20)*c
    J_r[:,0,3] = s*R01 + c*R00
    J_r[:,0,4] = s*R11 + c*R10
    J_r[:,0,5] = s*R21 + c*R20
    J_r[:,1,0] = s
    J_r[:,1,1] = -c
    J_r[:,1,2] = -lx*(s*R01 + c*R00) - ly*(s*R11 + c*R10) - lz*(s*R21 + c*R20) + x*c + y*s - (R00*t00 + R10*t10 + R20*t20)*c - (R01*t00 + R11*t10 + R21*t20)*s
    J_r[:,1,3] = -s*R00 + c*R01
    J_r[:,1,4] = -s*R10 + c*R11
    J_r[:,1,5] = -s*R20 + c*R21
    J_r[:,2,3] = R02
    J_r[:,2,4] = R12
    J_r[:,2,5] = R22

    res = reduce(np.matmul,
            [J_l, K, P_b2c, J_r])
    return res

#def extract_block_diag(a, n, k=0):
#    a = np.asarray(a)
#    if a.ndim != 2:
#        raise ValueError("Only 2-D arrays handled")
#    if not (n > 0):
#        raise ValueError("Must have n >= 0")
#
#    if k > 0:
#        a = a[:,n*k:] 
#    else:
#        a = a[-n*k:]
#
#    n_blocks = min(a.shape[0]//n, a.shape[1]//n)
#
#    new_shape = (n_blocks, n, n)
#    new_strides = (n*a.strides[0] + n*a.strides[1],
#                   a.strides[0], a.strides[1])
#    return np.lib.stride_tricks.as_strided(a, new_shape, new_strides)

#def block_view(A, b_n, b_m):
#    n, m = A.shape
#    np.lib.as_strided(Ci,
#            new_shape=(n_blocks, n, n)


#def reduced_camera_matrix(B, Ci, E, ET, s_c=3):
#    q = len(Ci) # NOTE :: Ci is the raveled block diag. matrix.
#    S = []
#    for (bi, bj) in np.triu_indices(q):
#        i0, i1 = bi*s_c, (bi+1)*s_c
#        j0, j1 = bj*s_c, (bj+1)*s_c
#        # ideal : Nx3x3 * Nx3x3 * Nx3x3
#        res = E[i0:i1].dot(Ci).dot(ET[:,j0:j1])
#        S[i0:i1, j0:j1] = np.matmul(E[i0:i1], Ci, ET[:,j0:j1])

def block_inv(A, n_rows, n_cols):
    q = A.shape[0] / n_rows # (# of blocks)
    #Ai = np.zeros(A.shape, dtype=np.float32)
    Ai = lil_matrix(A.shape, dtype=np.float32)
    for bi in range(q):
        i0, i1 = bi*n_rows, (bi+1)*n_rows
        j0, j1 = bi*n_cols, (bi+1)*n_cols
        
        a = np.linalg.inv(A[i0:i1,j0:j1].todense())
        Ai[i0:i1,j0:j1] = a
    print Ai.shape, Ai.nnz, Ai.size
    return Ai

def schur_trick(J, F, n_c, n_l, s_c=3, s_l=3, mu=1.0):
    """
    """
    ts = []
    # H.dot(dx) = -g

    # H of form [[B E],[E.T,C]] 
    # note, F = residual_BA(...)
    # B : block diagonal with p blocks of size cxc, c=3 (x,y,h)
    # C : block diagonal with q blocks of size sxs, s=3 (x,y,z)o

    # S : E.C^{-1}.E', Schur complement of C in H, "reduced camera matrix"
    # B - S.dy = v - E.C^{-1}.w (NOTE: x' = x.T)

    #H0 = J.T.dot(J)
    ts.append(time.time())
    H0 = J.T.dot(J) # << most time-consuming step
    ts.append(time.time())

    # opt1
    n = H0.shape[0]
    i = np.arange(n)
    H0[i,i] *= (1.0+mu)
    H=H0
    ts.append(time.time())

    # opt2
    #D = np.diag(np.sqrt(np.diag(H0)))
    #H = H0 + mu * D.T.dot(D) # mu, D are regularization terms
    #print 'dbg -1'
    #H = H0
    g = J.T.dot(F) # << sometimes consumes a lot of time as well
    ts.append(time.time())

    o_l = n_c*s_c # landmark index offset

    ts.append(time.time())
    # partition matrices
    B  = H[:o_l, :o_l]
    E  = H[:o_l, o_l:]
    ET = H[o_l:, :o_l]
    C  = H[o_l:, o_l:]
    v, w = -g[:o_l], -g[o_l:]

    # IMPORTANT : this is a hack on a lot of levels.
    # 1. Directly manipulates Ci data
    # 2. original H is not preserved since copy=False
    # 3. Relies on data to be in a certain order
    # FAST, but hacky. worth it? definitely.
    Ci = bsr_matrix(C, shape=C.shape, blocksize=(s_c,s_c))
    Ci.data = np.linalg.inv(Ci.data)

    # validation
    # Ci0 = block_inv(C, n_rows=s_c, n_cols=s_c) # should be pretty cheap?
    # print np.square(Ci0.todense() - Ci.todense()).sum()

    #Cib = np.linalg.inv(Cb.data)
    ts.append(time.time())

    # -- opt1 : direct multiple
    #ECi = E.dot(Ci) # << problem

    # -- opt2 : construct blockwise dot product
    #t0 = time.time()
    #Eci = np.zeros((n_c*s_c, n_l*s_l), dtype=np.float32)
    #for i in range(n_c):
    #    i0 = i * s_c
    #    i1 = (i+1) * s_c
    #    for j in range(n_l):
    #        j0 = j * s_l
    #        j1 = (j+1) * s_l
    #        Eci[i0:i1,j0:j1] = E[i0:i1,j0:j1].dot(Ci[j0:j1,j0:j1])
    ts.append(time.time())
    #t1 = time.time()
    #Eci = E.dot(Ci)
    #t2 = time.time()
    ECi = E.dot(Ci) # Note ECi is dense here.
    ts.append(time.time())
    #t3 = time.time()

    #delta = Eci - Eci2
    #print np.abs(delta).sum()
    #print (Eci != 0).sum(), Eci.size
    # S,v,w are all dense at this point
    S = (B - E.dot(ECi.T).T).todense()
    ts.append(time.time())
    b = (v - ECi.dot(w))
    ts.append(time.time())

    dy = None
    if np.linalg.cond(S) < (1.0 / np.finfo(np.float32).eps):
        try:
            cf = cho_factor(S)
            dy = cho_solve(cf, b)
            #dy = np.linalg.solve(S, b) # << pose optimization
        except Exception as e:
            print 'solve failed ; fallback to lstsq : {}'.format(e)
    ts.append(time.time())
    if dy is None:
        # probably ill-conditioned
        # TODO : maybe check residuals.
        dy = np.linalg.lstsq(S, b)[0]
    ts.append(time.time())
    dz = Ci.dot(w-ET.dot(dy)) # << landmark optimization
    ts.append(time.time())

    dt = np.diff(ts)
    print 'times', dt
    print 'times (normalized)', dt / dt.max()
    return np.concatenate([dy,dz], axis=0)

def main():
    n_test_p = 16
    n_test_l = 100

    # boilerplate parameters setup
    Ks = 1.0
    K = np.reshape([
            499.114583 * Ks, 0.000000, 325.589216 * Ks,
            0.000000, 498.996093 * Ks, 238.001597 * Ks,
            0.000000, 0.000000, 1.000000], (3,3))
    T_c2b = tx.compose_matrix(
            angles=[-np.pi/2 - np.deg2rad(10),0.0,-np.pi/2],
            translate=[0.174,0,0.113])
    T_b2c = tx.inverse_matrix(T_c2b)

    e = None
    p = np.random.uniform(size=(n_test_p,3))
    l = np.random.uniform(size=(n_test_l,3))

    #T_o2b = [
    #        tx.inverse_matrix( tx.compose_matrix(
    #        angles=[0,0,p_[-1]],
    #        translate=[p_[0],p_[1],0] )) for p_ in p]
    #T_o2b = np.stack(T_o2b, axis=0)
    #print 'validation-truth'
    #print T_o2b[0]

    ba_J(p, l, K, T_b2c, T_c2b)
    R_b2c = T_b2c[:3,:3]
    t_b2c = T_b2c[:3,3:]
    ba_J_v2(p, l, K, R_b2c, t_b2c, pt_h)

if __name__ == "__main__":
    main()
