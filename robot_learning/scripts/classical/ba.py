from scipy.sparse import bsr_matrix
import numpy as np
from tf import transformations as tx
import sparse

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

    ba_J(p, l,
        K, T_b2c, T_c2b)

if __name__ == "__main__":
    main()