import numpy as np
from tf import transformations as tx
from matplotlib import pyplot as plt

def scale_c(s_b, R_c, t_c,
        T_c2b, T_b2c,
        ):
    R_c2b = T_c2b[..., :3, :3]
    t_c2b = T_c2b[..., :3, 3:]
    R_b2c = T_b2c[..., :3, :3]
    t_b2c = T_b2c[..., :3, 3:]

    R_b = reduce(np.matmul, [R_c2b, R_c, R_b2c])

    v1 = np.matmul(R_c2b, t_c)
    v2 = t_c2b - reduce(np.matmul,[R_b, t_c2b])

    # c_a*s_c^2 + c_b*s_c^1 + c_c = s_b**2
    c_a = (v1*v1).sum(axis=(-2,-1))
    c_b = 2*(v1*v2).sum(axis=(-2,-1))
    c_c = (v2*v2).sum(axis=(-2,-1)) - s_b**2

    #print c_a
    #print c_b
    #print c_c

    det = c_b**2-4*c_a*c_c # determinant part
    
    sol_1 = (-c_b + np.sqrt(det) ) / (2*c_a) # apparently always >0?
    print 'sol_1', sol_1
    sol_2 = (-c_b - np.sqrt(det) ) / (2*c_a)
    print 'sol_2', sol_2

    # check by forward propagation
    # actually, both solutions are considered valid. what?
    stc1 = sol_1[:,None,None] * t_c #(t_c / np.linalg.norm(t_c, axis=(-2,-1)))
    stb1 = np.matmul(R_c2b, stc1) - np.matmul(R_b, t_c2b) + t_c2b
    print 'stb1', stb1[...,0]
    stcr1 = np.matmul(R_b2c, stb1 - t_c2b + np.matmul(R_b, t_c2b))
    print 'scale_b1', np.linalg.norm(stb1, axis=(-2,-1))
    print np.abs(stcr1 - stc1).sum()
    # turns out, for whatever reason, sol_1 is ALWAYS correct.

    stc2 = sol_2[:,None,None] * t_c #(t_c / np.linalg.norm(t_c, axis=(-2,-1)))
    stb2 = np.matmul(R_c2b, stc2) - np.matmul(R_b, t_c2b) + t_c2b
    print 'stb2', stb2[...,0]
    stcr2 = np.matmul(R_b2c, stb2 - t_c2b + np.matmul(R_b, t_c2b))
    print 'scale_b2', np.linalg.norm(stb2, axis=(-2,-1))
    print np.abs(stcr2 - stc2).sum()

def main():
    N = 10

    # sample T_b2c
    rvec = np.random.uniform(-np.pi, np.pi, size=(N,3))
    tvec = np.random.uniform(-np.pi, np.pi, size=(N,3))

    # super lazy way to construct the matrix. works, anyways.
    T_b2c = np.float32([
        tx.compose_matrix(
            angles=r,
            translate=t)
        for (r,t) in zip(rvec, tvec)
        ])
    T_c2b = np.linalg.inv(T_b2c)

    # sample Tbb (base_link motion)
    r_z = np.random.uniform(-np.pi, np.pi, size=(N))
    dx, dy  = np.random.uniform(-np.pi, np.pi, size=(2,N))
    c_z, s_z = np.cos(r_z), np.sin(r_z)
    Tbb  = np.zeros((N,4,4), dtype=np.float32)

    # rotation part
    Tbb[:,0,0] = c_z
    Tbb[:,0,1] = -s_z
    Tbb[:,1,0] = s_z
    Tbb[:,1,1] = c_z
    Tbb[:,2,2] = 1

    # translation part
    Tbb[:,0,3] = dx
    Tbb[:,1,3] = dy
    #dz=0

    # homogeneous part
    Tbb[:,3,3] = 1

    # base scale
    print 't_b (ground truth)', Tbb[:,:3,3]
    s_b = np.linalg.norm([dx,dy], axis=0)
    print 'scale_b (ground truth)', s_b.ravel()

    Tcc = reduce(np.matmul, [T_b2c, Tbb, T_c2b])
    tcc = Tcc[:, :3, 3]
    s_c = np.linalg.norm(tcc, axis=-1)

    R_c = Tcc[:, :3, :3]
    t_c = Tcc[:, :3, 3:]

    print 'scale_c (ground truth)', np.linalg.norm(t_c, axis=1).ravel()

    # anticipated scale_c
    s_c_r = scale_c(s_b, R_c, t_c / np.linalg.norm(t_c, axis=1, keepdims=True),
        T_c2b, T_b2c
        )

    #plt.plot(r_z, s_c/s_b, '+')
    plt.plot(s_c, s_b, '+')
    plt.show()

if __name__ == "__main__":
    main()
