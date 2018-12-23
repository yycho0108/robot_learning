import numpy as np
import cv2
from tf import transformations as tx

def pt_to_pth(pt):
    # copied
    return np.pad(pt, [(0,0),(0,1)],
            mode='constant',
            constant_values=1.0
            )

def project_BA(cam, lmk, T_b2c, T_c2b, K):
    """
    cam = np.array(Nx3) camera 2d pose (x,y,h) (WARN: actually base_link pose)
    lmk = np.array(Nx3) landmark position (x,y,z) in <map> coordinates
    """
    # naive iterative method: call the following repeatedly:
    # self.cvt_.pt3_pose_to_pt2_msk(lmk, cam)
    # alternative ...
    # K = (3x3) projection matrix
    # P = (Nx3x4) (R|t) normalized projection matrix

    n = len(cam)

    x = cam[:,0]
    y = cam[:,1]
    h = cam[:,2]

    # z-axis heading
    c = np.cos(h)
    s = np.sin(h)

    # directly construct batchwise T_o2b
    T_o2b = np.zeros((n,4,4), dtype=np.float32)

    # Rotation Part
    T_o2b[:,0,0] = c
    T_o2b[:,0,1] = s
    T_o2b[:,1,0] = -s
    T_o2b[:,1,1] = c
    T_o2b[:,2,2] = 1

    # Translation part
    T_o2b[:,0,3] = -y*s - x*c
    T_o2b[:,1,3] = x*s - y*c
    #T_o2b[:,2,3] = 0

    # Homogeneous part
    T_o2b[:,3,3] = 1

    one = np.ones_like(lmk[..., :1])
    lmk_h = np.concatenate([lmk,one], axis=-1)

    pt3 = reduce(np.matmul,[
        T_b2c,
        T_o2b,
        T_c2b,
        lmk_h[...,None]])

    #print 'where is the landmark in cam coord?'
    #print pt3.ravel()

    pt2 = reduce(np.matmul,[
        K,
        T_b2c[:3],
        T_o2b,
        T_c2b,
        lmk_h[...,None]])

    pt2 = pt2[...,0] # point 3x1 -> 3
    return pt2[:,:2] / pt2[:,2:]

def main():
    #np.random.seed(0)
    Ks = 1.0
    K = np.reshape([
            499.114583 * Ks, 0.000000, 325.589216 * Ks,
            0.000000, 498.996093 * Ks, 238.001597 * Ks,
            0.000000, 0.000000, 1.000000], (3,3))
    T_c2b = tx.compose_matrix(
            angles=[-np.pi/2 - np.deg2rad(10),0.0,-np.pi/2],
            translate=[0.174,0,0.113])
    T_b2c = tx.inverse_matrix(T_c2b)

    cam = np.random.uniform(-5.0, 5.0, size=(1,3))
    #cam = np.float32([0,0,0]).reshape(1,3)
    print 'cam', cam.ravel()
    lmk = np.random.uniform(-5.0, 5.0, size=(1,3))
    #lmk = np.float32([0,0,1]).reshape(1,3)
    print 'lmk', lmk

    # try to create rvec/tvec
    x, y, h = cam[0]
    T_b2o = tx.compose_matrix(
            angles=[0,0,h],
            translate=[x, y, 0])

    print 'cv2'
    #print tx.inverse_matrix(T_b2o)
    Ti = np.linalg.multi_dot([
        T_b2c,
        tx.inverse_matrix(T_b2o),
        T_c2b])

    rvec = cv2.Rodrigues(Ti[:3,:3])[0]
    tvec = Ti[:3,3:]

    # lmk in cam coord
    #print Ti.dot(lmk)
    #print 'where is the landmark in cam coord?'
    #print lmk.dot(Ti[:3,:3].T) + Ti[:3,3:].T

    #one = np.ones_like(lmk[..., :1])
    #lmk_h = np.concatenate([lmk,one], axis=-1)
    #print Ti.dot(lmk_h.T).ravel()

    pt2, _ = cv2.projectPoints(
            lmk,
            rvec,
            tvec,
            cameraMatrix=K,
            distCoeffs=np.zeros(5),
            # TODO : verify if it is appropriate to apply distortion
            )
    print pt2

    print '===='
    print 'custom'
    print project_BA(cam, lmk, T_b2c, T_c2b, K)

if __name__ == "__main__":
    main()
