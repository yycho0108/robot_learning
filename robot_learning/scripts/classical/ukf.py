import numpy as np
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import ExtendedKalmanFilter as EKF
from filterpy.kalman import MerweScaledSigmaPoints, JulierSigmaPoints

import sys
sys.path.append('../')
from misc import Rmat
from utils.vo_utils import add_p3d, sub_p3d

def get_QR(pose, dt):
    # Get appropriate Q/R Matrices from current pose.
    # Mostly just deals with getting the right orientation.
    Q0 = np.diag(np.square([5e-2, 2e-2, 6e-2, 2.5e-2, 1e-2, 8e-1]))
    R0 = np.diag(np.square([5e-2, 7e-2, 4e-2]))
    T = Rmat(pose[-1])

    Q = Q0.copy()

    # constant acceleration model?
    # NOTE : currently results in non-positive-definite matrix
    #g = [dt**2/2, dt]
    #G = np.outer(g,g)
    # x-part
    # Q[np.ix_([0,3],[0,3])] = G * (0.3)**2 # 10 cm/s^2
    # Q[np.ix_([1,4],[1,4])] = G * (0.1)**2 # 2 cm/s^2
    # Q[np.ix_([2,5],[2,5])] = G * (0.5)**2 # ~ 6 deg/s^2

    # apply rotation to translational parts
    Q[:2,:2] = T.dot(Q[:2,:2]).dot(T.T)
    #Q[3:5,3:5] = T.dot(Q[3:5,3:5]).dot(T.T)
    #NOTE: vx-vy components are invariant to current pose

    #Q = (Q+Q.T)/2.0
    #if np.any(Q<0):
    #    Q -= np.min(Q)

    # R has nothing to do with time
    R = R0.copy()
    R[:2,:2] = T.dot(R[:2,:2]).dot(T.T)

    return Q, R

def ukf_fx(s, dt):
    x,y,h,vx,vy,w = s

    dx = vx * dt
    dy = vy * dt
    dh = w * dt
    x,y,h = add_p3d([x,y,h], [dx,dy,dh])
    #x += v * np.cos(h) * dt
    #y += v * np.sin(h) * dt
    #h += w * dt
    #v *= 0.999
    return np.asarray([x,y,h,vx,vy,w])

def ukf_hx(s):
    return s[:3].copy()

def ukf_mean(xs, wm):
    """
    Runs circular mean for angular states, which is critical to preventing issues related to linear assumptions. 
    WARNING : do not replace with the default mean function
    """
    # Important : SUM! not mean.
    mx = np.sum(xs * np.expand_dims(wm, -1), axis=0)
    ms = np.mean(np.sin(xs[:,2])*wm)
    mc = np.mean(np.cos(xs[:,2])*wm)
    mx[2] = np.arctan2(ms,mc)
    return mx

def ukf_residual(a, b):
    """
    Runs circular residual for angular states, which is critical to preventing issues related to linear assumptions.
    WARNING : do not replace with the default residual function.
    """
    d = np.subtract(a, b)
    d[2] = np.arctan2(np.sin(d[2]), np.cos(d[2]))
    return d

def build_ukf(x0=None, P0=None,
        Q = None, R = None
        ):
    # build ukf
    if x0 is None:
        x0 = np.zeros(6)
    if P0 is None:
        # initial pose is very accurate, but velocity is unknown.
        # considering vehicle dynamics, it's most likely constrained
        # within +-0.5m/s in x direction, +-0.1m/s in y direction,
        # and 0.5rad/s in angular velocity.
        P0 = np.diag(np.square(
            [1e-6, 1e-6, 1e-6, 5e-1, 1e-1, 5e-1]
            ))
    if Q is None:
        # treat Q as a "tuning parameter"
        # low Q = high confidence in general state estimation
        # high Q = high confidence in measurement
        # from https://www.researchgate.net/post/How_can_I_find_process_noise_and_measurement_noise_in_a_Kalman_filter_if_I_have_a_set_of_RSSI_readings
        # Higher Q, the higher gain, more weight to the noisy measurements
        # and the estimation accuracy is compromised; In case of lower Q, the
        # better estimation accuracy is achieved and time lag may be introduced in the estimated value.
        # process noise
        Q = np.diag(np.square(
            [5e-2, 5e-2, 1e-1, 1e-1, 1e-1, 1e-1]
            )) #xyhvw
    if R is None:
        # in general anticipate much lower heading error than positional error
        # 1e-2 ~ 0.57 deg
        R = np.diag(np.square(
            [2e-2, 2e-2, 5e-2]
            )) # xyh

    #spts = MerweScaledSigmaPoints(6, 1e-3, 2, 3-6, subtract=ukf_residual)
    spts = MerweScaledSigmaPoints(6, 1e-3, 2, 0, subtract=ukf_residual)
    #spts = JulierSigmaPoints(6, 6-2, sqrt_method=np.linalg.cholesky, subtract=ukf_residual)

    ukf = UKF(6, 3, (1.0 / 30.), # dt guess
            ukf_hx, ukf_fx, spts,
            x_mean_fn=ukf_mean,
            z_mean_fn=ukf_mean,
            residual_x=ukf_residual,
            residual_z=ukf_residual)
    ukf.x = x0.copy()
    ukf.P = P0.copy()
    ukf.Q = Q
    ukf.R = R

    return ukf

def ekf_FJ(x, dt):
    i_x = 0 
    i_y = 1 
    i_h = 2 
    i_vx = 3 
    i_vy = 4
    i_w = 5

    F = np.eye(6)

    x,y,h,vx,vy,w = x.reshape(-1)

    c = np.cos(h)
    s = np.sin(h)

    F[i_x, i_h]  = dt * (-s*vx -c*vy) # F[0] = d(x') / d(s)
    F[i_x, i_vx] = dt * c
    F[i_x, i_vy] = dt * -s

    F[i_y, i_h]  = dt * (c*vx - s*vy)
    F[i_y, i_vx] = dt * s
    F[i_y, i_vy] = dt * c

    F[i_h, i_w] = dt

    return F

def ekf_HJ(x):
    return np.eye(3,6)

class EKFWrapper(EKF):
    def __init__(self, *args, **kwargs):
        super(EKFWrapper, self).__init__(*args, **kwargs)

    def predict(self, dt):
        self.F = F = ekf_FJ(self.x, dt)

        self.P = np.linalg.multi_dot([
            F, self.P, F.T]) + self.Q
        self.x = ukf_fx(self.x, dt) # uses the same transition function

        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)
        return

    def update(self, z):
        return super(EKFWrapper,self).update(z,
                ekf_HJ,
                ukf_hx,
                residual=ukf_residual)

def build_ekf(x0=None, P0=None,
        Q=None, R=None
        ):

    # build ukf
    if x0 is None:
        x0 = np.zeros(6)
    if P0 is None:
        # initial pose is very accurate, but velocity is unknown.
        # considering vehicle dynamics, it's most likely constrained
        # within +-0.5m/s in x direction, +-0.1m/s in y direction,
        # and 0.5rad/s in angular velocity.
        P0 = np.diag(np.square(
            [1e-6, 1e-6, 1e-6, 5e-1, 1e-1, 5e-1]
            ))
    if Q is None:
        # treat Q as a "tuning parameter"
        # low Q = high confidence in general state estimation
        # high Q = high confidence in measurement
        # from https://www.researchgate.net/post/How_can_I_find_process_noise_and_measurement_noise_in_a_Kalman_filter_if_I_have_a_set_of_RSSI_readings
        # Higher Q, the higher gain, more weight to the noisy measurements
        # and the estimation accuracy is compromised; In case of lower Q, the
        # better estimation accuracy is achieved and time lag may be introduced in the estimated value.
        # process noise
        Q = np.diag(np.square(
            [5e-2, 5e-2, 1e-1, 1e-1, 1e-1, 1e-1]
            )) #xyhvw
    if R is None:
        # in general anticipate much lower heading error than positional error
        # 1e-2 ~ 0.57 deg
        R = np.diag(np.square(
            [2e-2, 2e-2, 5e-2]
            )) # xyh

    ekf = EKFWrapper(6, 3)
    ekf.x = x0.copy()
    ekf.P = P0.copy()
    ekf.Q = Q
    ekf.R = R
    return ekf
