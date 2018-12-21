import numpy as np
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints, JulierSigmaPoints

import sys
sys.path.append('../')
from utils.vo_utils import add_p3d, sub_p3d

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
    spts = JulierSigmaPoints(6, 6-2, sqrt_method=np.linalg.cholesky, subtract=ukf_residual)

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

