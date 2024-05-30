"""Code adapted from mmtracking kalman_filter.py"""

import warnings
import numpy as np
import torch
import scipy.linalg

from mmtrack.models.builder import MOTION

@MOTION.register_module()
class KalmanFilterWithUncertainty(object):
    """A Kalman filter with measurement noise for tracking bounding boxes in image space.
    Adapts KalmanFilter from mmtracking kalman_filter.py.
    Changes:
        - Added measurement noise to the Kalman filter.
        - States are now: [x, y, w, h, vx, vy, vw, vh]
    """
    
    def __init__(self, fps=30):
        warnings.warn("""KalmanFilterWithUncertainty: Make sure the noise factors are 
                        tuned for the dataset by specifying the fps. 
                        The current values are for 30FPS datasets.""")
        ndim, dt = 4, 1
        #? Create Kalman filter model matrices.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        
        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. 
        self._std_weight_position = 1. / 20 * (30 / fps)
        self._std_weight_velocity = 1. / 160 * (30 / fps)
        
        
    def initiate(self, measurement, measurement_cov=None):
        """Create track from unassociated measurement with noise.

        Args:
            measurement (ndarray):  Bounding box coordinates (x, y, w, h) with
                                    center position (x, y), width w, and height h.
            measurement_cov (ndarray): Covariance matrix of the measurement.

        Returns:
             (ndarray, ndarray): Returns the mean vector (8 dimensional) and
                covariance matrix (8x8 dimensional) of the new track.
                Unobserved velocities are initialized to 0 mean.
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        if measurement_cov is not None:
            std_vel = [
                10 * self._std_weight_velocity * measurement[3],
                10 * self._std_weight_velocity * measurement[3],
                1e-5,
                10 * self._std_weight_velocity * measurement[3]
            ]
            vel_covariance = np.diag(np.square(std_vel))
            #* Top-right block is zero because the correlation between position
            #* and velocity cannot be easily determined.
            covariance = np.block([[measurement_cov, np.zeros_like(measurement_cov)],
                                [np.zeros_like(measurement_cov), vel_covariance]])
            
        else:
            std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3], 1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3], 1e-5,
            10 * self._std_weight_velocity * measurement[3]
            ]
            covariance = np.diag(np.square(std))
        return mean, covariance
    
    def predict(self, mean, covariance):
        """Run Kalman filter prediction step.
        
        Args:
            mean (ndarray): The 8 dimensional mean vector of the object
                state at the previous time step.

            covariance (ndarray): The 8x8 dimensional covariance matrix
                of the object state at the previous time step.

        Returns:
            (ndarray, ndarray): Returns the mean vector and covariance
                matrix of the predicted state. Unobserved velocities are
                initialized to 0 mean.
        """
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3], 1e-2,
            self._std_weight_position * mean[3]
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3], 1e-5,
            self._std_weight_velocity * mean[3]
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance
    
    def project(self, mean, covariance, measurement_covs=None):
        """Project state distribution to measurement space.

        Args:
            mean (ndarray): The state's mean vector (8 dimensional array).
            covariance (ndarray): The state's covariance matrix (8x8
                dimensional).
            measurement_covs (ndarray): The measurement's covariance matrix (Nx4x4
                or 4x4 dimensional).

        Returns:
            (ndarray, ndarray):  Returns the projected mean and covariance
            matrix of the given state estimate.
        """
        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot(
            (self._update_mat, covariance, self._update_mat.T))

        if measurement_covs is not None:
            if measurement_covs.ndim == 3:
                #* Multiple measurements each with their own covariance matrix.
                covariance = np.tile(covariance, (measurement_covs.shape[0], 1, 1))
            covariance += measurement_covs
        else:
            std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3], 1e-1,
            self._std_weight_position * mean[3]
            ]
            innovation_cov = np.diag(np.square(std))
            covariance += innovation_cov
        return mean, covariance
    
    def update(self, mean, covariance, measurement, measurement_cov=None):
        """Run Kalman filter correction step.

        Args:
            mean (ndarray): The predicted state's mean vector (8 dimensional).
            covariance (ndarray): The state's covariance matrix (8x8
                dimensional).
            measurement (ndarray): The 4 dimensional measurement vector
                (x, y, w, h), where (x, y) is the center position, w and h 
                the width and height of the bounding box.
            measurement_cov (ndarray): The measurement's covariance matrix (4x4 
                dimensional).

        Returns:
             (ndarray, ndarray): Returns the measurement-corrected state
             distribution.
        """
        projected_mean, projected_cov = self.project(mean, covariance, measurement_cov)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve((chol_factor, lower),
                                             np.dot(covariance,
                                                    self._update_mat.T).T,
                                             check_finite=False).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot(
            (kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance