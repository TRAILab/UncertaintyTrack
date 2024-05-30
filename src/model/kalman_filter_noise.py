"""Code adapted from mmtracking kalman_filter.py"""

import numpy as np
import torch
import scipy.linalg

from mmtrack.models.builder import MOTION
from mmtrack.models.motion.kalman_filter import KalmanFilter

from core.utils import (
    KLDivergence, JRDivergence, GIoU, Bhattacharyya, Wasserstein, Mahalanobis
)

@MOTION.register_module()
class KalmanFilterWithNoise(KalmanFilter):
    """A Kalman filter with measurement noise for tracking bounding boxes in image space.
    Adapted from mmtracking kalman_filter.py
    
    Args:
        center_only (bool): If set to True, the filter only tracks the bounding
            box centers instead of the whole boxes.
        distance_type (str): Distance type, KL, JR, or mahalanobis.
        threshold (float): Gating threshold. Overwrite the default value in
            KalmanFilter if distance_type is not mahalanobis.
        mahalanobis_cfg (dict): Custom config for distance function. Specifies
            the direction of mahalanobis distance calculation.
    """
    _distance_fns = {
        "kl": KLDivergence(),
        "jr": JRDivergence(),
        "giou": GIoU(mode='giou'),  #* one2one because track mean is already expanded; set to False if no covariance
        "iou": GIoU(mode='iou'),
        "giou with sampling": GIoU(with_sampling=True),
        "bhattacharyya": Bhattacharyya(),
        "wasserstein": Wasserstein()
    }   

    def __init__(self,  
                 center_only=False, 
                 distance_type="mahalanobis", 
                 threshold=None, 
                 mahalanobis_cfg=dict()):
        super(KalmanFilterWithNoise, self).__init__(center_only=center_only)
        if distance_type in self._distance_fns:
            self.distance = self._distance_fns[distance_type.lower()]
            self.use_mahalanobis = False
        else:
            if distance_type.lower() != "mahalanobis":
                print(f"Distance type {distance_type} not supported. \
                        Using Mahalanobis distance as default.")
            self.use_mahalanobis = True
            self.mahalanobis = Mahalanobis()
        self.use_custom = not self.use_mahalanobis
        if threshold is not None and self.use_custom:
            #? Overwrite gating_threshold from KalmanFilter
            self.gating_threshold = self.distance.get_threshold(threshold) \
                                    if hasattr(self.distance, "get_threshold") \
                                    else threshold
        
        if not self.use_mahalanobis:
            print("Distance config will be ignored for other distances.")
            self.mahalanobis_cfg = dict()
        else:
            self.mahalanobis_cfg = mahalanobis_cfg

    def initiate(self, measurement, measurement_cov):
        """Create track from unassociated measurement with noise.

        Args:
            measurement (ndarray):  Bounding box coordinates (x, y, a, h) with
            center position (x, y), aspect ratio a, and height h.
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
                10 * self._std_weight_velocity * measurement[3], 1e-5,
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

    def project(self, mean, covariance, measurement_covs):
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
                mean = np.tile(mean, (measurement_covs.shape[0], 1))
                covariance = np.tile(covariance, (measurement_covs.shape[0], 1, 1))
            covariance += measurement_covs
        return mean, covariance

    def update(self, mean, covariance, measurement, measurement_cov):
        """Run Kalman filter correction step.

        Args:
            mean (ndarray): The predicted state's mean vector (8 dimensional).
            covariance (ndarray): The state's covariance matrix (8x8
                dimensional).
            measurement (ndarray): The 4 dimensional measurement vector
                (x, y, a, h), where (x, y) is the center position, a the
                aspect ratio, and h the height of the bounding box.
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
    
    def measurement2track_mahalanobis(self,
                                   mean,
                                   covariance,
                                   measurements,
                                   one2one=True):
        """Computes the squared mahalanobis distance between a single state distribution
        and a set of measurements. This function is the inverted version of 
        `track2measurement_mahalanobis`. This function is the normal version of
        mahalanobis distance.

        Args:
            mean (ndarray): Mean vector over the state distribution (8
                dimensional).
            covariance (ndarray): Covariance of the state distribution (8x8
                dimensional).
            measurements (ndarray): An Nx4 dimensional matrix of N
                measurements, each in format (x, y, a, h) where (x, y) is the
                bounding box center position, a the aspect ratio, and h the
                height.
            one2one (bool, optional): Flag to determine if the distance is
                computed in a one-to-one fashion or in a pairwise manner. 
                If True, then the distance is only computed in order. 
                If False, then the distance is computed for all pairs.
                Defaults to True.

        Returns:
            ndarray: Returns an array of length N, where the i-th element
            contains the computed distance.
        """
        return self.mahalanobis(mean, covariance, measurements, None, one2one=one2one)
    
    def track2measurement_mahalanobis(self,
                                   mean,
                                   measurements,
                                   measurement_covs,
                                   one2one=True):
        """Computes the squared mahalanobis distance between a set of measurements and
        a single state distribution. This function is the inverted version of 
        `measurement2track_mahalanobis`. The state distribution is treated as
        a single measurement and the measurements are treated as multiple states.

        Args:
            mean (ndarray): Mean vector over the state distribution (8
                dimensional).
            measurements (ndarray): An Nx4 dimensional matrix of N
                measurements, each in format (x, y, a, h) where (x, y) is the
                bounding box center position, a the aspect ratio, and h the
                height.
            measurement_covs (ndarray): An Nx4x4 dimensional tensor containing
                the measurement covariance matrices for each of the N
                measurements.
            one2one (bool, optional): Flag to determine if the distance is
                computed in a one-to-one fashion or in a pairwise manner. 
                If True, then the distance is only computed in order. 
                If False, then the distance is computed for all pairs.
                Defaults to True.

        Returns:
            ndarray: Returns an array of length N, where the i-th element
            contains the computed distance.
        """
        return self.mahalanobis(measurements, measurement_covs, mean, None, one2one=one2one)

    def gating_distance(self,
                        mean,
                        covariance,
                        measurements,
                        measurement_covs,
                        only_position=False):
        """Compute gating distance between state distribution and measurements.
        
        If mahalanobis distance is used:
            If direction is 'normal', the distance of measurements from the predicted
            state distribution is computed. If direction is 'flipped', the distance 
            of the predicted state distribution from the measurements is computed.
            If direction is 'symmetric' or `bidirectional_gating` is True, both
            distances are computed and combined.
        
            A suitable distance threshold can be 
            obtained from `chi2inv95`. If `only_position` is False, the chi-square 
            distribution has 4 degrees of freedom, otherwise 2.
        If custom distance is used:
            The distance between the state distribution and the measurement distributions
            is computed using the custom distance function provided. The custom distance
            functions are symmetric and the direction is ignored.

        Args:
            mean (ndarray): Mean vector over the state distribution (8
                dimensional).
            covariance (ndarray): Covariance of the state distribution (8x8
                dimensional).
            measurements (ndarray): An Nx4 dimensional matrix of N
                measurements, each in format (x, y, a, h) where (x, y) is the
                bounding box center position, a the aspect ratio, and h the
                height.
            measurement_covs (ndarray): An Nx4x4 dimensional tensor containing
                the measurement covariance matrices for each of the N
                measurements.
            only_position (bool, optional): If True, distance computation is
                done with respect to the bounding box center position only.
                Defaults to False.

        Returns:
            measurement_track_dist (ndarray | None): Returns an array of length N, 
                where the i-th element contains the distance between state and measurements
                in the normal direction.
            track_measurement_dist (ndarray | None): Returns an array of length N, 
                where the i-th element contains the distance between state and measurements
                in the inverted direction.
        """
        projected_means, projected_covs = self.project(mean, covariance, measurement_covs)
        
        if only_position:
            #TODO: Implement only_position
            raise NotImplementedError
        
        #? Get diagonal elements to simplify calculations
        projected_means = torch.from_numpy(projected_means)
        projected_covs = torch.diagonal(torch.from_numpy(projected_covs), dim1=-2, dim2=-1)
        measurements = torch.from_numpy(measurements)
        measurement_covs = torch.diagonal(torch.from_numpy(measurement_covs), dim1=-2, dim2=-1) \
                            if measurement_covs is not None else None
        #* projected_means and projected_covs won't be expanded
        one2one = False if measurement_covs is None else True
        if not self.use_mahalanobis:
            if measurement_covs is None:
                raise ValueError(
                    "Measurement covariance must be provided for distribution comparison \
                        using other distance functions."
                )
            distance = self.distance(projected_means,
                                    projected_covs,
                                    measurements,
                                    measurement_covs,
                                    one2one=one2one)
            return distance, None
        else:
            #? Mahalanobis distance as default
            measurement_track_dist, track_measurement_dist = None, None
            bidirectional = any([
                self.mahalanobis_cfg.get('bidirectional_gating', False),
                self.mahalanobis_cfg.get('direction', 'normal') == 'symmetric'
            ])
            #* Normal direction (measurement -> track)
            if bidirectional or self.mahalanobis_cfg.get('direction', 'normal') == 'normal':
                measurement_track_dist = self.measurement2track_mahalanobis(projected_means,
                                                                            projected_covs,
                                                                            measurements,
                                                                            one2one=one2one)
            #* Inverted direction (track -> measurement)
            if bidirectional or self.mahalanobis_cfg.get('direction', 'normal') == 'flipped':
                track_measurement_dist = self.track2measurement_mahalanobis(projected_means,
                                                                            measurements,
                                                                            measurement_covs,
                                                                            one2one=one2one)
            
            return measurement_track_dist, track_measurement_dist
        
    def _apply_gating(self, m2t_cost, t2m_cost):
        """Apply gating threshold to cost matrices and return 
        the appropriate cost matrix specified by the mahalanobis_cfg.

        Args:
            m2t_cost (ndarray): Cost matrix from measurements to tracks.
                Normal direction. Shape (M, N)
            t2m_cost (ndarray): Cost matrix from tracks to measurements.
                Inverted direction. Shape (M, N)

        Returns:
            ndarray: Cost matrix with gating threshold applied.
        """
        if m2t_cost is not None:
            m2t_cost[m2t_cost > self.gating_threshold] = np.nan
        if t2m_cost is not None:
            t2m_cost[t2m_cost > self.gating_threshold] = np.nan
        
        if self.mahalanobis_cfg.get('direction', 'normal') == 'symmetric':
            cost = m2t_cost + t2m_cost
        elif self.mahalanobis_cfg.get('direction', 'normal') == 'flipped':
            cost = t2m_cost
            if self.mahalanobis_cfg.get('bidirectional_gating', False):
                nan_mask = np.isnan(m2t_cost)
                cost[nan_mask] = np.nan
        else:
            cost = m2t_cost
            if self.mahalanobis_cfg.get('bidirectional_gating', False):
                nan_mask = np.isnan(t2m_cost)
                cost[nan_mask] = np.nan
        
        return cost

    def track(self, tracks, bboxes, bbox_covs):
        """Track forward.

        Args:
            tracks (dict[int:dict]): Track buffer.
            bboxes (Tensor): Detected bounding boxes.
            bbox_covs (Tensor): Covariance matrices of detected bounding boxes.

        Returns:
            (dict[int:dict], Tensor): Updated tracks and distance cost matrix.
        """
        m2t_costs, t2m_costs = [], []
        if isinstance(bbox_covs, torch.Tensor):
            bbox_covs = bbox_covs.cpu().numpy()
        else:
            bbox_covs = None

        for id, track in tracks.items():
            track.mean, track.covariance = self.predict(
                track.mean, track.covariance)
            measurement2track, track2measurement = self.gating_distance(track.mean,
                                                    track.covariance,
                                                    bboxes.cpu().numpy(),
                                                    bbox_covs,
                                                    only_position=self.center_only)
            if measurement2track is not None:
                m2t_costs.append(measurement2track)
            if track2measurement is not None:
                t2m_costs.append(track2measurement)

        #? Accumulate costs based on direction in mahalanobis_cfg
        m2t_costs = np.stack(m2t_costs, 0) if len(m2t_costs) > 0 else None
        t2m_costs = np.stack(t2m_costs, 0) if len(t2m_costs) > 0 else None
        costs = self._apply_gating(m2t_costs, t2m_costs)
        
        return tracks, costs
