from abc import ABC, abstractmethod
import torch
import numpy as np
import lap

from mmdet.core import bbox_overlaps
from mmtrack.core.bbox import bbox_xyxy_to_cxcyah
from core.utils import (
    Bhattacharyya, Wasserstein, Mahalanobis, Hellinger, AIRM,
    gaussian_entropy, bbox_xyxy_to_cxcywh, bbox_cov_xyxy_to_cxcywh,
    bbox_cov_xyxy_to_cxcyah,
    get_ellipse_box
)
from core.visualization import get_ellipse_params

class ProbabilisticTracker(ABC):
    _distance_fns = {
        "bhattacharyya": Bhattacharyya(),
        "wasserstein": Wasserstein(),
        "mahalanobis": Mahalanobis(),
        "hellinger": Hellinger()
    }
    
    def __init__(self, 
                 with_covariance=True,
                 primary_cascade=None,
                 secondary_fn=None,
                 secondary_cascade=False,
                 **kwargs):
        """Initialize the tracker.
        Args:
            with_covariance (bool): Whether to use measurement covariance matrix for tracking.
            primary_fn (dict | None): Primary matching method. Defaults to None.
                - type (str): Matching method (e.g. BhattaCharyya).
                - threshold (float): Threshold for matching.
            primary_cascade (dict | None): matching cascade mode for primary assignment.
                Should have keys:
                - num_bins (int): Number of bins for matching cascade.
                    If None, the matching is done greedily.
                Defaults to None.
            secondary_fn (dict | None): Secondary matching method. Defaults to None.
                - type (str): Matching method (e.g. BhattaCharyya).
                - threshold (float): Threshold for matching.
            secondary_cascade (dict | None): matching cascade mode for secondary assignment.
        """
        self.with_covariance = with_covariance
        #? Matching cascade
        self.with_primary_cascade = True if primary_cascade is not None \
                                            and not self.with_reid \
                                            else False
        self.primary_cascade = primary_cascade if self.with_primary_cascade else dict()
        self.with_secondary_cascade = True if secondary_cascade is not None else False
        self.secondary_cascade = secondary_cascade if self.with_secondary_cascade else dict()
        
        #? Custom distance functions
        self.custom_secondary = True if secondary_fn is not None else False
        if self.custom_secondary:
            if isinstance(secondary_fn, dict):
                distance_type = secondary_fn.get('type', '').lower()
                if distance_type in self._distance_fns:
                    self.secondary_distance = self._distance_fns[distance_type]
                    threshold = secondary_fn['threshold']
                    self.secondary_threshold = self.secondary_distance.get_threshold(threshold) \
                                    if hasattr(self.secondary_distance, "get_threshold") \
                                    else threshold
                else:
                    raise NotImplementedError(f'secondary_fn type {distance_type} \
                                                is not implemented')
            else:
                raise TypeError(f'secondary_fn must be a dict, \
                                    but got {type(secondary_fn)}')
        
        self._detailed_results = []
    @property
    def analysis_cfg(self):
        return self._analysis_cfg
    
    @analysis_cfg.setter
    def analysis_cfg(self, value):
        self._analysis_cfg = value
        
    @property
    def detailed_results(self):
        return self._detailed_results
    
    @detailed_results.setter
    def detailed_results(self, value):
        self._detailed_results = value
        
    def get_track_bbox_and_cov(self, id):
        bbox = bbox_xyxy_to_cxcyah(self.tracks[id].bboxes[-1])
        assert bbox.ndim == 2 and bbox.shape[0] == 1
        bbox = bbox.squeeze(0).cpu().numpy()
        if self.with_covariance:
            bbox_cov = self.tracks[id].bbox_covs[-1]
            if (bbox_cov - 1e-4 * torch.eye(4).to(bbox_cov)).abs().max() < 1e-4:
                bbox_cov = None
            else:
                bbox_cov = bbox_cov_xyxy_to_cxcyah(bbox_cov)
                assert bbox_cov.ndim == 3 and bbox_cov.shape[0] == 1
                bbox_cov = bbox_cov.squeeze(0).cpu().numpy()
        else:
            bbox_cov = None
        return bbox, bbox_cov
    
    def update_track_scores(self, frame_id, active_ids):
        for id in active_ids:
            if self.primary_cascade.get('num_bins', None) is None:
                self.tracks[id].age = (frame_id - self.tracks[id].frame_ids[-1]).item()
            self.tracks[id].entropy = gaussian_entropy(self.tracks[id].covariance)
        
    # def get_covariance_scores(self, det_bbox_covs, step):
    #     if step == "primary":
    #         mode = self.primary_cascade.get('mode', '')
    #         is_diagonal = True
    #         det_bbox_covs = det_bbox_covs.cpu()
    #     elif step == "secondary":
    #         mode = self.secondary_cascade.get('mode', '')
    #         is_diagonal = False
    #     else:
    #         raise ValueError(f"Invalid step: {step}. Must be 'primary' or 'secondary'.")
    #     if mode == 'entropy':
    #         scores = gaussian_entropy(det_bbox_covs.numpy(), is_diagonal=is_diagonal)
    #     elif mode == 'eigenvalue':
    #         scores = max_eigenvalue(det_bbox_covs.numpy(), is_diagonal=is_diagonal)
    #     else:
    #         raise ValueError(f"Invalid mode for covariance score: {mode}.")
    #     return scores
    def get_covariance_entropy(self, det_bbox_covs, is_diagonal=False):
        is_torch = isinstance(det_bbox_covs, torch.Tensor)
        det_bbox_covs_ = det_bbox_covs.cpu().numpy() if is_torch else det_bbox_covs
        entropy = gaussian_entropy(det_bbox_covs_, is_diagonal=is_diagonal)
        entropy = torch.from_numpy(entropy).to(det_bbox_covs) if is_torch else entropy
        return entropy
    
    def get_covariance_trace(self, det_bbox_covs):
        return torch.diagonal(det_bbox_covs, dim1=-2, dim2=-1).sum(dim=-1)
    
    # def matching_by_detection(self, cost_matrix, covariance_scores, step):
    #     if step == "primary":
    #         num_bins = self.primary_cascade['num_bins']
    #     elif step == "secondary":
    #         num_bins = self.secondary_cascade['num_bins']
    #     else:
    #         raise ValueError(f"Invalid step: {step}. Must be 'primary' or 'secondary'.")
        
    #     return matching_cascade_by_detection(covariance_scores, cost_matrix, num_bins)
    
    def _group_by_value(self, values, num_bins, min_value=None, max_value=None):
        """Group values into bins.
        Indices are in range (1, num_bins) inclusive.
        Indices are 0 if value is less than min_value.
        Indices are num_bins+1 if value is greater than max_value.
        """
        min_value = values.min() if min_value is None else min_value
        max_value = values.max() if max_value is None else max_value
        bins = np.linspace(min_value-2*np.finfo(np.float32).eps, 
                        max_value+2*np.finfo(np.float32).eps, 
                        num_bins+1)
        inds = np.digitize(values, bins)
        return inds
    
    def matching_by_bin(self, cost_matrix, scores, threshold, num_bins):
        """Bins detections based on `scores` and performs matching
        within each bin.
    
        Args:
            cost_matrix (np.array): cost matrix.
            scores (np.array): array of scores to group detections by.
            threshold (float): threshold for matching.
            num_bins (int, optional): number of bins to group detections into.
                Defaults to None.
        """
        use_greedy = True if num_bins is None else False
        if scores is None:
            raise ValueError("Scores must be specified for matching by bin.")
        cost = cost_matrix.copy()
        rows = np.full((cost_matrix.shape[0],), -1)
        cols = np.full((cost_matrix.shape[1],), -1)

        if use_greedy:
            cost[cost > threshold] = np.nan
            inds = np.argsort(scores)
            for c in inds:
                track_costs = cost[:, c]
                if len(np.isfinite(track_costs).nonzero()[0]) == 0:
                    #* No possible matches with current detection
                    continue
                r = np.nanargmin(track_costs)
                rows[r] = c
                cols[c] = r
                cost[r, :] = np.nan
        else:
            inds = self._group_by_value(scores, num_bins)
            for bin in range(1, inds.max()+1):
                dets_idx_in_bin = (inds == bin).nonzero()[0]
                if len(dets_idx_in_bin) == 0:
                    #* No detections in this bin
                    continue
                cost_bin = cost[:, dets_idx_in_bin]
                r_bin, c_bin = lap.lapjv(cost_bin, extend_cost=True, 
                                         cost_limit=threshold, return_cost=False)
                matched_tracks = r_bin > -1
                matched_detections = c_bin > -1
                if matched_detections.nonzero()[0].size == 0:
                    continue
                
                rows[matched_tracks] = dets_idx_in_bin[r_bin[matched_tracks]]
                cols[dets_idx_in_bin[matched_detections]] = c_bin[matched_detections]
                
                #? Remove matched tracks from next possible assignments
                cost[matched_tracks, :] = np.inf
                
        return rows, cols
    
    def custom_distance(self, 
                        track_bboxes, 
                        track_bbox_covs, 
                        det_bboxes, 
                        det_bbox_covs, 
                        one2one=False):
        #? Simplify custom distance functions using diagonal covariance
        track_bbox_covs = torch.diagonal(track_bbox_covs,
                                            dim1=-2,
                                            dim2=-1)
        det_bbox_covs = torch.diagonal(det_bbox_covs,
                                            dim1=-2,
                                            dim2=-1)
        dists = self.secondary_distance(track_bboxes,
                                        track_bbox_covs,
                                        det_bboxes,
                                        det_bbox_covs,
                                        one2one=one2one)
        return dists, self.secondary_threshold
    
    def mahalanobis(self, det_bboxes, det_bbox_covs, track_bboxes, track_bbox_covs, one2one=False):
        threshold = self._distance_fns['mahalanobis'].get_threshold()
        det_to_track = self._distance_fns['mahalanobis'](det_bboxes, det_bbox_covs, 
                                                         track_bboxes, track_bbox_covs,
                                                         one2one=one2one)
        # invalid_inds = det_to_track > threshold
        
        # if self.bidirectional:
        #     track_to_det = self._distance_fns['mahalanobis'](track_bboxes, track_bbox_covs, 
        #                                                      det_bboxes_cxcyah, det_bbox_covs_cxcyah,
        #                                                      one2one=True)
        #     invalid_inds = invalid_inds | (track_to_det > threshold)
        
        return det_to_track, threshold
    
    def get_outer_bboxes(self, bboxes, bbox_covs):
        extra_cols = bboxes[:, 4:] if bboxes.shape[-1] > 4 \
                    else torch.empty((bboxes.shape[0], 0)).to(bboxes)
        if bboxes.numel() > 0:
            x1y1 = get_ellipse_box(bboxes[:, :2], bbox_covs[:, :2, :2])[:, :2]
            x2y2 = get_ellipse_box(bboxes[:, 2:4], bbox_covs[:, 2:, 2:])[:, 2:]
            new_bboxes = torch.cat((x1y1, x2y2, extra_cols), dim=-1)
            new_bbox_covs = 1e-4 * torch.eye(4).reshape(1, 4, 4).repeat(
                                bbox_covs.size(0), 1, 1).to(bbox_covs)
            return new_bboxes, new_bbox_covs
        else:
            return bboxes, bbox_covs
    
    def get_inner_bboxes(self, bboxes, bbox_covs):
        extra_cols = bboxes[:, 4:] if bboxes.shape[-1] > 4 \
                    else torch.empty((bboxes.shape[0], 0)).to(bboxes)
        if bboxes.numel() > 0:
            x1y1 = get_ellipse_box(bboxes[:, :2], bbox_covs[:, :2, :2])[:, 2:]
            x2y2 = get_ellipse_box(bboxes[:, 2:4], bbox_covs[:, 2:, 2:])[:, :2]
            new_bboxes = torch.cat((x1y1, x2y2, extra_cols), dim=-1)
            new_bbox_covs = 1e-4 * torch.eye(4).reshape(1, 4, 4).repeat(
                                bbox_covs.size(0), 1, 1).to(bbox_covs)
            return new_bboxes, new_bbox_covs
        else:
            return bboxes, bbox_covs
        
    def check_width_height(self, img_meta, bboxes, bbox_covs, percent=0.25):
        """Check if radius of ellipse is less than percent of width and height.
        """
        img_h, img_w = img_meta[0]['ori_shape'][:-1]
        bboxes_ = bboxes.clone()
        bboxes_[:, 0::2] = torch.clamp(bboxes_[:, 0::2], 0, img_w)
        bboxes_[:, 1::2] = torch.clamp(bboxes_[:, 1::2], 0, img_h)
        
        width, height = bboxes_[:, 2] - bboxes_[:, 0], bboxes_[:, 3] - bboxes_[:, 1]
        tl_w, tl_h, _ = get_ellipse_params(bbox_covs[:, :2, :2], q=0.95)
        br_w, br_h, _ = get_ellipse_params(bbox_covs[:, 2:, 2:], q=0.95)
        
        tl_valid = (tl_w/2 < percent * width) & (tl_h/2 < percent * height)
        br_valid = (br_w/2 < percent * width) & (br_h/2 < percent * height)
        
        return tl_valid & br_valid
    
    # def construct_cov_from_eig(self, eigenvalues, eigenvectors):
    #     """Construct covariance matrix from eigenvalues and eigenvectors.
    #     """
    #     eigenvalues = torch.clamp(eigenvalues, min=0)
    #     covariance = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.transpose(-1, -2)
    #     return covariance
    
    def get_better_boxes(self, det_bboxes, det_bbox_covs, track_bboxes, mode='iou'):
        """Find a better bounding box from distribution that has higher iou with track.
        """
        outer_bboxes, outer_bbox_covs = self.get_outer_bboxes(det_bboxes, det_bbox_covs)
        inner_bboxes, inner_bbox_covs = self.get_inner_bboxes(det_bboxes, det_bbox_covs)
        det_ious = bbox_overlaps(track_bboxes, det_bboxes[:, :4], mode=mode, is_aligned=True)
        #TODO: get ious from association to avoid re-computing
        #TODO: compute ious all together instead of separately
        outer_ious = bbox_overlaps(track_bboxes, outer_bboxes[:, :4], mode=mode, is_aligned=True)
        inner_ious = bbox_overlaps(track_bboxes, inner_bboxes[:, :4], mode=mode, is_aligned=True)
        
        all_bboxes = torch.stack((det_bboxes, outer_bboxes, inner_bboxes))
        all_bbox_covs = torch.stack((det_bbox_covs, outer_bbox_covs, inner_bbox_covs))
        ious = torch.stack((det_ious, outer_ious, inner_ious))
        best_bboxes = all_bboxes[torch.argmax(ious, dim=0), torch.arange(ious.shape[1])]
        best_bbox_covs = all_bbox_covs[torch.argmax(ious, dim=0), torch.arange(ious.shape[1])]
        return best_bboxes, best_bbox_covs