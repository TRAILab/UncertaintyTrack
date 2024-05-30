"""Code adapted from mmtrack.models.trackers.ocsort_tracker.py"""

import lap
import numpy as np
import torch
from addict import Dict
from mmcv.runner import force_fp32
from mmdet.core import bbox_overlaps

from mmtrack.core.bbox import bbox_cxcyah_to_xyxy, bbox_xyxy_to_cxcyah
from mmtrack.models import TRACKERS
from mmtrack.models.trackers import SortTracker, OCSORTTracker

from core.utils import bbox_cov_xyxy_to_cxcyah
from .prob_tracker import ProbabilisticTracker


@TRACKERS.register_module()
class ProbabilisticOCSORTTracker(OCSORTTracker, ProbabilisticTracker):
    _score_modes = ['confidence', 'entropy']
    def __init__(self, 
                 with_covariance=True,
                 det_score_mode='confidence',
                 primary_fn=None,
                 primary_cascade=None,
                 secondary_fn=None,
                 secondary_cascade=False,
                 final_matching=False,
                 expand_boxes=False,
                 init_percent=0.6,
                 final_percent=0.3,
                 init_ellipse_filter=False,
                 second_ellipse_filter=False,
                 final_ellipse_filter=False,
                 use_mahalanobis=False,
                 return_expanded=False,
                 return_remaining_expanded=False,
                 **kwargs):
        ProbabilisticTracker.__init__(self,
                                      with_covariance=with_covariance,
                                      primary_fn=primary_fn,
                                      primary_cascade=primary_cascade,
                                      secondary_fn=secondary_fn,
                                      secondary_cascade=secondary_cascade)
        OCSORTTracker.__init__(self, **kwargs)
        if det_score_mode not in self._score_modes:
            raise ValueError(f"Invalid det_score_mode: {det_score_mode}. "
                             f"Must be one of {self._score_modes}.")
        self.det_score_mode = det_score_mode
        self.compute_entropy = det_score_mode == 'entropy' \
                                or self.with_primary_cascade \
                                or self.with_secondary_cascade
        self.final_matching = final_matching
        self.expand_boxes = expand_boxes
        self.init_percent = init_percent
        self.final_percent = final_percent
        self.init_ellipse_filter = init_ellipse_filter
        self.second_ellipse_filter = second_ellipse_filter
        self.final_ellipse_filter = final_ellipse_filter
        self.use_mahalanobis = use_mahalanobis
        self.return_expanded = return_expanded and expand_boxes
        self.return_remaining_expanded = return_remaining_expanded and expand_boxes
        
    def init_track(self, id, obj):
        """Initialize a track."""
        super(OCSORTTracker, self).init_track(id, obj)
        if self.tracks[id].frame_ids[-1] == 0:
            self.tracks[id].tentative = False
        else:
            self.tracks[id].tentative = True
        bbox, bbox_cov = self.get_track_bbox_and_cov(id)
        self.tracks[id].mean, self.tracks[id].covariance = self.kf.initiate(
            bbox, bbox_cov)
        
        # track.obs maintains the history associated detections to this track
        self.tracks[id].obs = []
        bbox_id = self.memo_items.index('bboxes')
        self.tracks[id].obs.append(obj[bbox_id])
        # a placefolder to save mean/covariance before losing tracking it
        # parameters to save: mean, covariance, measurement
        self.tracks[id].tracked = True
        self.tracks[id].saved_attr = Dict()
        self.tracks[id].velocity = torch.tensor(
            (-1, -1)).to(obj[bbox_id].device)  # placeholder
        
    def update_track(self, id, obj):
        """Update a track."""
        super(OCSORTTracker, self).update_track(id, obj)
        if self.tracks[id].tentative:
            if len(self.tracks[id]['bboxes']) >= self.num_tentatives:
                self.tracks[id].tentative = False
        
        bbox, bbox_cov = self.get_track_bbox_and_cov(id)
        self.tracks[id].mean, self.tracks[id].covariance = self.kf.update(
            self.tracks[id].mean, self.tracks[id].covariance, bbox, bbox_cov)
        
        self.tracks[id].tracked = True
        bbox_id = self.memo_items.index('bboxes')
        self.tracks[id].obs.append(obj[bbox_id])

        bbox1 = self.k_step_observation(self.tracks[id])
        bbox2 = obj[bbox_id]
        self.tracks[id].velocity = self.vel_direction(bbox1, bbox2).to(
            obj[bbox_id].device)
        
    def ocm_assign_ids(self,
                       ids,
                       det_bboxes,
                       det_bbox_covs,
                       weight_iou_with_det_scores=False,
                       match_iou_thr=0.5,
                       mahalanobis=False,
                       cascade=False,
                       num_bins=None,
                       score_mode='confidence',
                       iou_mode='iou'):
        """Apply Observation-Centric Momentum (OCM) to assign ids.

        OCM adds movement direction consistency into the association cost
        matrix. This term requires no additional assumption but from the
        same linear motion assumption as the canonical Kalman Filter in SORT.

        Args:
            ids (list[int]): Tracking ids.
            det_bboxes (Tensor): of shape (N, 5)
            det_bbox_covs (Tensor): of shape (N, 4, 4)
            weight_iou_with_det_scores (bool, optional): Whether using
                detection scores to weight IOU which is used for matching.
                Defaults to False.
            match_iou_thr (float, optional): Matching threshold.
                Defaults to 0.5.
            cascade (bool, optional): Whether to use matching cascade.
                Defaults to False.
            num_bins (int, optional): Number of bins for matching cascade.

        Returns:
            tuple(int): The assigning ids.
                - row (ndarray): `row[i]` specifies the column to which row `i` is assigned.
                - col (ndarray): `col[i]` specifies the row to which column `i` is assigned.

        OC-SORT uses velocity consistency besides IoU for association
        """
        #? Gather all track bboxes
        track_bboxes = np.zeros((0, 4))
        track_bbox_covs = np.zeros((0, 4, 4))
        for id in ids:
            track_mean, track_covariance = self.kf.project(self.tracks[id].mean,
                                            self.tracks[id].covariance,
                                            det_bbox_covs.cpu().numpy() if self.with_covariance else None)
            track_bboxes = np.concatenate(
                (track_bboxes, track_mean[None]), axis=0)
            if self.with_covariance:
                track_bbox_covs = np.concatenate(
                    (track_bbox_covs, track_covariance), axis=0)
        track_bboxes = torch.from_numpy(track_bboxes).to(det_bboxes)
        track_bboxes_xyxy = bbox_cxcyah_to_xyxy(track_bboxes)

        #? Compute distance
        if mahalanobis:
            track_bbox_covs = torch.from_numpy(track_bbox_covs).to(det_bbox_covs)
            track_bboxes = torch.repeat_interleave(track_bboxes, det_bboxes.shape[0], dim=0)
            det_bboxes_cxcyah = bbox_xyxy_to_cxcyah(det_bboxes[:, :4])
            det_bbox_covs_cxcyah = bbox_cov_xyxy_to_cxcyah(det_bbox_covs)
            det_bboxes_rep = det_bboxes_cxcyah[:, :4].repeat(len(ids), 1)
            det_bbox_covs_rep = det_bbox_covs_cxcyah.repeat(len(ids), 1, 1)
            maha, threshold = self.mahalanobis(det_bboxes_rep[:, :4], det_bbox_covs_rep, 
                                    track_bboxes, track_bbox_covs, one2one=True)
            dists = maha.reshape(len(ids), det_bboxes.shape[0])
            
            #? Scale Mahalanobis distance with covariance scores
            det_entropy = self.get_covariance_entropy(det_bbox_covs)
            # if det_bboxes.numel() > 0:
            #     det_scores = det_entropy / det_entropy.max()
            #     dists *= det_scores[None]
            dists = dists.cpu().numpy()
        else:
            ious = bbox_overlaps(track_bboxes_xyxy, det_bboxes[:, :4], mode=iou_mode)
        
            if self.compute_entropy:
                det_entropy = self.get_covariance_entropy(det_bbox_covs)
            
            #? Scale IOU with detection scores
            if weight_iou_with_det_scores and det_bboxes.numel() > 0:
                if score_mode == 'entropy':
                    det_scores = det_entropy / det_entropy.min()
                    ious /= det_scores[None]
                else:
                    ious *= det_bboxes[:, 4][None]
            
            dists = (1 - ious).cpu().numpy()
            threshold = 1 - match_iou_thr

        if len(ids) > 0 and len(det_bboxes) > 0:
            track_velocities = torch.stack(
                [self.tracks[id].velocity for id in ids]).to(det_bboxes.device)
            k_step_observations = torch.stack([
                self.k_step_observation(self.tracks[id]) for id in ids
            ]).to(det_bboxes.device)
            # valid1: if the track has previous observations to estimate speed
            # valid2: if the associated observation k steps ago is a detection
            valid1 = track_velocities.sum(dim=1) != -2
            valid2 = k_step_observations.sum(dim=1) != -4
            valid = valid1 & valid2

            vel_to_match = self.vel_direction_batch(k_step_observations[:, :4],
                                                    det_bboxes[:, :4])
            track_velocities = track_velocities[:, None, :].repeat(
                1, det_bboxes.shape[0], 1)

            angle_cos = (vel_to_match * track_velocities).sum(dim=-1)
            angle_cos = torch.clamp(angle_cos, min=-1, max=1)
            angle = torch.acos(angle_cos)  # [0, pi]
            norm_angle = (angle - np.pi / 2.) / np.pi  # [-0.5, 0.5]
            valid_matrix = valid[:, None].int().repeat(1, det_bboxes.shape[0])
            # set non-valid entries 0
            valid_norm_angle = norm_angle * valid_matrix

            dists += valid_norm_angle.cpu().numpy() * self.vel_consist_weight

        #? Solve the linear assignment problem
        if dists.size > 0:
            if cascade:
                row, col = self.matching_by_bin(dists, det_entropy.cpu().numpy(), threshold, num_bins)
            else:
                row, col = lap.lapjv(
                    dists, extend_cost=True, cost_limit=threshold, return_cost=False)
        else:
            row = np.zeros(len(ids)).astype(np.int32) - 1
            col = np.zeros(len(det_bboxes)).astype(np.int32) - 1
        return row, col
    
    def ocr_assign_ids(self,
                       track_obs,
                       track_obs_covs,
                       det_bboxes,
                       det_bbox_covs,
                       weight_iou_with_det_scores=False,
                       match_iou_thr=0.5,
                       mahalanobis=False,
                       cascade=False,
                       num_bins=None,
                       score_mode='confidence',
                       iou_mode='iou'):
        """association for Observation-Centric Recovery.

        As try to recover tracks from being lost whose estimated velocity is
        out- to-date, we use IoU-only matching strategy.

        Args:
            track_obs (Tensor): the list of historical associated
                detections of tracks
            track_obs_covs (Tensor): the list of historical associated
                detection covariance matrices of tracks
            det_bboxes (Tensor): of shape (N, 5), unmatched detections
            det_bbox_covs (Tensor): of shape (N, 4, 4)
            weight_iou_with_det_scores (bool, optional): Whether using
                detection scores to weight IOU which is used for matching.
                Defaults to False.
            match_iou_thr (float, optional): Matching threshold.
                Defaults to 0.5.
            cascade (bool, optional): Whether to use matching cascade.
                Defaults to False.
            num_bins (int, optional): Number of bins for matching cascade.

        Returns:
            tuple(int): The assigning ids.
                - row (ndarray): `row[i]` specifies the column to which row `i` is assigned.
                - col (ndarray): `col[i]` specifies the row to which column `i` is assigned.
        """
        det_entropy = self.get_covariance_entropy(det_bbox_covs)
        
        #? Compute distance
        if mahalanobis:
            maha, threshold = self.mahalanobis(det_bboxes[:, :4], det_bbox_covs, 
                                track_obs[:, :4], track_obs_covs)
            dists = maha.T.cpu().numpy()
        else:
            ious = bbox_overlaps(track_obs[:, :4], det_bboxes[:, :4], mode=iou_mode)
            
            #? Scale IOU with detection scores
            if weight_iou_with_det_scores and det_bboxes.numel() > 0:
                if self.det_score_mode == 'entropy':
                    det_scores = det_entropy / det_entropy.min()
                    ious /= det_scores[None]
                else:
                    ious *= det_bboxes[:, 4][None]

            dists = (1 - ious).cpu().numpy()
            threshold = 1 - match_iou_thr
        
        #? Solve the linear assignment problem
        if dists.size > 0:
            if cascade:
                row, col = self.matching_by_bin(dists, det_entropy.cpu().numpy(), threshold, num_bins)
            else:
                row, col = lap.lapjv(
                    dists, extend_cost=True, cost_limit=threshold, return_cost=False)
        else:
            row = np.zeros(len(track_obs)).astype(np.int32) - 1
            col = np.zeros(len(det_bboxes)).astype(np.int32) - 1
        return row, col
    
    @force_fp32(apply_to=('img', 'bboxes', 'bbox_covs'))
    def track(self,
              img,
              img_metas,
              model,
              bboxes,
              bbox_covs,
              labels,
              frame_id,
              rescale=False,
              **kwargs):
        """Tracking forward function.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            model (nn.Module): MOT model.
            bboxes (Tensor): of shape (N, 5).
            bbox_covs (Tensor): bbox covariance matrices of shape (N, 4, 4).
            labels (Tensor): of shape (N, ).
            frame_id (int): The id of current frame, 0-index.
            rescale (bool, optional): If True, the bounding boxes should be
                rescaled to fit the original scale of the image. Defaults to
                False.

        Returns:
            tuple: Tracking results (bboxes, labels, ids)
        """
        assert model.with_motion, "motion model is required."
        if not hasattr(self, 'kf'):
            self.kf = model.motion
        
        if self.empty or bboxes.size(0) == 0:
            valid_inds = bboxes[:, -1] > self.init_track_thr
            bboxes = bboxes[valid_inds]
            labels = labels[valid_inds]
            if bbox_covs is not None:
                bbox_covs = bbox_covs[valid_inds]
            
            if self.init_ellipse_filter:
                valid_inds = self.check_width_height(img_metas, 
                                                    bboxes, 
                                                    bbox_covs, 
                                                    percent=self.init_percent)
                bboxes = bboxes[valid_inds]
                labels = labels[valid_inds]
                if bbox_covs is not None:
                    bbox_covs = bbox_covs[valid_inds]
            
            num_new_tracks = bboxes.size(0)
            ids = torch.arange(self.num_tracks,
                               self.num_tracks + num_new_tracks).to(labels)
            self.num_tracks += num_new_tracks
        else:
            #? 0. init
            ids = torch.full((bboxes.size(0), ),
                             -1,
                             dtype=labels.dtype,
                             device=labels.device)
            
            #? Get the detection bboxes for the first association
            det_inds = bboxes[:, -1] > self.obj_score_thr
            det_bboxes = bboxes[det_inds]
            det_bbox_covs = bbox_covs[det_inds]
            det_labels = labels[det_inds]
            det_ids = ids[det_inds]
            
            if self.init_ellipse_filter:
                valid_inds = self.check_width_height(img_metas, 
                                                     det_bboxes, 
                                                     det_bbox_covs, 
                                                     percent=self.init_percent)
                det_bboxes = det_bboxes[valid_inds]
                det_bbox_covs = det_bbox_covs[valid_inds]
                det_labels = det_labels[valid_inds]
                det_ids = det_ids[valid_inds]
            
            #? 0. use Kalman Filter to predict current location
            for id in self.confirmed_ids:
                # track is lost in previous frame
                if self.tracks[id].frame_ids[-1] != frame_id - 1:
                    self.tracks[id].mean[7] = 0
                if self.tracks[id].tracked:
                    self.tracks[id].saved_attr.mean = self.tracks[id].mean
                    self.tracks[id].saved_attr.covariance = self.tracks[id].covariance
                self.tracks[id].mean, self.tracks[id].covariance = self.kf.predict(
                        self.tracks[id].mean, self.tracks[id].covariance)
            
            #? 1. first match
            first_ids_to_match = self.confirmed_ids
            _, first_match_det_inds = self.ocm_assign_ids(
                first_ids_to_match, det_bboxes, det_bbox_covs, 
                self.weight_iou_with_det_scores, self.match_iou_thr)
            # '-1' mean a detection box is not matched with tracklets in previous frame
            valid = first_match_det_inds > -1   # matched this round
            det_ids[valid] = torch.tensor(first_ids_to_match)[first_match_det_inds[valid]].to(labels)
            
            first_match_det_bboxes = det_bboxes[valid]
            first_match_det_bbox_covs = det_bbox_covs[valid]
            first_match_det_labels = det_labels[valid]
            first_match_det_ids = det_ids[valid]
            assert (first_match_det_ids > -1).all()

            first_unmatch_det_bboxes = det_bboxes[~valid]
            first_unmatch_det_bbox_covs = det_bbox_covs[~valid]
            first_unmatch_det_labels = det_labels[~valid]
            first_unmatch_det_ids = det_ids[~valid]
            assert (first_unmatch_det_ids == -1).all()
            
            #? 2. use unmatched detection bboxes from the first match to match the unconfirmed tracks
            tentative_ids_to_match = self.unconfirmed_ids
            _, tentative_match_det_inds = self.ocm_assign_ids(
                tentative_ids_to_match, first_unmatch_det_bboxes, first_unmatch_det_bbox_covs,
                self.weight_iou_with_det_scores, self.match_iou_thr)
            valid = tentative_match_det_inds > -1
            first_unmatch_det_ids[valid] = torch.tensor(tentative_ids_to_match)[tentative_match_det_inds[valid]].to(labels)
            
            match_det_bboxes = torch.cat(
                (first_match_det_bboxes, first_unmatch_det_bboxes[valid]), dim=0)
            match_det_bbox_covs = torch.cat(
                (first_match_det_bbox_covs, first_unmatch_det_bbox_covs[valid]), dim=0)
            match_det_labels = torch.cat(
                (first_match_det_labels, first_unmatch_det_labels[valid]), dim=0)
            match_det_ids = torch.cat(
                (first_match_det_ids, first_unmatch_det_ids[valid]), dim=0)
            assert (match_det_ids > -1).all()
            
            #* Remaining unmatched detections after step 1-2
            unmatch_det_bboxes = first_unmatch_det_bboxes[~valid]
            unmatch_det_bbox_covs = first_unmatch_det_bbox_covs[~valid]
            unmatch_det_labels = first_unmatch_det_labels[~valid]
            unmatch_det_ids = first_unmatch_det_ids[~valid]
            assert (unmatch_det_ids == -1).all()
            
            all_track_ids = [id for id, _ in self.tracks.items()]
            unmatched_track_ids = torch.tensor(
                [ind for ind in all_track_ids if ind not in match_det_ids])
            
            #? 3. use OCR to associate remaining unmatched tracks and detections
            if len(unmatched_track_ids) > 0:
                last_observations = []
                last_observation_covs = []
                for id in unmatched_track_ids:
                    last_box = self.tracks[id.item()].bboxes[-1].squeeze(0)[:4]
                    last_box_cov = self.tracks[id.item()].bbox_covs[-1].squeeze(0)
                    last_observations.append(last_box)
                    last_observation_covs.append(last_box_cov)
                last_observations = torch.stack(last_observations)
                last_observation_covs = torch.stack(last_observation_covs)

                remain_det_ids = torch.full((unmatch_det_bboxes.size(0), ),
                                            -1,
                                            dtype=labels.dtype,
                                            device=labels.device)
                ocr_match_track_inds, ocr_match_det_inds  = self.ocr_assign_ids(
                    last_observations, last_observation_covs, unmatch_det_bboxes, unmatch_det_bbox_covs,
                    self.weight_iou_with_det_scores, self.match_iou_thr)
                valid = ocr_match_det_inds > -1
                remain_det_ids[valid] = unmatched_track_ids.clone()[
                    ocr_match_det_inds[valid]].to(labels)
                
                ocr_match_det_bboxes = unmatch_det_bboxes[valid]
                ocr_match_det_bbox_covs = unmatch_det_bbox_covs[valid]
                ocr_match_det_labels = unmatch_det_labels[valid]
                ocr_match_det_ids = remain_det_ids[valid]
                assert (ocr_match_det_ids > -1).all()

                ocr_unmatch_det_bboxes = unmatch_det_bboxes[~valid]
                ocr_unmatch_det_bbox_covs = unmatch_det_bbox_covs[~valid]
                ocr_unmatch_det_labels = unmatch_det_labels[~valid]
                ocr_unmatch_det_ids = remain_det_ids[~valid]
                assert (ocr_unmatch_det_ids == -1).all()
                
                match_det_bboxes = torch.cat(
                    (match_det_bboxes, ocr_match_det_bboxes), dim=0)
                match_det_bbox_covs = torch.cat(
                    (match_det_bbox_covs, ocr_match_det_bbox_covs), dim=0)
                match_det_labels = torch.cat(
                    (match_det_labels, ocr_match_det_labels), dim=0)
                match_det_ids = torch.cat(
                    (match_det_ids, ocr_match_det_ids), dim=0)
                
                unmatch_det_bboxes = ocr_unmatch_det_bboxes
                unmatch_det_bbox_covs = ocr_unmatch_det_bbox_covs
                unmatch_det_labels = ocr_unmatch_det_labels
                unmatch_det_ids = ocr_unmatch_det_ids
                
                unmatched_track_inds = ocr_match_track_inds == -1
                unmatched_track_ids = unmatched_track_ids[unmatched_track_inds]
                unmatched_track_bboxes = last_observations[unmatched_track_inds]
                unmatched_track_bbox_covs = last_observation_covs[unmatched_track_inds]
                
            if self.final_ellipse_filter:
                valid_inds = self.check_width_height(img_metas, 
                                                    unmatch_det_bboxes, 
                                                    unmatch_det_bbox_covs, 
                                                    percent=self.final_percent)
                unmatch_det_bboxes = unmatch_det_bboxes[valid_inds]
                unmatch_det_bbox_covs = unmatch_det_bbox_covs[valid_inds]
                unmatch_det_labels = unmatch_det_labels[valid_inds]
                unmatch_det_ids = unmatch_det_ids[valid_inds]
            
            #? Final matching
            if self.final_matching and len(unmatched_track_ids) > 0:
                if len(unmatch_det_ids) > 0:
                    if self.expand_boxes:
                        unmatch_det_bboxes_exp, unmatch_det_bbox_covs_exp = \
                            self.get_outer_bboxes(unmatch_det_bboxes,
                                                    unmatch_det_bbox_covs)
                        unmatched_track_bboxes, _ = self.get_outer_bboxes(unmatched_track_bboxes,
                                                                            unmatched_track_bbox_covs)
                    else:
                        unmatch_det_bboxes_exp = unmatch_det_bboxes
                        unmatch_det_bbox_covs_exp = unmatch_det_bbox_covs
                    
                    remain_det_ids = torch.full((unmatch_det_bboxes.size(0), ),
                                            -1,
                                            dtype=labels.dtype,
                                            device=labels.device)

                    _, final_match_det_inds = self.ocr_assign_ids(
                        unmatched_track_bboxes, unmatched_track_bbox_covs, 
                        unmatch_det_bboxes_exp, unmatch_det_bbox_covs_exp,
                        self.weight_iou_with_det_scores, self.match_iou_thr,
                        self.use_mahalanobis, self.with_secondary_cascade,
                        self.secondary_cascade.get('num_bins', None), iou_mode='giou')
                    
                    valid = final_match_det_inds > -1
                    remain_det_ids[valid] = unmatched_track_ids.clone()[
                        final_match_det_inds[valid]].to(labels)
                    
                    final_match_det_bboxes = unmatch_det_bboxes[valid]
                    final_match_det_bbox_covs = unmatch_det_bbox_covs[valid]
                    final_match_det_labels = unmatch_det_labels[valid]
                    final_match_det_ids = remain_det_ids[valid]
                    assert (final_match_det_ids > -1).all()
                    
                    unmatch_det_bboxes = unmatch_det_bboxes[~valid]
                    unmatch_det_bbox_covs = unmatch_det_bbox_covs[~valid]
                    unmatch_det_labels = unmatch_det_labels[~valid]
                    unmatch_det_ids = remain_det_ids[~valid]
                    assert (unmatch_det_ids == -1).all()
                    
                    match_det_bboxes = torch.cat(
                        (match_det_bboxes, final_match_det_bboxes), dim=0)
                    match_det_bbox_covs = torch.cat(
                        (match_det_bbox_covs, final_match_det_bbox_covs), dim=0)
                    match_det_labels = torch.cat(
                        (match_det_labels, final_match_det_labels), dim=0)
                    match_det_ids = torch.cat(
                        (match_det_ids, final_match_det_ids), dim=0)
            else:
                if self.return_remaining_expanded:
                    unmatch_det_bboxes, unmatch_det_bbox_covs = \
                        self.get_outer_bboxes(unmatch_det_bboxes,
                                                unmatch_det_bbox_covs)
            
            #? 4. maintain tracks for online smoothing
            for i in range(len(match_det_ids)):
                det_bbox = match_det_bboxes[i]
                track_id = match_det_ids[i].item()
                if not self.tracks[track_id].tracked:
                    #* the track is lost before this step
                    self.online_smooth(self.tracks[track_id], det_bbox)

            for track_id in all_track_ids:
                if track_id not in match_det_ids:
                    self.tracks[track_id].tracked = False
                    self.tracks[track_id].obs.append(None)
            
            #? 5. gather all tracking results
            bboxes = torch.cat(
                (match_det_bboxes, unmatch_det_bboxes), dim=0)
            bbox_covs = torch.cat(
                (match_det_bbox_covs, unmatch_det_bbox_covs), dim=0)
            labels = torch.cat(
                (match_det_labels, unmatch_det_labels), dim=0)
            ids = torch.cat(
                (match_det_ids, unmatch_det_ids), dim=0)
            
            #? 6. assign new ids
            new_track_inds = ids == -1
            ids[new_track_inds] = torch.arange(
                self.num_tracks,
                self.num_tracks + new_track_inds.sum()).to(labels)
            self.num_tracks += new_track_inds.sum()
        
        self.update(
            ids=ids,
            bboxes=bboxes,
            bbox_covs=bbox_covs,
            labels=labels,
            frame_ids=frame_id
        )
        return bboxes, bbox_covs, labels, ids