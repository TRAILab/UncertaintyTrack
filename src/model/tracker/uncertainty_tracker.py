

import lap
import numpy as np
import torch
from addict import Dict
from mmcv.runner import force_fp32
from mmdet.core import bbox_overlaps

from mmtrack.models import TRACKERS
from mmtrack.models.trackers import SortTracker, OCSORTTracker
from mmtrack.core.bbox import bbox_cxcyah_to_xyxy, bbox_xyxy_to_cxcyah

from core.utils import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh, bbox_cov_xyxy_to_cxcyah
from .prob_tracker import ProbabilisticTracker

@TRACKERS.register_module()
class UncertaintyTracker(OCSORTTracker, ProbabilisticTracker):
    """Uncertainty tracker for multi-object tracking that leverages
    spatial uncertainty from the detector. Extends OCSORTTracker for 
    track-measurement association and uses the same track management.
    """
    def __init__(self, 
                 with_covariance=True,
                 use_giou=False,
                 det_score_mode='entropy',
                 primary_fn=None,
                 primary_cascade=None,
                 secondary_fn=None,
                 secondary_cascade=False,
                 expand_boxes=False,
                 percent=0.25,
                 ellipse_filter=False,
                 filter_output=False,
                 combine_mahalanobis=False,
                 **kwargs):
        ProbabilisticTracker.__init__(self,
                                      with_covariance=with_covariance,
                                      primary_fn=primary_fn,
                                      primary_cascade=primary_cascade,
                                      secondary_fn=secondary_fn,
                                      secondary_cascade=secondary_cascade)
        OCSORTTracker.__init__(self, **kwargs)
        self.iou_mode = 'giou' if use_giou else 'iou'
        self.det_score_mode = det_score_mode
        self.compute_entropy = det_score_mode == 'entropy' \
                                or self.with_primary_cascade \
                                or self.with_secondary_cascade
        self.expand_boxes = expand_boxes
        self.percent = percent
        self.ellipse_filter = ellipse_filter
        self.combine_mahalanobis = combine_mahalanobis
        self.filter_output = filter_output

    def init_track(self, id, obj):
        """Initialize a track."""
        super(SortTracker, self).init_track(id, obj)
        if self.tracks[id].frame_ids[-1] == 0:
            #* Track is initialized on the first frame
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
        super(SortTracker, self).update_track(id, obj)
        
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
    
    def state_association(self,
                       ids,
                       det_bboxes,
                       det_bbox_covs,
                       det_labels,
                       weight_iou_with_det_scores=False,
                       match_iou_thr=0.5,
                       mode='iou'):
        """Extends `ocm_assign_ids` in OCSORTTracker to use covariance
        matrices for matching. Also supports multi-class classification.

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
        for id in ids:
            track_bboxes = np.concatenate(
                (track_bboxes, self.tracks[id].mean[:4][None]), axis=0)
        track_bboxes = torch.from_numpy(track_bboxes).to(det_bboxes)
        track_bboxes = bbox_cxcyah_to_xyxy(track_bboxes)

        #? Compute distance
        ious = bbox_overlaps(track_bboxes, det_bboxes[:, :4], mode=mode)
        
        if self.compute_entropy:
            det_entropy = self.get_covariance_entropy(det_bbox_covs)
        
        #? Scale IOU with detection scores
        if weight_iou_with_det_scores and det_bboxes.numel() > 0:
            if self.det_score_mode == 'entropy':
                det_scores = det_entropy / det_entropy.min()
                ious /= det_scores[None]
            else:
                ious *= det_bboxes[:, 4][None]
        
        #* support multi-class association
        track_labels = torch.tensor([
            self.tracks[id]['labels'][-1] for id in ids
        ]).to(det_labels)
        
        cat_match = det_labels[None, :] == track_labels[:, None]
        # to avoid det and track of different categories are matched
        cat_mismatch_cost = (1 - cat_match.int()) * 1e6
        
        dists = (1 - ious + cat_mismatch_cost).cpu().numpy()

        #TODO: Add eigenvector direction consistency cost
        #TODO: need to check threshold
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
            threshold = 1 - match_iou_thr
            row, col = lap.lapjv(
                dists, extend_cost=True, cost_limit=threshold, return_cost=False)
        else:
            row = np.zeros(len(ids)).astype(np.int32) - 1
            col = np.zeros(len(det_bboxes)).astype(np.int32) - 1
        return row, col, track_bboxes
    
    def detection_matching(self,
                           ids,
                           track_bboxes,
                           track_bbox_covs,
                           det_bboxes,
                           det_bbox_covs,
                           det_labels,
                           weight_iou_with_det_scores=False,
                           match_iou_thr=0.5,
                           cascade=False,
                           num_bins=None,
                           mode='iou'):
        """Extends `ocr_assign_ids` in OCSORTTracker to...

        Args:
            ids (list[int]): Tracking ids.
            track_bboxes (Tensor): last detected bboxes of tracks, 
                of shape (N, 4)
            track_bbox_covs (Tensor): bbox covariance matrices of shape (N, 4, 4)
            det_bboxes (Tensor): of shape (N, 5), unmatched detections
            det_bbox_covs (Tensor): of shape (N, 4, 4)
            det_labels (Tensor): of shape (N, )
            weight_iou_with_det_scores (bool, optional): Whether using
                detection scores to weight IOU which is used for matching.
                Defaults to False.
            match_iou_thr (float, optional): Matching threshold.
                Defaults to 0.5.
            cascade (bool, optional): Whether to use matching cascade.
                Defaults to False.
            num_bins (int, optional): Number of bins for matching cascade.
                Defaults to None.
        """
        #TODO: Fix when bbox_covs is None
        #? Compute distance
        if self.custom_secondary:
            track_bbox_covs_diag = torch.diagonal(track_bbox_covs, dim1=-2, dim2=-1)
            det_bbox_covs_diag = torch.diagonal(det_bbox_covs, dim1=-2, dim2=-1)
            custom_dists = self.secondary_distance(track_bboxes[:, :4], track_bbox_covs_diag, 
                                                        det_bboxes[:, :4], det_bbox_covs_diag)
            
            # dists = self.secondary_distance(det_bboxes[:, :4], det_bbox_covs,
            #                                 track_bboxes[:, :4], track_bbox_covs)
            
            # breakpoint()
            # dists = dists.T #* because opposite direction
            # dists[dists1 > self.secondary_threshold] = np.inf
            custom_dists = custom_dists.cpu().numpy()
            # threshold = self.secondary_threshold
            # cascade = False
        else:
            if self.compute_entropy:
                det_entropy = self.get_covariance_entropy(det_bbox_covs)
            
            #? Compute distance
            ious = bbox_overlaps(track_bboxes[:, :4], det_bboxes[:, :4], mode=mode)
            
            #? Scale IOU with detection scores
            if weight_iou_with_det_scores and det_bboxes.numel() > 0:
                if self.det_score_mode == 'entropy':
                    det_scores = det_entropy / det_entropy.min()
                    ious /= det_scores[None]
                    ious = torch.clamp(ious, max=1)
                    pass
                else:
                    ious *= det_bboxes[:, 4][None]
                    
            #* support multi-class association
            #TODO: move this out of else clause
            track_labels = torch.tensor([
                self.tracks[id]['labels'][-1] for id in ids
            ]).to(det_labels)
            
            cat_match = det_labels[None, :] == track_labels[:, None]
            # to avoid det and track of different categories are matched
            cat_mismatch_cost = (1 - cat_match.int()) * 1e6
            
            dists = (1 - ious + cat_mismatch_cost).cpu().numpy()
            threshold = 1 - match_iou_thr
        
        #? Solve the linear assignment problem
        if dists.size > 0:
            if cascade:
                row, col = self.matching_by_bin(dists, det_entropy.cpu().numpy(), threshold, num_bins)
            else:
                row, col = lap.lapjv(
                    dists, extend_cost=True, cost_limit=threshold, return_cost=False)
        else:
            row = np.zeros(len(track_bboxes)).astype(np.int32) - 1
            col = np.zeros(len(det_bboxes)).astype(np.int32) - 1
        return row, col, track_bboxes
    
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
            valid_inds = bboxes[:, -1] > self.init_track_thr    #* initial score threshold
            bboxes = bboxes[valid_inds]
            bbox_covs = bbox_covs[valid_inds]
            labels = labels[valid_inds]
            num_new_tracks = bboxes.size(0)
            ids = torch.arange(self.num_tracks,
                               self.num_tracks + num_new_tracks).to(labels)
            self.num_tracks += num_new_tracks
        else:
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
            
            #? 0. use Kalman Filter to predict current location
            for id in self.confirmed_ids:
                #* track is lost in previous frame
                if self.tracks[id].frame_ids[-1] != frame_id - 1:
                    self.tracks[id].mean[7] = 0
                if self.tracks[id].tracked:
                    self.tracks[id].saved_attr.mean = self.tracks[id].mean
                    self.tracks[id].saved_attr.covariance = self.tracks[id].covariance
                self.tracks[id].mean, self.tracks[id].covariance = self.kf.predict(
                        self.tracks[id].mean, self.tracks[id].covariance)
            
            #? 1. first: predicted confirmed track states and detections
            first_ids_to_match = self.confirmed_ids
            first_match_track_inds, first_match_det_inds, first_track_bboxes = self.state_association(
                first_ids_to_match, det_bboxes, det_bbox_covs, det_labels,
                self.weight_iou_with_det_scores, self.match_iou_thr)
            # '-1' mean a detection box is not matched with tracklets in previous frame
            matched = first_match_det_inds > -1   #* matched this round
            det_ids[matched] = torch.tensor(first_ids_to_match)[first_match_det_inds[matched]].to(labels)
            
            first_match_det_bboxes = det_bboxes[matched]
            first_match_det_bbox_covs = det_bbox_covs[matched]
            first_match_det_labels = det_labels[matched]
            first_match_det_ids = det_ids[matched]
            assert (first_match_det_ids > -1).all()

            first_unmatch_det_bboxes = det_bboxes[~matched]
            first_unmatch_det_bbox_covs = det_bbox_covs[~matched]
            first_unmatch_det_labels = det_labels[~matched]
            first_unmatch_det_ids = det_ids[~matched]
            assert (first_unmatch_det_ids == -1).all()
            
            # if len(first_match_det_ids) > 0:
            #     first_match_track_bboxes = first_track_bboxes[first_match_det_inds[matched]]
            #     first_match_det_bboxes, first_match_det_bbox_covs = \
            #         self.get_better_boxes(first_match_det_bboxes, first_match_det_bbox_covs,
            #                                 first_match_track_bboxes[:, :4])
            
            #? 2. second: predicted unconfirmed (tentative) track states
            #? and unmatched detections
            second_ids_to_match = self.unconfirmed_ids
            second_match_track_inds, second_match_det_inds, second_track_bboxes = self.state_association(
                second_ids_to_match, first_unmatch_det_bboxes, 
                first_unmatch_det_bbox_covs, first_unmatch_det_labels,
                self.weight_iou_with_det_scores, self.match_iou_thr)
            matched = second_match_det_inds > -1
            first_unmatch_det_ids[matched] = torch.tensor(second_ids_to_match)[second_match_det_inds[matched]].to(labels)
            
            second_match_det_bboxes = first_unmatch_det_bboxes[matched]
            second_match_det_bbox_covs = first_unmatch_det_bbox_covs[matched]
            second_match_det_labels = first_unmatch_det_labels[matched]
            second_match_det_ids = first_unmatch_det_ids[matched]
            assert (second_match_det_ids > -1).all()
            
            # if len(second_match_det_ids) > 0:
            #     second_match_track_bboxes = second_track_bboxes[second_match_det_inds[matched]]
            #     second_match_det_bboxes, second_match_det_bbox_covs = \
            #         self.get_better_boxes(second_match_det_bboxes, second_match_det_bbox_covs,
            #                                 second_match_track_bboxes[:, :4])
            
            match_det_bboxes = torch.cat(
                (first_match_det_bboxes, second_match_det_bboxes), dim=0)
            match_det_bbox_covs = torch.cat(
                (first_match_det_bbox_covs, second_match_det_bbox_covs), dim=0)
            match_det_labels = torch.cat(
                (first_match_det_labels, second_match_det_labels), dim=0)
            match_det_ids = torch.cat(
                (first_match_det_ids, second_match_det_ids), dim=0)
            assert (match_det_ids > -1).all()
            
            #* Remaining unmatched detections after step 1-2
            unmatch_det_bboxes = first_unmatch_det_bboxes[~matched]
            unmatch_det_bbox_covs = first_unmatch_det_bbox_covs[~matched]
            unmatch_det_labels = first_unmatch_det_labels[~matched]
            unmatch_det_ids = first_unmatch_det_ids[~matched]
            assert (unmatch_det_ids == -1).all()
            
            all_track_ids = [id for id, _ in self.tracks.items()]
            unmatched_track_ids = torch.tensor(
                [id for id in all_track_ids 
                 if id not in match_det_ids and 
                 (id in self.confirmed_ids or self.tracks[id].tracked)])
            
            #? 3. third: remaining unmatched tracks and detections
            #* tracks: confirmed tracks + unlost tentative tracks
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
                
                third_match_track_inds, third_match_det_inds, third_track_bboxes  = self.detection_matching(
                    unmatched_track_ids.tolist(), last_observations, last_observation_covs,
                    unmatch_det_bboxes, unmatch_det_bbox_covs, unmatch_det_labels, 
                    self.weight_iou_with_det_scores, self.match_iou_thr,
                    self.with_secondary_cascade, self.secondary_cascade.get('num_bins', None))
                
                matched = third_match_det_inds > -1
                remain_det_ids[matched] = unmatched_track_ids.clone()[
                    third_match_det_inds[matched]].to(labels)
                
                third_match_det_bboxes = unmatch_det_bboxes[matched]
                third_match_det_bbox_covs = unmatch_det_bbox_covs[matched]
                third_match_det_labels = unmatch_det_labels[matched]
                third_match_det_ids = remain_det_ids[matched]
                assert (third_match_det_ids > -1).all()
                
                # if len(third_match_det_ids) > 0:
                #     third_match_track_bboxes = third_track_bboxes[third_match_det_inds[matched]]
                #     third_match_det_bboxes, third_match_det_bbox_covs = \
                #         self.get_better_boxes(third_match_det_bboxes, third_match_det_bbox_covs,
                #                                 third_match_track_bboxes[:, :4])

                third_unmatch_det_bboxes = unmatch_det_bboxes[~matched]
                third_unmatch_det_bbox_covs = unmatch_det_bbox_covs[~matched]
                third_unmatch_det_labels = unmatch_det_labels[~matched]
                third_unmatch_det_ids = remain_det_ids[~matched]
                assert (third_unmatch_det_ids == -1).all()
                
                unmatch_det_bboxes = third_unmatch_det_bboxes
                unmatch_det_bbox_covs = third_unmatch_det_bbox_covs
                unmatch_det_labels = third_unmatch_det_labels
                unmatch_det_ids = third_unmatch_det_ids
                
                match_det_bboxes = torch.cat(
                    (match_det_bboxes, third_match_det_bboxes), dim=0)
                match_det_bbox_covs = torch.cat(
                    (match_det_bbox_covs, third_match_det_bbox_covs), dim=0)
                match_det_labels = torch.cat(
                    (match_det_labels, third_match_det_labels), dim=0)
                match_det_ids = torch.cat(
                    (match_det_ids, third_match_det_ids), dim=0)
                
                #* Remaining unmatched tracks after step 1-3 (new)
                #* ----------------------------------------------
                unmatched_track_inds = third_match_track_inds == -1
                unmatched_track_ids = unmatched_track_ids[unmatched_track_inds]
                unmatched_track_bboxes = last_observations[unmatched_track_inds]
                unmatched_track_bbox_covs = last_observation_covs[unmatched_track_inds]
            
            #? 6.0. Filter out detections with large ellipse radius
            if self.ellipse_filter:
                valid_inds = self.check_width_height(img_metas, 
                                                     unmatch_det_bboxes, 
                                                     unmatch_det_bbox_covs, 
                                                     percent=self.percent)
                unmatch_det_bboxes = unmatch_det_bboxes[valid_inds]
                unmatch_det_bbox_covs = unmatch_det_bbox_covs[valid_inds]
                unmatch_det_labels = unmatch_det_labels[valid_inds]
                unmatch_det_ids = unmatch_det_ids[valid_inds]
            
            if self.expand_boxes and unmatch_det_bboxes.numel() > 0:
                unmatch_det_bboxes, unmatch_det_bbox_covs = self.get_outer_bboxes(unmatch_det_bboxes,
                                                                                    unmatch_det_bbox_covs)
                #? 4. fourth: match tracks with expanded det boxes using outer bounds of ellipses
                if len(unmatched_track_ids) > 0:
                    unmatched_track_bboxes, _ = self.get_outer_bboxes(unmatched_track_bboxes,
                                                                unmatched_track_bbox_covs)
                    remain_det_ids = torch.full((unmatch_det_bboxes.size(0), ),
                                            -1,
                                            dtype=labels.dtype,
                                            device=labels.device)
                
                    _, fourth_match_det_inds, _ = self.detection_matching(
                        unmatched_track_ids.tolist(), unmatched_track_bboxes, None,
                        unmatch_det_bboxes, None, unmatch_det_labels, 
                        self.weight_iou_with_det_scores, self.match_iou_thr,
                        None, None, mode='giou')
                    
                    matched = fourth_match_det_inds > -1
                    remain_det_ids[matched] = unmatched_track_ids.clone()[
                        fourth_match_det_inds[matched]].to(labels)
                    
                    fourth_match_det_bboxes = unmatch_det_bboxes[matched]
                    fourth_match_det_bbox_covs = unmatch_det_bbox_covs[matched]
                    fourth_match_det_labels = unmatch_det_labels[matched]
                    fourth_match_det_ids = remain_det_ids[matched]
                    assert (fourth_match_det_ids > -1).all()

                    fourth_unmatch_det_bboxes = unmatch_det_bboxes[~matched]
                    fourth_unmatch_det_bbox_covs = unmatch_det_bbox_covs[~matched]
                    fourth_unmatch_det_labels = unmatch_det_labels[~matched]
                    fourth_unmatch_det_ids = remain_det_ids[~matched]
                    assert (fourth_unmatch_det_ids == -1).all()
                    
                    unmatch_det_bboxes = fourth_unmatch_det_bboxes
                    unmatch_det_bbox_covs = fourth_unmatch_det_bbox_covs
                    unmatch_det_labels = fourth_unmatch_det_labels
                    unmatch_det_ids = fourth_unmatch_det_ids
                    
                    match_det_bboxes = torch.cat(
                        (match_det_bboxes, fourth_match_det_bboxes), dim=0)
                    match_det_bbox_covs = torch.cat(
                        (match_det_bbox_covs, fourth_match_det_bbox_covs), dim=0)
                    match_det_labels = torch.cat(
                        (match_det_labels, fourth_match_det_labels), dim=0)
                    match_det_ids = torch.cat(
                        (match_det_ids, fourth_match_det_ids), dim=0)
            #* ----------------------------------------------
            #? 4. maintain tracks for online smoothing
            for i in range(len(match_det_ids)):
                det_bbox = match_det_bboxes[i]
                det_bbox_cov = match_det_bbox_covs[i]
                track_id = match_det_ids[i].item()
                if not self.tracks[track_id].tracked:
                    #* the track is lost before this step
                    self.online_smooth(self.tracks[track_id], det_bbox)
            
            #? 5. Flag the tracks that are lost in this frame
            for track_id in all_track_ids:
                if track_id not in match_det_ids:
                    self.tracks[track_id].tracked = False
                    self.tracks[track_id].obs.append(None)
            
            # #? 6.0. Filter out detections with large ellipse radius
            # if self.ellipse_filter:
            #     valid_inds = self.check_width_height(img_metas, 
            #                                          unmatch_det_bboxes, 
            #                                          unmatch_det_bbox_covs, 
            #                                          percent=self.percent)
            #     unmatch_det_bboxes = unmatch_det_bboxes[valid_inds]
            #     unmatch_det_bbox_covs = unmatch_det_bbox_covs[valid_inds]
            #     unmatch_det_labels = unmatch_det_labels[valid_inds]
            #     unmatch_det_ids = unmatch_det_ids[valid_inds]
            
            #? 6.1. gather all tracking results
            bboxes = torch.cat(
                (match_det_bboxes, unmatch_det_bboxes), dim=0)
            bbox_covs = torch.cat(
                (match_det_bbox_covs, unmatch_det_bbox_covs), dim=0)
            labels = torch.cat(
                (match_det_labels, unmatch_det_labels), dim=0)
            ids = torch.cat(
                (match_det_ids, unmatch_det_ids), dim=0)
            
            #? 7. assign new ids
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