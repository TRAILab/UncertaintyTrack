"""Code adapted from mmtrack.models.trackers.byte_tracker.py"""

import lap
import numpy as np
import torch
from mmcv.runner import force_fp32
from mmdet.core import bbox_overlaps

from mmtrack.core.bbox import bbox_cxcyah_to_xyxy, bbox_xyxy_to_cxcyah
from mmtrack.models import TRACKERS
from mmtrack.models.trackers import ByteTracker

from core.utils import bbox_cov_xyxy_to_cxcyah
from .prob_tracker import ProbabilisticTracker


@TRACKERS.register_module()
class ProbabilisticByteTracker(ByteTracker, ProbabilisticTracker):
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
        ByteTracker.__init__(self, **kwargs)
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
        super(ByteTracker, self).init_track(id, obj)
        if self.tracks[id].frame_ids[-1] == 0:
            self.tracks[id].tentative = False
        else:
            self.tracks[id].tentative = True
        bbox, bbox_cov = self.get_track_bbox_and_cov(id)
        self.tracks[id].mean, self.tracks[id].covariance = self.kf.initiate(
            bbox, bbox_cov)
        self.tracks[id].tracked = True
        
    def update_track(self, id, obj):
        """Update a track."""
        super(ByteTracker, self).update_track(id, obj)
        if self.tracks[id].tentative:
            if len(self.tracks[id]['bboxes']) >= self.num_tentatives:
                self.tracks[id].tentative = False
        track_label = self.tracks[id]['labels'][-1]
        label_idx = self.memo_items.index('labels')
        obj_label = obj[label_idx]
        assert obj_label == track_label
        
        bbox, bbox_cov = self.get_track_bbox_and_cov(id)

        self.tracks[id].mean, self.tracks[id].covariance = self.kf.update(
            self.tracks[id].mean, self.tracks[id].covariance, bbox, bbox_cov)
        self.tracks[id].tracked = True
        
    def assign_ids(self, 
                   ids,
                   det_bboxes,
                   det_bbox_covs,
                   det_labels,
                   weight_iou_with_det_scores=False,
                   match_iou_thr=0.5,
                   mahalanobis=False,
                   cascade=False,
                   num_bins=None,
                   score_mode='confidence',
                   iou_mode='iou'):
        """Assign ids.

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
            tuple(ndarray): Returns matched indices of tracks and detections.
                - row (ndarray): `row[i]` specifies the column to which row `i` is assigned.
                - col (ndarray): `col[i]` specifies the row to which column `i` is assigned.
        """
        #? Gather all track bboxes
        track_bboxes = np.zeros((0, 4))
        track_bbox_covs = np.zeros((0, 4, 4))
        for id in ids:
            track_mean, track_covariance = self.kf.project(self.tracks[id].mean,
                                            self.tracks[id].covariance,
                                            det_bbox_covs.cpu().numpy() if self.with_covariance else None)
            track_bboxes = np.concatenate(
                (track_bboxes, track_mean[:4][None]), axis=0)
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
            
        # support multi-class association
        track_labels = torch.tensor([
            self.tracks[id]['labels'][-1] for id in ids
        ]).to(det_bboxes.device)
        
        cat_match = det_labels[None, :] == track_labels[:, None]
        # to avoid det and track of different categories are matched
        cat_mismatch_cost = (1 - cat_match.int()) * 1e6
        
        dists += cat_mismatch_cost.cpu().numpy()
        
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
    
    def assign_ids_obs(self, 
                        ids,
                        track_bboxes,
                        track_bbox_covs,
                        det_bboxes,
                        det_bbox_covs,
                        det_labels,
                        weight_iou_with_det_scores=False,
                        match_iou_thr=0.3,
                        mahalanobis=False,
                        cascade=False,
                        num_bins=None,
                        score_mode='confidence',
                        iou_mode='iou'):
        """Assign ids.

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
            tuple(ndarray): Returns matched indices of tracks and detections.
                - row (ndarray): `row[i]` specifies the column to which row `i` is assigned.
                - col (ndarray): `col[i]` specifies the row to which column `i` is assigned.
        """
        det_entropy = self.get_covariance_entropy(det_bbox_covs)
        
        #? Compute distance
        if mahalanobis:
            maha, threshold = self.mahalanobis(det_bboxes[:, :4], det_bbox_covs, 
                                track_bboxes[:, :4], track_bbox_covs)
            dists = maha.T.cpu().numpy()
        else:
            ious = bbox_overlaps(track_bboxes[:, :4], det_bboxes[:, :4], mode=iou_mode)

            #? Scale IOU with detection scores
            if weight_iou_with_det_scores and det_bboxes.numel() > 0:
                if score_mode == 'entropy':
                    det_scores = det_entropy / det_entropy.min()
                    ious /= det_scores[None]
                else:
                    ious *= det_bboxes[:, 4][None]
            
            dists = (1 - ious).cpu().numpy()
            threshold = 1 - match_iou_thr
            
        # support multi-class association
        track_labels = torch.tensor([
            self.tracks[id]['labels'][-1] for id in ids
        ]).to(det_bboxes.device)
        
        cat_match = det_labels[None, :] == track_labels[:, None]
        # to avoid det and track of different categories are matched
        cat_mismatch_cost = (1 - cat_match.int()) * 1e6
        
        dists += cat_mismatch_cost.cpu().numpy()
        
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
            #? 0.0: Init ids
            ids = torch.full((bboxes.size(0), ),
                             -1,
                             dtype=labels.dtype,
                             device=labels.device)
            
            #? 0.1: Get high score boxes
            first_det_inds = bboxes[:, -1] > self.obj_score_thrs['high']
            first_det_bboxes = bboxes[first_det_inds]
            first_det_bbox_covs = bbox_covs[first_det_inds]
            first_det_labels = labels[first_det_inds]
            first_det_ids = ids[first_det_inds]

            #? 0.2: Get low score boxes
            second_det_inds = (~first_det_inds) & (
                bboxes[:, -1] > self.obj_score_thrs['low'])
            second_det_bboxes = bboxes[second_det_inds]
            second_det_bbox_covs = bbox_covs[second_det_inds]
            second_det_labels = labels[second_det_inds]
            second_det_ids = ids[second_det_inds]
            
            if self.second_ellipse_filter:
                valid_inds = self.check_width_height(img_metas, 
                                                     second_det_bboxes, 
                                                     second_det_bbox_covs, 
                                                     percent=self.init_percent)
                second_det_bboxes = second_det_bboxes[valid_inds]
                second_det_bbox_covs = second_det_bbox_covs[valid_inds]
                second_det_labels = second_det_labels[valid_inds]
                second_det_ids = second_det_ids[valid_inds]
            
            #? 0.3: Use Kalman Filter to predict current location
            for id in self.confirmed_ids:
                #* track is lost in previous frame
                if self.tracks[id].frame_ids[-1] != frame_id - 1:
                    self.tracks[id].mean[7] = 0
                self.tracks[id].mean, self.tracks[id].covariance = self.kf.predict(
                        self.tracks[id].mean, self.tracks[id].covariance)
            
            #? 1. first: match high score boxes with confirmed tracks
            first_ids_to_match = self.confirmed_ids
            first_match_track_inds, first_match_det_inds = self.assign_ids(
                first_ids_to_match, first_det_bboxes, first_det_bbox_covs, first_det_labels,
                self.weight_iou_with_det_scores, self.match_iou_thrs['high'], False,
                self.with_primary_cascade, self.primary_cascade.get('num_bins', None))
            # '-1' mean a detection box is not matched with tracklets in previous frame
            valid = first_match_det_inds > -1   # matched this round
            first_det_ids[valid] = torch.tensor(first_ids_to_match)[first_match_det_inds[valid]].to(labels)
            
            first_match_det_bboxes = first_det_bboxes[valid]
            first_match_det_bbox_covs = first_det_bbox_covs[valid]
            first_match_det_labels = first_det_labels[valid]
            first_match_det_ids = first_det_ids[valid]
            assert (first_match_det_ids > -1).all()

            first_unmatch_det_bboxes = first_det_bboxes[~valid]
            first_unmatch_det_bbox_covs = first_det_bbox_covs[~valid]
            first_unmatch_det_labels = first_det_labels[~valid]
            first_unmatch_det_ids = first_det_ids[~valid]
            assert (first_unmatch_det_ids == -1).all()
            
            #? 2. second: match remaining high score boxes with tentative tracks
            tentative_ids_to_match = self.unconfirmed_ids
            _, tentative_match_det_inds = self.assign_ids(
                tentative_ids_to_match, first_unmatch_det_bboxes, first_unmatch_det_bbox_covs,
                first_unmatch_det_labels, self.weight_iou_with_det_scores, 
                self.match_iou_thrs['tentative'])
            
            valid = tentative_match_det_inds > -1
            first_unmatch_det_ids[valid] = torch.tensor(tentative_ids_to_match)[tentative_match_det_inds[valid]].to(labels)
            
            first_match_det_bboxes = torch.cat((first_match_det_bboxes, first_unmatch_det_bboxes[valid]), dim=0)
            first_match_det_bbox_covs = torch.cat((first_match_det_bbox_covs, first_unmatch_det_bbox_covs[valid]), dim=0)
            first_match_det_labels = torch.cat((first_match_det_labels, first_unmatch_det_labels[valid]), dim=0)
            first_match_det_ids = torch.cat((first_match_det_ids, first_unmatch_det_ids[valid]), dim=0)
            assert (first_match_det_ids > -1).all()
            
            first_unmatch_det_bboxes = first_unmatch_det_bboxes[~valid]
            first_unmatch_det_bbox_covs = first_unmatch_det_bbox_covs[~valid]
            first_unmatch_det_labels = first_unmatch_det_labels[~valid]
            first_unmatch_det_ids = first_unmatch_det_ids[~valid]
            assert (first_unmatch_det_ids == -1).all()
            
            #? 3.0: Get unmatched confirmed tracks that aren't lost
            # second match for unmatched tracks from the first match
            second_ids_to_match = []
            for i, id in enumerate(first_ids_to_match):
                case_1 = first_match_track_inds[i] == -1    #* not matched in first step
                case_2 = self.tracks[id].frame_ids[-1] == frame_id - 1  #* not lost in previous frame
                if case_1 and case_2:
                    second_ids_to_match.append(id)
            
            #? 3.1 third: match low score boxes with tracks in 3.0
            # weight_iou_with_det = self.weight_iou_with_det_scores and self.det_score_mode == 'entropy'
            _, second_match_det_inds = self.assign_ids(
                second_ids_to_match, second_det_bboxes, second_det_bbox_covs, 
                second_det_labels, False, self.match_iou_thrs['low'], False,
                None, None)

            valid = second_match_det_inds > -1
            second_det_ids[valid] = torch.tensor(second_ids_to_match)[second_match_det_inds[valid]].to(ids)
            
            #? 4. Gather all matched detections
            #* from second match, only matched detections are kept
            valid = second_det_ids > -1
            second_match_det_bboxes = second_det_bboxes[valid]
            second_match_det_bbox_covs = second_det_bbox_covs[valid]
            second_match_det_labels = second_det_labels[valid]
            second_match_det_ids = second_det_ids[valid]
            assert (second_match_det_ids > -1).all()
            
            #? 4.1: remaining high score boxes are filtered using ellipses
            if self.final_ellipse_filter:
                valid_inds = self.check_width_height(img_metas, 
                                                    first_unmatch_det_bboxes, 
                                                    first_unmatch_det_bbox_covs, 
                                                    percent=self.final_percent)
                first_unmatch_det_bboxes = first_unmatch_det_bboxes[valid_inds]
                first_unmatch_det_bbox_covs = first_unmatch_det_bbox_covs[valid_inds]
                first_unmatch_det_labels = first_unmatch_det_labels[valid_inds]
                first_unmatch_det_ids = first_unmatch_det_ids[valid_inds]
            
            bboxes = torch.cat(
                            (first_match_det_bboxes, second_match_det_bboxes), 
                            dim=0)
            bbox_covs = torch.cat(
                            (first_match_det_bbox_covs, second_match_det_bbox_covs), 
                            dim=0)
            labels = torch.cat(
                        (first_match_det_labels, second_match_det_labels), 
                        dim=0)
            ids = torch.cat((first_match_det_ids, second_match_det_ids),
                    dim=0)
            
            unmatched_track_ids = [
                id for id, _ in self.tracks.items()
                if id not in ids and 
                (id in self.confirmed_ids or self.tracks[id].frame_ids[-1] == frame_id - 1)
            ]
            
            #? Final matching
            if self.final_matching and len(unmatched_track_ids) > 0:
                #* Assume ellipse filter was applied before
                second_unmatch_det_bboxes = second_det_bboxes[~valid]
                second_unmatch_det_bbox_covs = second_det_bbox_covs[~valid]
                second_unmatch_det_labels = second_det_labels[~valid]
                second_unmatch_det_ids = second_det_ids[~valid]
                assert (second_unmatch_det_ids == -1).all()
                
                #? Filter using ellipses before matching
                valid_inds = self.check_width_height(img_metas, 
                                                    second_unmatch_det_bboxes, 
                                                    second_unmatch_det_bbox_covs, 
                                                    percent=self.final_percent)
                second_unmatch_det_bboxes = second_unmatch_det_bboxes[valid_inds]
                second_unmatch_det_bbox_covs = second_unmatch_det_bbox_covs[valid_inds]
                second_unmatch_det_labels = second_unmatch_det_labels[valid_inds]
                second_unmatch_det_ids = second_unmatch_det_ids[valid_inds]
                
                unmatch_det_bboxes = torch.cat((first_unmatch_det_bboxes, second_unmatch_det_bboxes), dim=0)
                unmatch_det_bbox_covs = torch.cat((first_unmatch_det_bbox_covs, second_unmatch_det_bbox_covs), dim=0)
                unmatch_det_labels = torch.cat((first_unmatch_det_labels, second_unmatch_det_labels), dim=0)
                unmatch_det_ids = torch.cat((first_unmatch_det_ids, second_unmatch_det_ids), dim=0)
                
                if len(unmatch_det_ids) > 0:
                    #? Gather all track bboxes
                    unmatch_track_bboxes = []
                    unmatch_track_bbox_covs = []
                    for id in unmatched_track_ids:
                        last_box = self.tracks[id].bboxes[-1].squeeze(0)[:4]
                        last_box_cov = self.tracks[id].bbox_covs[-1].squeeze(0)
                        unmatch_track_bboxes.append(last_box)
                        unmatch_track_bbox_covs.append(last_box_cov)
                    unmatch_track_bboxes = torch.stack(unmatch_track_bboxes)
                    unmatch_track_bbox_covs = torch.stack(unmatch_track_bbox_covs)
                    
                    if self.expand_boxes:
                        unmatch_det_bboxes_exp, unmatch_det_bbox_covs_exp = self.get_outer_bboxes(unmatch_det_bboxes,
                                                                                    unmatch_det_bbox_covs)
                        unmatch_track_bboxes, _ = self.get_outer_bboxes(unmatch_track_bboxes,
                                                            unmatch_track_bbox_covs)
                    else:
                        unmatch_det_bboxes_exp = unmatch_det_bboxes
                        unmatch_det_bbox_covs_exp = unmatch_det_bbox_covs
                    
                    _, final_match_det_inds = self.assign_ids_obs(
                        unmatched_track_ids, unmatch_track_bboxes, unmatch_track_bbox_covs,
                        unmatch_det_bboxes_exp, unmatch_det_bbox_covs_exp, unmatch_det_labels,
                        False, 0.3, self.use_mahalanobis, self.with_secondary_cascade,
                        self.secondary_cascade.get('num_bins', None), iou_mode='giou')
                    
                    valid = final_match_det_inds > -1
                    unmatch_det_ids[valid] = torch.tensor(unmatched_track_ids)[final_match_det_inds[valid]].to(ids)
                    
                    valid = unmatch_det_ids > -1
                    if self.return_expanded:
                        final_match_det_bboxes = unmatch_det_bboxes_exp[valid]
                        final_match_det_bbox_covs = unmatch_det_bbox_covs_exp[valid]
                    else:
                        final_match_det_bboxes = unmatch_det_bboxes[valid]
                        final_match_det_bbox_covs = unmatch_det_bbox_covs[valid]
                    final_match_det_labels = unmatch_det_labels[valid]
                    final_match_det_ids = unmatch_det_ids[valid]
                    assert (final_match_det_ids > -1).all()

                    if self.return_remaining_expanded:
                        first_unmatch_det_bboxes = unmatch_det_bboxes[:len(first_unmatch_det_ids)][~valid[:len(first_unmatch_det_ids)]]
                        first_unmatch_det_bbox_covs = unmatch_det_bbox_covs[:len(first_unmatch_det_ids)][~valid[:len(first_unmatch_det_ids)]]
                        first_unmatch_det_labels = unmatch_det_labels[:len(first_unmatch_det_ids)][~valid[:len(first_unmatch_det_ids)]]
                        first_unmatch_det_ids = unmatch_det_ids[:len(first_unmatch_det_ids)][~valid[:len(first_unmatch_det_ids)]]
                    else:
                        first_unmatch_det_bboxes = first_unmatch_det_bboxes[~valid[:len(first_unmatch_det_bboxes)]]
                        first_unmatch_det_bbox_covs = first_unmatch_det_bbox_covs[~valid[:len(first_unmatch_det_bbox_covs)]]
                        first_unmatch_det_labels = first_unmatch_det_labels[~valid[:len(first_unmatch_det_labels)]]
                        first_unmatch_det_ids = first_unmatch_det_ids[~valid[:len(first_unmatch_det_ids)]]

                    bboxes = torch.cat((bboxes, final_match_det_bboxes, first_unmatch_det_bboxes), dim=0)
                    bbox_covs = torch.cat((bbox_covs, final_match_det_bbox_covs, first_unmatch_det_bbox_covs), dim=0)
                    labels = torch.cat((labels, final_match_det_labels, first_unmatch_det_labels), dim=0)
                    ids = torch.cat((ids, final_match_det_ids, first_unmatch_det_ids), dim=0)
                
            else:
                if self.return_remaining_expanded and len(first_unmatch_det_ids) > 0:
                    first_unmatch_det_bboxes, first_unmatch_det_bbox_covs = \
                        self.get_outer_bboxes(first_unmatch_det_bboxes,
                                                first_unmatch_det_bbox_covs)
                bboxes = torch.cat((bboxes, first_unmatch_det_bboxes), dim=0)
                bbox_covs = torch.cat((bbox_covs, first_unmatch_det_bbox_covs), dim=0)
                labels = torch.cat((labels, first_unmatch_det_labels), dim=0)
                ids = torch.cat((ids, first_unmatch_det_ids), dim=0)

            #? 5. Assign new ids
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