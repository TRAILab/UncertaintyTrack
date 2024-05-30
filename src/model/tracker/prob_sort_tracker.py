"""Code adapted from mmtrack.models.trackers.sort_tracker.py"""
import numpy as np
import torch
from mmcv.runner import force_fp32
from mmdet.core import bbox_overlaps
from motmetrics.lap import linear_sum_assignment

from mmtrack.core import imrenormalize
from mmtrack.core.bbox import bbox_xyxy_to_cxcyah
from mmtrack.models import TRACKERS
from mmtrack.models.trackers import SortTracker

from core.utils import bbox_cov_xyxy_to_cxcyah
from .prob_tracker import ProbabilisticTracker
from core.utils.analysis_utils import get_active_inactive   #! For analysis; remove later

@TRACKERS.register_module()
class ProbabilisticSortTracker(SortTracker, ProbabilisticTracker):
    """Tracker for UncertainMOT.
    Extends SortTracker for DeepSORT.
    """
    def __init__(self, 
                 primary_cascade=None,
                 secondary_fn=None,
                 secondary_cascade=False,
                 det_entropy_range=None,
                 **kwargs):
        ProbabilisticTracker.__init__(self,
                                      primary_cascade=primary_cascade,
                                      secondary_fn=secondary_fn,
                                      secondary_cascade=secondary_cascade,
                                      det_entropy_range=det_entropy_range)
        SortTracker.__init__(self, **kwargs)

    def init_track(self, id, obj):
        """Initialize a track."""
        super(SortTracker, self).init_track(id, obj)
        self.tracks[id].tentative = True

        bbox = bbox_xyxy_to_cxcyah(self.tracks[id].bboxes[-1])
        assert bbox.ndim == 2 and bbox.shape[0] == 1
        bbox = bbox.squeeze(0).cpu().numpy()

        bbox_cov = self.tracks[id].bbox_covs[-1]
        if isinstance(bbox_cov, torch.Tensor):
            assert bbox_cov.ndim == 3 and bbox_cov.shape[0] == 1
            #? convert bbox_cov to cxcyah cov
            bbox_cov = bbox_cov_xyxy_to_cxcyah(bbox_cov)
            bbox_cov = bbox_cov.squeeze(0).cpu().numpy()
            bbox_cov = bbox_cov \
                        if self.analysis_cfg.get('with_covariance', True) \
                        else None
        else:
            bbox_cov = None

        self.tracks[id].mean, self.tracks[id].covariance = self.kf.initiate(
            bbox, bbox_cov)

        if self.analysis_cfg.get('name', '') == 'init entropy':
            _, default_covariance = self.kf.initiate(bbox, None)
            self.detailed_results.append(
                {
                    'default': default_covariance,
                    'predicted': self.tracks[id].covariance
                }
            )

    def update_track(self, id, obj):
        """Update a track."""
        super(SortTracker, self).update_track(id, obj)
        if self.tracks[id].tentative:
            if len(self.tracks[id]['bboxes']) >= self.num_tentatives:
                self.tracks[id].tentative = False

        bbox = bbox_xyxy_to_cxcyah(self.tracks[id].bboxes[-1])
        assert bbox.ndim == 2 and bbox.shape[0] == 1
        bbox = bbox.squeeze(0).cpu().numpy()

        bbox_cov = self.tracks[id].bbox_covs[-1]
        if isinstance(bbox_cov, torch.Tensor):
            assert bbox_cov.ndim == 3 and bbox_cov.shape[0] == 1
            #? convert bbox_cov to cxcyah cov
            bbox_cov = bbox_cov_xyxy_to_cxcyah(bbox_cov)
            bbox_cov = bbox_cov.squeeze(0).cpu().numpy()
            bbox_cov = bbox_cov \
                        if self.analysis_cfg.get('with_covariance', True) \
                        else None
        else:
            bbox_cov = None

        if self.analysis_cfg.get('name', '') == 'before after entropy':
            detailed_result = {'before': self.tracks[id].covariance}
        self.tracks[id].mean, self.tracks[id].covariance = self.kf.update(
            self.tracks[id].mean, self.tracks[id].covariance, bbox, bbox_cov)
        if self.analysis_cfg.get('name', '') == 'before after entropy':
            detailed_result.update({'after': self.tracks[id].covariance})
            self.detailed_results.append(detailed_result)

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
            tuple: Tracking results.
        """
        self.analysis_cfg = kwargs.get('analysis_cfg', {})
        self.detailed_results = []
        assert model.with_motion, "motion model is required."
        if not hasattr(self, 'kf'):
            self.kf = model.motion

        if self.with_reid:
            if self.reid.get('img_norm_cfg', False):
                reid_img = imrenormalize(img, img_metas[0]['img_norm_cfg'],
                                         self.reid['img_norm_cfg'])
            else:
                reid_img = img.clone()

        valid_inds = bboxes[:, -1] > self.obj_score_thr
        bboxes = bboxes[valid_inds]
        labels = labels[valid_inds]
        if bbox_covs is not None:
            bbox_covs = bbox_covs[valid_inds]

        #? No existing tracks or no detections; just create and delete tracks
        if self.empty or bboxes.size(0) == 0:
            num_new_tracks = bboxes.size(0)
            ids = torch.arange(
                self.num_tracks,
                self.num_tracks + num_new_tracks,
                dtype=torch.long)
            self.num_tracks += num_new_tracks
            if self.with_reid:
                embeds = model.reid.simple_test(
                    self.crop_imgs(reid_img, img_metas, bboxes[:, :4].clone(),
                                   rescale))
        else:
            ids = torch.full((bboxes.size(0), ), -1, dtype=torch.long)
            
            #* Associate confirmed tracks first
            active_ids = self.confirmed_ids
            
            #? Separate active and inactive tracks for analysis
            if self.analysis_cfg.get('name', '') == 'average entropy':
                if len(active_ids) > 0:
                    active_inactive_dict = get_active_inactive(self.tracks, active_ids, frame_id)
                    self.detailed_results.append(active_inactive_dict)

            # motion
            bboxes_cxcyah = bbox_xyxy_to_cxcyah(bboxes)
            bbox_covs_cxcyah = bbox_cov_xyxy_to_cxcyah(bbox_covs) \
                                if bbox_covs is not None else None
            bbox_covs_cxcyah = bbox_covs_cxcyah \
                                if self.analysis_cfg.get('with_covariance', True) \
                                else None
            
            self.tracks, costs = model.motion.track(
                self.tracks, 
                bboxes_cxcyah, 
                bbox_covs_cxcyah)

            #? Update track and/or detection covariance scores
            if self.with_primary_cascade or self.with_reid:
                covariance_scores = self.get_covariance_scores(bbox_covs_cxcyah, step="primary")
                self.update_track_scores(frame_id, active_ids)

            valid_inds = [list(self.ids).index(_) for _ in active_ids]
            
            if self.with_reid:
                embeds = model.reid.simple_test(
                        self.crop_imgs(reid_img, img_metas, bboxes[:, :4].clone(),
                            rescale))
            if len(active_ids) > 0:
                #* Motion + appearance
                if self.with_reid:
                    track_embeds = self.get(
                        'embeds',
                        active_ids,
                        self.reid.get('num_samples', None),
                        behavior='mean')
                    reid_dists = torch.cdist(track_embeds,embeds).cpu().numpy()

                    #? Filter reid dists with infeasible motion associations
                    reid_dists[~np.isfinite(costs[valid_inds, :])] = np.nan
                    
                    #? Filter reid dists with infeasible appearance associations
                    match_score_thr = self.reid.get('match_score_thr', None)
                    assert match_score_thr is not None, \
                        "reid.match_score_thr not set"
                    assert match_score_thr >= 0, \
                        "reid.match_score_thr must be non-negative"
                    reid_dists[reid_dists > match_score_thr] = np.nan
                    
                costs = costs[valid_inds, :]
                #? Filter motion cost with infeasible appearance associations
                if self.with_reid:
                    costs[~np.isfinite(reid_dists)] = np.nan
                
                #? Assignment with Hungarian algorithm
                confirmed_tracks = {k: v for k, v in self.tracks.items() 
                                        if k in active_ids}
                if self.with_reid:
                    # #TODO: remove this later
                    # row, col = adaptive_motion_reid_assignment(costs,
                    #                                            reid_dists,
                    #                                            confirmed_tracks,
                    #                                            covariance_scores,
                    #                                            self.det_entropy_range)
                    pass
                elif self.with_primary_cascade:
                    row, col = self.matching_by_detection(costs,
                                                            covariance_scores,
                                                            step="primary")
                else:
                    row, col = linear_sum_assignment(costs)
                
                for r, c in zip(row, col):
                    ids[c] = active_ids[r]

            #* Associate remaining confirmed tracks with tentative
            active_ids = [
                id for id in self.ids if id not in ids
            ]
            active_dets = torch.nonzero(ids == -1).squeeze(1)
            
            #? Secondary matching
            if len(active_ids) > 0 and len(active_dets) > 0:
                track_bboxes = self.get('bboxes', active_ids).cpu()
                det_bboxes = bboxes[active_dets][:, :-1].cpu()
                track_bbox_covs = self.get('bbox_covs', active_ids).cpu()
                det_bbox_covs = bbox_covs[active_dets, :, :].cpu()
                
                if self.with_secondary_cascade:
                    covariance_scores = self.get_covariance_scores(det_bbox_covs, step="secondary")

                if self.custom_secondary:
                    dists = self.custom_distance(track_bboxes, track_bbox_covs,
                                                    det_bboxes, det_bbox_covs)
                else:
                    #* GIoU matching
                    ious = bbox_overlaps(
                        track_bboxes, det_bboxes, mode='giou').cpu().numpy()
                    dists = 1 - ious
                    threshold = 1 - self.match_iou_thr
                    dists[dists > threshold] = np.nan
                
                if self.with_secondary_cascade:
                    row, col = self.matching_by_detection(dists,
                                                            covariance_scores,
                                                            step="secondary")
                else:
                    row, col = linear_sum_assignment(dists)
                
                for r, c in zip(row, col):
                    ids[active_dets[c]] = active_ids[r]

            new_track_inds = ids == -1
            ids[new_track_inds] = torch.arange(
                self.num_tracks,
                self.num_tracks + new_track_inds.sum(),
                dtype=torch.long)
            self.num_tracks += new_track_inds.sum()

        self.update(
            ids=ids,
            bboxes=bboxes[:, :4],
            bbox_covs=bbox_covs,
            scores=bboxes[:, -1],
            labels=labels,
            embeds=embeds if self.with_reid else None,
            frame_ids=frame_id)
        
        #! For analysis; remove later
        results = {
            'regular': (bboxes, labels, ids)
        }
        if self.analysis_cfg:
            results['analysis'] = self.detailed_results
        
        return results
