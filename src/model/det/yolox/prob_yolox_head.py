import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import bias_init_with_prob
from mmcv.ops.nms import batched_nms
from mmcv.runner import force_fp32

from mmdet.core import multi_apply, reduce_mean, bbox_overlaps, select_single_mlvl
from mmdet.models.builder import HEADS
from mmdet.models.dense_heads import YOLOXHead

from core.utils import (
    covariance2cholesky, compute_mean_covariance_torch,
    clamp_log_variance, compute_mean_variance_torch,
    product_of_gaussians, covariance_intersection
)


@HEADS.register_module()
class ProbabilisticYOLOXHead(YOLOXHead):
    """Probabilistic YOLOXHead head that extends
    YOLOXHead in `YOLOX <https://arxiv.org/abs/2107.08430>`_ 
    to predict the mean and covariance of bounding box outputs.
    
    Args:
        post_process (str): Post-processing method. Default: "nms".
            Options are "nms", "bayesian", and "covariance intersection".
        compute_cls_var (bool): Whether to compute classification variance.
            Default: False.
        affinity_thr (float): Threshold for affinity in post-processing. 
            Default: 0.9.
        separate_levels (bool): Whether to handle detections in each scale level
            separately then fuse, or combine all detections then fuse.
            If True, the former is used. If False, the latter is used.
            Default: False.
        post_process_mlvl (str): Post-processing method for fusing outputs across
            scale levels. Default: "bayesian".
            If bayesian, bayesian inference is used to fuse outputs across scale levels.
            This assumes that the bounding box predictions across scale levels are
            independent. If covariance intersection, covariance intersection is used
            to fuse outputs across scale levels. This assumes that the bounding box
            predictions across scale levels are correlated, but the correlation is unknown.
    """
    
    def __init__(self,
                 post_process="nms",
                 compute_cls_var=False,
                 affinity_thr=0.9,
                 separate_levels=False,
                 post_process_mlvl="bayesian",
                 **kwargs):
        if post_process not in ['nms', 'bayesian', 'covariance_intersection']:
            raise ValueError(f"{post_process} is an invalid \
                                bounding box post-processing method.")
        self.post_process = post_process
        self.compute_cls_var = compute_cls_var
        self.affinity_thr = affinity_thr
        self.separate_levels = separate_levels and post_process != "nms"
        if post_process_mlvl not in ['nms', 'bayesian', 'covariance_intersection']:
            raise ValueError(f"{post_process_mlvl} is an invalid \
                                bounding box multi-scale post-processing method.")
        self.post_process_mlvl = post_process_mlvl
        
        super(ProbabilisticYOLOXHead, self).__init__(**kwargs)
    
    def _init_layers(self):
        self.multi_level_cls_convs = nn.ModuleList()
        self.multi_level_reg_convs = nn.ModuleList()
        self.multi_level_conv_cls = nn.ModuleList()
        #? Create module for classification variance estimation
        self.multi_level_conv_cls_var = nn.ModuleList()
        self.multi_level_conv_reg = nn.ModuleList()
        #? Create module for bounding box covariance estimation
        self.multi_level_conv_reg_cov = nn.ModuleList()
        self.multi_level_conv_obj = nn.ModuleList()
        for _ in self.strides:
            self.multi_level_cls_convs.append(self._build_stacked_convs())
            self.multi_level_reg_convs.append(self._build_stacked_convs())
            (conv_cls, conv_cls_var, 
             conv_reg, conv_reg_cov, 
             conv_obj) = self._build_predictor()
            self.multi_level_conv_cls.append(conv_cls)
            self.multi_level_conv_cls_var.append(conv_cls_var)
            self.multi_level_conv_reg.append(conv_reg)
            self.multi_level_conv_reg_cov.append(conv_reg_cov)
            self.multi_level_conv_obj.append(conv_obj)
    
    def _build_predictor(self):
        """Initialize predictor layers of a single level head."""
        conv_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 1)
        #* Same shape as classification output
        conv_cls_var = nn.Conv2d(self.feat_channels, self.cls_out_channels, 1) \
                        if self.compute_cls_var else None
        conv_reg = nn.Conv2d(self.feat_channels, 4, 1)
        #* Same shape as bounding box output (diagonal covariance)
        conv_reg_cov = nn.Conv2d(self.feat_channels, 4, 1)
        conv_obj = nn.Conv2d(self.feat_channels, 1, 1)
        return conv_cls, conv_cls_var, conv_reg, conv_reg_cov, conv_obj
    
    def init_weights(self):
        """Initialize weights of the head.
        Note: Same weight initialization for bbox covariance as bbox regression"""
        super().init_weights()
        if self.compute_cls_var:
            var_bias_init = bias_init_with_prob(0.0001)
            for conv_cls_var in self.multi_level_conv_cls_var:
                conv_cls_var.bias.data.fill_(var_bias_init)
    
    def forward_single(self, x, cls_convs, reg_convs, conv_cls, conv_cls_var, 
                       conv_reg, conv_reg_cov, conv_obj):
        """Forward feature of a single scale level."""
        cls_feat = cls_convs(x)
        reg_feat = reg_convs(x)
        
        cls_score = conv_cls(cls_feat)
        cls_score_var = conv_cls_var(cls_feat) if self.compute_cls_var else None
        bbox_pred = conv_reg(reg_feat)
        bbox_cov = conv_reg_cov(reg_feat)
        objectness = conv_obj(reg_feat)
        
        return cls_score, cls_score_var, bbox_pred, bbox_cov, objectness
    
    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            tuple[Tensor]: A tuple of multi-level predication map, each is a
                4D-tensor of shape (batch_size, 5+num_classes, height, width).
        """
        return multi_apply(self.forward_single, feats,
                           self.multi_level_cls_convs,
                           self.multi_level_reg_convs,
                           self.multi_level_conv_cls,
                           self.multi_level_conv_cls_var,
                           self.multi_level_conv_reg,
                           self.multi_level_conv_reg_cov,
                           self.multi_level_conv_obj)
    
    @force_fp32(apply_to=('cls_scores', 'cls_vars', 'bbox_preds', 'bbox_covs', 'objectnesses'))
    def get_bboxes(self,
                   cls_scores,
                   cls_vars,
                   bbox_preds,
                   bbox_covs,
                   objectnesses,
                   img_metas=None,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        """Transform network outputs of a batch into bbox results.
        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            cls_vars (list[Tensor] | list[None]): Box score variances for 
                all scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
                List of None if compute_cls_var is False.
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            bbox_covs (list[Tensor]): Box covariance for all scale levels,
                each is a 4D-tensor, has shape 
                (batch_size, num_priors * 4, H, W).
            objectnesses (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, 1, H, W).
            img_metas (list[dict], Optional): Image meta info. Default None.
            cfg (mmcv.Config, Optional): Test / postprocessing configuration,
                if None, test_cfg would be used.  Default None.
            rescale (bool): If True, return boxes in original image space.
                Default False.
            with_nms (bool): If True, do nms before return boxes.
                Default True.
        Returns:
            list[list[Tensor, Tensor, Tensor]]: Each item in result_list is 3-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is an
                (n, 4, 4) tensor representing the bounding box covariance matrices. 
                The third item is a (n,) tensor where each item is the predicted 
                class label of the corresponding box.
        """
        assert len(cls_scores) == len(bbox_preds) == len(objectnesses) \
            == len(cls_vars) == len(bbox_covs)
            
        cfg = self.test_cfg if cfg is None else cfg

        num_imgs = len(img_metas)
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device,
            with_stride=True)
        
        mlvl_cls_scores_list = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                  self.cls_out_channels)
            for cls_score in cls_scores
        ]
        mlvl_cls_vars_list = [
            cls_var.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                self.cls_out_channels)
            if cls_var is not None else None for cls_var in cls_vars 
        ]
        mlvl_bbox_preds_list = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        mlvl_bbox_covs_list = [
            bbox_cov.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_cov in bbox_covs
        ]
        mlvl_objectness_list = [
            objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
            for objectness in objectnesses
        ]
        
        result_list = []
        for img_id in range(len(img_metas)):
            mlvl_cls_scores = select_single_mlvl(mlvl_cls_scores_list, img_id)
            mlvl_cls_vars = select_single_mlvl(mlvl_cls_vars_list, img_id) \
                            if self.compute_cls_var else [None] * len(mlvl_cls_vars_list)
            mlvl_bboxes = select_single_mlvl(mlvl_bbox_preds_list, img_id)
            mlvl_bbox_covs = select_single_mlvl(mlvl_bbox_covs_list, img_id)
            mlvl_score_factors = select_single_mlvl(mlvl_objectness_list, img_id)
            scale_factors = np.array(img_metas[img_id]['scale_factor'])

            result_list.append(
                self._get_bboxes_single(mlvl_cls_scores, mlvl_cls_vars, mlvl_bboxes, 
                                   mlvl_bbox_covs, mlvl_score_factors, mlvl_priors, 
                                   scale_factors, cfg, rescale, with_nms))
        return result_list
    
    def _get_bboxes_single(self,
                      mlvl_cls_scores,
                      mlvl_cls_vars,
                      mlvl_bbox_preds,
                      mlvl_bbox_covs,
                      mlvl_score_factors,
                      mlvl_priors,
                      scale_factors,
                      cfg,
                      rescale=False,
                      with_nms=True):
        """Post-processing for probabilistic YOLOXHead.
        Transforms outputs of a single image into bounding box predictions with
        scores and bounding box covariance matrices.
        
        Args:
            mlvl_cls_scores (list[Tensor]): Box scores from all scale levels 
                of a single image, each has shape (num_priors * H * W, num_classes).
            mlvl_cls_vars (list[Tensor] | list[None]): Box score variances from all
                scale levels of a single image, each has shape 
                (num_priors * H * W, num_classes). List of None if compute_cls_var 
                is False.
            mlvl_bbox_preds (list[Tensor]): Box energies / deltas from all scale 
                levels of a single image, each has shape (num_priors * H * W, 4).
            mlvl_bbox_covs (list[Tensor]): Box covariance matrices from all scale
                levels of a single image, each has shape (num_priors * H * W, 4).
            mlvl_score_factors (list[Tensor]): Score factors from all scale levels
                of a single image, each has shape (num_priors * H * W, 1).
            mlvl_priors (Tensor): Prior boxes of all scale levels of a single image,
                has shape (num_priors * H * W, 4).
            scale_factors (ndarray): Scale factors of the image used to scale back
                the bounding boxes to the original size. Shape (4,).
            cfg (mmcv.Config): Test / postprocessing configuration.
            rescale (bool): If True, return boxes in original image space.
                Default False.
            with_nms (bool): If True, filter results before returning boxes.
        
        Returns:
            tuple[Tensor]: Results of detected bboxes, associated covariance
                matrices and labels.

                - det_bboxes (Tensor): Predicted bboxes with shape \
                    [num_bboxes, 5], where the first 4 columns are bounding \
                    box positions (tl_x, tl_y, br_x, br_y) and the 5-th \
                    column are class confidence scores between 0 and 1.
                - det_bbox_covs (Tensor): Predicted bbox covariance matrices \
                    with shape [num_bboxes, 4, 4].
                - det_labels (Tensor): Predicted labels of the corresponding \
                    box with shape [num_bboxes].
                - det_score_vars (Tensor | None): Predicted score variances of the \
                    corresponding box with shape [num_bboxes].
        """
        mlvl_scores_tensor = torch.empty(0).to(mlvl_cls_scores[0])
        mlvl_score_vars_tensor = torch.empty(0).to(mlvl_cls_vars[0]) \
                                    if self.compute_cls_var else None
        mlvl_bboxes_tensor = torch.empty((0, 4)).to(mlvl_bbox_preds[0])
        mlvl_bbox_covs_tensor = torch.empty((0, 4, 4)).to(mlvl_bbox_covs[0])
        mlvl_labels_tensor = torch.empty(0).to(mlvl_cls_scores[0]).long()
        
        #? Iterate through each scale level
        for i, (cls_logits, cls_vars, bboxes, bbox_covs, score_factors, priors) in \
            enumerate(zip(mlvl_cls_scores, mlvl_cls_vars, mlvl_bbox_preds, 
                           mlvl_bbox_covs, mlvl_score_factors, mlvl_priors)):
            #? Perform MC-Sampling to generate logits using classification variance
            if cls_vars is not None:
                cls_logit_dists = torch.distributions.Normal(
                                    cls_logits, scale=torch.sqrt(torch.exp(cls_vars)))
                cls_logits = cls_logit_dists.sample((self.loss_cls.num_samples,))
            
            cls_scores = cls_logits.sigmoid()
            score_factors = score_factors.sigmoid()
            
            #? Compute mean and variance of scores
            cls_scores, cls_score_vars = compute_mean_variance_torch(cls_scores)
            
            #? Filter results with score threshold
            scores, labels = torch.max(cls_scores, dim=1)
            valid_mask = score_factors * scores >= cfg.score_thr
            scores = scores[valid_mask] * score_factors[valid_mask]
            labels = labels[valid_mask]
            if self.compute_cls_var:
                score_vars = cls_score_vars[valid_mask] * torch.pow(score_factors[valid_mask], 2)
                score_vars = score_vars[torch.arange(score_vars.shape[0]), labels]
            else:
                score_vars = None
            bboxes = bboxes[valid_mask]
            bbox_covs = bbox_covs[valid_mask]
            priors = priors[valid_mask]
            
            bbox_covs = clamp_log_variance(bbox_covs)
            if bboxes.numel() > 0:
                assert bbox_covs.shape[0] == bboxes.shape[0]
                
                #? Construct cholesky factor matrix from covariance matrix vector
                bbox_chol = covariance2cholesky(bbox_covs)
                
                #? MC-Sampling to generate bounding box predictions
                bbox_dists = torch.distributions.MultivariateNormal(
                                bboxes, scale_tril=bbox_chol)
                bbox_samples = bbox_dists.sample((1000,))   # (1000, N, 4)
                bbox_samples = self._bbox_decode(priors, bbox_samples)
                bboxes, bbox_covs = compute_mean_covariance_torch(bbox_samples)
            else:
                bboxes = self._bbox_decode(priors, bboxes)
                bbox_covs = torch.empty((0, 4, 4), 
                                        device=bbox_covs.device)
            
            #? Rescale bboxes
            if rescale:
                bboxes, bbox_covs = \
                    self._rescale_preds(bboxes, bbox_covs, scale_factors)
            
            if self.separate_levels:
                #? Perform post-processing for each scale level
                slvl_results = self._post_process_slvl(scores, score_vars,
                                                    bboxes, bbox_covs,
                                                    labels, cfg)
                bboxes, bbox_covs, scores, score_vars, labels = slvl_results
            
            mlvl_scores_tensor = torch.cat((mlvl_scores_tensor, scores), dim=0)
            mlvl_score_vars_tensor = torch.cat((mlvl_score_vars_tensor, score_vars), dim=0) \
                                        if self.compute_cls_var else None
            mlvl_bboxes_tensor = torch.cat((mlvl_bboxes_tensor, bboxes), dim=0)
            mlvl_bbox_covs_tensor = torch.cat((mlvl_bbox_covs_tensor, bbox_covs), dim=0)
            mlvl_labels_tensor = torch.cat((mlvl_labels_tensor, labels), dim=0)
        
        return self._post_process_mlvl(mlvl_scores_tensor, mlvl_score_vars_tensor,
                                     mlvl_bboxes_tensor, mlvl_bbox_covs_tensor,
                                     mlvl_labels_tensor, cfg, 
                                     self.post_process_mlvl if self.separate_levels else None)
    
    def _rescale_preds(self, bboxes, bbox_covs, scale_factors):
        """Rescale bounding box prediction results to original image scale.

        Args:
            bboxes (torch.Tensor): Shape (N, 4).
            bbox_covs (torch.Tensor): Shape (N, 4, 4).
            scale_factors (ndarray): Shape (4,)
        """
        scale_factors_inv = torch.reciprocal(torch.from_numpy(scale_factors).to(bboxes))
        scale_matrix = torch.diag_embed(scale_factors_inv).to(bbox_covs) # (4, 4)
        bboxes = bboxes @ scale_matrix
        
        #* Add small value to make sure covariance matrix is well conditioned
        bbox_covs += 1e-4 * torch.eye(4, device=bbox_covs.device)
        bbox_covs = scale_matrix @ bbox_covs @ scale_matrix.transpose(-1, -2)
        return bboxes, bbox_covs
    
    def _fuse_cluster(self, bboxes, bbox_covs, scores, score_vars, 
                      labels, center_inds, post_process):
        #? Compute iou affinity matrix to find cluster members
        ious = bbox_overlaps(bboxes, bboxes) # (N, N)
        box_cluster_inds = ious[center_inds, :] > self.affinity_thr
        
        #? Compute mean and covariance for every cluster
        center_labels = labels[center_inds]
        cluster_bboxes, cluster_bbox_covs = [], []
        for cluster, center_label in zip(box_cluster_inds, center_labels):
            cluster_labels = labels[cluster] # (|cluster|,)
            valid_class_mask = cluster_labels == center_label
            
            #* Switch to numpy as torch.inverse is slower than np.linalg.inv
            member_bboxes = bboxes[cluster][valid_class_mask].cpu().numpy()
            member_bbox_covs = bbox_covs[cluster][valid_class_mask].cpu().numpy()
            
            if post_process == 'bayesian':
                cluster_bbox, cluster_bbox_cov = product_of_gaussians(member_bboxes,
                                                                        member_bbox_covs)
            elif post_process == 'covariance_intersection':
                cluster_bbox, cluster_bbox_cov = covariance_intersection(member_bboxes,
                                                                            member_bbox_covs)
            
            cluster_bboxes.append(torch.from_numpy(cluster_bbox))
            cluster_bbox_covs.append(torch.from_numpy(cluster_bbox_cov))
        
        det_bboxes = torch.stack(cluster_bboxes, dim=0).to(bboxes)
        det_bbox_covs = torch.stack(cluster_bbox_covs, dim=0).to(bbox_covs)
        det_scores = scores[center_inds]
        det_score_vars = score_vars[center_inds] \
                            if score_vars is not None else None
        det_labels = center_labels
        
        return det_bboxes, det_bbox_covs, det_scores, det_score_vars, det_labels
    
    def _post_process_slvl(self,
                            slvl_scores,
                            slvl_score_vars,
                            slvl_bboxes,
                            slvl_bbox_covs,
                            slvl_labels,
                            cfg):
        """Post-process predictions of a single scale level using fusion methods
        such as bayesian inference or covariance intersection (specified by `self.post_process`).
        Bayesian inference assumes that the bounding box predictions are independent.
        Covariance intersection assumes that the bounding box predictions are correlated,
        but the correlation is unknown.
        
        Let N be the total number of detected bounding boxes in this scale level.
        
        Args:
            slvl_scores (Tensor): Box scores of a single scale of a single image, 
                with shape (N,).
            slvl_score_vars (Tensor | None): Box score variances of a single scale 
                of a single image, with shape (N,). 
                None if compute_cls_var is False.
            slvl_bboxes (Tensor): Box predictions of a single scale of a single
                image, with shape (N, 4).
            slvl_bbox_covs (Tensor): Box covariance matrices of a single scale
                of a single image, with shape (N, 4, 4).
            slvl_labels (Tensor): Box class labels of a single scale of a single 
                image, with shape (N,).
            cfg (mmcv.Config): Test / postprocessing configuration.

        Returns:
            tuple[Tensor]: Results of detected bboxes, associated covariance
                matrices, class scores and labels.

                - det_bboxes (Tensor): Predicted bboxes with shape \
                    [num_bboxes, 4], representing the bounding \
                    box positions (tl_x, tl_y, br_x, br_y).
                - det_bbox_covs (Tensor): Predicted bbox covariance matrices \
                    with shape [num_bboxes, 4, 4].
                - det_scores (Tensor): Predicted class confidence scores between
                    0 and 1 of the corresponding box with shape [num_bboxes].
                - det_score_vars (Tensor | None): Predicted score variances of the \
                    corresponding box with shape [num_bboxes].
                - det_labels (Tensor): Predicted labels of the corresponding \
                    box with shape [num_bboxes].
        """
        if slvl_bboxes.numel() == 0:
            det_bbox_covs = torch.empty(
                tuple(slvl_bbox_covs.shape),
                dtype=slvl_bbox_covs.dtype,
                device=slvl_bbox_covs.device
            )
            det_score_vars = torch.empty(
                tuple(slvl_score_vars.shape),
                dtype=slvl_score_vars.dtype,
                device=slvl_score_vars.device
            ) if slvl_score_vars is not None else None
            return slvl_bboxes, det_bbox_covs, slvl_scores, det_score_vars, slvl_labels
        
        #? Perform NMS to get cluster centers
        _, keep_idxs = batched_nms(slvl_bboxes, slvl_scores,
                                            slvl_labels, cfg.nms)

        #? Fuse cluster members
        return self._fuse_cluster(slvl_bboxes, slvl_bbox_covs, slvl_scores,
                                slvl_score_vars, slvl_labels, keep_idxs,
                                self.post_process)
        
    def _post_process_mlvl(self, 
                         mlvl_scores, 
                         mlvl_score_vars,
                         mlvl_bboxes, 
                         mlvl_bbox_covs, 
                         mlvl_labels, 
                         cfg,
                         post_process_override=None):
        """
        Post-process predictions of all scale levels using nms or fusion methods
        such as bayesian inference or covariance intersection (specified by `self.post_process`
        or overridden by `post_process_override`).
        Bayesian inference assumes that the bounding box predictions are independent.
        Covariance intersection assumes that the bounding box predictions are correlated,
        but the correlation is unknown.
        
        If self.separate_levels is True, the bounding box predictions of each scale level 
        are fused in the _post_process_slvl method. Then, the fused bounding box predictions
        of each scale level are concatenated and fused again using this method.
        Otherwise, the bounding box predictions of all scale levels are processed all-together.
        
        Let M be the total number of bounding boxes from all scale levels whether that is 
        with or without single-level fusion.
        
        Args:
            mlvl_scores (Tensor): Box scores from all scale levels 
                of a single image, with shape (M,).
            mlvl_score_vars (Tensor | None): Box score variances from all
                scale levels of a single image, with shape (M,). 
                None if compute_cls_var is False.
            mlvl_bboxes (Tensor): Box predictions from all scale 
                levels of a single image, with shape (M, 4).
            mlvl_bbox_covs (Tensor): Box covariance matrices from all scale
                levels of a single image, with shape (M, 4, 4).
            mlvl_labels (Tensor): Box class labels from all scale levels 
                of a single imwith shape (M,).
            cfg (mmcv.Config): Test / postprocessing configuration.
            post_process_override (str | None): Post-processing method to override
                self.post_process. Default None.

        Returns:
            tuple[Tensor]: Results of detected bboxes, associated covariance
                matrices and labels.

                - det_bboxes (Tensor): Predicted bboxes with shape \
                    [num_bboxes, 5], where the first 4 columns are bounding \
                    box positions (tl_x, tl_y, br_x, br_y) and the 5-th \
                    column are class confidence scores between 0 and 1.
                - det_bbox_covs (Tensor): Predicted bbox covariance matrices \
                    with shape [num_bboxes, 4, 4].
                - det_labels (Tensor): Predicted labels of the corresponding \
                    box with shape [num_bboxes].
                - det_score_vars (Tensor | None): Predicted score variances of the \
                    corresponding box with shape [num_bboxes].
        """
        post_process = post_process_override if post_process_override is not None \
                        else self.post_process
        if mlvl_bboxes.numel() == 0:
            det_bboxes = torch.cat([mlvl_bboxes, mlvl_scores[:, None]], -1)
            det_bbox_covs = torch.empty(
                tuple(mlvl_bbox_covs.shape),
                dtype=mlvl_bbox_covs.dtype,
                device=mlvl_bbox_covs.device
            )
            det_score_vars = torch.empty(
                tuple(mlvl_score_vars.shape),
                dtype=mlvl_score_vars.dtype,
                device=mlvl_score_vars.device
            ) if mlvl_score_vars is not None else None
            return det_bboxes, det_bbox_covs, mlvl_labels, det_score_vars
        
        #? Perform NMS to get cluster centers
        det_bboxes, keep_idxs = batched_nms(mlvl_bboxes, mlvl_scores,
                                            mlvl_labels, cfg.nms)
        if post_process == 'nms':
            det_bbox_covs = mlvl_bbox_covs[keep_idxs]
            det_labels = mlvl_labels[keep_idxs]
            det_score_vars = mlvl_score_vars[keep_idxs] \
                                if mlvl_score_vars is not None else None
            return det_bboxes, det_bbox_covs, det_labels, det_score_vars

        elif post_process in ['bayesian', 'covariance_intersection']:
            #? Fuse cluster members
            cluster_bboxes, cluster_bbox_covs, \
            cluster_scores, cluster_score_vars, cluster_labels = \
                self._fuse_cluster(mlvl_bboxes, mlvl_bbox_covs, mlvl_scores,
                                        mlvl_score_vars, mlvl_labels, keep_idxs,
                                        post_process)
            
            cluster_bboxes = torch.cat([cluster_bboxes, cluster_scores[:, None]], -1)
            return cluster_bboxes, cluster_bbox_covs, cluster_labels, cluster_score_vars
        else:
            raise ValueError(f"{self.post_process} is an invalid bounding box post-processing method.")
    
    @force_fp32(apply_to=('cls_scores', 'cls_vars', 'bbox_preds', 'bbox_covs', 'objectnesses'))
    def loss(self,
             cls_scores,
             cls_vars,
             bbox_preds,
             bbox_covs,
             objectnesses,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head.
        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_priors * num_classes.
            cls_vars (list[Tensor] | list[None]): Box score variances for 
                each scale level, each is a 4D-tensor, the channel number is
                num_priors * num_classes.
                List of None if self.compute_cls_var is False.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_priors * 4.
            bbox_covs (list[Tensor]): Box covariance for each scale level,
                each is a 4D-tensor, the channel number is 
                num_priors * 4.
            objectnesses (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, 1, H, W).
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
        """
        num_imgs = len(img_metas)
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device,
            with_stride=True)
        
        #? Flatten cls_scores, cls_vars, bbox_preds, bbox_covs, objectnesses
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                  self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_cls_vars = [
            cls_var.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                self.cls_out_channels)
            for cls_var in cls_vars 
        ] if self.compute_cls_var else None
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_bbox_covs = [
            bbox_cov.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_cov in bbox_covs
        ]
        flatten_objectness = [
            objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
            for objectness in objectnesses
        ]
        
        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1)
        flatten_cls_vars = torch.cat(flatten_cls_vars, dim=1) \
                            if self.compute_cls_var else None
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_bbox_covs = torch.cat(flatten_bbox_covs, dim=1)
        flatten_objectness = torch.cat(flatten_objectness, dim=1)
        flatten_priors = torch.cat(mlvl_priors)
        flatten_bboxes = self._bbox_decode(flatten_priors, flatten_bbox_preds)
        priors = flatten_priors.unsqueeze(0).repeat(num_imgs, 1, 1)
        
        #? Get targets
        (pos_masks, cls_targets, obj_targets, bbox_targets, l1_targets,
         num_fg_imgs) = multi_apply(
             self._get_target_single, flatten_cls_scores.detach(),
             flatten_objectness.detach(),
             priors,
             flatten_bboxes.detach(), gt_bboxes, gt_labels)
        num_pos = torch.tensor(
            sum(num_fg_imgs),
            dtype=torch.float,
            device=flatten_cls_scores.device)
        num_total_samples = max(reduce_mean(num_pos), 1.0)
        pos_masks = torch.cat(pos_masks, 0)
        cls_targets = torch.cat(cls_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        bbox_targets = torch.cat(bbox_targets, 0)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)
        
        #? Clamp log covariances to avoid numerical instability
        bbox_covs = clamp_log_variance(flatten_bbox_covs.view(-1, 4)[pos_masks])
        
        #? Compute losses
        #! Assumes deltas are decoded in the loss function
        #! Basically only works for SampleIoULoss
        if self.loss_bbox.__class__.__name__ != 'SampleIoULoss':
            raise NotImplementedError("Only SampleIoULoss is supported for probabilistic YOLOXHead for now.")
        loss_bbox = self.loss_bbox(
            flatten_bbox_preds.view(-1, 4)[pos_masks],  #* Deltas
            bbox_covs,
            bbox_targets,   #* Decoded targets
            priors.view(-1, 4)[pos_masks],
            self._bbox_decode) / num_total_samples
        
        loss_obj = self.loss_obj(flatten_objectness.view(-1, 1),
                                 obj_targets) / num_total_samples
        
        cls_vars = flatten_cls_vars.view(-1, self.num_classes)[pos_masks] \
                    if flatten_cls_vars is not None else None
        loss_cls = self.loss_cls(
            flatten_cls_scores.view(-1, self.num_classes)[pos_masks],
            cls_vars,
            cls_targets) / num_total_samples

        loss_dict = dict(
            loss_cls=loss_cls, loss_bbox=loss_bbox, loss_obj=loss_obj)
        
        if self.use_l1:
            loss_l1 = self.loss_l1(
                flatten_bbox_preds.view(-1, 4)[pos_masks],
                bbox_covs,
                l1_targets) / num_total_samples
            loss_dict.update(loss_l1=loss_l1)
        
        return loss_dict