import torch
import torch.nn as nn
import numpy as np
from mmcv.runner import force_fp32
from mmcv.cnn.utils import xavier_init, normal_init
from mmcv.cnn import ConvModule
from mmcv.ops import batched_nms

from mmdet.core import (images_to_levels, multi_apply, BboxOverlaps2D)
from mmdet.core.utils import filter_scores_and_topk, select_single_mlvl
from mmdet.models.builder import HEADS
from mmdet.models.dense_heads import RetinaHead

from core.utils import (
    compute_probabilistic_weight, covariance2cholesky, compute_mean_covariance_torch,
    clamp_log_variance, compute_mean_variance_torch
)


@HEADS.register_module()
class ProbabilisticRetinaHead(RetinaHead):
    r"""
    An anchor-based probabilistic head used in `BayesOD
    <https://arxiv.org/pdf/1903.03838.pdf>`.

    It extends the RetinaHead to predict the mean and covariance of the anchor box 
    classes and deltas.
    """
    def __init__(self,
                 num_classes,
                 in_channels,
                 epoch_step=2,
                 iters_in_epoch=3996,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 covariance_type="diagonal",
                 post_process="nms",
                 with_sampling=False,
                 compute_cls_var=False,
                 affinity_thr=0.9,
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     octave_base_scale=4,
                     scales_per_octave=3,
                     ratios=[0.5, 1.0, 2.0],
                     strides=[8, 16, 32, 64, 128]),
                 init_cfg=None,
                 **kwargs):
        # Diagonal covariance matrix has N elements
        # Number of elements required to describe an NxN covariance matrix is
        # computed as: (N*(N+1))/2 (lower triangular matrix)
        if covariance_type == "diagonal":
            self.bbox_cov_dims = 4
        else:
            self.bbox_cov_dims = 10
            
        self.with_sampling = with_sampling

        self.post_process = post_process
        self.compute_cls_var = compute_cls_var
        self.affinity_thr = affinity_thr
        self.current_iter = 0
        self.current_epoch = 0
        self.epoch_step = epoch_step
        self.iters_in_epoch = iters_in_epoch
        
        self.iou_calc = BboxOverlaps2D()

        super(ProbabilisticRetinaHead, self).__init__(
            num_classes,
            in_channels,
            stacked_convs=stacked_convs,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            anchor_generator=anchor_generator,
            init_cfg=init_cfg,
            **kwargs)

        if hasattr(self.loss_bbox, "covariance_type"):
            self.loss_bbox.covariance_type = covariance_type

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
                
        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.num_base_priors * self.cls_out_channels,
            3,
            padding=1)
        
        self.retina_reg = nn.Conv2d(
            self.feat_channels, self.num_base_priors * 4, 3, padding=1)
        
        #? Create module for classification variance estimation
        if self.compute_cls_var:
            #* Same shape as the classification output (diagonal covariance matrix)
            self.retina_cls_var = nn.Conv2d(
                self.feat_channels,
                self.num_base_priors * self.cls_out_channels,
                3,
                padding=1)
        else:
            self.retina_cls_var = None
        
        #? Create module for bounding box covariance estimation
        #* Same shape as the bounding box output (diagonal covariance matrix)
        self.retina_reg_cov = nn.Conv2d(
            self.feat_channels,
            self.num_base_priors * self.bbox_cov_dims,
            3,
            padding=1)
        
        #? Initialize retina_cls_var and retina_reg_cov modules
        #? init_cfg may not be called if the model is loaded from pretrained weights
        #TODO: clean up this using `load_from` in the config file and modifying init_weights method instead
        if isinstance(self.init_cfg, dict):
            if hasattr(self.init_cfg, "override"):
                override_init_cfg = self.init_cfg.override
                if override_init_cfg is not None and isinstance(override_init_cfg, list):
                    for _init in override_init_cfg:
                        if _init.name in ["retina_cls_var", "retina_reg_cov"]:
                            #? Initialize the module
                            module = getattr(self, _init.name)
                            if module is not None:
                                if _init.type == "Xavier":
                                    xavier_init(module, bias=_init.bias)
                                else:
                                    normal_init(module, mean=0, std=_init.std)
                        else:
                            raise ValueError("Invalid override init config.")

    def forward_single(self, x):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level;
                    the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale
                    level, the channels number is num_anchors * 4.
                cls_var (Tensor | None): Cls score covariances for a single scale level;
                    the channels number is num_anchors * num_classes.
                    None if self.compute_cls_var is False.
                bbox_cov (Tensor): Box covariances for a single scale level,
                    the channels number is num_anchors * bbox_cov_dims
        """
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)
        
        #? cls variance and bbox covariance estimation
        cls_var = self.retina_cls_var(cls_feat) if self.compute_cls_var else None
        bbox_cov = self.retina_reg_cov(reg_feat)

        return cls_score, bbox_pred, cls_var, bbox_cov

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: A tuple of classification scores and bbox prediction.

                - cls_scores (list[Tensor]): Classification scores for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_base_priors * num_classes.
                - bbox_preds (list[Tensor]): Box energies / deltas for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_base_priors * 4.
                - cls_vars (list[Tensor] | list[None]): Classification score covariances for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is self.num_base_priors * num_classes.
                    List of None if self.compute_cls_var is False.
                - bbox_covs (list[Tensor]): Box covariances for all scale \
                    levels, each is a 4-D tensor, the channels number is \
                    num_base_priors * bbox_cov_dims
        """
        return multi_apply(self.forward_single, feats)
    
    def get_targets(self,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    label_channels=1,
                    unmap_outputs=True,
                    return_sampling_results=False):
        """Compute regression and classification targets for anchors in
        multiple images.

        Args:
            anchor_list (list[list[Tensor]]): Multi level anchors of each
                image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, 4).
            valid_flag_list (list[list[Tensor]]): Multi level valid flags of
                each image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, )
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            img_metas (list[dict]): Meta info of each image.
            gt_bboxes_ignore_list (list[Tensor]): Ground truth bboxes to be
                ignored.
            gt_labels_list (list[Tensor]): Ground truth labels of each box.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: Usually returns a tuple containing learning targets.

                - labels_list (list[Tensor]): Labels of each level.
                - label_weights_list (list[Tensor]): Label weights of each
                  level.
                - bbox_targets_list (list[Tensor]): BBox targets of each level.
                - bbox_weights_list (list[Tensor]): BBox weights of each level.
                - num_total_pos (int): Number of positive samples in all
                  images.
                - num_total_neg (int): Number of negative samples in all
                  images.

            additional_returns: This function enables user-defined returns from
                `self._get_targets_single`. These returns are currently refined
                to properties at each feature map (i.e. having HxW dimension).
                The results will be concatenated after the end
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors to a single tensor
        concat_anchor_list = []
        concat_valid_flag_list = []
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            concat_anchor_list.append(torch.cat(anchor_list[i]))
            concat_valid_flag_list.append(torch.cat(valid_flag_list[i]))

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        results = multi_apply(
            self._get_targets_single,
            concat_anchor_list,
            concat_valid_flag_list,
            gt_bboxes_list,
            gt_bboxes_ignore_list,
            gt_labels_list,
            img_metas,
            label_channels=label_channels,
            unmap_outputs=unmap_outputs)
        (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
         pos_inds_list, neg_inds_list, sampling_results_list) = results[:7]
        rest_results = list(results[7:])  # user-added return values
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        #* all_: list[Tensor] for each image
        #* _list: list[Tensor] for each level, where each tensor is contains all 4 images stacked
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights, num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets, num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights, num_level_anchors)

        res = (labels_list, label_weights_list, bbox_targets_list, 
                bbox_weights_list, num_total_pos, num_total_neg)
        if return_sampling_results:
            res = res + (sampling_results_list, )
        for i, r in enumerate(rest_results):  # user-added return values
            rest_results[i] = images_to_levels(r, num_level_anchors)

        return res + tuple(rest_results)
    
    def loss_single(self, cls_score, bbox_pred, cls_var, bbox_cov, anchors, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples):
        """Compute loss of a single scale level.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            cls_var (Tensor | None): Box score covariances for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
                None if compute_cls_var is False.
            bbox_cov (Tensor): Box corner covariances for each scale
                level with shape (N, num_anchors * self.bbox_cov_dims, H, W).
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor
                weight shape (N, num_total_anchors, 4).
            bbox_weights (Tensor): BBox regression loss weights of each anchor
                with shape (N, num_total_anchors, 4).
            num_total_samples (int): If sampling, num total samples equal to
                the number of total anchors; Otherwise, it is the number of
                positive anchors.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        #? Calculate weight factor for loss attenuation
        probabilistic_weight = self.current_epoch / self.epoch_step + \
                        ((self.current_iter+1) % self.iters_in_epoch) / self.iters_in_epoch
        probabilistic_weight = min(compute_probabilistic_weight(probabilistic_weight), 1.0)
        
        #? classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        if cls_var is not None:
            cls_var = cls_var.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, cls_var, labels, probabilistic_weight, label_weights, avg_factor=num_total_samples)
        
        #? regression loss
        target_dim = bbox_targets.size(-1)
        bbox_targets = bbox_targets.reshape(-1, target_dim)
        bbox_weights = bbox_weights.reshape(-1, target_dim)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, target_dim)
        if bbox_cov is not None:
            bbox_cov = bbox_cov.permute(0, 2, 3, 1).reshape(-1, self.bbox_cov_dims)
        
        #? Clamp log covariances to avoid numerical instability
        bbox_cov = clamp_log_variance(bbox_cov)

        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_cov,
            bbox_targets,
            probabilistic_weight,
            bbox_weights,
            avg_factor=num_total_samples)
        return loss_cls, loss_bbox
    
    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'cls_vars', 'bbox_covs'))
    def loss(self,
             cls_scores,
             bbox_preds,
             cls_vars,
             bbox_covs,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            cls_vars (list[Tensor] | list[None]): Box score variances for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
                List of none if compute_cls_var is False.
            bbox_covs (list[Tensor]): Box covariances for each scale
                level with shape (N, num_anchors * bbox_cov_dims, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, 
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)

        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            cls_vars,
            bbox_covs,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples)
        
        return dict(loss_cls=losses_cls, 
                    loss_bbox=losses_bbox)
        
    def simple_test(self, feats, img_metas, rescale=False):
        """Test function without test-time augmentation.

        Args:
            feats (tuple[torch.Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is ``bboxes`` with shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n, ).
        """
        return self.simple_test_bboxes(feats, img_metas, rescale=rescale)
    
    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'cls_vars', 'bbox_covs'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   cls_vars,
                   bbox_covs,
                   score_factors=None,
                   img_metas=None,
                   cfg=None,
                   rescale=False,
                   with_nms=True,
                   **kwargs):
        """Transform network outputs of a batch into bbox results.

        Note: When score_factors is not None, the cls_scores are
        usually multiplied by it then obtain the real score used in NMS,
        such as CenterNess in FCOS, IoU branch in ATSS.

        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            cls_vars (list[Tensor] | list[None]): Box score variances for all scale levels
                Has shape (batch_size, num_priors * num_classes, H, W)
                List of None if compute_cls_var is False.
            bbox_covs (list[Tensor]): Box covariances for all scale
                levels with shape (batch_size, num_priors * bbox_cov_dims, H, W)
            score_factors (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, num_priors * 1, H, W). Default None.
            img_metas (list[dict], Optional): Image meta info. Default None.
            cfg (mmcv.Config, Optional): Test / postprocessing configuration,
                if None, test_cfg would be used.  Default None.
            rescale (bool): If True, return boxes in original image space.
                Default False.
            with_nms (bool): If True, perform post-processing before returning boxes.
                Default True.

        Returns:
            list[list[Tensor, Tensor, Tensor]]: Each item in result_list is 3-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is an
                (n, bbox_cov_dims, bbox_cov_dims) tensor representing the bounding
                box covariance matrices. The third item is a (n,) tensor where each
                item is the predicted class label of the corresponding box.
        """
        assert len(cls_scores) == len(bbox_preds)

        if score_factors is None:
            # e.g. Retina, FreeAnchor, Foveabox, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, ATSS, etc.
            raise ValueError('Score factor is not supported for RetinaNet.')

        num_levels = len(cls_scores)

        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device)

        result_list = []

        for img_id in range(len(img_metas)):
            img_meta = img_metas[img_id]
            cls_score_list = select_single_mlvl(cls_scores, img_id)
            if cls_vars[0] is not None:
                cls_score_var_list = select_single_mlvl(cls_vars, img_id)
            else:
                cls_score_var_list = [None for _ in range(num_levels)]
            bbox_pred_list = select_single_mlvl(bbox_preds, img_id)
            bbox_cov_list = select_single_mlvl(bbox_covs, img_id)
            if with_score_factors:
                score_factor_list = select_single_mlvl(score_factors, img_id)
            else:
                score_factor_list = [None for _ in range(num_levels)]

            results = self._get_bboxes_single(cls_score_list, bbox_pred_list,
                                              cls_score_var_list, bbox_cov_list,
                                              score_factor_list, mlvl_priors,
                                              img_meta, cfg, rescale, with_nms,
                                              **kwargs)
            result_list.append(results)
        return result_list

    def _get_bboxes_single(self,
                           cls_score_list,
                           bbox_pred_list,
                           cls_score_var_list,
                           bbox_cov_list,
                           score_factor_list,
                           mlvl_priors,
                           img_meta,
                           cfg,
                           rescale=False,
                           with_nms=True,
                           **kwargs):
        """Transform outputs of a single image into bbox predictions.

        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_priors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has shape
                (num_priors * 4, H, W).
            cls_score_var_list (list[Tensor] | list[None]): Box score variances from all
                scale levels of a single image, each item has shape
                (num_priors * num_classes, H, W).
                List of None if compute_cls_var is False.
            bbox_cov_list (list[Tensor]): Box covariances from all scale
                levels of a single image, each item has shape 
                (num_priors * bbox_cov_dims, H, W).
            score_factor_list (list[Tensor]): Score factor from all scale
                levels of a single image, each item has shape
                (num_priors * 1, H, W).
            mlvl_priors (list[Tensor]): Each element in the list is
                the priors of a single level in feature pyramid. In all
                anchor-based methods, it has shape (num_priors, 4). In
                all anchor-free methods, it has shape (num_priors, 2)
                when `with_stride=True`, otherwise it still has shape
                (num_priors, 4).
            img_meta (dict): Image meta info.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, perform post-processing before returning boxes.
                Default True.

        Returns:
            tuple[Tensor]: Results of detected bboxes, associated covariance
                matrices and labels. If with_nms is False and mlvl_score_factor 
                is None, return mlvl_bboxes, mlvl_bbox_covs and mlvl_scores, 
                else return mlvl_bboxes, mlvl_scores and mlvl_score_factor. 
                Usually with_nms is False is used for aug test. 
                If with_nms is True, then return the following format:

                - det_bboxes (Tensor): Predicted bboxes with shape \
                    [num_bboxes, 5], where the first 4 columns are bounding \
                    box positions (tl_x, tl_y, br_x, br_y) and the 5-th \
                    column are scores between 0 and 1.
                - det_bbox_covs (Tensor): Predicted bbox covariance matrices \
                    with shape [num_bboxes, 4, 4].
                - det_labels (Tensor): Predicted labels of the corresponding \
                    box with shape [num_bboxes].
                - det_score_vars (Tensor | None): Predicted score variances of the \
                    corresponding box with shape [num_bboxes].
        """
        if score_factor_list[0] is None:
            # e.g. Retina, FreeAnchor, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, etc.
            with_score_factors = True

        cfg = self.test_cfg if cfg is None else cfg
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bboxes = []
        mlvl_bbox_covs = []
        mlvl_scores = []
        mlvl_score_vars = []
        mlvl_labels = []
        if with_score_factors:
            mlvl_score_factors = []
        else:
            mlvl_score_factors = None
        
        #? Iterate through each scale level
        for level_idx, (cls_score, bbox_pred, cls_score_var, bbox_cov, 
                        score_factor, priors) in \
                enumerate(zip(cls_score_list, bbox_pred_list, cls_score_var_list,
                              bbox_cov_list, score_factor_list, mlvl_priors)):

            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            bbox_cov = bbox_cov.permute(1, 2, 0).reshape(-1, self.bbox_cov_dims)
            
            if with_score_factors:
                score_factor = score_factor.permute(1, 2,
                                                    0).reshape(-1).sigmoid()
            cls_logits = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)

            #? Perform MC-Sampling to generate logits using cls_score_var
            if cls_score_var is not None:
                cls_logits_var = cls_score_var.permute(1, 2, 0).reshape(-1, self.cls_out_channels)
                cls_logits_dists = torch.distributions.normal.Normal(
                    cls_logits, scale=torch.sqrt(torch.exp(cls_logits_var)))
                cls_logits = cls_logits_dists.sample((self.loss_cls.num_samples, ))
            
            if self.use_sigmoid_cls:
                scores = cls_logits.sigmoid()
            else:
                # remind that we set FG labels to [0, num_class-1]
                # since mmdet v2.0
                # BG cat_id: num_class
                scores = cls_logits.softmax(-1)[:, :-1]
            
            #? Compute mean and variance of scores
            scores, score_vars = compute_mean_variance_torch(scores)
            
            # After https://github.com/open-mmlab/mmdetection/pull/6268/,
            # this operation keeps fewer bboxes under the same `nms_pre`.
            # There is no difference in performance for most models. If you
            # find a slight drop in performance, you can set a larger
            # `nms_pre` than before.
            results = filter_scores_and_topk(
                scores, cfg.score_thr, nms_pre,
                dict(bbox_pred=bbox_pred, bbox_cov=bbox_cov, priors=priors))
            scores, labels, keep_idxs, filtered_results = results

            bbox_pred = filtered_results['bbox_pred']
            priors = filtered_results['priors']
            bbox_cov = filtered_results['bbox_cov']
            if score_vars is not None:
                score_vars = score_vars[keep_idxs]
                score_vars = score_vars[torch.arange(score_vars.size(0)), labels]

            if with_score_factors:
                score_factor = score_factor[keep_idxs]
            
            bbox_cov = clamp_log_variance(bbox_cov)
            if bbox_pred.numel() > 0 and self.with_sampling:
                assert bbox_cov.numel() > 0, "bbox_cov is empty"
                #? Construct cholesky factor of covariance matrix vector
                bbox_chol = covariance2cholesky(bbox_cov)

                #? Monte-Carlo sampling of predicted deltas and covariances
                mvn = torch.distributions.MultivariateNormal(
                    bbox_pred, scale_tril=bbox_chol)
                delta_samples = mvn.sample((1000,))    # (1000, N, 4)
                priors_repeated = torch.repeat_interleave(priors.unsqueeze(0), 1000, dim=0) # (1000, N, 4)
                
                #? Get bboxes from the bbox deltas and priors
                bboxes_samples = self.bbox_coder.decode(
                    priors_repeated, delta_samples, max_shape=img_shape)
                bboxes, bbox_covs = compute_mean_covariance_torch(bboxes_samples)
            else:
                bboxes = self.bbox_coder.decode(priors, bbox_pred, max_shape=img_shape)
                #? Construct covariance matrix with diagonal only;
                bbox_covs = torch.diag_embed(torch.exp(bbox_cov[:, :4]))

            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_score_vars.append(score_vars)
            mlvl_labels.append(labels)
            mlvl_bbox_covs.append(bbox_covs)
            if with_score_factors:
                mlvl_score_factors.append(score_factor)

        return self._bbox_post_process(mlvl_scores, mlvl_score_vars, mlvl_labels, 
                                       mlvl_bboxes, mlvl_bbox_covs, img_meta['scale_factor'], 
                                       cfg, rescale, with_nms, mlvl_score_factors,
                                       **kwargs)
    
    def _bbox_fusion(self, cluster_means, cluster_covs, merge_mode):
        """Fuse cluster means and covariances.
        Args:
            cluster_means (nd array): cluster box means.
                Has shape (|cluster|, 4)
            cluster_covs (nd array): cluster box covariance matrices.
                Has shape (|cluster|, 4, 4)
            merge_mode (str): mode of merging clusters.
                Options are "bayesian" and "covariance_intersection".
        Returns:
            final_mean (nd array): cluster fused mean.
                Has shape (4,)
            final_cov (nd array): cluster fused covariance matrix.
                Has shape (4, 4)
        """
        precision_mats = np.linalg.inv(cluster_covs)
        if merge_mode == "bayesian":
            final_cov = np.linalg.inv(precision_mats.sum(0))
            final_mean = np.matmul(
                precision_mats, np.expand_dims(cluster_means, -1)).sum(0)
            final_mean = np.squeeze(np.matmul(final_cov, final_mean))
        elif merge_mode == "covariance_intersection":
            """Fast covariance intersection algorithm.
            See "Improved Fast Covariance Intersection for Distributed Data Fusion"
            (Franken, D. et al., 2005) for more details. 
            """
            precision_diff = precision_mats.sum(0) - precision_mats
            precision_det = np.linalg.det(precision_mats)
            total_precision_det = np.linalg.det(precision_mats.sum(0))
            precision_diff_det = np.linalg.det(precision_diff)
            weights = (total_precision_det - precision_diff_det + precision_det) / \
                (precision_mats.shape[0] * total_precision_det + \
                    (precision_det - precision_diff_det).sum(0))
            weighted_precisions = (
                np.expand_dims(weights, (1,2)) * precision_mats)
            final_cov = np.linalg.inv(weighted_precisions.sum(0))
            final_mean = np.squeeze(final_cov @ (
                            (weighted_precisions @ 
                                np.expand_dims(cluster_means, -1)).sum(0)),
                            axis=-1)
        return final_mean, final_cov

    def _bbox_post_process(self,
                           mlvl_scores,
                           mlvl_score_vars,
                           mlvl_labels,
                           mlvl_bboxes,
                           mlvl_bbox_covs,
                           scale_factor,
                           cfg,
                           rescale=False,
                           with_nms=True,
                           mlvl_score_factors=None,
                           **kwargs):
        """bbox post-processing method.

        The boxes would be rescaled to the original image scale and do
        the nms operation. Usually `with_nms` is False is used for aug test.

        Args:
            mlvl_scores (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_bboxes, ).
            mlvl_score_vars (list[Tensor]): Box score variances from all scale
                levels of a single image, each item has shape (num_bboxes, ).
            mlvl_labels (list[Tensor]): Box class labels from all scale
                levels of a single image, each item has shape
                (num_bboxes, ).
            mlvl_bboxes (list[Tensor]): Decoded bboxes from all scale
                levels of a single image, each item has shape (num_bboxes, 4).
            mlvl_bbox_covs (list[Tensor]): Decoded bbox covariance matrices from
                all scale levels of a single image, each item has shape 
                (num_bboxes, bbox_cov_dims, bbox_cov_dims).
            scale_factor (ndarray, optional): Scale factor of the image arange
                as (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, perform post-processing before returning boxes.
                Default True.
            mlvl_score_factors (list[Tensor], optional): Score factor from
                all scale levels of a single image, each item has shape
                (num_bboxes, ). Default: None.

        Returns:
            tuple[Tensor]: Results of detected bboxes and labels. If with_nms
                is False and mlvl_score_factor is None, return mlvl_bboxes, 
                mlvl_bbox_covs and mlvl_scores, else return mlvl_bboxes, 
                mlvl_scores and mlvl_score_factor. Usually with_nms is False 
                is used for aug test. If with_nms is True, then return the 
                following format

                - det_bboxes (Tensor): Predicted bboxes with shape \
                    [num_bboxes, 5], where the first 4 columns are bounding \
                    box positions (tl_x, tl_y, br_x, br_y) and the 5-th \
                    column are scores between 0 and 1.
                - det_bbox_covs (Tensor): Predicted bbox covariance matrices \
                    with shape [num_bboxes, bbox_cov_dims, bbox_cov_dims].
                - det_labels (Tensor): Predicted labels of the corresponding \
                    box with shape [num_bboxes].
                - det_score_vars (Tensor | None): Predicted score variances of the \
                    corresponding box with shape [num_bboxes].
        """
        assert len(mlvl_scores) == len(mlvl_score_vars) \
                == len(mlvl_bboxes) == len(mlvl_labels) == len(mlvl_bbox_covs)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_bbox_covs = torch.cat(mlvl_bbox_covs)

        #? Rescale bboxes and bbox_covs to the original image size
        if rescale:
            scale_factors_inv = torch.reciprocal(
                torch.from_numpy(scale_factor).to(mlvl_bboxes))
            scale_matrix = torch.diag_embed(scale_factors_inv
                                            ).to(mlvl_bbox_covs) # (4, 4)
            mlvl_bboxes = mlvl_bboxes @ scale_matrix # (N, 4)
            
            #? Rescale covariance matrix
            # Add small value to make sure covariance matrix is well conditioned
            mlvl_bbox_covs += 1e-4 * torch.eye(mlvl_bbox_covs.shape[2], 
                                                device=mlvl_bbox_covs.device)
            mlvl_bbox_covs = scale_matrix @ mlvl_bbox_covs \
                              @ scale_matrix.transpose(1, 0)   # (N, 4, 4)

        mlvl_scores = torch.cat(mlvl_scores)
        #TODO: check if mlvl_score_vars is not a list of None
        mlvl_score_vars = torch.cat(mlvl_score_vars) if mlvl_score_vars[0] is not None else None
        mlvl_labels = torch.cat(mlvl_labels)

        if mlvl_score_factors is not None:
            # TODO： Add sqrt operation in order to be consistent with
            #  the paper.
            mlvl_score_factors = torch.cat(mlvl_score_factors)
            mlvl_scores = mlvl_scores * mlvl_score_factors
            mlvl_score_vars = mlvl_score_vars * mlvl_score_factors.pow(2) \
                                if mlvl_score_vars is not None else None

        if with_nms:
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
            
            if self.post_process == 'nms':
                #? Perform NMS
                det_bboxes, keep_idxs = batched_nms(mlvl_bboxes, mlvl_scores,
                                                    mlvl_labels, cfg.nms)
                det_bboxes = det_bboxes[:cfg.max_per_img]
                det_bbox_covs = mlvl_bbox_covs[keep_idxs][:cfg.max_per_img]
                det_labels = mlvl_labels[keep_idxs][:cfg.max_per_img]
                det_score_vars = mlvl_score_vars[keep_idxs][:cfg.max_per_img] \
                                    if mlvl_score_vars is not None else None
                return det_bboxes, det_bbox_covs, det_labels, det_score_vars

            elif self.post_process in ['bayesian', 'covariance_intersection']:
                #? Get indices of bboxes after NMS
                _, keep_idxs = batched_nms(mlvl_bboxes, mlvl_scores, 
                                        mlvl_labels, cfg.nms)
                keep_idxs = keep_idxs[:cfg.max_per_img] # (min(N,max_per_img), )
                
                #? Get box cluster indices
                #* Each row is a cluster, and columns with iou greater than
                #* affinity_thr are in the same cluster.
                ious = self.iou_calc(mlvl_bboxes, mlvl_bboxes)  # (N, N)
                box_cluster_idxs = ious[keep_idxs,:]        # (min(N,max_per_img), N)
                box_cluster_idxs = box_cluster_idxs > self.affinity_thr
                
                #? Compute mean and covariance for every cluster.
                bboxes_list = []
                bbox_covs_list = []
                mlvl_labels_filtered = mlvl_labels[keep_idxs] # (min(N,max_per_img), )
                
                for box_cluster, center_cls_label in zip(box_cluster_idxs, mlvl_labels_filtered):
                    cluster_labels = mlvl_labels[box_cluster] # (|box_cluster|, )
                    
                    #? Compare the class of each element in cluster with the center class
                    same_class_idx = center_cls_label == cluster_labels

                    # Switch to numpy as torch.inverse is slower than np.linalg.inv
                    cluster_bboxes = mlvl_bboxes[box_cluster, :][same_class_idx].to('cpu').numpy()
                    cluster_bbox_covs = mlvl_bbox_covs[box_cluster, :][same_class_idx].to('cpu').numpy()
                    det_bbox, det_bbox_cov = self._bbox_fusion(cluster_bboxes, cluster_bbox_covs,
                                                                self.post_process)
                    bboxes_list.append(torch.from_numpy(det_bbox))
                    bbox_covs_list.append(torch.from_numpy(det_bbox_cov))

                det_labels = mlvl_labels[keep_idxs]
                det_bboxes = torch.stack(bboxes_list).to(mlvl_bboxes.device)
                det_bboxes = torch.cat([det_bboxes, mlvl_scores[keep_idxs].unsqueeze(-1)], -1)
                det_bbox_covs = torch.stack(bbox_covs_list).to(mlvl_bbox_covs.device)
                det_score_vars = mlvl_score_vars[keep_idxs] if mlvl_score_vars is not None else None
                return det_bboxes, det_bbox_covs, det_labels, det_score_vars
            else:
                raise ValueError(f"{self.post_process} is an invalid bounding box post-processing method.")
        else:
            return mlvl_bboxes, mlvl_bbox_covs, mlvl_scores, mlvl_labels, mlvl_score_vars

    def aug_test(self, feats, img_metas, rescale=False):
        """Test function with test time augmentation. Found in anchor_head.py.
        Not implemented yet for Probabilistic RetinaNet.
        """
        raise NotImplementedError("Not implemented yet for Probabilistic RetinaNet.")