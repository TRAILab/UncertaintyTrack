import torch
import torch.nn as nn

from mmdet.models.builder import LOSSES
from mmdet.models.losses.smooth_l1_loss import smooth_l1_loss, l1_loss
from mmdet.models.losses.utils import weight_reduce_loss

from core.utils import covariance2cholesky


@LOSSES.register_module()
class NLL(nn.Module):
    """Negative log-likelihood loss for bounding box delta regression.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to "mean".
        covariance_type (str, optional): The type of covariance matrix.
            Options are "diagonal" and "full". Defaults to "diagonal".
        attenuated (bool, optional): Whether to use loss attenuation.
            Defaults to False.
        loss_type (str, optional): The type of loss. 
            Options are "L1" and "SmoothL1". Defaults to "L1".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, 
                 reduction='mean', 
                 covariance_type='diagonal',
                 attenuated=False,
                 loss_type='L1',
                 loss_weight=1.0,
                 **kwargs):
        super(NLL, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self._covariance_type = covariance_type
        self.attenuated = attenuated
        if loss_type == 'L1':
            self.loss_func = l1_loss
        elif loss_type == 'SmoothL1':
            self.loss_func = smooth_l1_loss
        else:
            raise ValueError('loss_type should be "L1" or "SmoothL1"')
        
    @property
    def covariance_type(self):
        return self._covariance_type
    
    @covariance_type.setter
    def covariance_type(self, type):
        self._covariance_type = type

    def forward(self,
                pred,
                pred_cov,
                target,
                probabilistic_weight=1.0,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            prev_cov (torch.Tensor): The covariance of the prediction in log scale.
            target (torch.Tensor): The learning target of the prediction.
            probabilistic_weight (float, optional): The probabilistic weight for 
                loss attenuation. Computed using current iteration and epoch. 
                Defaults to 1.0.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        probabilistic_weight = probabilistic_weight if self.attenuated else 1.0
        
        #? NLL loss for bounding box regression
        if self.covariance_type == 'diagonal':
            loss_bbox = 0.5 * torch.exp(-pred_cov) * self.loss_func(
                pred,
                target,
                weight,
                reduction='none',
                avg_factor=None,
                **kwargs)
            loss_cov_regularization = 0.5 * pred_cov
            loss_bbox += loss_cov_regularization
        elif self.covariance_type == 'full':
            #* Use log prob from multivariate normal distribution
            if weight is not None:
                #? Apply weight early to avoid broadcasting later
                pred = pred * weight
                nonzero_mask = (pred != 0).any(dim=1)
                pred = pred[nonzero_mask]
                pred_cov = pred_cov[nonzero_mask]
                target = target[nonzero_mask]
                weight = None

            if pred.numel() > 0:
                #? Define multivariate Gaussian distributions to compute log prob
                pred_cov_cholesky = covariance2cholesky(pred_cov)
                mvn = torch.distributions.MultivariateNormal(pred, scale_tril=pred_cov_cholesky)
                loss_bbox = -mvn.log_prob(target)
            else:
                loss_bbox = self.loss_func(
                                pred,
                                target,
                                weight,
                                reduction='none',
                                avg_factor=None)
        else:
            raise ValueError('covariance_type in NLL should be "diagonal" or "full"')
        loss_bbox = weight_reduce_loss(loss_bbox, weight, reduction, avg_factor)
        loss_bbox = probabilistic_weight * loss_bbox
        
        if probabilistic_weight < 1.0:
            #? Perform loss annealing for training with NLL
            standard_loss_bbox = self.loss_func(
                pred,
                target,
                weight,
                reduction=reduction,
                avg_factor=avg_factor,
                **kwargs
            )
            loss_bbox += (1.0 - probabilistic_weight) * standard_loss_bbox
        
        loss_bbox = self.loss_weight * loss_bbox
        return loss_bbox