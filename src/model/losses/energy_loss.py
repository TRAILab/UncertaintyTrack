import torch
import torch.nn as nn
import warnings

from mmdet.models.builder import LOSSES
from mmdet.models.losses.smooth_l1_loss import smooth_l1_loss, l1_loss
from mmdet.models.losses import IoULoss
from mmdet.models.losses.iou_loss import iou_loss

from core.utils import covariance2cholesky

def draw_samples(mean, cov, num_samples=1000):
    """Draw samples from the multivariate Gaussian distribution parameterized by mean and covariance.
    
    Args:
        mean (torch.Tensor): The mean of the multivariate Gaussian distribution.
        cov (torch.Tensor): The covariance of the multivariate Gaussian distribution.
        num_samples (int, optional): Number of samples drawn from the multivariate
            Gaussian distribution defined by predicted box means and covariance matrices.
            Defaults to 1000.
            
    Returns:
        tuple(torch.Tensor, torch.Tensor): 
            Two sets of samples of the multivariate Gaussian distribution, each of shape (num_samples, N, 4)
    """
    #? Define multivariate Gaussian distributions to draw MC-samples from
    cov_cholesky = covariance2cholesky(cov)
    mvn = torch.distributions.MultivariateNormal(mean, scale_tril=cov_cholesky)
    samples = mvn.rsample((num_samples+1,))
    sample_set1 = samples[:-1, :, :]
    sample_set2 = samples[1:, :, :]
    
    return sample_set1, sample_set2

@LOSSES.register_module()
class ESLoss(nn.Module):
    """Energy loss for bounding box delta regression.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to "mean".
        loss_type (str, optional): The type of loss. 
            Options are ("L1", "SmoothL1"). Defaults to "L1".
        attenuated (bool, optional): Whether to use loss attenuation.
        num_samples (int, optional): Number of samples drawn from the
            multivariate Gaussian distribution defined by predicted box
            means and covariance matrices. Defaults to 1000.
        loss_weight (float, optional): The weight of loss.
    """
    def __init__(self, 
                    reduction='mean',  
                    loss_type='L1',
                    attenuated=False,
                    num_samples=1000,
                    loss_weight=1.0,
                    **kwargs):
        super(ESLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

        if loss_type == 'L1':
            self.loss_fn = l1_loss
        elif loss_type == 'SmoothL1':
            self.loss_fn = smooth_l1_loss
        else:
            raise ValueError('loss_type should be "L1" or "SmoothL1"')
        self.attenuated = attenuated
        self.num_samples = num_samples

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
        
        #? Apply weight early to reduce memory usage in sampling
        if weight is not None:
            pred = pred * weight
            nonzero_mask = (pred != 0).any(dim=1)
            pred = pred[nonzero_mask]
            pred_cov = pred_cov[nonzero_mask]
            target = target[nonzero_mask]

        if pred.numel() > 0:
            sample_set1, sample_set2 = draw_samples(pred, pred_cov, self.num_samples)
            
            #? Compute energy loss
            target_set = torch.repeat_interleave(target.unsqueeze(0),
                                             self.num_samples,
                                             dim=0).reshape(-1, 4)
            first = self.loss_fn(
                sample_set1.reshape(-1, 4),
                target_set,
                None,
                reduction=reduction,
                avg_factor=avg_factor,
                **kwargs
            )
            second = self.loss_fn(
                sample_set1.reshape(-1, 4),
                sample_set2.reshape(-1, 4),
                None,
                reduction=reduction,
                avg_factor=avg_factor,
                **kwargs
            )
            
            loss_bbox = (first - 0.5 * second) / self.num_samples
            loss_bbox = probabilistic_weight * loss_bbox
            
            #* Loss annealing might be needed for regression
            if probabilistic_weight < 1.0:
                standard_loss_bbox = self.loss_fn(
                    pred,
                    target,
                    None,
                    reduction=reduction,
                    avg_factor=avg_factor,
                    **kwargs
                )
                loss_bbox += (1.0 - probabilistic_weight) * standard_loss_bbox
        else:
            loss_bbox = pred.sum()
        return self.loss_weight * loss_bbox
    
@LOSSES.register_module()
class SampleIoULoss(IoULoss):
    """IoU loss for bounding boxes samples drawn from multivariate Gaussian distributions.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to "mean".
        num_samples (int, optional): Number of samples drawn from the
            multivariate Gaussian distribution defined by predicted box
            means and covariance matrices. Defaults to 1000.
        loss_weight (float, optional): The weight of loss.
    """
    def __init__(self, 
                 num_samples=1000,
                 **kwargs):
        super(SampleIoULoss, self).__init__(**kwargs)
        self.num_samples = num_samples
    
    def forward(self,
                pred,
                pred_cov,
                target,
                priors,
                decoder,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            prev_cov (torch.Tensor): The covariance of the prediction in log scale.
            target (torch.Tensor): The learning target of the prediction.
            prior (torch.Tensor): The prior boxes of the prediction.
            decoder (func): The decoder to convert sample deltas into
                sample bboxes. Defaults to None.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        if pred_cov is None:
            #? Use deterministic IoULoss
            pred = decoder(priors, pred)
            return super().forward(pred, 
                                   target, 
                                   weight=weight, 
                                   avg_factor=avg_factor, 
                                   reduction_override=reduction_override)
        
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        
        #? Apply weight early to reduce memory usage in sampling
        if weight is not None:
            pred = pred * weight
            nonzero_mask = (pred != 0).any(dim=1)
            pred = pred[nonzero_mask]
            pred_cov = pred_cov[nonzero_mask]
            target = target[nonzero_mask]
            priors = priors[nonzero_mask]

        if pred.numel() > 0:
            sample_set1, sample_set2 = draw_samples(pred, pred_cov, self.num_samples)
            
            #? Decode samples to get sample boxes
            sample_set1 = decoder(priors, sample_set1)
            sample_set2 = decoder(priors, sample_set2)
            
            #? Compute sample IoU loss using energy loss
            target = torch.repeat_interleave(target.unsqueeze(0),
                                             self.num_samples,
                                             dim=0).reshape(-1, 4)
            first = iou_loss(
                sample_set1.reshape(-1, 4),
                target,
                None,
                mode=self.mode,
                eps=self.eps,
                reduction=reduction,
                avg_factor=avg_factor,
                **kwargs)
            second = iou_loss(
                sample_set1.reshape(-1, 4),
                sample_set2.reshape(-1, 4),
                None,
                mode=self.mode,
                eps=self.eps,
                reduction=reduction,
                avg_factor=avg_factor,
                **kwargs)
            loss_bbox = (first - 0.5 * second) / self.num_samples
        else:
            loss_bbox = pred.sum()
        return self.loss_weight * loss_bbox