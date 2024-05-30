import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.builder import LOSSES
from mmdet.models.losses import FocalLoss
from mmdet.models.losses.focal_loss import sigmoid_focal_loss, py_focal_loss_with_prob


@LOSSES.register_module()
class SampleFocalLoss(FocalLoss):

    def __init__(self,
                 attenuated=False,
                 num_samples=10,
                 **kwargs):
        """FocalLoss from mmdet extended to support sampling
        from distribution defined by mean and variance.
        
        Args:
            attenuated (bool, optional): Whether to use loss attenuation.
                Defaults to False.
            num_samples (int, optional): Number of samples drawn from the 
                Gaussian distribution defined by mean and covariance. 
                Defaults to 10.
        """
        self.num_samples = num_samples
        self.attenuated = attenuated
        super(SampleFocalLoss, self).__init__(**kwargs)

    def forward(self,
                pred_logits,
                pred_logits_var,
                target,
                probabilistic_weight=1.0,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred_logits (torch.Tensor): The predicted logits.
            pred_logits_var (torch.Tensor | None): The predicted logits covariance in log scale. 
                The shape should be the same as pred.
            target (torch.Tensor): The learning label of the prediction.
            probabilistic_weight (float, optional): The probabilistic weight for 
                loss attenuation. Computed using current iteration and epoch. 
                Defaults to 1.0.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.activated:
            calculate_loss_func = py_focal_loss_with_prob
        else:
            if torch.cuda.is_available() and pred_logits.is_cuda:
                calculate_loss_func = sigmoid_focal_loss
            else:
                #* The following is for debugging; not implemented for Probabilistic Retinanet
                raise NotImplementedError()
        probabilistic_weight = probabilistic_weight if self.attenuated else 1.0
        
        if pred_logits_var is not None:
            #? Check shapes
            assert pred_logits.shape == pred_logits_var.shape
        
            #? Produce normal samples using logits as the mean and variance
            pred_logits_std = torch.sqrt(torch.exp(pred_logits_var))    #(N, num_classes)
            univariate_normal_dists = torch.distributions.normal.Normal(
                                        pred_logits, scale=pred_logits_std) 
            pred_logits_samples = univariate_normal_dists.rsample((self.num_samples,)) #(num_samples, N, num_classes) 
            pred_logits_samples = pred_logits_samples.reshape(pred_logits_samples.shape[1]*self.num_samples,
                                                        pred_logits_samples.shape[2]) #(N*num_samples, num_classes)
            
            #? Produce copies of the target classes to match the number of samples
            target_repeated = torch.unsqueeze(target, 0)    #(1, N)
            target_repeated = torch.repeat_interleave(target_repeated, self.num_samples, dim=0).reshape(
                                    target_repeated.shape[1]*self.num_samples)  #(N*num_samples)
            if weight is not None:
                weight_repeated = torch.unsqueeze(weight, 0)    #(1, N)
                weight_repeated = torch.repeat_interleave(weight_repeated, self.num_samples, dim=0).reshape(
                                        weight_repeated.shape[1]*self.num_samples)  #(N*num_samples)
            else:
                weight_repeated = None
            
            loss_cls = calculate_loss_func(
                pred_logits_samples,
                target_repeated,
                weight_repeated,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor) / self.num_samples

        #? Compute standard focal loss
        if pred_logits_var is None or probabilistic_weight < 1.0:
            standard_loss_cls = calculate_loss_func(pred_logits,
                                                    target,
                                                    weight,
                                                    gamma=self.gamma,
                                                    alpha=self.alpha,
                                                    reduction=reduction,
                                                    avg_factor=avg_factor)
            if pred_logits_var is None:
                loss_cls = standard_loss_cls
            else:
                loss_cls =  (1 - probabilistic_weight) * standard_loss_cls + \
                            probabilistic_weight * loss_cls

        return self.loss_weight * loss_cls