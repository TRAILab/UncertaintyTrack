import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.builder import LOSSES
from mmdet.models.losses import CrossEntropyLoss


@LOSSES.register_module()
class SampleCrossEntropyLoss(CrossEntropyLoss):
    
    def __init__(self,
                 num_samples=10,
                 attenuated=False,
                 **kwargs):
        """CrossEntropyLoss from mmdet extended to support sampling
        from distribution defined by mean and variance.

        Args:
            num_samples (int, optional): Number of samples drawn from the 
                Gaussian distribution defined by mean and covariance. 
                Defaults to 10.
            attenuated (bool, optional): Whether to use loss attenuation.
                Defaults to False.
        """
        self.num_samples = num_samples
        self.attenuated = attenuated
        super(SampleCrossEntropyLoss, self).__init__(**kwargs)
    
    def forward(self,
                cls_logits,
                cls_logits_var,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=None,
                **kwargs):
        """Forward function.

        Args:
            cls_logits (torch.Tensor): The predicted logits.
            cls_logits_var (torch.Tensor | None): The predicted logits variance in log scale. 
                The shape should be the same as pred.
            label (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): Sample-wise loss weight.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The method used to reduce the
                loss. Options are "none", "mean" and "sum".
            ignore_index (int | None): The label index to be ignored.
                If not None, it will override the default value. Default: None.
        Returns:
            torch.Tensor: The calculated loss.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if ignore_index is None:
            ignore_index = self.ignore_index

        if self.class_weight is not None:
            class_weight = cls_logits.new_tensor(
                self.class_weight, device=cls_logits.device)
        else:
            class_weight = None
        probabilistic_weight = probabilistic_weight if self.attenuated else 1.0
        
        if cls_logits_var is not None:
            assert cls_logits.shape == cls_logits_var.shape
            
            #? Produce normal samples using logits as the mean and logits_var as the variance
            logits_std = torch.sqrt(torch.exp(cls_logits_var))    #(N, num_classes)
            uni_normal_dists = torch.distributions.normal.Normal(
                                    cls_logits, scale=logits_std) 
            logits_samples = uni_normal_dists.rsample((self.num_samples,)) #(num_samples, N, num_classes) 
            logits_samples = logits_samples.reshape(
                                logits_samples.shape[1]*self.num_samples,
                                logits_samples.shape[2]) #(N*num_samples, num_classes)
            
            #? Produce copies of the label to match the number of samples
            label_repeated = torch.unsqueeze(label, 0)    #(1, N)
            label_repeated = torch.repeat_interleave(label_repeated, self.num_samples, dim=0).reshape(
                                    label_repeated.shape[1]*self.num_samples)  #(N*num_samples)
            if weight is not None:
                weight_repeated = torch.unsqueeze(weight, 0)    #(1, N)
                weight_repeated = torch.repeat_interleave(weight_repeated, self.num_samples, dim=0).reshape(
                                        weight_repeated.shape[1]*self.num_samples)  #(N*num_samples)
            else:
                weight_repeated = None
            
            loss_cls = self.cls_criterion(
                logits_samples,
                label_repeated,
                weight_repeated,
                class_weight=class_weight,
                reduction=reduction,
                avg_factor=avg_factor,
                ignore_index=ignore_index,
                avg_non_ignore=self.avg_non_ignore,
                **kwargs) / self.num_samples
        
        #? Compute standard cross entropy loss
        if cls_logits_var is None or probabilistic_weight < 1.0:
            standard_loss_cls = self.cls_criterion(
                cls_logits,
                label,
                weight,
                class_weight=class_weight,
                reduction=reduction,
                avg_factor=avg_factor,
                ignore_index=ignore_index,
                avg_non_ignore=self.avg_non_ignore,
                **kwargs)
            if cls_logits_var is None:
                loss_cls = standard_loss_cls
            else:
                loss_cls =  (1 - probabilistic_weight) * standard_loss_cls + \
                            probabilistic_weight * loss_cls
        
        return self.loss_weight * loss_cls