import torch
import numpy as np


def energy_score(det_bboxes, det_bbox_covs, gt_bboxes, num_samples=1000, max_per_gpu=100000):
    """Computes the energy score between true-positive predictions 
    and ground truth targets.

    Args:
        det_bboxes (torch.tensor)   : Predicted bounding boxes. Shape (n, 4).
        det_bbox_covs (torch.tensor): Predicted bounding box covariance matrices.
                                        Shape (n, 4, 4).
        gt_bboxes (torch.tensor)    : Ground truth bounding boxes. Shape (n, 4).
        num_samples (int, optional) : Number of samples to use for computation.
                                        Defaults to 1000.
        max_per_gpu (int, optional) : Maximum number of detections to consider at once.
                                        Defaults to 100000.
    """
    #? Split input to ensure we don't run out of memory
    #* 100000 * 1000 fits under 12GB of GPU memory.
    det_bboxes_split = torch.split(det_bboxes, max_per_gpu, dim=0)
    det_bbox_covs_split = torch.split(det_bbox_covs, max_per_gpu, dim=0)
    gt_bboxes_split = torch.split(gt_bboxes, max_per_gpu, dim=0)
    energy_scores = []
    for _det_bboxes, _det_bbox_covs, _gt_bboxes in zip(det_bboxes_split, det_bbox_covs_split, gt_bboxes_split):
        _det_bbox_cov_chols = torch.linalg.cholesky(_det_bbox_covs.float() + 
                                                    1e-4 * torch.eye(_det_bbox_covs.shape[-1], 
                                                                    device=_det_bbox_covs.device).float())
        mvn = torch.distributions.multivariate_normal.MultivariateNormal(
            loc=_det_bboxes.float(), scale_tril=_det_bbox_cov_chols)
        samples = mvn.sample((num_samples + 1,))
        sample_set1 = samples[:-1]
        sample_set2 = samples[1:]
        
        es = torch.norm(sample_set1 - _gt_bboxes.float(), dim=-1).mean(0) \
                - 0.5 * torch.norm(sample_set1 - sample_set2, dim=-1).mean(0)
        es = es.mean()
        energy_scores.append(es)
        del _det_bboxes, _det_bbox_covs, _gt_bboxes, \
            mvn, samples, sample_set1, sample_set2
    return torch.stack(energy_scores).mean().item()

def negative_loglikelihood(det_bboxes, det_bbox_covs, gt_bboxes):
    """Computes the negative log-likelihood between true-positive predictions
    and ground truth targets.

    Args:
        det_bboxes (torch.tensor)   : Predicted bounding boxes. Shape (n, 4).
        det_bbox_covs (torch.tensor): Predicted bounding box covariance matrices.
                                        Shape (n, 4, 4).
        gt_bboxes (torch.tensor)    : Ground truth bounding boxes. Shape (n, 4).
    """
    det_bbox_cov_chols = torch.linalg.cholesky(det_bbox_covs.float() + 
                                                    1e-4 * torch.eye(det_bbox_covs.shape[-1], 
                                                                    device=det_bbox_covs.device).float())
    mvn = torch.distributions.multivariate_normal.MultivariateNormal(
        loc=det_bboxes.float(), scale_tril=det_bbox_cov_chols)
    nll = -mvn.log_prob(gt_bboxes.float()).mean()
    return nll.item()

def entropy(det_bbox_covs, reduction='mean'):
    """Computes the average entropy of the predicted bounding box covariance matrices.
    
    Args:
        det_bbox_covs (torch.tensor): Predicted bounding box covariance matrices.
                                        Shape (n, 4, 4).
        reduction (str, optional)        : Mode of entropy computation. Defaults to 'mean'.
                                            Options: 'mean', 'minimum', 'maximum', 'std'
    """
    D = det_bbox_covs.shape[-1]
    _, logdet = torch.slogdet(det_bbox_covs + \
                                1e-4 * (torch.eye(D, device=det_bbox_covs.device)
                                        .float()))
    entropy = 0.5 * logdet + 0.5 * D * (1 + torch.log(torch.tensor(2*np.pi)))
    
    if reduction == 'minimum':
        return entropy.min().item()
    elif reduction == 'maximum':
        return entropy.max().item()
    elif reduction == 'std':
        return entropy.std().item()
    else:
        return entropy.mean().item()