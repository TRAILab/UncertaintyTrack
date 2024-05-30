import torch


def compute_probabilistic_weight(weight):
    """Compute the probabilistic weight for loss annealing.
    
    Args:
        weight (float): The initial weight for loss annealing.
    
    Returns:
        float: The probabilistic weight.
    """
    return (100**weight-1.0)/(100.0-1.0)

def clamp_log_variance(log_covariance, min_log_variance=-9.21, max_log_variance=9.21):
    """Clamp log variance.
    
    Args:
        log_covariance (torch.Tensor): The log variance to clamp.
        min_log_variance (float, optional): The minimum log variance. Defaults to -9.21.
        max_log_variance (float, optional): The maximum log variance. Defaults to 9.21.
    
    Returns:
        torch.Tensor: The clamped log variance.
    """
    log_covariance_diag = torch.clamp(log_covariance[:, :4], min_log_variance, max_log_variance)
    return torch.cat([log_covariance_diag, log_covariance[:, 4:]], dim=1)