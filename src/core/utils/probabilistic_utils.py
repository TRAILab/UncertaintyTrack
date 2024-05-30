import torch
import numpy as np
from scipy.stats import norm, chi2


def compute_mean_covariance_torch(samples):
    """
    Function for efficient computation of mean and covariance matrix from samples in pytorch.

    Args:
        samples(Tensor | list(tensor)): tensor of shape M x N x k where M is the number of 
            Monte-Carlo samples or a list of M tensors of shape N x k.

    Returns:
        mean(Tensor): a N x k tensor containing the predicted mean.
        covariance(Tensor): a N x k x k tensor containing the predicted covariance matrix.

    """
    if isinstance(samples, torch.Tensor):
        num_samples = samples.shape[0]
    elif isinstance(samples, list):
        num_samples = len(samples)
        samples = torch.stack(samples, 0) # M x N x k
    else:
        raise ValueError("Samples must be a tensor or a list of tensors.")

    #? Compute Mean
    mean = torch.mean(samples, 0, keepdim=True) # 1 x N x k

    #? Compute Covariance
    residuals = torch.unsqueeze(samples - mean, 3)  # M x N x k x 1
    covariance = torch.matmul(residuals, torch.transpose(residuals, 2, 3))  # M x N x k x k
    covariance = torch.sum(covariance, 0) / (num_samples - 1)   # N x k x k

    return mean.squeeze(0), covariance


def gaussian_entropy(sigma, is_diagonal=False):
    """Compute the entropy of Multivariate Gaussian distribution.

    Args:
        sigma (np.array): The covariance matrices. Shape (*, D, D).
        is_diagonal (bool): Whether the covariance matrices are diagonal.
    """
    D = sigma.shape[-1]
    if is_diagonal:
        sigma_diag = np.diagonal(sigma, axis1=-2, axis2=-1)
        logdet = np.log(sigma_diag).sum(-1)
    else:
        _, logdet = np.linalg.slogdet(sigma + 1e-4 * np.identity(D))
    entropy = 0.5 * logdet + 0.5 * D * (1 + np.log(2*np.pi))
    return entropy

def max_eigenvalue(pd_mat, is_diagonal=False):
    """Compute the maximum eigenvalue of positive-definite matrices.

    Args:
        pd_mat (np.array): The pd matrices. Shape (*, D, D).
        is_diagonal (bool): Whether the matrices are diagonal.
    """
    if is_diagonal:
        max_eigenvalue = np.diagonal(pd_mat, axis1=-2, axis2=-1).max(-1)
    else:
        max_eigenvalue = np.linalg.eigvalsh(pd_mat)[-1]
    return max_eigenvalue

def trace(pd_mat, is_diagonal=False):
    """Compute the trace of positive-definite matrices.

    Args:
        pd_mat (np.array): The pd matrices. Shape (*, D, D).
        is_diagonal (bool): Whether the matrices are diagonal.
    """
    if is_diagonal:
        trace = np.diagonal(pd_mat, axis1=-2, axis2=-1).sum(-1)
    else:
        trace = np.trace(pd_mat, axis1=-2, axis2=-1)
    return trace


def compute_mean_variance_torch(samples):
    """
    Function for efficient computation of mean and variance from samples in pytorch.

    Args:
        samples(Tensor | list(tensor)): tensor of shape M x N x k where M is the number of 
            Monte-Carlo samples or a list of M tensors of shape N x k

    Returns:
        mean(Tensor): a N x k tensor containing the predicted mean.
        variance(Tensor): a N x k tensor containing the predicted variance.
    """
    if isinstance(samples, list):
        samples = torch.stack(samples, 0) # M x N x k
    elif not isinstance(samples, torch.Tensor):
        raise ValueError("Samples must be a tensor or a list of tensors.")

    if samples.dim() < 3:   #* No samples
        return samples, None
    else:
        num_samples = samples.shape[0]
    
    #? Compute Mean
    mean = samples.mean(0) # N x k

    #? Compute Variance
    residuals = samples - mean
    variance = torch.pow(residuals, 2).sum(0) / (num_samples - 1)   # N x k
    
    return mean, variance


def covariance2cholesky(covariance):
    """Transforms the predicted covariance matrices to
    the Cholesky decomposition of the covariance matrices.

    Args:
        covariance (Tensor): Predicted covariance matrices in log-scale
            of shape (K, 4) or (K, 10).
            If (K, 4), then the covariance matrices are assumed to be diagonal.
            If (K, 10), then the covariance is assumed to be lower triangular
            factor matrix of cholesky decomposition.
    Returns:
        cholesky (K, 4, 4): Cholesky factor matrices
    """
    if covariance.ndim > 2:
        raise ValueError("Covariance matrix must be represented as flat vector.")
    
    covariance_exp = torch.exp(covariance)
    
    #? Embed diagonal elements of covariance matrix first
    diag = covariance_exp[:, :4]
    cholesky = torch.diag_embed(diag)
    
    if covariance_exp.shape[1] > 4:
        #? Embed off-diagonal elements of covariance matrix
        if covariance_exp.shape[1] != 10:
            raise ValueError("Covariance matrix must be of shape (K, 4) or (K, 10).")
        tril_indices = torch.tril_indices(row=4, col=4, offset=-1)
        cholesky[:, tril_indices[0], tril_indices[1]] = covariance_exp[:, 4:]
    else:
        #? Predicted diagonal elements are squared in nll
        #? Need to take square root for cholesky decomposition
        cholesky = torch.sqrt(cholesky)

    return cholesky


def product_of_gaussians(means, covariances):
    """Compute the mean and covariance of the product of N multivariate Gaussian distributions.
    See Section 8.1.8 of "The Matrix Cookbook" for more details.
    
    Args:
        means (nd array): Means of the N Gaussian distributions.
            Has shape (N, D).
        covariances (nd array): Covariance matrices of the N Gaussian distributions.
            Has shape (N, D, D).
    Returns:
        product_mean (nd array): Mean of the product of the N Gaussian distributions.
            Has shape (D,).
        product_covariance (nd array): Covariance matrix of the product of the 
            N Gaussian distributions. Has shape (D, D).
    """
    precision_mats = np.linalg.inv(covariances)
    product_covariance = np.linalg.inv(precision_mats.sum(0))
    product_mean = (precision_mats @ np.expand_dims(means, -1)).sum(0)
    product_mean = np.squeeze(product_covariance @ product_mean)
    
    return product_mean, product_covariance

def covariance_intersection(means, covariances):
    """Implementation of the Fast covariance intersection algorithm.
    See "Improved Fast Covariance Intersection for Distributed Data Fusion"
    (Franken, D. et al., 2005) for more details.
    
    Args:
        means (nd array): Means of the N Gaussian distributions.
            Has shape (N, D).
        covariances (nd array): Covariance matrices of the N Gaussian distributions.
            Has shape (N, D, D).
    Returns:
        fused_mean (nd array): Fused mean of the N Gaussian distributions.
            Has shape (D,).
        fused_covariance (nd array): Fused covariance matrix of the N Gaussian 
            distributions. Has shape (D, D).
    """
    precision_mats = np.linalg.inv(covariances)
    precision_diff = precision_mats.sum(0) - precision_mats
    precision_det = np.linalg.det(precision_mats)
    total_precision_det = np.linalg.det(precision_mats.sum(0))
    precision_diff_det = np.linalg.det(precision_diff)
    weights = (total_precision_det - precision_diff_det + precision_det) / \
        (precision_mats.shape[0] * total_precision_det + \
            (precision_det - precision_diff_det).sum(0))
    weighted_precisions = (
        np.expand_dims(weights, (1,2)) * precision_mats)
    
    fused_covariance = np.linalg.inv(weighted_precisions.sum(0))
    fused_mean = np.squeeze(fused_covariance @ (
                    (weighted_precisions @ 
                        np.expand_dims(means, -1)).sum(0)),
                    axis=-1)
    
    return fused_mean, fused_covariance

def get_ellipse_box(xy, covs, q=0.95):
    """Get the top-left and bottom-right corners of the tight bounding box of the ellipse
    that defines the confidence region (specified by `q`) of the distribution.
    Notes:
        - The ellipse is assumed to be centered at xy.
        - Large eigenvalue corresponds to the major axis.
        - Angle is the angle between the major axis and the x-axis.
        - Assumes the ellipse is vertically symmetric, which is should be.

    Args:
        xy (torch.tensor): Center coordinates of the ellipse. Shape (N, 2).
        covs (torch.tensor): Covariance matrices of the ellipse. Shape (N, 2, 2).
        q (float, optional): Confidence level of the distribution, should be in (0, 1)
                                Defaults to 0.95.

    Returns:
        boxes (torch.tensor): Top-left and bottom-right corners of the bounding box of the ellipse.
                            Shape (N, 4).
    """
    q = np.asarray(q)
    r2 = chi2.ppf(q, 2)
    cx, cy = xy[..., 0], xy[..., 1]
    val, vec = torch.linalg.eigh(covs)  #* Returns in ascending order; eigenvectors are column vectors
    major_radius = torch.sqrt(val[:, 1] * r2)
    minor_radius = torch.sqrt(val[:, 0] * r2)
    theta = torch.atan2(*(torch.split(vec[..., 1].flip(-1), 
                                      [1, 1], dim=-1))
                        ).squeeze(-1)
    
    #? Compute corners of tight ellipse bounding box
    t1 = torch.atan(-minor_radius * torch.tan(theta) / major_radius)
    t2 = t1 + np.pi
    x1, x2 = [cx + major_radius * torch.cos(t) * torch.cos(theta) \
                - minor_radius * torch.sin(t) * torch.sin(theta) \
                for t in (t1, t2)]
    min_x, max_x = torch.minimum(x1, x2), torch.maximum(x1, x2)
    
    t1 = torch.atan(minor_radius / (major_radius * torch.tan(theta)))
    t2 = t1 + np.pi
    y1, y2 = [cy + minor_radius * torch.sin(t) * torch.cos(theta) \
                + major_radius * torch.cos(t) * torch.sin(theta) \
                for t in (t1, t2)]
    min_y, max_y = torch.minimum(y1, y2), torch.maximum(y1, y2)
    
    #? Verify vertical symmetry
    assert ((max_y - cy) - (cy - min_y)).abs().max() < 1e-2, \
        "The ellipse is not vertically symmetric."
    
    boxes = torch.stack([min_x, min_y, max_x, max_y], dim=-1)
    return boxes
