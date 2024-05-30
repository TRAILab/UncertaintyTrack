import torch
import numpy as np
from scipy.stats import norm, chi2

def get_ellipse_params(cov, q=None, nsig=2):
    """
    Computes the widths, heights and rotation angles of 2d ellipses for plotting.
    
    Parameters
    ----------
    cov : (*, 2, 2) array
        Covariance matrices.
    q : float, optional
        Confidence level, should be in (0, 1).
    nsig : int, optional
        Confidence level in unit of standard deviations.
        E.g. 1 stands for 68.3% and 2 stands for 95.4%.

    Returns
    -------
    widths, heights, rotations :
        The lengths of two axes and the rotation angles in degree
        for the ellipses (angle between major axis and x-axis).
    """
    if q is not None:
        q = np.asarray(q)
    elif nsig is not None:
        q = 2 * norm.cdf(nsig) - 1
    else:
        raise ValueError('One of `q` and `nsig` should be specified.')
    r2 = chi2.ppf(q, 2)
    if isinstance(cov, np.ndarray):
        vals, vecs = np.linalg.eigh(cov)  #* Returns in ascending order; eigenvectors are column vectors
        radius = (2 * np.sqrt(vals * r2)).astype(np.int32)  #* width: major axis (x); height: minor axis (y)
        heights, widths = radius[..., 0], radius[..., 1]
        rotations = np.degrees(np.arctan2(vecs[..., ::-1, 1][...,0],
                                        vecs[..., ::-1, 1][...,1]))
        rotations = rotations.astype(np.int32)
    elif isinstance(cov, torch.Tensor):
        vals, vecs = torch.linalg.eigh(cov)
        radius = (2 * torch.sqrt(vals * r2)).int()
        heights, widths = radius[..., 0], radius[..., 1]
        rotations = torch.rad2deg(torch.atan2(torch.flip(vecs,[1])[...,1, 0],
                                            torch.flip(vecs,[1])[..., 1, 1]))
        rotations = rotations.int()
    return widths, heights, rotations