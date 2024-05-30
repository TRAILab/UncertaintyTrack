from abc import ABC, abstractmethod
import torch
import numpy as np

from mmdet.core import bbox_overlaps
from mmtrack.core.bbox import bbox_cxcyah_to_xyxy


class Distance(ABC):
    """Abstract class for distribution distance functions.
    Properties:
        mode (str, optional): Flag for training or inference to determine if
            covariance matrices are in log-scale or not. If 'train', then they
            are in log-scale. If 'inference', then they are not in log-scale.
            Defaults to 'inference'.
        log_scale (bool, optional): Flag to determine if the distance function
            for diagonal covariance matrices is expects the covariance matrices
            to be in log-scale or not. Defaults to False.
    """
    def __init__(self):
        self.log_scale = False

    def __call__(self, mu_p, sigma_p, mu_q, sigma_q, one2one=False):
        """Computes the one-to-one or pairwise distance between two sets of 
        multivariate Gaussian distributions. Diagonal covariance matrices are 
        assumed to be only used during training, so it is implemented in PyTorch only.

        Args:
            mu_p (torch.Tensor | np.ndarray): Mean of distribution P of shape (M, K).
                If during training, mu_p must be a torch.Tensor.
            sigma_p (torch.Tensor | np.ndarray): Covariance matrix of distribution P.
                If diagonal, shape is (M, K). If full, shape is (M, K, K).
                If during training, sigma_p must be a torch.Tensor and is in log-scale.
            mu_q (torch.Tensor | np.ndarray): Mean of distribution Q of shape (N, K).
                If during training, mu_q must be a torch.Tensor.
            sigma_q (torch.Tensor | np.ndarray): Covariance matrix of distribution Q.
                If diagonal, shape is (N, K). If full, shape is (N, K, K).
                If during training, sigma_q must be a torch.Tensor and is in log-scale.
            one2one (bool, optional): Flag to determine if the distance is
                computed in a one-to-one fashion or in a pairwise manner between
                two sets of distributions. If True, then the distance is only
                computed between two distributions in order. If False, then the distance
                is computed between all pairs of distributions. Defaults to False.

        Returns:
            (torch.Tensor | np.ndarray): Pairwise distance of shape (M, N) or (M,) if
                one2one is True.
        """
        #? Check shapes
        if mu_p.shape[1:] != mu_q.shape[1:]:
            raise ValueError('Distributions P and Q should have same mean dimension.')
        if sigma_p is not None and sigma_q is not None:
            if sigma_p.shape[1:] != sigma_q.shape[1:]:
                raise ValueError('Distributions P and Q should have same covariance matrix shape.')

        #? Check if full or diagonal covariance matrices
        if len(sigma_p.shape[1:]) != 1 and len(sigma_p.shape[1:]) != 2:
            raise ValueError('Covariance matrices must be either diagonal or full.\
                Got shape {}.'.format(sigma_p.shape))
        else:
            is_diagonal = len(sigma_p.shape[1:]) == 1
            is_numpy = isinstance(mu_p, np.ndarray)
            if is_numpy:
                mu_p = torch.from_numpy(mu_p)
                mu_q = torch.from_numpy(mu_q)
                sigma_p = torch.from_numpy(sigma_p) if sigma_p is not None else None
                sigma_q = torch.from_numpy(sigma_q) if sigma_q is not None else None
            if not one2one:
                #? Expand dimensions to enable broadcasting
                mu_p = mu_p.unsqueeze(1)
                sigma_p = sigma_p.unsqueeze(1) if sigma_p is not None else None
                mu_q = mu_q.unsqueeze(0)
                sigma_q = sigma_q.unsqueeze(0) if sigma_q is not None else None
            #? Diagonal
            if is_diagonal:
                if self.log_scale and self.mode == 'inference':
                    sigma_p = torch.log(sigma_p) if sigma_p is not None else None
                    sigma_q = torch.log(sigma_q) if sigma_q is not None else None
                dist = self.forward_diag(mu_p, sigma_p, mu_q, sigma_q)
            else:
                dist = self.forward_full(mu_p, sigma_p, mu_q, sigma_q)
            dist = dist.numpy() if is_numpy and isinstance(dist, torch.Tensor) \
                                else dist
            return dist

    @abstractmethod
    def forward_full(self, mu_p, sigma_p, mu_q, sigma_q):
        pass
    
    @abstractmethod
    def forward_diag(self, mu_p, sigma_p, mu_q, sigma_q):
        pass
    
    def get_threshold(self, threshold):
        return threshold
    
class KLDivergence(Distance):
    def __init__(self, mode='inference'):
        super(KLDivergence, self).__init__()
        self.mode = mode
        self.log_scale = True

    #TODO: simplify using torch & cholesky decomposition
    def forward_full(self, mu_p, sigma_p, mu_q, sigma_q):
        raise NotImplementedError(
            "KL Divergence for full covariance matrices is not implemented.")
        K = mu_p.shape[-1]
        sigma_q_inv = np.linalg.inv(sigma_q)
        residual = np.expand_dims((mu_p - mu_q),axis=-1)
        trace = np.trace(sigma_q_inv @ sigma_p, axis1=-2, axis2=-1)
        maha = (np.swapaxes(residual, -1, -2) @ sigma_q_inv @ residual).squeeze((-1, -2))
        _, logdet_p = np.linalg.slogdet(sigma_p)
        _, logdet_q = np.linalg.slogdet(sigma_q)
        kl = 0.5 * (trace + maha - K + logdet_q - logdet_p)
        return kl

    def forward_diag(self, mu_p, sigma_p, mu_q, sigma_q):
        #TODO: convert input to log-scale if not training
        raise NotImplementedError(
            "KL Divergence for diagonal covariance matrices is not implemented.")
        K = mu_p.shape[-1]
        sigma_q_inv = torch.exp(-sigma_q)
        residual = mu_p - mu_q
        trace = torch.exp(-sigma_q + sigma_p).sum(dim=-1)
        log_p_q = (sigma_q - sigma_p).sum(dim=-1)
        maha = (residual * sigma_q_inv * residual).sum(dim=-1)
        kl = 0.5 * (trace + maha - K + log_p_q)
        return kl

class JRDivergence(Distance):
    def __init__(self, mode='inference', beta=0.85):
        super(JRDivergence, self).__init__()
        self.mode = mode
        self.beta = beta
        self.log_scale = True

    #TODO: simplify using torch & cholesky decomposition
    #* sqrtm =  cholesky factorization of the matrix
    def forward_full(self, mu_p, sigma_p, mu_q, sigma_q):
        raise NotImplementedError(
            "JR Divergence for full covariance matrices is not implemented.")
        K = mu_p.shape[-1]
        sigma_p_inv = np.linalg.inv(sigma_p)
        sigma_q_inv = np.linalg.inv(sigma_q)
        residual = np.expand_dims((mu_q - mu_p),axis=-1)
        maha = np.sqrt(
                np.swapaxes(residual, -1, -2) @ (sigma_p_inv + sigma_q_inv) \
                    @ residual).squeeze()
        vectorized_sqrtm = np.vectorize(sqrtm, signature='(n,m)->(n,m)')
        sigma_p_inv_sqrt =  vectorized_sqrtm(sigma_p)
        sigma_p_inv_sqrt = np.linalg.inv(sigma_p_inv_sqrt)
        squared_geo_mean = sigma_p_inv_sqrt @ sigma_q @ sigma_p_inv_sqrt
        #* np.vectorize and logm causes "buffer source array is read-only" error
        sgm_shape = squared_geo_mean.shape
        squared_geo_mean = squared_geo_mean.reshape(-1, K, K)
        geo_mean_log = np.array([logm(sgm) for sgm in squared_geo_mean])
        geo_mean_log = geo_mean_log.reshape(sgm_shape)
        riemannian = np.linalg.norm(geo_mean_log, 
                                    ord='fro',
                                    axis=(-2, -1))
        jr = (1 - self.beta) * maha + self.beta * riemannian
        return jr
    
    def forward_diag(self, mu_p, sigma_p, mu_q, sigma_q):
        #TODO: convert input to log-scale if not training
        raise NotImplementedError(
            "JR Divergence for diagonal covariance matrices is not implemented.")
        K = mu_p.shape[-1]
        sigma_p_inv = torch.exp(-sigma_p)
        sigma_q_inv = torch.exp(-sigma_q)
        residual = mu_q - mu_p
        maha = (residual * (sigma_p_inv + sigma_q_inv) * residual).sum(dim=-1)
        maha = maha.clamp(min=1e-12).sqrt()
        riemannian = torch.norm(-sigma_p + sigma_q, dim=-1)
        jr = (1 - self.beta) * maha + self.beta * riemannian
        return jr

class Wasserstein(Distance):
    def __init__(self, mode='inference'):
        super(Wasserstein, self).__init__()
        self.mode = mode
    
    def forward_full(self, mu_p, sigma_p, mu_q, sigma_q):
        # residual = mu_p - mu_q
        # first = (residual * residual).sum(dim=-1)
        
        # cholesky_q = torch.cholesky(sigma_q)
        # inner = cholesky_q @ sigma_p @ torch.transpose(cholesky_q, -1, -2)
        # cholesky_inner = torch.cholesky(inner)
        # trace_p = torch.trace(sigma_p, axis1=-2, axis2=-1)
        # trace_q = torch.trace(sigma_q, axis1=-2, axis2=-1)
        # trace_inner = torch.trace(cholesky_inner, axis1=-2, axis2=-1)   #! not sure
        # second = trace_p + trace_q - 2 * trace_inner
        raise NotImplementedError(
            "Wasserstein distance is not implemented yet for full covariance matrices."
        )
        
        return first + second
    
    def forward_diag(self, mu_p, sigma_p, mu_q, sigma_q):
        residual = mu_p - mu_q
        first = (residual * residual).sum(dim=-1)
        inner = (sigma_p + sigma_q).clamp(min=1e-12).sqrt()
        trace_p = sigma_p.sum(dim=-1)
        trace_q = sigma_q.sum(dim=-1)
        trace_inner = inner.sum(dim=-1)
        second = trace_p + trace_q - 2 * trace_inner
        return first + second

class Bhattacharyya(Distance):
    def __init__(self, mode='inference'):
        super(Bhattacharyya, self).__init__()
        self.mode = mode
        self.log_scale = True

    #TODO: simplify using torch & cholesky decomposition
    def forward_full(self, mu_p, sigma_p, mu_q, sigma_q):
        raise NotImplementedError(
            "Bhattacharyya distance is not implemented yet for full covariance matrices."
        )
        #* Squared Mahalanobis distance
        # sigma_q_inv = np.linalg.inv(sigma_q)
        # residual = np.expand_dims((mu_p - mu_q),axis=-1)
        # first = (np.swapaxes(residual, -1, -2) @ sigma_q_inv @ residual).squeeze((-1, -2))
        
        # sigma = 0.5 * (sigma_p + sigma_q)
        # _, logdet_sigma = np.linalg.slogdet(sigma)
        # _, logdet_sigma_p = np.linalg.slogdet(sigma_p)
        # _, logdet_sigma_q = np.linalg.slogdet(sigma_q)
        # second = logdet_sigma - 0.5 * logdet_sigma_p - 0.5 * logdet_sigma_q
        # return 0.125 * first + 0.5 * second
    
    def forward_diag(self, mu_p, sigma_p, mu_q, sigma_q):
        sigma = torch.logaddexp(0.5 * sigma_p, 0.5 * sigma_q)
        sigma_inv = torch.exp(-sigma)
        residual = mu_p - mu_q
        first = (residual * sigma_inv * residual).sum(dim=-1)
        
        logdet_sigma = sigma.sum(dim=-1)
        logdet_sigma_p = sigma_p.sum(dim=-1)
        logdet_sigma_q = sigma_q.sum(dim=-1)
        second = logdet_sigma - 0.5 * logdet_sigma_p - 0.5 * logdet_sigma_q
        return 0.125 * first + 0.5 * second
    
    def get_threshold(self, bc_threshold):
        if bc_threshold < 0. or bc_threshold > 1.:
            raise ValueError(f'Threshold for Bhattacharyya must be in [0,1], \
                                but got {bc_threshold}')
        #* BD = -ln(BC)
        return -np.log(bc_threshold)
    
class Hellinger(Bhattacharyya):
    def __init__(self, mode='inference'):
        super(Hellinger, self).__init__(mode=mode)
    
    def forward_full(self, mu_p, sigma_p, mu_q, sigma_q):
        raise NotImplementedError(
            "Hellinger distance is not implemented yet for full covariance matrices."
        )

    def forward_diag(self, mu_p, sigma_p, mu_q, sigma_q):
        bhattacharyya = super(Hellinger, self).forward_diag(mu_p, sigma_p, mu_q, sigma_q)
        hell = (1 - torch.exp(-bhattacharyya)).clamp(min=1e-12).sqrt()
        return hell
    
    def get_threshold(self, bc_threshold):
        return np.sqrt(1 - bc_threshold)
        
class Mahalanobis(Distance):
    #* Squared Mahalanobis distance
    def __init__(self, mode='inference'):
        super(Mahalanobis, self).__init__()
        self.mode = mode
        self.log_scale = True

    def forward_full(self, mu_p, sigma_p, mu_q, sigma_q=None):
        chol_p = torch.linalg.cholesky(sigma_p)
        residual = mu_p - mu_q
        z = torch.linalg.solve_triangular(
            chol_p,
            residual.unsqueeze(-1),
            upper=False).squeeze(-1)
        return torch.sum(z*z, dim=-1)

    def forward_diag(self, mu_p, sigma_p, mu_q, sigma_q=None):
        sigma_p_inv = torch.exp(-sigma_p)
        residual = mu_p - mu_q
        maha = (residual * sigma_p_inv * residual).sum(dim=-1)
        return maha
    
    def get_threshold(self, *args):
        #* Fixed threshold based on chi-square distribution;
        #* See kalman filter for more details
        return 9.4877

class AIRM(Distance):
    def __init__(self, mode='inference'):
        super(AIRM, self).__init__()
        self.mode = mode
        self.log_scale = True
    
    def forward_full(self, mu_p, sigma_p, mu_q, sigma_q):
        raise NotImplementedError(
            "AIRM is not implemented yet for full covariance matrices."
        )
    
    def forward_diag(self, mu_p, sigma_p, mu_q, sigma_q):
        diff = sigma_q - sigma_p
        airm = torch.norm(diff, dim=-1)
        return airm
    
class GIoU:
    #TODO: document
    def __init__(self, with_sampling=False, num_samples=100, mode='giou'):
        self.with_sampling = with_sampling
        self.num_samples = num_samples
        self.mode = mode
    
    def __call__(self, bboxes1, bbox_covs1, bboxes2, bbox_covs2, one2one=False):
        if self.with_sampling:
            mvn1 = torch.distributions.Normal(bboxes1,
                                              scale=bbox_covs1.sqrt())
            mvn2 = torch.distributions.Normal(bboxes2,
                                              scale=bbox_covs2.sqrt())
            bbox_mean1 = bbox_cxcyah_to_xyxy(
                            mvn1.sample((self.num_samples,)).mean(0))
            bbox_mean2 = bbox_cxcyah_to_xyxy(
                            mvn2.sample((self.num_samples,)).mean(0))
        else:
            if bboxes1.dim() == 1:
                bboxes1 = bboxes1.unsqueeze(0)
            bbox_mean1 = bbox_cxcyah_to_xyxy(bboxes1)
            bbox_mean2 = bbox_cxcyah_to_xyxy(bboxes2)
        
        distance = 1 - bbox_overlaps(bbox_mean1, bbox_mean2, mode=self.mode, is_aligned=one2one)
        
        if distance.dim() > 1:
            distance = distance.mean(0)
        if distance.device != 'cpu':
            distance = distance.cpu()
        return distance.numpy()