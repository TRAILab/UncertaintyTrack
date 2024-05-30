import torch
import numpy as np

from mmdet.core import bbox2result

#* --------------------------------------------------------
#* Detection utils
#* --------------------------------------------------------

def bbox_and_cov2result(bboxes, bbox_covs, labels, num_classes):
    """
    Convert detection results to lists of numpy arrays.
    Adapted bbox2result function from mmdet to handle covariance matrices.
    Outputs dictionary instead of tuple to bypass mmdet's single_gpu_test function
    that checks for tuple output for masks.

    Args:
        bboxes (torch.Tensor | np.ndarray): shape (n, 5)
        bbox_covs (torch.Tensor | np.ndarray): shape (n, 4, 4)
        labels (torch.Tensor | np.ndarray): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        dict[str, list[ndarray]]: dictionary containing results of each class.
            - 'bbox': bbox results of each class
            - 'bbox_cov': bbox covariance results of each class
    """
    if bboxes.shape[0] == 0:
        bbox_out = [np.zeros((0, 5), dtype=np.float32) for i in range(num_classes)]
        bbox_cov_out = [np.zeros((0, 4, 4), dtype=np.float32) for i in range(num_classes)]
    else:
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.detach().cpu().numpy()
            assert isinstance(bbox_covs, torch.Tensor), \
                f'bbox_covs should be torch.Tensor, but got {type(bbox_covs)}'
            bbox_covs = bbox_covs.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
        bbox_out = [bboxes[labels == i, :] for i in range(num_classes)]
        bbox_cov_out = [bbox_covs[labels == i, :] for i in range(num_classes)]

    return {'bbox': bbox_out, 'bbox_cov': bbox_cov_out}

def bbox_cov_xyxy_to_cxcyah(cov_xyxy, a_var=1e-1):
    """Convert covariance matrices of bbox coordinates from (x1, y1, x2, y2) to (cx, cy, a, h),
    where a is the aspect ratio and h is the height.

    Args:
        cov_xyxy (Tensor): Shape (n, 4, 4) for covariance matrices in (x1, y1, x2, y2) format.
        a_var (float): Variance of aspect ratio. Default: 1e-2. Assumed to be constant.

    Returns:
        Tensor: Converted covariance matrices in (cx, cy, a, h) format.
    """
    #? Jacobian matrix for conversion from (x1, y1, x2, y2) to (cx, cy, a, h)
    #* Simplify non-linearity with aspect ratio by assuming a constant variance
    J = torch.tensor([
        [0.5, 0, 0.5, 0],
        [0, 0.5, 0, 0.5],
        [0, 0, 0, 0],
        [0, -1, 0, 1]
    ], dtype=cov_xyxy.dtype, device=cov_xyxy.device)
    
    #* Covariance matrix conversion using Jacobian: cov_cxcyah = J * cov_xyxy * J^T
    cov_cxcyah = torch.matmul(torch.matmul(J, cov_xyxy), J.T)
    cov_cxcyah[:, 2, 2] = a_var
    
    #? aspect ratio is assumed to be constant
    #? set elements to zero to preserve positive definiteness
    cov_cxcyah[:, 3, :3] = 0
    cov_cxcyah[:, :3, 3] = 0
    # cov_cxcyah = torch.diag_embed(torch.diagonal(cov_cxcyah, dim1=-2, dim2=-1))
    
    return cov_cxcyah

# def bbox_cov_cxcyah_to_xyxy(cov_cxcyah, bbox_cxcyah):
#     """Convert covariance matrices of bbox coordinates from (cx, cy, a, h) to (x1, y1, x2, y2),
#     where a is the aspect ratio and h is the height.

#     Args:
#         cov_cxcyah (Tensor): Shape (n, 4, 4) for covariance matrices in (cx, cy, a, h) format.
#         bbox_cxcyah (Tensor): Shape (n, 4) for bbox coordinates in (cx, cy, a, h) format.

#     Returns:
#         Tensor: Converted covariance matrices in (x1, y1, x2, y2) format.
#     """
#     raise NotImplementedError
#     a, h = bbox_cxcyah[:, 2], bbox_cxcyah[:, 3]
#     #? Jacobian matrix for conversion from (cx, cy, a, h) to (x1, y1, x2, y2)
#     #* Assume a constant variance for aspect ratio
#     J = torch.zeros_like(cov_cxcyah)
#     J[:, 0, 0] = J[:, 2, 0] = 1
#     J[:, 1, 1] = J[:, 3, 1] = 1
#     # J[:, 0, 2] = -0.5 * h
#     # J[:, 2, 2] = 0.5 * h
#     J[:, 0, 3] = -0.5 * a
#     J[:, 2, 3] = 0.5 * a
#     J[:, 1, 3] = -0.5
#     J[:, 3, 3] = 0.5
    
#     #* Covariance matrix conversion using Jacobian: cov_xyxy = J * cov_cxcyah * J^T
#     cov_xyxy = torch.matmul(torch.matmul(J, cov_cxcyah), torch.transpose(J, 1, 2))
#     return cov_xyxy

def bbox_xyxy_to_cxcywh(bbox_xyxy):
    """Convert bbox coordinates from (x1, y1, x2, y2) to (cx, cy, w, h).

    Args:
        bbox (Tensor): Shape (n, 4) or (4, ) for bboxes.

    Returns:
        Tensor: Converted bboxes.
    """
    cx = (bbox_xyxy[:, 0] + bbox_xyxy[:, 2]) / 2
    cy = (bbox_xyxy[:, 1] + bbox_xyxy[:, 3]) / 2
    w = bbox_xyxy[:, 2] - bbox_xyxy[:, 0]
    h = bbox_xyxy[:, 3] - bbox_xyxy[:, 1]
    bbox_cxcywh = torch.stack([cx, cy, w, h], dim=-1)
    return bbox_cxcywh

def bbox_xyxy_to_xywh(bbox_xyxy):
    """Convert bbox coordinates from (x1, y1, x2, y2) to (x1, y1, w, h).

    Args:
        bbox (Tensor): Shape (n, 4) or (4, ) for bboxes.

    Returns:
        Tensor: Converted bboxes.
    """
    x = bbox_xyxy[..., 0]
    y = bbox_xyxy[..., 1]
    w = bbox_xyxy[..., 2] - bbox_xyxy[..., 0]
    h = bbox_xyxy[..., 3] - bbox_xyxy[..., 1]
    bbox_xywh = torch.stack([x, y, w, h], dim=-1)
    return bbox_xywh

def bbox_cov_xyxy_to_cxcywh(cov_xyxy):
    """Convert covariance matrices of bbox coordinates from (x1, y1, x2, y2) to (cx, cy, w, h),
    where w and h are the width and height of the bounding boxes.

    Args:
        cov_xyxy (Tensor): Shape (n, 4, 4) for covariance matrices in (x1, y1, x2, y2) format.

    Returns:
        Tensor: Converted covariance matrices in (cx, cy, w, h) format.
    """
    #? Jacobian matrix for conversion from (x1, y1, x2, y2) to (cx, cy, w, h)
    #* Simplify non-linearity with aspect ratio by assuming a constant variance
    J = torch.tensor([
        [0.5, 0, 0.5, 0],
        [0, 0.5, 0, 0.5],
        [-1, 0, 1, 0],
        [0, -1, 0, 1]
    ], dtype=cov_xyxy.dtype, device=cov_xyxy.device)
    
    #* Covariance matrix conversion using Jacobian: cov_cxcywh = J * cov_xyxy * J^T
    cov_cxcywh = J @ cov_xyxy @ J.T
    return cov_cxcywh

def bbox_cxcywh_to_xyxy(bbox_cxcywh):
    """Convert bbox coordinates from (cx, cy, w, h) to (x1, y1, x2, y2).

    Args:
        bbox (Tensor): Shape (n, 4) or (4, ) for bboxes.

    Returns:
        Tensor: Converted bboxes.
    """
    x1 = bbox_cxcywh[:, 0] - bbox_cxcywh[:, 2] / 2
    y1 = bbox_cxcywh[:, 1] - bbox_cxcywh[:, 3] / 2
    x2 = bbox_cxcywh[:, 0] + bbox_cxcywh[:, 2] / 2
    y2 = bbox_cxcywh[:, 1] + bbox_cxcywh[:, 3] / 2
    bbox_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)
    return bbox_xyxy

#* --------------------------------------------------------
#* Tracker utils
#* --------------------------------------------------------

def results2outs(bbox_results=None,
                 bbox_cov_results=None,
                 **kwargs):
    """Restore the results (list of results of each category) into the results
    of the model forward.

    Args:
        bbox_results (list[np.ndarray]): A list of boxes of each category.
        bbox_cov_results (list[np.ndarray]): A list of covariance matrices of each category.
        
    Returns:
        tuple: tracking results of each class. It may contain keys as belows:

        - bboxes (np.ndarray): shape (n, 5)
        - bbox_covs (np.ndarray): shape (n, 4, 4)
        - labels (np.ndarray): shape (n, )
        - ids (np.ndarray): shape (n, )
    """
    outputs = dict()

    if bbox_results is not None:
        labels = []
        for i, bbox in enumerate(bbox_results):
            labels.extend([i] * bbox.shape[0])
        labels = np.array(labels, dtype=np.int64)
        outputs['labels'] = labels

        bboxes = np.concatenate(bbox_results, axis=0).astype(np.float32)
        if bboxes.shape[1] == 5:
            outputs['bboxes'] = bboxes
        elif bboxes.shape[1] == 6:
            ids = bboxes[:, 0].astype(np.int64)
            bboxes = bboxes[:, 1:]
            outputs['bboxes'] = bboxes
            outputs['ids'] = ids
        else:
            raise NotImplementedError(
                f'Not supported bbox shape: (N, {bboxes.shape[1]})')
    
    if bbox_cov_results is not None:
        bbox_covs = np.concatenate(bbox_cov_results, axis=0).astype(np.float32)
        outputs['bbox_covs'] = bbox_covs
    else:
        outputs['bbox_covs'] = None
    
    return outputs

def outs2results(bboxes=None,
                 bbox_covs=None,
                 labels=None,
                 ids=None,
                 num_classes=None,
                 **kwargs):
    """Convert tracking/detection results to a list of numpy arrays.

    Args:
        bboxes (torch.Tensor | np.ndarray): shape (n, 5)
        bbox_covs (torch.Tensor | np.ndarray): shape (n, 4, 4)
        labels (torch.Tensor | np.ndarray): shape (n, )
        ids (torch.Tensor | np.ndarray): shape (n, )
        num_classes (int): class number, not including background class

    Returns:
        dict[str : list(ndarray) | list[list[np.ndarray]]]: tracking/detection
        results of each class. It may contain keys as belows:

        - bbox_results (list[np.ndarray]): Each list denotes bboxes of one
            category.
        - bbox_cov_results (list[np.ndarray]): Each list denotes covariance matrices of one
            category.
        - mask_results (list[list[np.ndarray]]): Each outer list denotes masks
            of one category. Each inner list denotes one mask belonging to
            the category. Each mask has shape (h, w).
    """
    assert labels is not None
    assert num_classes is not None

    results = dict()

    if ids is not None:
        valid_inds = ids > -1
        ids = ids[valid_inds]
        labels = labels[valid_inds]

    if bboxes is not None:
        if ids is not None:
            bboxes = bboxes[valid_inds]
            bbox_covs = bbox_covs[valid_inds]
            if bboxes.shape[0] == 0:
                bbox_results = [
                    np.zeros((0, 6), dtype=np.float32)
                    for i in range(num_classes)
                ]
                bbox_cov_results = [np.zeros((0, 4, 4), dtype=np.float32) 
                                for i in range(num_classes)]
            else:
                if isinstance(bboxes, torch.Tensor):
                    bboxes = bboxes.cpu().numpy()
                    bbox_covs = bbox_covs.cpu().numpy()
                    labels = labels.cpu().numpy()
                    ids = ids.cpu().numpy()
                bbox_results = [
                    np.concatenate(
                        (ids[labels == i, None], bboxes[labels == i, :]),
                        axis=1) for i in range(num_classes)
                ]
                bbox_cov_results = [bbox_covs[labels == i, :] for i in range(num_classes)]
        else:
            bbox_results = bbox2result(bboxes, labels, num_classes)
            bbox_cov_results = None
        results['bbox_results'] = bbox_results
        results['bbox_cov_results'] = bbox_cov_results

    return results