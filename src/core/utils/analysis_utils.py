import torch
import numpy as np
import json
from collections import defaultdict


def results2json(results_dict, out_file):
    """Dump the analysis results to a json file.

    Args:
        results_dict (dict): The analysis results.
        out_file (str): The filename of the output json file.
    """
    assert isinstance(results_dict, dict)
    assert out_file.endswith('.json')
    
    with open(out_file, 'w') as f:
        json.dump(results_dict, f, indent=4, separators=(',', ': '))

def gaussian_entropy(sigma):
    """Compute the entropy of Multivariate Gaussian distribution.

    Args:
        sigma (np.array): The covariance matrices. Shape (N, D, D).
    """
    D = sigma.shape[-1]
    _, logdet = np.linalg.slogdet(sigma + 1e-2 * np.identity(D))
    entropy = 0.5 * logdet + 0.5 * D * (1 + np.log(2*np.pi))
    return entropy

#TODO: modify this
def tracks2entropy(bboxes=None,
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
            if bboxes.shape[0] == 0:
                bbox_results = [
                    np.zeros((0, 6), dtype=np.float32)
                    for i in range(num_classes)
                ]
            else:
                if isinstance(bboxes, torch.Tensor):
                    bboxes = bboxes.cpu().numpy()
                    labels = labels.cpu().numpy()
                    ids = ids.cpu().numpy()
                bbox_results = [
                    np.concatenate(
                        (ids[labels == i, None], bboxes[labels == i, :]),
                        axis=1) for i in range(num_classes)
                ]
        else:
            # bbox_results = bbox2result(bboxes, labels, num_classes)
            # results = bbox_and_cov2result()
            pass
        results['bbox_results'] = bbox_results

    return results

def init_entropy(results):
    """Compute entropy of the state distribution and predicted distribution
    in the track initiation of Kalman Filter.
    
    Args:
        results (list[dict]): The list of outputs.
            - default: The default covariance. Shape (D,D).
            - predicted: The predicted covariance. Shape (D,D).
            
    Returns:
        dict:
            - default: The entropy of the default covariance.
            - predicted: The entropy of the predicted covariance.
    """
    entropy_dict = dict()
    
    if len(results) > 0:
        assert isinstance(results[0], dict)
        assert 'default' in results[0]
        assert 'predicted' in results[0]
        
        default = np.stack([res['default'] for res in results])
        predicted = np.stack([res['predicted'] for res in results])
        entropy_dict['default'] = gaussian_entropy(default).mean()
        entropy_dict['predicted'] = gaussian_entropy(predicted).mean()
    
    return entropy_dict

def observation_entropy(results):
    """Compute entropy of the state estimation distribution before 
    and after update with observation.
    
    Args:
        results (list[dict]): The list of outputs.
            - before: The covariance before update. Shape (D,D).
            - after: The covariance after update. Shape (D,D).
            
    Returns:
        dict:
            - before: The entropy of the distribution before update.
            - after: The entropy of the distribution after update.
    """
    entropy_dict = dict()
    
    if len(results) > 0:
        assert isinstance(results[0], dict)
        assert 'before' in results[0]
        assert 'after' in results[0]
        
        before = np.stack([res['before'] for res in results])
        after = np.stack([res['after'] for res in results])
        entropy_dict['before'] = gaussian_entropy(before).mean()
        entropy_dict['after'] = gaussian_entropy(after).mean()
    
    return entropy_dict

def average_entropy(results):
    """Compute the average entropy of the state estimation distributions 
    of active and inactive tracks.
    
    Args:
        results (list[dict]): The list of outputs (length 1).
            - active: List of covariances of active tracks. Shape (D,D).
            - inactive_2: List of covariances of inactive tracks of age 2. 
                            Shape (D,D).
            - inactive_3: List of covariances of inactive tracks of age 3. 
                            Shape (D,D).
            - inactive_4: List of covariances of inactive tracks of age 4. 
                            Shape (D,D).
            - inactive_5: List of covariances of inactive tracks of age 5 or more.
                            Shape (D,D).
    Returns:
        dict of entropy values with same keys as input.
    """
    entropy_dict = dict()
    if len(results) > 0:
        results = results[0]
        assert isinstance(results, dict)
        keys = ['active', 'inactive_2', 'inactive_3', 'inactive_4', 'inactive_5']
        assert any([key in results for key in keys])
        for key in keys:
            if len(results.get(key, [])) > 0:
                entropy_dict[key] = gaussian_entropy(
                                        np.stack(results[key])).mean()
    return entropy_dict

def analyze_results(analysis_cfg, results):
    analysis_dict = dict()
    if analysis_cfg.get('type', '') == 'entropy':
        entropy_fn = {
            'init entropy': init_entropy,
            'before after entropy': observation_entropy,
            'average entropy': average_entropy
            }
        assert results is not None, \
            "Detailed results not found for entropy analysis."
        if analysis_cfg.get('name', '') in entropy_fn:
            entropy = entropy_fn[analysis_cfg['name']](results)
            analysis_dict['entropy'] = entropy
        else:
            raise ValueError('Unknown analysis name: {}'.format(analysis_cfg.get('name', '')))
    return analysis_dict

def accumuluate_analysis(analysis_cfg, analysis_results):
    analysis_dict = dict()
    keys = []
    list_dict = defaultdict(list)
    if analysis_cfg.get('name', '') == 'init entropy':
        keys.extend(['default', 'predicted'])
    elif analysis_cfg.get('name', '') == 'before after entropy':
        keys.extend(['before', 'after'])
    elif analysis_cfg.get('name', '') == 'average entropy':
        keys.extend(['active', 'inactive_2', 'inactive_3', 
                    'inactive_4', 'inactive_5'])
    
    for i, result in enumerate(analysis_results):
        for key in keys:
            if key in result:
                list_dict[key].append(result[key])
    for key in keys:
        analysis_dict[key] = np.mean(list_dict[key])
    
    return analysis_dict

def get_active_inactive(tracks, ids, frame_id):
    #* active: associated last frame
    #* inactive_2: associated 2 frames ago
    #* inactive_3: associated 3 frames ago
    #* inactive_4: associated 4 frames ago
    #* inactive_5: associated 5 or more frames ago
    out_dict = defaultdict(list)
    for id in ids:
        if tracks[id].frame_ids[-1] == frame_id - 1:
            out_dict['active'].append(tracks[id]['covariance'])
        elif tracks[id].frame_ids[-1] == frame_id - 2:
            out_dict['inactive_2'].append(tracks[id]['covariance'])
        elif tracks[id].frame_ids[-1] == frame_id - 3:
            out_dict['inactive_3'].append(tracks[id]['covariance'])
        elif tracks[id].frame_ids[-1] == frame_id - 4:
            out_dict['inactive_4'].append(tracks[id]['covariance'])
        else:
            out_dict['inactive_5'].append(tracks[id]['covariance'])
    
    return out_dict

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)