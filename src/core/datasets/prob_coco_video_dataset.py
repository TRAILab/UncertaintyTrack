import os
import os.path as osp
import tempfile

import numpy as np
import pandas as pd
import motmetrics as mm
from mmcv.utils import print_log
from mmdet.datasets import DATASETS
from mmdet.core import eval_map
from mmtrack.datasets import CocoVideoDataset
import trackeval

from core.utils import results2outs
from core.visualization import get_ellipse_params

@DATASETS.register_module()
class ProbabilisticCocoVideoDataset(CocoVideoDataset):
    """Dataset for CocoVideoDataset with support for covariance matrices."""
    
    def format_results(self, results, resfile_path=None, metrics=['track']):
        """Format the results to txts (standard format for MOT Challenge).

        Args:
            results (dict(list[ndarray])): Testing results of the dataset.
            resfile_path (str, optional): Path to save the formatted results.
                Defaults to None.
            metrics (list[str], optional): The results of the specific metrics
                will be formatted. Defaults to ['track'].

        Returns:
            tuple: (resfile_path, resfiles, names, tmp_dir), resfile_path is
            the path to save the formatted results, resfiles is a dict
            containing the filepaths, names is a list containing the name of
            the videos, tmp_dir is the temporal directory created for saving
            files.
        """
        breakpoint()
        assert isinstance(results, dict), 'results must be a dict.'
        if resfile_path is None:
            tmp_dir = tempfile.TemporaryDirectory()
            resfile_path = tmp_dir.name
        else:
            tmp_dir = None
            if osp.exists(resfile_path):
                print_log('remove previous results.', self.logger)
                import shutil
                shutil.rmtree(resfile_path)

        resfiles = dict()
        for metric in metrics:
            resfiles[metric] = osp.join(resfile_path, metric)
            os.makedirs(resfiles[metric], exist_ok=True)

        inds = [i for i, _ in enumerate(self.data_infos) if _['frame_id'] == 0]
        num_vids = len(inds)
        assert num_vids == len(self.vid_ids)
        inds.append(len(self.data_infos))
        vid_infos = self.coco.load_vids(self.vid_ids)
        names = [_['name'] for _ in vid_infos]

        for i in range(num_vids):
            for metric in metrics:
                if metric == 'track':
                    formatter = getattr(self, f'format_{metric}_results')
                    bbox_results = results[f'{metric}_bboxes'][inds[i]:inds[i + 1]]
                    bbox_cov_results = results[f'{metric}_bbox_covs'][inds[i]:inds[i + 1]]
                    if len(bbox_cov_results) == 0:
                        bbox_cov_results = [None] * len(bbox_results)
                    
                    formatter(bbox_results,
                            bbox_cov_results,
                            self.data_infos[inds[i]:inds[i + 1]],
                            f'{resfiles[metric]}/{names[i]}.txt')
                else:
                    pass

        return resfile_path, resfiles, names, tmp_dir
    
    def format_track_results(self, bbox_results, bbox_cov_results, infos, resfile):
        """Format tracking results."""
        results_per_video = []
        ellipses_per_video = []
        for frame_id, (bbox_result, bbox_cov_result) \
            in enumerate(zip(bbox_results, bbox_cov_results)):
            
            outs_track = results2outs(bbox_results=bbox_result, 
                                      bbox_cov_results=bbox_cov_result)
            
            track_ids, bboxes, labels = outs_track['ids'], outs_track['bboxes'], outs_track['labels']
            bbox_covs = outs_track['bbox_covs']
            frame_ids = np.full_like(track_ids, frame_id)
            
            results_per_frame = np.concatenate(
                (frame_ids[:, None], track_ids[:, None], bboxes, labels[:, None]), axis=1)

            #? Get ellipse parameters
            if bbox_covs is not None:
                tl_widths, tl_heights, tl_rotations = get_ellipse_params(bbox_covs[:, :2, :2],
                                                                        q=0.95)
                br_widths, br_heights, br_rotations = get_ellipse_params(bbox_covs[:, 2:, 2:],
                                                                        q=0.95)
                ellipses_per_frame = np.concatenate(
                    (tl_widths[:, None], tl_heights[:, None], tl_rotations[:, None],
                    br_widths[:, None], br_heights[:, None], br_rotations[:, None]),
                    axis=1)
            else:
                ellipses_per_frame = np.zeros((len(results_per_frame), 6))
            
            results_per_video.append(results_per_frame)
            ellipses_per_video.append(ellipses_per_frame)
        # `results_per_video` is a ndarray with shape (N, 8). Each row denotes
        # (frame_id, track_id, x1, y1, x2, y2, score, label.
        # `ellipses_per_video` is a ndarray with shape (N, 6). Each row denotes
        # (tl_width, tl_height, tl_rotation, br_width, br_height, br_rotation)
        results_per_video = np.concatenate(results_per_video)
        ellipses_per_video = np.concatenate(ellipses_per_video)
        
        with open(resfile, 'wt') as f1, \
            open(resfile.replace('.txt', '_ellipse.txt'), 'wt') as f2:
            for frame_id, _ in enumerate(infos):
                results_per_frame = \
                    results_per_video[results_per_video[:, 0] == frame_id]
                ellipses_per_frame = \
                    ellipses_per_video[results_per_video[:, 0] == frame_id]
                for i in range(len(results_per_frame)):
                    _, track_id, x1, y1, x2, y2, conf, label = results_per_frame[i]
                    w1, h1, r1, w2, h2, r2 = ellipses_per_frame[i]
                    f1.writelines(
                        f'{frame_id},{track_id},{x1:.3f},{y1:.3f},' +
                        f'{(x2-x1):.3f},{(y2-y1):.3f},{conf:.3f},{label},-1,-1\n')
                    f2.writelines(
                        f'{frame_id},{track_id},' + 
                        f'{w1},{h1},{r1},{w2},{h2},{r2}\n')
            f1.close()
            f2.close()
            
    def get_preds_and_anns(self, results=None):
        inds = [i for i, _ in enumerate(self.data_infos) if _['frame_id'] == 0]
        num_vids = len(inds)
        inds.append(len(self.data_infos))
        ann_infos = [self.get_ann_info(_) for _ in self.data_infos]
        ann_infos = [ann_infos[inds[i]:inds[i + 1]] for i in range(num_vids)]
        data_infos = [self.data_infos[inds[i]:inds[i + 1]] for i in range(num_vids)]
        if results is not None:
            track_bboxes = [
                    results['track_bboxes'][inds[i]:inds[i + 1]]
                    for i in range(num_vids)
                ]
        else:
            track_bboxes = None
        return ann_infos, data_infos, track_bboxes
    
    def anns2dataframe(self, ann_info, data_info):
        """Convert the annotation info to a pandas dataframe."""
        columns = ['FrameId', 'Id', 'X', 'Y', 'Width', 'Height', 
                'Confidence', 'ClassId', 'Visibility']
        df = pd.DataFrame(columns=columns)
        
        for i, (ann, data) in enumerate(zip(ann_info, data_info)):
            frame_id = data['frame_id']
            ids, bboxes, class_ids = ann['instance_ids'], ann['bboxes'], ann['labels']
            bboxes_x1y1wh = np.asarray(bboxes)
            bboxes_x1y1wh[:, 2:] = bboxes_x1y1wh[:, 2:] - bboxes_x1y1wh[:, :2]
            confidences = -1 * np.ones(len(ids))     #* made up
            visibilities = -1 * np.ones(len(ids))    #* made up
            frame_id_array = np.full(len(ids), frame_id, dtype=np.int32)
            rows = np.concatenate([frame_id_array[:, None], ids[:, None],
                                bboxes_x1y1wh, confidences[:, None],
                                class_ids[:, None], visibilities[:, None]], axis=1)
            df = df.append(pd.DataFrame(rows, columns=columns))
        df = df.astype({'FrameId': np.int32, 'Id': np.int32, 'ClassId': np.int32})
        df.set_index(['FrameId', 'Id'], inplace=True)
        return df