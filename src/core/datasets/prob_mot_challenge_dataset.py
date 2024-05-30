import os
import os.path as osp
import tempfile

import numpy as np
from mmcv.utils import print_log
from mmdet.datasets import DATASETS
from mmtrack.datasets import MOTChallengeDataset

from core.utils import results2outs
from core.visualization import get_ellipse_params
from core.inference import interpolate_tracks


@DATASETS.register_module()
class ProbabilisticMOTChallengeDataset(MOTChallengeDataset):
    """Dataset for MOTChallenge with support for covariance matrices."""
    
    def format_results(self, results, resfile_path=None, metrics=['track'], with_ellipse=True):
        """Format the results to txts (standard format for MOT Challenge).

        Args:
            results (dict(list[ndarray])): Testing results of the dataset.
            resfile_path (str, optional): Path to save the formatted results.
                Defaults to None.
            metrics (list[str], optional): The results of the specific metrics
                will be formatted.. Defaults to ['track'].

        Returns:
            tuple: (resfile_path, resfiles, names, tmp_dir), resfile_path is
            the path to save the formatted results, resfiles is a dict
            containing the filepaths, names is a list containing the name of
            the videos, tmp_dir is the temporal directory created for saving
            files.
        """
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
                formatter = getattr(self, f'format_{metric}_results')
                if metric == 'track':
                    formatter(results[f'{metric}_bboxes'][inds[i]:inds[i + 1]],
                            results[f'{metric}_bbox_covs'][inds[i]:inds[i + 1]],
                            self.data_infos[inds[i]:inds[i + 1]],
                            f'{resfiles[metric]}/{names[i]}.txt',
                            with_ellipse=with_ellipse)

        return resfile_path, resfiles, names, tmp_dir
    
    def format_track_results(self, bbox_results, bbox_cov_results, infos, resfile, with_ellipse=True):
        """Format tracking results."""

        results_per_video = []
        ellipses_per_video = []
        for frame_id, (bbox_result, bbox_cov_result) \
            in enumerate(zip(bbox_results, bbox_cov_results)):
            
            outs_track = results2outs(bbox_results=bbox_result, 
                                      bbox_cov_results=bbox_cov_result)
            track_ids, bboxes = outs_track['ids'], outs_track['bboxes']
            bbox_covs = outs_track['bbox_covs']
            frame_ids = np.full_like(track_ids, frame_id)
            
            results_per_frame = np.concatenate(
                (frame_ids[:, None], track_ids[:, None], bboxes), axis=1)

            #? Get ellipse parameters
            if bbox_covs is not None and with_ellipse:
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
            if with_ellipse:
                ellipses_per_video.append(ellipses_per_frame)
        # `results_per_video` is a ndarray with shape (N, 8). Each row denotes
        # (frame_id, track_id, x1, y1, x2, y2, score, label.
        # `ellipses_per_video` is a ndarray with shape (N, 6). Each row denotes
        # (tl_width, tl_height, tl_rotation, br_width, br_height, br_rotation)
        results_per_video = np.concatenate(results_per_video)
        if with_ellipse:
            ellipses_per_video = np.concatenate(ellipses_per_video)
        
        if self.interpolate_tracks_cfg is not None:
            results_per_video, ellipses_per_video = interpolate_tracks(
                results_per_video, 
                ellipses_per_video, 
                **self.interpolate_tracks_cfg)
        
        if with_ellipse:
            with open(resfile, 'wt') as f1, \
                open(resfile.replace('.txt', '_ellipse.txt'), 'wt') as f2:
                for frame_id, info in enumerate(infos):
                    if 'mot_frame_id' in info:
                        mot_frame_id = info['mot_frame_id']
                    else:
                        mot_frame_id = info['frame_id'] + 1
                    
                    results_per_frame = \
                        results_per_video[results_per_video[:, 0] == frame_id]
                    ellipses_per_frame = \
                        ellipses_per_video[results_per_video[:, 0] == frame_id]
                    for i in range(len(results_per_frame)):
                        _, track_id, x1, y1, x2, y2, conf = results_per_frame[i]
                        w1, h1, r1, w2, h2, r2 = ellipses_per_frame[i]
                        f1.writelines(
                            f'{mot_frame_id},{int(track_id)},{x1:.3f},{y1:.3f},' +
                            f'{(x2-x1):.3f},{(y2-y1):.3f},{conf:.3f},-1,-1,-1\n')
                        f2.writelines(
                            f'{mot_frame_id},{int(track_id)},' + 
                            f'{w1},{h1},{r1},{w2},{h2},{r2}\n')
                f1.close()
                f2.close()
        else:
            with open(resfile, 'wt') as f1:
                for frame_id, info in enumerate(infos):
                    if 'mot_frame_id' in info:
                        mot_frame_id = info['mot_frame_id']
                    else:
                        mot_frame_id = info['frame_id'] + 1
                    results_per_frame = \
                        results_per_video[results_per_video[:, 0] == frame_id]
                    for i in range(len(results_per_frame)):
                        _, track_id, x1, y1, x2, y2, conf = results_per_frame[i]
                        f1.writelines(
                            f'{mot_frame_id},{int(track_id)},{x1:.3f},{y1:.3f},' +
                            f'{(x2-x1):.3f},{(y2-y1):.3f},{conf:.3f},-1,-1,-1\n')
                f1.close()