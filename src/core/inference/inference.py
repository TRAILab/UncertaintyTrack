import os
import os.path as osp
import torch
import numpy as np
import pandas as pd
import mmcv
from mmcv.image import tensor2imgs
from collections import defaultdict

from ..utils.analysis_utils import results2json, accumuluate_analysis 
from ..visualization import show_track_result

def mmdet_single_gpu_test(model,
                    data_loader,
                    model_results=None,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3,
                    **kwargs):
    """Single gpu test function for inference from mmdet.apis
    Adapted to handle covariance matrices.
    
    Saves sample images every 500 iterations if out_dir is specified
    or show is True.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    PALETTE = getattr(dataset, 'PALETTE', None)
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        batch_size = len(result)
        if show or out_dir:
            if i % 500 == 0:
                if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
                    img_tensor = data['img'][0]
                else:
                    img_tensor = data['img'][0].data[0]
                img_metas = data['img_metas'][0].data[0]
                imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
                assert len(imgs) == len(img_metas)

                for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                    h, w, _ = img_meta['img_shape']
                    img_show = img[:h, :w, :]

                    ori_h, ori_w = img_meta['ori_shape'][:-1]
                    img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                    if out_dir:
                        out_file = osp.join(out_dir, img_meta['ori_filename'])
                    else:
                        out_file = None

                    model.module.show_result(
                        img_show,
                        result[i],
                        bbox_color=PALETTE,
                        text_color=PALETTE,
                        mask_color=PALETTE,
                        show=show,
                        out_file=out_file,
                        score_thr=show_score_thr)

        results.extend(result)
        for _ in range(batch_size):
            prog_bar.update()
    return results

def mmtrack_single_gpu_test(model,
                    data_loader,
                    model_results=None,
                    show=False,
                    out_dir=None,
                    fps=3,
                    show_score_thr=0.3,
                    analysis_cfg=None,
                    **kwargs):
    """Test model with single gpu.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        show (bool, optional): If True, visualize the prediction results.
            Defaults to False.
        out_dir (str, optional): Path of directory to save the
            visualization results. Defaults to None.
        fps (int, optional): FPS of the output video.
            Defaults to 3.
        show_score_thr (float, optional): The score threshold of visualization
            (Only used in VID for now). Defaults to 0.3.

    Returns:
        dict[str, list]: The prediction results.
    """
    model.eval()
    analysis_cfg = analysis_cfg or {}
    results = defaultdict(list)
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    
    #? Show track results specified by txt file if it exists
    # hard-coded path for now :(
    filter_txt = '/home/misc/show_viz.txt'
    if osp.exists(filter_txt):
        df = pd.read_csv(filter_txt,
                sep=r'\s+|\t+|,',
                skipinitialspace=True,
                header=None,
                names=['Sequence', 'Frame'],
                engine='python')
    else:
        df = None
    
    prev_result = None
    for i, data in enumerate(data_loader):
        #* each `i` is a frame
        #* the loop iterates over all frames in the dataset (i.e. all videos)
        frame_id = data['img_metas'][0].data[0][0]['frame_id']
        if model_results is None:
            with torch.no_grad():
                result = model(return_loss=False, rescale=True, **data,
                            analysis_cfg=analysis_cfg)
        else:
            result = {k: v[i] for k, v in model_results.items()}
        batch_size = data['img'][0].size(0)
        
        if show or out_dir:
            assert batch_size == 1, 'Only support batch_size=1 when testing.'
            img_tensor = data['img'][0]
            img_meta = data['img_metas'][0].data[0][0]
            
            if df is not None:
                seq = osp.dirname(img_meta['ori_filename']).split('/')[0]
                #* Parsing frame_id is a bit hard-coded for MOT and BDD
                frame_id = int(osp.basename(img_meta['ori_filename']).split("-")[-1].split('.')[0])
                if seq in df.Sequence.values and \
                    frame_id in df[df.Sequence == seq].Frame.values:
                    save_viz = True
                else:
                    save_viz = False
            else:
                save_viz = True
            
            if save_viz:
                img = tensor2imgs(img_tensor, **img_meta['img_norm_cfg'])[0]

                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))
                
                out_file = osp.join(out_dir, img_meta['ori_filename'])
                parent_dir = osp.dirname(out_file)
                os.makedirs(parent_dir, exist_ok=True)
                out_file = osp.basename(out_file)
                show_score_thr = 0.1    #* hard-coded the lowest threshold from byte from now
                # prev_result = None
                show_track_result(
                    img_show,
                    result,
                    prev_result,
                    model.module.CLASSES,
                    show=show,
                    parent_dir=parent_dir,
                    out_file=out_file,
                    score_thr=show_score_thr)
                
        prev_result = result    #* only works if the current frame is not the first frame in seq
        for k, v in result.items():
            results[k].append(v)

        for _ in range(batch_size):
            prog_bar.update()

    #? Filter out extra analysis results here
    analysis_results = results.pop(analysis_cfg.get('type', ''), [])
    analysis_dict = accumuluate_analysis(analysis_cfg, analysis_results)
    print(f"\n{analysis_dict}")
    #? Save analysis to file
    if analysis_cfg.get('save_dir', None):
        try:
            results2json(analysis_dict, analysis_cfg['save_dir'])
        except:
            print('Failed to save analysis results to file')
        
    return results

def _interpolate_track(track, track_id, max_num_frames=20):
    """Interpolate a track linearly to make the track more complete.

    Args:
        track (ndarray): With shape (N, 7). Each row denotes
            (frame_id, track_id, x1, y1, x2, y2, score).
        max_num_frames (int, optional): The maximum disconnected length in the
            track. Defaults to 20.

    Returns:
        ndarray: The interpolated track with shape (N, 7). Each row denotes
            (frame_id, track_id, x1, y1, x2, y2, score)
    """
    assert (track[:, 1] == track_id).all(), \
        'The track id should not changed when interpolate a track.'

    frame_ids = track[:, 0]
    interpolated_track = np.zeros((0, 7))
    # perform interpolation for the disconnected frames in the track.
    for i in np.where(np.diff(frame_ids) > 1)[0]:
        left_frame_id = frame_ids[i]
        right_frame_id = frame_ids[i + 1]
        num_disconnected_frames = int(right_frame_id - left_frame_id)

        if 1 < num_disconnected_frames < max_num_frames:
            left_bbox = track[i, 2:6]
            right_bbox = track[i + 1, 2:6]

            # perform interpolation for two adjacent tracklets.
            for j in range(1, num_disconnected_frames):
                cur_bbox = j / (num_disconnected_frames) * (
                    right_bbox - left_bbox) + left_bbox
                cur_result = np.ones((7, ))
                cur_result[0] = j + left_frame_id
                cur_result[1] = track_id
                cur_result[2:6] = cur_bbox

                interpolated_track = np.concatenate(
                    (interpolated_track, cur_result[None]), axis=0)

    interpolated_track = np.concatenate((track, interpolated_track), axis=0)
    return interpolated_track

def interpolate_tracks(tracks, det_ellipses, min_num_frames=5, max_num_frames=20):
    """Interpolate tracks linearly to make tracks more complete.

    This function is proposed in
    "ByteTrack: Multi-Object Tracking by Associating Every Detection Box."
    `ByteTrack<https://arxiv.org/abs/2110.06864>`_.

    Args:
        tracks (ndarray): With shape (N, 7). Each row denotes
            (frame_id, track_id, x1, y1, x2, y2, score).
        det_ellipses (ndarray): With shape (N, 6). Each row denotes
            (tl_width, tl_height, tl_rotation, 
            br_width, br_height, br_rotation).
        min_num_frames (int, optional): The minimum length of a track that will
            be interpolated. Defaults to 5.
        max_num_frames (int, optional): The maximum disconnected length in
            a track. Defaults to 20.

    Returns:
        ndarray: The interpolated tracks with shape (N, 7). Each row denotes
            (frame_id, track_id, x1, y1, x2, y2, score)
    """
    max_track_id = int(np.max(tracks[:, 1]))
    min_track_id = int(np.min(tracks[:, 1]))

    # perform interpolation for each track
    interpolated_tracks = []
    interpolated_ellipses = []
    for track_id in range(min_track_id, max_track_id + 1):
        inds = tracks[:, 1] == track_id
        track = tracks[inds]
        ellipse = det_ellipses[inds]
        num_frames = len(track)
        if num_frames <= 2:
            continue

        if num_frames > min_num_frames:
            interpolated_track = _interpolate_track(track, track_id,
                                                    max_num_frames)
            extra_rows = interpolated_track.shape[0] - track.shape[0]
            interpolated_ellipse = np.zeros((extra_rows, 6), dtype=np.int32)
            interpolated_ellipse = np.concatenate(
                (ellipse, interpolated_ellipse), axis=0)
        else:
            interpolated_track = track
            interpolated_ellipse = ellipse
        interpolated_tracks.append(interpolated_track)
        interpolated_ellipses.append(interpolated_ellipse)

    interpolated_tracks = np.concatenate(interpolated_tracks)
    interpolated_ellipses = np.concatenate(interpolated_ellipses)
    return interpolated_tracks[interpolated_tracks[:, 0].argsort()], \
            interpolated_ellipses[interpolated_tracks[:, 0].argsort()]