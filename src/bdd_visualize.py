import argparse
import os
import os.path as osp

import mmcv
import motmetrics as mm
import numpy as np
import pandas as pd
from mmcv import Config
from mmcv.utils import print_log

from mmtrack.datasets import build_dataset

from core import *


BDD_CATEGORIES = [
    'pedestrian', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
    'bicycle'
]

def parse_args():
    parser = argparse.ArgumentParser(
        description='visualize errors for multiple object tracking')
    parser.add_argument('config', help='path of the config file')
    parser.add_argument('--result-file', help='path of the inference result')
    parser.add_argument(
        '--out-dir',
        help='directory where painted images or videos will be saved')
    parser.add_argument(
        '--show',
        action='store_true',
        help='whether to show the results on the fly')
    parser.add_argument(
        '--skip-fn',
        action='store_true',
        help='whether to skip frames with FNs errors only'
    )
    parser.add_argument(
        '--fps', type=int, default=3, help='FPS of the output video')
    parser.add_argument(
        '--backend',
        type=str,
        choices=['cv2', 'plt'],
        default='plt',
        help='backend of visualization')
    args = parser.parse_args()
    return args


def load_ellipse_from_txt(ellipse_resfile):
    """Load the ellipse parameters from the txt file
    into a pandas dataframe similar to `load_motchallenge`
    from mm.io.loadtxt"""
    df = pd.read_csv(ellipse_resfile,
                     sep=r'\s+|\t+|,',
                     index_col=[0,1],
                     skipinitialspace=True,
                     header=None,
                     names=['FrameId', 'Id', 'tl_w', 'tl_h', 'tl_theta', 
                            'br_w', 'br_h', 'br_theta'],
                     engine='python')
    return df

def get_results_df(resfiles, video_name):
    """Get results into a dataframe.

    Args:
        resfiles (dict): A dict containing the directory of the MOT results.
        video_name (str): Name of the video to be evaluated.

    Returns:
        res (pandas.DataFrame): The tracking results of the video.
    """
    res_file = osp.join(resfiles['track'], f'{video_name}.txt')
    ellipse_res_file = osp.join(resfiles['track'], f'{video_name}_ellipse.txt')
    res = mm.io.loadtxt(res_file)
    ellipse_res = load_ellipse_from_txt(ellipse_res_file)
    res = pd.concat([res, ellipse_res], axis=1, join='inner')
    return res


def main():
    args = parse_args()

    assert args.show or args.out_dir, \
        ('Please specify at least one operation (show the results '
         '/ save the results) with the argument "--show" or "--out-dir"')

    if not args.result_file.endswith(('.pkl', 'pickle')):
        raise ValueError('The result file must be a pkl file.')

    if args.out_dir is not None:
        os.makedirs(args.out_dir, exist_ok=True)

    print_log('This script visualizes the error for multiple object tracking. '
              'By Default, the red bounding box denotes false positive, '
              'the yellow bounding box denotes the false negative '
              'and the blue bounding box denotes ID switch.')

    cfg = Config.fromfile(args.config)
    dataset = build_dataset(cfg.data.val, dict(test_mode=True))
    results = mmcv.load(args.result_file)
    
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
    
    # create index from frame_id to filename
    filenames_dict = dict()
    for data_info in dataset.data_infos:
        video_name = data_info['filename'].split(os.sep, 1)[0]
        frame_id = data_info['frame_id']
        if video_name not in filenames_dict:
            filenames_dict[video_name] = dict()
        filenames_dict[video_name][frame_id] = data_info['filename']
    
    ann_infos, data_infos, bboxes = dataset.get_preds_and_anns(results)
    acc_results = get_mot_acc(bboxes, ann_infos, classes=dataset.CLASSES)
    # format the results to txts
    _, resfiles, video_names, _ = dataset.format_results(
        results, None, ['track'])
    
    for i, video_name in enumerate(video_names):
        print_log(f'Start processing video {video_name}')
        ann_info = ann_infos[i]
        data_info = data_infos[i]
        acc_per_cls = acc_results[i][0]   #* len(# classes)
        frame_ids_per_cls = acc_results[i][1]   #* len(# classes)
        assert len(acc_per_cls) == len(dataset.CLASSES)
        
        gt = dataset.anns2dataframe(ann_info, data_info)
        res = get_results_df(resfiles, video_name)
        
        for frame_id in range(len(data_info)):
            # path of image
            img = osp.join(dataset.img_prefix,
                        filenames_dict[video_name][frame_id])
            
            if df is not None:
                if video_name in df.Sequence.values and \
                    frame_id in df[df.Sequence == video_name].Frame.values:
                    save_viz = True
                else:
                    save_viz = False
            else:
                save_viz = True
            if save_viz:
                # events in the current frame
                #* Each class has its own acc and appears in different frames
                #? For each class, add bboxes, ellipses
                bboxes, ellipse_params, ids, error_types = [], [], [], []
                for acc_cls, frame_ids in zip(acc_per_cls, frame_ids_per_cls):
                    mot_frame_id = frame_ids[frame_id]
                    if mot_frame_id > -1:
                        events = acc_cls.mot_events.xs(mot_frame_id)
                        cur_res = res.loc[frame_id] if frame_id in res.index else None
                        cur_gt = gt.loc[frame_id] if frame_id in gt.index else None

                        fps = events[events.Type == 'FP']
                        fns = events[events.Type == 'MISS']
                        idsws = events[events.Type == 'SWITCH']
                        
                        for hid in fps.HId:
                            bboxes.append([
                                cur_res.loc[hid].X, cur_res.loc[hid].Y,
                                cur_res.loc[hid].X + cur_res.loc[hid].Width,
                                cur_res.loc[hid].Y + cur_res.loc[hid].Height,
                                cur_res.loc[hid].Confidence,
                                cur_res.loc[hid].ClassId
                            ])
                            ellipse_params.append([
                                cur_res.loc[hid].tl_w, cur_res.loc[hid].tl_h, 
                                cur_res.loc[hid].tl_theta, cur_res.loc[hid].br_w,
                                cur_res.loc[hid].br_h, cur_res.loc[hid].br_theta
                            ])
                            ids.append(hid)
                            # error_type = 0 denotes false positive error
                            error_types.append(0)
                        for oid in fns.OId:
                            bboxes.append([
                                cur_gt.loc[oid].X, cur_gt.loc[oid].Y,
                                cur_gt.loc[oid].X + cur_gt.loc[oid].Width,
                                cur_gt.loc[oid].Y + cur_gt.loc[oid].Height,
                                cur_gt.loc[oid].Confidence,
                                cur_gt.loc[oid].ClassId
                            ])
                            ellipse_params.append([0, 0, 0, 0, 0, 0])
                            ids.append(-1)
                            # error_type = 1 denotes false negative error
                            error_types.append(1)
                        for hid in idsws.HId:
                            bboxes.append([
                                cur_res.loc[hid].X, cur_res.loc[hid].Y,
                                cur_res.loc[hid].X + cur_res.loc[hid].Width,
                                cur_res.loc[hid].Y + cur_res.loc[hid].Height,
                                cur_res.loc[hid].Confidence,
                                cur_res.loc[hid].ClassId
                            ])
                            ellipse_params.append([
                                cur_res.loc[hid].tl_w, cur_res.loc[hid].tl_h, 
                                cur_res.loc[hid].tl_theta, cur_res.loc[hid].br_w,
                                cur_res.loc[hid].br_h, cur_res.loc[hid].br_theta
                            ])
                            ids.append(hid)
                            # error_type = 2 denotes id switch
                            error_types.append(2)
                
                if len(bboxes) == 0:
                    bboxes = np.zeros((0, 6), dtype=np.float32)
                    ellipses = np.zeros((0, 6), dtype=np.int32)
                else:
                    bboxes = np.asarray(bboxes, dtype=np.float32)
                    ellipses = np.asarray(ellipse_params, dtype=np.int32)
                ids = np.asarray(ids, dtype=np.int32)
                error_types = np.asarray(error_types, dtype=np.int32)
                
                if error_types.size > 0:
                    if args.skip_fn and np.all(error_types == 1):
                        continue
                    imshow_mot_errors(
                        img,
                        bboxes,
                        ellipses,
                        ids,
                        error_types,
                        classes=BDD_CATEGORIES,
                        font_scale=5,
                        show=args.show,
                        out_file=osp.join(args.out_dir,
                                        f'{video_name}/{frame_id:06d}.jpg')
                        if args.out_dir else None,
                        backend=args.backend)

        print_log(f'Done! Visualization images are saved in '
                  f'\'{args.out_dir}/{video_name}\'')

        try:
            mmcv.frames2video(
                f'{args.out_dir}/{video_name}',
                f'{args.out_dir}/{video_name}.mp4',
                fps=args.fps,
                fourcc='mp4v',
                start=0,
                end=len(data_info) - 1,
                show_progress=False)
            print_log(
                f'Done! Visualization video is saved as '
                f'\'{args.out_dir}/{video_name}.mp4\' with a FPS of {args.fps}')
        except:
            print_log(
                f'Failed to convert images to video for {video_name}, '
                f'please check if the images are saved in '
                f'\'{args.out_dir}/{video_name}\'')


if __name__ == '__main__':
    main()