import os.path as osp
import random

import cv2
import matplotlib
import matplotlib.pyplot as plt
import mmcv
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle, Ellipse
from matplotlib.collections import PatchCollection
from mmcv.utils import mkdir_or_exist

from ..utils import results2outs
from .utils import get_ellipse_params

def _random_color(seed):
    """Random a color according to the input seed."""
    random.seed(seed)
    colors = sns.color_palette()
    color = random.choice(colors)
    return color

def imshow_mot_errors(*args, backend='plt', **kwargs):
    """Show the wrong tracks on the input image
    Include bbox ellipses on corners.

    Args:
        backend (str, optional): Backend of visualization.
            Defaults to 'plt'.
    """
    if backend == 'plt':
        return _plt_show_wrong_tracks(*args, **kwargs)
    else:
        raise NotImplementedError()

def imshow_tracks(*args, backend='plt', **kwargs):
    """Show the tracks on the input image."""
    if backend == 'plt':
        return _plt_show_tracks(*args, **kwargs)
    else:
        raise NotImplementedError()

def _plt_show_wrong_tracks(img,
                           bboxes,
                           ellipses,
                           ids,
                           error_types,
                           classes=None,
                           thickness=1.0,
                           font_scale=3,
                           text_width=8,
                           text_height=13,
                           show=False,
                           wait_time=100,
                           out_file=None):
    """Show the wrong tracks with matplotlib.

    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (ndarray): A ndarray of shape (k, 5).
        ellipses (ndarray): A ndarray of shape (k, 6) where each
            row is [tl_w, tl_h, tl_theta, br_w, br_h, br_theta].
        ids (ndarray): A ndarray of shape (k, ).
        error_types (ndarray): A ndarray of shape (k, ), where 0 denotes
            false positives, 1 denotes false negative and 2 denotes ID switch.
        thickness (float, optional): Thickness of lines.
            Defaults to 1.0.
        font_scale (float, optional): Font scale to draw id and score.
            Defaults to 3.
        text_width (int, optional): Width to draw id and score.
            Defaults to 8.
        text_height (int, optional): Height to draw id and score.
            Defaults to 13.
        show (bool, optional): Whether to show the image on the fly.
            Defaults to False.
        wait_time (int, optional): Value of waitKey param.
            Defaults to 100.
        out_file (str, optional): The filename to write the image.
            Defaults to None.

    Returns:
        ndarray: Original image.
    """
    assert bboxes.ndim == 2, \
        f' bboxes ndim should be 2, but its ndim is {bboxes.ndim}.'
    assert ids.ndim == 1, \
        f' ids ndim should be 1, but its ndim is {ids.ndim}.'
    assert error_types.ndim == 1, \
        f' error_types ndim should be 1, but its ndim is {error_types.ndim}.'
    assert bboxes.shape[0] == ids.shape[0], \
        'bboxes.shape[0] and ids.shape[0] should have the same length.'
    assert bboxes.shape[1] == 5 or bboxes.shape[1] == 6, \
        f' bboxes.shape[1] should be 5 or 6, but its {bboxes.shape[1]}.'
    assert bboxes.shape[0] == ellipses.shape[0], \
        'bboxes.shape[0] and ellipses.shape[0] should have the same length.'

    bbox_colors = sns.color_palette()
    # red, yellow, blue
    bbox_colors = [bbox_colors[3], bbox_colors[1], bbox_colors[0]]

    if isinstance(img, str):
        img = plt.imread(img)
    else:
        assert img.ndim == 3
        img = mmcv.bgr2rgb(img)

    img_shape = img.shape
    bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
    bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])

    plt.imshow(img)
    plt.gca().set_axis_off()
    plt.autoscale(False)
    plt.subplots_adjust(
        top=1, bottom=0, right=1, left=0, hspace=None, wspace=None)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.rcParams['figure.figsize'] = img_shape[1], img_shape[0]
    
    for bbox, ellipse, error_type, id in zip(bboxes, ellipses, 
                                             error_types, ids):
        if len(bbox) == 5:
            x1, y1, x2, y2, score = bbox
            label = 0
        else:
            x1, y1, x2, y2, score, label = bbox
        w, h = int(x2 - x1), int(y2 - y1)
        left_top = (int(x1), int(y1))
        right_bot = (int(x2), int(y2))

        # bbox
        plt.gca().add_patch(
            Rectangle(
                left_top,
                w,
                h,
                linewidth=thickness,
                edgecolor=bbox_colors[error_type],
                facecolor='none'))
        
        # ellipse
        tl_w, tl_h, tl_theta, br_w, br_h, br_theta = ellipse
        if not (np.isnan(tl_w) or np.isnan(tl_h) or np.isnan(tl_theta)):
            e1 = Ellipse(
                xy=left_top,
                width=tl_w,
                height=tl_h,
                angle=tl_theta)
        if not (np.isnan(br_w) or np.isnan(br_h) or np.isnan(br_theta)):
            e2 = Ellipse(
                xy=right_bot,
                width=br_w,
                height=br_h,
                angle=br_theta)
        p = PatchCollection([e1, e2], 
                            facecolor='none',
                            edgecolors=bbox_colors[error_type],
                            linewidths=1.5)
        plt.gca().add_collection(p)

        # FN does not have id and score
        if error_type == 1:
            continue

        # # score
        # #TODO: show variance as well
        # text = '{:.02f}'.format(score)
        # width = len(text) * text_width
        # plt.gca().add_patch(
        #     Rectangle((left_top[0], left_top[1]),
        #               width,
        #               text_height,
        #               thickness,
        #               edgecolor=bbox_colors[error_type],
        #               facecolor=bbox_colors[error_type]))

        # plt.text(
        #     left_top[0],
        #     left_top[1] + text_height + 2,
        #     text,
        #     fontsize=font_scale)

        # # id
        # if classes is not None:
        #     text = f"{classes[int(label)]}_{str(id)}"
        # else:
        #     text = str(id)
        # width = len(text) * text_width
        # plt.gca().add_patch(
        #     Rectangle((left_top[0], left_top[1] + text_height + 1),
        #               width,
        #               text_height,
        #               thickness,
        #               edgecolor=bbox_colors[error_type],
        #               facecolor=bbox_colors[error_type]))
        # plt.text(
        #     left_top[0],
        #     left_top[1] + 2 * (text_height + 1),
        #     text,
        #     fontsize=font_scale)

    if out_file is not None:
        mkdir_or_exist(osp.abspath(osp.dirname(out_file)))
        plt.savefig(out_file, dpi=300, bbox_inches='tight', pad_inches=0.0)

    if show:
        plt.draw()
        plt.pause(wait_time / 1000.)

    plt.clf()
    return img

def _plt_show_tracks(img,
                     bboxes,
                     ellipses,
                     labels,
                     ids,
                     masks=None,
                     classes=None,
                     score_thr=0.0,
                     thickness=0.1,
                     font_scale=5,
                     show=False,
                     wait_time=0,
                     parent_dir=None,
                     out_file=None):
    """Show the tracks with matplotlib."""
    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert ids.ndim == 1
    assert bboxes.shape[0] == ids.shape[0]
    assert bboxes.shape[1] == 5

    if isinstance(img, str):
        img = plt.imread(img)
    else:
        img = mmcv.bgr2rgb(img)

    img_shape = img.shape
    bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
    bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])

    inds = np.where(bboxes[:, -1] > score_thr)[0]
    bboxes = bboxes[inds]
    labels = labels[inds]
    ids = ids[inds]

    if not show:
        matplotlib.use('Agg')

    plt.imshow(img)
    plt.gca().set_axis_off()
    plt.autoscale(False)
    plt.subplots_adjust(
        top=1, bottom=0, right=1, left=0, hspace=None, wspace=None)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.rcParams['figure.figsize'] = img_shape[1], img_shape[0]

    # text_width, text_height = 12, 16
    text_width = 12
    text_height = 14
    
    for i, (bbox, ellipse, label, id) in enumerate(zip(bboxes, ellipses, labels, ids)):
        x1, y1, x2, y2 = bbox[:4].astype(np.int32)
        score = float(bbox[-1])
        w, h = int(x2 - x1), int(y2 - y1)
        left_top = (int(x1), int(y1))
        right_bot = (int(x2), int(y2))

        # bbox
        bbox_color = _random_color(id)
        plt.gca().add_patch(
            Rectangle((x1, y1),
                      w,
                      h,
                      thickness,
                      edgecolor=bbox_color,
                      facecolor='none'))
        # ellipse
        tl_w, tl_h, tl_theta, br_w, br_h, br_theta = ellipse
        if not (np.isnan(tl_w) or np.isnan(tl_h) or np.isnan(tl_theta)):
            e1 = Ellipse(
                xy=left_top,
                width=tl_w,
                height=tl_h,
                angle=tl_theta)
        if not (np.isnan(br_w) or np.isnan(br_h) or np.isnan(br_theta)):
            e2 = Ellipse(
                xy=right_bot,
                width=br_w,
                height=br_h,
                angle=br_theta)
        p = PatchCollection([e1, e2], 
                            facecolor='none',
                            edgecolors=bbox_color,
                            linewidths=1.5)
        plt.gca().add_collection(p)
        
        # score
        #* trace of ellipse covariance matrix
        # q_095 = 5.991
        # # 2 * np.sqrt(5.991 * eigenvalues)
        # trace = np.array([(l/2)**2 / q_095 for l in [tl_w, tl_h, br_w, br_h]]).sum()
        # # text = '{:.02f}'.format(score)
        # text = '{:.01f}'.format(trace)
        # if classes is not None:
        #     text += f'|{classes[label]}'
        # width = len(text) * text_width
        # plt.gca().add_patch(
        #     Rectangle((x1, y1),
        #             #   width,
        #                 w,
        #               text_height,
        #               thickness,
        #               edgecolor=bbox_color,
        #               facecolor=bbox_color))
        # plt.text(x1, y1 + text_height, text, fontsize=4)

        # # id
        # text = str(id)
        # width = len(text) * text_width
        # plt.gca().add_patch(
        #     Rectangle((x1, y1 + text_height + 1),
        #               width,
        #               text_height,
        #               thickness,
        #               edgecolor=bbox_color,
        #               facecolor=bbox_color))
        # plt.text(x1, y1 + 2 * text_height + 2, text, fontsize=4)
        
    # In order to show the mask.
    plt.imshow(img)

    if out_file is not None and parent_dir is not None:
        plt.savefig(osp.join(parent_dir, out_file), 
                    dpi=300, bbox_inches='tight', pad_inches=0.0)

    if show:
        plt.draw()
        plt.pause(wait_time / 1000.)
    else:
        plt.show()
    plt.clf()
    return img

def show_track_result(img,
                    result,
                    prev_result,
                    classes,
                    score_thr=0.0,
                    thickness=1,
                    font_scale=0.5,
                    show=False,
                    parent_dir=None,
                    out_file=None,
                    wait_time=0,
                    backend='plt',
                    **kwargs):
        """Visualize tracking results.

        Args:
            img (str | ndarray): Filename of loaded image.
            result (dict): Tracking result.
                - The value of key 'track_bboxes' is list with length
                num_classes, and each element in list is ndarray with
                shape(n, 6) in [id, tl_x, tl_y, br_x, br_y, score] format.
                - The value of key 'track_bbox_covs' is list with length
                num_classes, and each element in list is ndarray with
                shape (n, 4, 4) in xyxy format.
                - The value of key 'det_bboxes' is list with length
                num_classes, and each element in list is ndarray with
                shape(n, 5) in [tl_x, tl_y, br_x, br_y, score] format.
            classes (tuple[str]): Names of each classes.
            thickness (int, optional): Thickness of lines. Defaults to 1.
            font_scale (float, optional): Font scales of texts. Defaults
                to 0.5.
            show (bool, optional): Whether show the visualizations on the
                fly. Defaults to False.
            parent_dir (str | None, optional): Parent directory of output
            out_file (str | None, optional): Output filename. Defaults to None.
            backend (str, optional): Backend to draw the bounding boxes,
                options are `cv2` and `plt`. Defaults to 'cv2'.

        Returns:
            ndarray: Visualized image.
        """
        assert isinstance(result, dict)
        track_bboxes = result.get('track_bboxes', None)
        track_bbox_covs = result.get('track_bbox_covs', None)
        if isinstance(img, str):
            img = mmcv.imread(img)
        outs_track = results2outs(
            bbox_results=track_bboxes,
            bbox_cov_results=track_bbox_covs)
        bboxes = outs_track.get('bboxes', None)
        labels = outs_track.get('labels', None)
        ids = outs_track.get('ids', None)
        
        #? Convert covariance matrices to ellipse parameters
        bbox_covs = outs_track.get('bbox_covs', None)
        if bbox_covs is not None:
            tl_widths, tl_heights, tl_rotations = get_ellipse_params(bbox_covs[:, :2, :2],
                                                                     q=0.95)
            br_widths, br_heights, br_rotations = get_ellipse_params(bbox_covs[:, 2:, 2:],
                                                                     q=0.95)
            ellipses = np.concatenate(
                (tl_widths[:, None], tl_heights[:, None], tl_rotations[:, None],
                br_widths[:, None], br_heights[:, None], br_rotations[:, None]),
                axis=1)
        else:
            ellipses = None
        
        if prev_result is not None:
            assert isinstance(prev_result, dict)
            prev_track_bboxes = prev_result.get('track_bboxes', None)
            prev_track_bbox_covs = prev_result.get('track_bbox_covs', None)
            prev_outs_track = results2outs(
                bbox_results=prev_track_bboxes,
                bbox_cov_results=prev_track_bbox_covs)
            bboxes = np.vstack([prev_outs_track['bboxes'], bboxes])
            labels = np.concatenate([prev_outs_track['labels'], labels])
            ids = np.concatenate([prev_outs_track['ids'], ids])
            
            #? Convert covariance matrices to ellipse parameters
            prev_bbox_covs = prev_outs_track.get('bbox_covs', None)
            if bbox_covs is not None:
                tl_widths, tl_heights, tl_rotations = get_ellipse_params(prev_bbox_covs[:, :2, :2],
                                                                        q=0.95)
                br_widths, br_heights, br_rotations = get_ellipse_params(prev_bbox_covs[:, 2:, 2:],
                                                                        q=0.95)
                prev_ellipses = np.concatenate(
                    (tl_widths[:, None], tl_heights[:, None], tl_rotations[:, None],
                    br_widths[:, None], br_heights[:, None], br_rotations[:, None]),
                    axis=1)
                ellipses = np.vstack([prev_ellipses, ellipses])
            else:
                prev_ellipses = None
        
        img = imshow_tracks(
            img,
            bboxes,
            ellipses,
            labels,
            ids,
            classes=classes,
            score_thr=score_thr,
            thickness=thickness,
            font_scale=font_scale,
            show=show,
            parent_dir=parent_dir,
            out_file=out_file,
            wait_time=wait_time,
            backend=backend)
        return img