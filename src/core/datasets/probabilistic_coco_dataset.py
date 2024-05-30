"""
Code adapted from mmdetection coco.py
"""
import contextlib
import io
import itertools
import logging
from collections import OrderedDict
from tqdm import tqdm

import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.coco import CocoDataset

from ..evaluation.prob_cocoeval import ProbCOCOeval


@DATASETS.register_module()
class ProbabilisticCocoDataset(CocoDataset):
    
    def __init__(self, classwise=False, **kwargs):
        super().__init__(**kwargs)
        self.classwise = classwise

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox annotation. Adapted from mmdetection.
        Simplified to handle only bounding boxes.

        Args:
            ann_info (list[dict]): Annotation info of an image.

        Returns:
            dict: A dict containing the following keys: 
                bboxes, bboxes_ignore, labels.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore)

        return ann

    def xyxy2xywh_cov(self, bbox_cov):
        """Convert ``x1y1x2y2`` style bounding boxes to ``x1y1wh`` style for
        COCO evaluation.

        Args:
            bbox_cov (np.ndarray): The bounding box covariance matrices,
                shape (4, 4), in the format of ``x1y1x2y2``.
        
        Returns:
            np.ndarray: The converted bounding boxes in the format of ``x1y1wh``.
        """
        transform_mat = np.array(
            [[1.0, 0, 0, 0],
             [0, 1.0, 0, 0],
             [-1.0, 0, 1.0, 0],
             [0, -1.0, 0, 1.0]],
            dtype=bbox_cov.dtype
        )
        out_bbox_cov = transform_mat @ bbox_cov @ transform_mat.T
        return out_bbox_cov

    def _det_with_cov2list(self, results):
        """Convert detection results with covariance to list of objects
            in COCO json style."""
        results_list = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            result = results[idx]
            for label in range(len(result['bbox'])):
                bboxes = result['bbox'][label]
                bbox_covs = result['bbox_cov'][label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['bbox_cov'] = self.xyxy2xywh_cov(bbox_covs[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    results_list.append(data)
        return results_list

    def results2list(self, results):
        """Convert results to list of objects in COCO json style.

        Args:
            results (list[list | dict]): Testing results of the dataset
                including bboxes and bbox covariance matrices.
                If list[dict], each dict contains 'bbox' and 'bbox_cov' keys.

        Returns:
            list[dict[str, int|float]]: List of all detection results.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        #? Check if results contain covariance matrices
        if isinstance(results[0], dict):
            return self._det_with_cov2list(results)
        elif isinstance(results[0], list):
            raise TypeError("Results should contain covariance matrices.")
        else:
            raise TypeError('Invalid type of results')

    def evaluate_det(self,
                        results_list,
                        coco_gt,
                        metrics,
                        logger=None,
                        classwise=False,
                        proposal_nums=(100, 300, 1000),
                        iou_thrs=None,
                        metric_items=None):
        """object detection evaluation in COCO protocol with 
        probabilistic detection results using proper scoring rules.

        Args:
            results_list (list[dict[str, int|float]]): List of all detection results.
            coco_gt (COCO): COCO API object with ground truth annotation.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'scoring'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when ``metric=='bbox'`` 
                and ``['NLL (0.5:0.95)', 'Energy Score (0.5:0.95)','NLL (0.7)', 
                'Energy Score (0.7)', 'Mean Entropy (0.7)']`` will be used when
                ``metric=='scoring'``.

        Returns:
            dict[str, float]: COCO style evaluation metric with scoring rules.
        """
        if iou_thrs is None:
            iou_thrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        if metric_items is not None:
            if not isinstance(metric_items, list):
                metric_items = [metric_items]

        eval_results = OrderedDict()
        try:
            coco_det = coco_gt.loadRes(results_list)
        except IndexError:
            print_log(
                'Cannot load predictions into COCO object.',
                logger=logger,
                level=logging.ERROR)
            return eval_results

        cocoEval = ProbCOCOeval(coco_gt, 
                                coco_det, 
                                iou_type='bbox',
                                catIds=self.cat_ids,
                                imgIds=self.img_ids,
                                maxDets=list(proposal_nums),
                                iouThrs=iou_thrs)

        for metric in metrics:
            msg = f'Evaluating {metric}...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)
            
            # mapping of cocoEval.stats
            coco_metric_names = {
                'mAP': 0,
                'mAP_50': 1,
                'mAP_75': 2,
                'mAP_s': 3,
                'mAP_m': 4,
                'mAP_l': 5,
                'AR@100': 6,
                'AR@300': 7,
                'AR@1000': 8,
                'AR_s@1000': 9,
                'AR_m@1000': 10,
                'AR_l@1000': 11
            }
            scoring_rules = ['NLL (0.5:0.95)', 'Energy Score (0.5:0.95)', 'Mean Entropy (0.5:0.95)',
                             'NLL (0.7)', 'Energy Score (0.7)', 'Mean Entropy (0.7)']
            
            if metric_items is not None:
                for metric_item in metric_items:
                    if metric_item not in (list(coco_metric_names.keys()) 
                                           + scoring_rules):
                        raise KeyError(
                            f'metric item {metric_item} is not supported')

            #? Run evaluation using COCO API
            cocoEval.evaluate(metric=metric)
            cocoEval.accumulate(metric=metric)
            
            #? Save coco summarize print information to logger
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                cocoEval.summarize(metric=metric)
            print_log('\n' + redirect_string.getvalue(), logger=logger)

            if metric == 'bbox':
                if classwise or self.classwise:  # Compute per-category AP
                    # Compute per-category AP
                    # from https://github.com/facebookresearch/detectron2/
                    precisions = cocoEval.eval['precision']
                    # precision: (iou, recall, cls, area range, max dets)
                    assert len(self.cat_ids) == precisions.shape[2]

                    results_per_category = []
                    for idx, catId in enumerate(self.cat_ids):
                        # area range index 0: all area ranges
                        # max dets index -1: typically 100 per image
                        nm = self.coco.loadCats(catId)[0]
                        precision = precisions[:, :, idx, 0, -1]
                        precision = precision[precision > -1]
                        if precision.size:
                            ap = np.mean(precision)
                        else:
                            ap = float('nan')
                        results_per_category.append(
                            (f'{nm["name"]}', f'{float(ap):0.3f}'))

                    num_columns = min(6, len(results_per_category) * 2)
                    results_flatten = list(
                        itertools.chain(*results_per_category))
                    headers = ['category', 'AP'] * (num_columns // 2)
                    results_2d = itertools.zip_longest(*[
                        results_flatten[i::num_columns]
                        for i in range(num_columns)
                    ])
                    table_data = [headers]
                    table_data += [result for result in results_2d]
                    table = AsciiTable(table_data)
                    print_log('\n' + table.table, logger=logger)

                coco_metric_items = metric_items
                if coco_metric_items is None:
                    coco_metric_items = [
                        'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                    ]

                for metric_item in coco_metric_items:
                    key = f'{metric}_{metric_item}'
                    val = float(
                        f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}'
                    )
                    eval_results[key] = val
                ap = cocoEval.stats[:6]
                eval_results[f'{metric}_mAP_copypaste'] = (
                    f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                    f'{ap[4]:.3f} {ap[5]:.3f}')
            
            elif metric == 'scoring':
                for scoring_rule in scoring_rules:
                    eval_results[scoring_rule] = cocoEval.scoring_stats[scoring_rule]

                if classwise:  # Compute per-category scoring rules
                    raise NotImplementedError(
                        "Classwise scoring rules are not supported yet.")
            else:
                raise KeyError(f'metric {metric} is not supported')

        return eval_results

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None,
                 **kwargs):
        """Evaluation in COCO protocol and probabilistic scoring rules
        such as NLL and Energy Score

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'scoring'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox'`` and ``['NLL', 'ES']`` will be used when
                ``metric=='scoring'``.

        Returns:
            dict[str, float]: COCO style evaluation metric with scoring rules.
        """
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'scoring', 'analysis']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        coco_gt = self.coco
        self.cat_ids = coco_gt.get_cat_ids(cat_names=self.CLASSES)
        results_list = self.results2list(results)
        eval_results = self.evaluate_det(results_list, coco_gt,
                                              metrics, logger, classwise,
                                              proposal_nums, iou_thrs,
                                              metric_items)

        return eval_results
    
    def analyze(self, 
                results, 
                iou_thr=0.3,
                proposal_nums=(100, 300, 1000), 
                **kwargs):
        coco_gt = self.coco
        results_list = self.results2list(results)
        coco_det = coco_gt.loadRes(results_list)
        self.cat_ids = coco_gt.get_cat_ids(cat_names=self.CLASSES)
        cocoEval = ProbCOCOeval(coco_gt, 
                                coco_det, 
                                iou_type='bbox',
                                catIds=self.cat_ids,
                                imgIds=self.img_ids,
                                maxDets=list(proposal_nums),
                                iouThrs=[iou_thr])
        cocoEval.evaluate(metric='scoring')
        
        print(f"Collecting results for analysis...")
        
        all_det_bboxes_tp, all_det_bbox_covs_tp, all_det_scores_tp, all_imgIds_tp = cocoEval.collect_results()
        assert len(all_imgIds_tp) == all_det_bboxes_tp.shape[0]
        
        all_det_bboxes_tp_ = all_det_bboxes_tp.copy()
        all_det_bboxes_tp_[:, 2:] = all_det_bboxes_tp_[:, 2:] + all_det_bboxes_tp_[:, :2]
        all_det_bboxes_tp_[:, 0::2] = np.clip(all_det_bboxes_tp_[:, 0::2], 0, 1280)
        all_det_bboxes_tp_[:, 1::2] = np.clip(all_det_bboxes_tp_[:, 1::2], 0, 720)
        all_det_bboxes_tp_[:, 2:] = all_det_bboxes_tp_[:, 2:] - all_det_bboxes_tp_[:, :2]
        
        
        # all_det_bboxes        = np.empty((0, 4))
        # all_det_bbox_covs     = np.empty((0, 4, 4))
        # all_det_scores        = np.empty((0,))
        
        # for i, res in tqdm(enumerate(results), total=len(results)):
        #     img_w = self.data_infos[i]['width']
        #     img_h = self.data_infos[i]['height']
        #     for c in range(len(self.cat_ids)):
        #         bboxes = res['bbox'][c][:, :4]
        #         scores = res['bbox'][c][:, 4]
        #         bbox_covs = res['bbox_cov'][c]
                
        #         #? Clip boxes to image boundaries
        #         # bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_w)
        #         # bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_h)
        #         # bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, :2]
                
        #         all_det_bboxes    = np.concatenate((all_det_bboxes, bboxes), axis=0)
        #         all_det_scores    = np.concatenate((all_det_scores, scores), axis=0)
        #         all_det_bbox_covs = np.concatenate((all_det_bbox_covs, bbox_covs), axis=0)
        # all_det_bboxes[:, 2:] = all_det_bboxes[:, 2:] - all_det_bboxes[:, :2]
        
        all_det_bboxes    = np.array([res['bbox'] for res in results_list])
        all_det_bboxes[:, 0::2] = np.clip(all_det_bboxes[:, 0::2], 0, 1280)
        all_det_bboxes[:, 1::2] = np.clip(all_det_bboxes[:, 1::2], 0, 720)
        all_det_bboxes[:, 2:] = all_det_bboxes[:, 2:] - all_det_bboxes[:, :2]
        all_det_bbox_covs = np.array([res['bbox_cov'] for res in results_list])
        all_det_scores    = np.array([res['score'] for res in results_list])
        
        return (all_det_bboxes_tp_, all_det_bbox_covs_tp, all_det_scores_tp, 
                all_det_bboxes, all_det_bbox_covs, all_det_scores)