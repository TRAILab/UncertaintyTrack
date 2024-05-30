"""
Code adapted from cocodataset/cocoapi/cocoeval.py
"""
import time
import copy
import numpy as np
import torch
from tqdm import tqdm

from mmdet.datasets.api_wrappers import COCOeval

from .metrics import energy_score, negative_loglikelihood, entropy


class ProbCOCOeval(COCOeval):
    def __init__(self, coco_gt=None, 
                        coco_dt=None, 
                        iou_type='bbox',
                        catIds=None,
                        imgIds=None,
                        maxDets=None,
                        iouThrs=None):
        super().__init__(coco_gt, coco_dt, iou_type)
        if catIds is not None:
            self.params.catIds = catIds
        if imgIds is not None:
            self.params.imgIds = imgIds
        if maxDets is not None:
            self.params.maxDets = maxDets
        if iouThrs is not None:
            self.params.iouThrs = iouThrs
        self._prepare()
        
    def _prepare(self):
        print('\nLoading and preparing results and ground-truths for evaluation...')
        tic = time.time()
        p = self.params
        p.iouType = 'bbox'
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params=p
        super()._prepare()
        
        print("Computing IoU per image...")
        #? loop through images, area range, max detection number
        catIds = p.catIds if p.useCats else [-1]
        computeIoU = self.computeIoU
        self.ious = {(imgId, catId): computeIoU(imgId, catId) \
                        for imgId in p.imgIds
                        for catId in catIds}
        #* self.ious is a dict of (imgId, catId) -> IoU matrix (nDt, nGt)
        #* nDt is the number of detections, nGt is the number of ground truths
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc-tic))
        
    def evaluate(self, metric='bbox'):
        '''
        Run per image evaluation on given images and store results 
        (a list of dict or a dict of dicts) in self.evalImgs
        :return: None
        '''
        tic = time.time()
        p = self.params
        evaluateImg = self.evaluateImg
        maxDet = p.maxDets[-1]
        catIds = p.catIds if p.useCats else [-1]
        
        print(f"Matching results to ground-truths for *{metric}* evaluation...")
        #TODO: probably not the most efficient way to do this...
        if metric == 'bbox':
            self.evalImgs = [evaluateImg(imgId, catId, areaRng, maxDet)
                    for catId in catIds
                    for areaRng in p.areaRng
                    for imgId in p.imgIds
                ]
        elif metric == 'scoring':
            areaRng = p.areaRng[0]
            self.evalImgs = {(imgId,catId): evaluateImg(imgId, catId, areaRng, maxDet)
                    for catId in catIds
                    for imgId in p.imgIds
            }
        else:
            raise ValueError('Unknown metric in COCOevalPOD evaluate: {}'.format(metric))

        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc-tic))
        
    def accumulate(self, metric='bbox'):
        """ Accumulate per image evaluation results and store the result in self.eval.
            Compute bbox metrics or scoring rules (e.g. Energy Score, NLL) and mean entropy.
            Scoring rules are thresholded at 0.5-0.95 IoU and also at 0.7 IoU for TPs.
        """
        if metric == 'bbox':
            super().accumulate()
        elif metric == 'scoring':
            self.eval = {}
            print(f"Evaluating Energy Score and NLL per class...")
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            for catIdx, catId in enumerate(self.params.catIds):
                gt_bboxes_per_cls        = torch.Tensor().to(device)
                det_bboxes_per_cls       = torch.Tensor().to(device)
                det_bbox_covs_per_cls    = torch.Tensor().to(device)
                gt_bboxes_per_cls_07     = torch.Tensor().to(device)
                det_bboxes_per_cls_07    = torch.Tensor().to(device)
                det_bbox_covs_per_cls_07 = torch.Tensor().to(device)

                imgId_pbar = tqdm(enumerate(self.params.imgIds), total=len(self.params.imgIds))
                for imgIdx, imgId in imgId_pbar:
                    gts = self._gts[imgId, catId]
                    dts = self._dts[imgId, catId]
                    if len(dts) == 0 or len(gts) == 0:
                        #* No detections or ground truths for this image; skip
                        continue
                    gt_bboxes     = (torch.cat([torch.tensor(gt['bbox'])
                                                .unsqueeze(0) for gt in gts], dim=0))
                    det_bboxes    = (torch.cat([torch.tensor(dt['bbox'])
                                                .unsqueeze(0) for dt in dts], dim=0))
                    det_bbox_covs = (torch.cat([torch.tensor(dt['bbox_cov'])
                                                .unsqueeze(0) for dt in dts], dim=0))
                    
                    dtm      = self.evalImgs[imgId, catId]['dtMatches']
                    dtIg     = self.evalImgs[imgId, catId]['dtIgnore']
                    tpIdx    = (np.logical_and(dtm, np.logical_not(dtIg))
                                .nonzero())
                    thrIdx, detIdx = tpIdx[0], tpIdx[1]
                    matched_gtIds  = dtm[thrIdx, detIdx]
                    
                    if len(matched_gtIds) == 0:
                        #* No TP for this image; only evaluate TPs
                        continue
                    
                    #? Convert gtIds to indices
                    gtIds = self.evalImgs[imgId, catId]['gtIds']
                    gtId_to_idx = {gtId: idx for idx, gtId in enumerate(gtIds)}
                    matched_gtIdx = (torch.tensor([gtId_to_idx[gtId.item()]
                                                for gtId in matched_gtIds]))

                    #? Gather true dtm positive matches
                    gt_bboxes_tp          = gt_bboxes[matched_gtIdx, :].to(device)
                    det_bboxes_tp         = det_bboxes[detIdx, :].to(device)
                    det_bbox_covs_tp      = det_bbox_covs[detIdx, :, :].to(device)
                    gt_bboxes_per_cls     = (torch.cat((gt_bboxes_per_cls, 
                                                    gt_bboxes_tp), dim=0))
                    det_bboxes_per_cls    = (torch.cat((det_bboxes_per_cls, 
                                                        det_bboxes_tp), dim=0))
                    det_bbox_covs_per_cls = (torch.cat((det_bbox_covs_per_cls, 
                                                        det_bbox_covs_tp), dim=0))
                    
                    #? Filter for TP that have IoU > 0.7
                    tpThr_07 = [0.7]
                    tp_07_IoUIdx = np.where(np.in1d(self.params.iouThrs, tpThr_07))[0]
                    dtm_07   = dtm[tp_07_IoUIdx, :]
                    dtIg_07  = dtIg[tp_07_IoUIdx, :]
                    tpIdx_07 = (np.logical_and(dtm_07, np.logical_not(dtIg_07))
                                .nonzero())
                    thrIdx_07, detIdx_07 = tpIdx_07[0], tpIdx_07[1]
                    matched_gtIds_07  = dtm_07[thrIdx_07, detIdx_07]
                    if len(matched_gtIds_07) > 0:
                        matched_gtIdx_07 = (torch.tensor([gtId_to_idx[gtId.item()]
                                                for gtId in matched_gtIds_07]))
                        gt_bboxes_tp_07          = gt_bboxes[matched_gtIdx_07, :].to(device)
                        det_bboxes_tp_07         = det_bboxes[detIdx_07, :].to(device)
                        det_bbox_covs_tp_07      = det_bbox_covs[detIdx_07, :, :].to(device)
                        gt_bboxes_per_cls_07     = (torch.cat((gt_bboxes_per_cls_07, 
                                                        gt_bboxes_tp_07), dim=0))
                        det_bboxes_per_cls_07    = (torch.cat((det_bboxes_per_cls_07, 
                                                            det_bboxes_tp_07), dim=0))
                        det_bbox_covs_per_cls_07 = (torch.cat((det_bbox_covs_per_cls_07, 
                                                            det_bbox_covs_tp_07), dim=0))
                        
                        del matched_gtIdx_07, gt_bboxes_tp_07, det_bboxes_tp_07, det_bbox_covs_tp_07
                    
                    del gts, dts, gt_bboxes, det_bboxes, det_bbox_covs, dtm, dtIg, \
                        tpIdx, thrIdx, detIdx, matched_gtIds, gtIds, gtId_to_idx, \
                        matched_gtIdx, gt_bboxes_tp, det_bboxes_tp, det_bbox_covs_tp, \
                        dtm_07, dtIg_07, tpIdx_07, thrIdx_07, detIdx_07, matched_gtIds_07

                #? Compute scoring rules
                compute_score = len(gt_bboxes_per_cls) > 0 and len(det_bboxes_per_cls) > 0
                compute_score_07 = len(gt_bboxes_per_cls_07) > 0 and len(det_bboxes_per_cls_07) > 0
                self.eval[catId] = {
                    'es': energy_score(det_bboxes_per_cls,
                                        det_bbox_covs_per_cls,
                                        gt_bboxes_per_cls) \
                            if compute_score else 0.0,
                    'nll': negative_loglikelihood(det_bboxes_per_cls,
                                                    det_bbox_covs_per_cls,
                                                    gt_bboxes_per_cls) \
                            if compute_score else 0.0,
                    'average_entropy': entropy(det_bbox_covs_per_cls) \
                            if compute_score else 0.0,
                    'es_07': energy_score(det_bboxes_per_cls_07,
                                            det_bbox_covs_per_cls_07,
                                            gt_bboxes_per_cls_07) \
                        if compute_score_07 else 0.0,
                    'nll_07': negative_loglikelihood(det_bboxes_per_cls_07,
                                                        det_bbox_covs_per_cls_07,
                                                        gt_bboxes_per_cls_07) \
                        if compute_score_07 else 0.0,
                    'average_entropy_07': entropy(det_bbox_covs_per_cls_07) \
                        if compute_score_07 else 0.0
                }
                #? Min and Max entropy
                min_entropy = entropy(det_bbox_covs_per_cls_07, 'minimum') \
                                if compute_score_07 else 0.0
                max_entropy = entropy(det_bbox_covs_per_cls_07, 'maximum') \
                                if compute_score_07 else 0.0
                std_entropy = entropy(det_bbox_covs_per_cls_07, 'std') \
                                if compute_score_07 else 0.0
                print(f"Min entropy (0.7): {min_entropy:.4f}, Max entropy (0.7): {max_entropy:.4f}, " + \
                        f"Std entropy (0.7): {std_entropy:.4f}")
                
                del gt_bboxes_per_cls, det_bboxes_per_cls, det_bbox_covs_per_cls, \
                    gt_bboxes_per_cls_07, det_bboxes_per_cls_07, det_bbox_covs_per_cls_07, \
                    imgId_pbar
        else:
            raise ValueError('Unknown metric in COCOevalPOD accumulate: {}'.format(metric))
        del self.evalImgs
        
    def collect_results(self):
        imgId_pbar = tqdm(enumerate(self.params.imgIds), total=len(self.params.imgIds))

        all_det_bboxes_tp        = np.empty((0, 4))
        all_det_bbox_covs_tp     = np.empty((0, 4, 4))
        all_det_scores_tp        = np.empty((0,))
        all_imgIds               = []
        
        for _, imgId in imgId_pbar:
            for _, catId in enumerate(self.params.catIds):
                gts = self._gts[imgId, catId]
                dts = self._dts[imgId, catId]
                if len(dts) == 0 or len(gts) == 0:
                    #* No detections or ground truths; skip
                    continue
                det_bboxes    = (np.vstack([np.expand_dims(dt['bbox'], axis=0)
                                            for dt in dts]))
                det_bbox_covs = (np.vstack([np.expand_dims(dt['bbox_cov'], axis=0)
                                            for dt in dts]))
                det_scores    = np.hstack([dt['score'] for dt in dts])
                dtm      = self.evalImgs[imgId, catId]['dtMatches']
                dtIg     = self.evalImgs[imgId, catId]['dtIgnore']
                tpIdx    = (np.logical_and(dtm, np.logical_not(dtIg))
                            .nonzero())
                thrIdx, detIdx = tpIdx[0], tpIdx[1]
                matched_gtIds  = dtm[thrIdx, detIdx]
                
                if len(matched_gtIds) == 0:
                    #* No TP for this image
                    continue

                #? Gather true positive results
                det_bboxes_tp         = det_bboxes[detIdx, :]       
                det_bbox_covs_tp      = det_bbox_covs[detIdx, :, :]
                det_scores_tp         = det_scores[detIdx]
                
                all_det_bboxes_tp    = (np.concatenate((all_det_bboxes_tp,
                                                        det_bboxes_tp), axis=0))
                all_det_bbox_covs_tp = (np.concatenate((all_det_bbox_covs_tp,
                                                        det_bbox_covs_tp), axis=0))
                all_det_scores_tp    = (np.concatenate((all_det_scores_tp,
                                                        det_scores_tp), axis=0))
                all_imgIds.extend([imgId-1] * len(detIdx))
        return all_det_bboxes_tp, all_det_bbox_covs_tp, all_det_scores_tp, all_imgIds
    
    def _summarizeScoring(self):
        #? Average across classes with non-empty ground truths
        es_array = np.array([self.eval[catId]['es']
                            for _, catId in enumerate(self.params.catIds)])
        es = es_array[es_array > 0].mean()
        
        es_07_array = np.array([self.eval[catId]['es_07']
                            for _, catId in enumerate(self.params.catIds)])
        es_07 = es_07_array[es_07_array > 0].mean()
        
        nll_array = np.array([self.eval[catId]['nll']
                            for _, catId in enumerate(self.params.catIds)])
        nll = nll_array[nll_array > 0].mean()
        
        nll_07_array = np.array([self.eval[catId]['nll_07']
                            for _, catId in enumerate(self.params.catIds)])
        nll_07 = nll_07_array[nll_07_array > 0].mean()
        
        mean_entropy_array = np.array([self.eval[catId]['average_entropy']
                            for _, catId in enumerate(self.params.catIds)])
        mean_entropy = mean_entropy_array[mean_entropy_array > 0].mean()
        
        mean_entropy_07_array = np.array([self.eval[catId]['average_entropy_07']
                            for _, catId in enumerate(self.params.catIds)])
        mean_entropy_07 = mean_entropy_07_array[mean_entropy_07_array > 0].mean()
        
        output_string = f"Mean Energy Score (0.5:0.95): {es:.4f}, " + \
                        f"Mean NLL (0.5:0.95): {nll:.4f}, " + \
                        f"Mean Entropy (0.5:0.95): {mean_entropy:.4f}\n" + \
                        f"Mean Energy Score (0.7): {es_07:.4f}, " + \
                        f"Mean NLL (0.7): {nll_07:.4f}, " + \
                        f"Mean Entropy (0.7): {mean_entropy_07:.4f}"
        print(output_string)
        return {'Energy Score (0.5:0.95)': es, 
                'NLL (0.5:0.95)': nll,
                'Mean Entropy (0.5:0.95)': mean_entropy,
                'Energy Score (0.7)': es_07,
                'NLL (0.7)': nll_07,
                'Mean Entropy (0.7)': mean_entropy_07}
    
    def summarize(self, metric='bbox'):
        if metric == 'bbox':
            super().summarize()
        elif metric == 'scoring':
            self.scoring_stats = self._summarizeScoring()
        else:
            raise ValueError('Unknown metric in COCOevalPOD summarize: {}'.format(metric))