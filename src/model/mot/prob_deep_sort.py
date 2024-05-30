"""Implementation of UncertainMOT. Adapted from mmtracking DeepSORT."""

from mmtrack.core import outs2results
from mmtrack.models.builder import MODELS
from mmtrack.models.mot.deep_sort import DeepSORT

from core.utils.analysis_utils import analyze_results


@MODELS.register_module()
class ProbabilisticDeepSORT(DeepSORT):
    """
    """
    def __init__(self,
                 detector=None,
                 reid=None,
                 tracker=None,
                 motion=None,
                 pretrains=None,
                 init_cfg=None):
        super(ProbabilisticDeepSORT, self).__init__(
                                    detector=detector,
                                    reid=reid,
                                    tracker=tracker,
                                    motion=motion,
                                    pretrains=pretrains,
                                    init_cfg=init_cfg)

    def simple_test(self,
                    img,
                    img_metas,
                    rescale=False,
                    public_bboxes=None,
                    **kwargs):
        """Test without augmentations.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            rescale (bool, optional): If False, then returned bboxes and masks
                will fit the scale of img, otherwise, returned bboxes and masks
                will fit the scale of original image shape. Defaults to False.
            public_bboxes (list[Tensor], optional): Public bounding boxes from
                the benchmark. Defaults to None.

        Returns:
            dict[str : list(ndarray)]: The tracking results.
        """
        analysis_cfg = kwargs.get('analysis_cfg', {})
        frame_id = img_metas[0].get('frame_id', -1)
        #? Reset tracks at the beginning of each sequence
        if frame_id == 0:
            self.tracker.reset()

        x = self.detector.extract_feat(img)
        if public_bboxes is not None:
            raise NotImplementedError("UncertainMOT does not support public_bboxes.")
        if hasattr(self.detector, 'roi_head'):
            raise NotImplementedError("UncertainMOT does not support roi_head.")
        elif hasattr(self.detector, 'bbox_head'):
            outs = self.detector.bbox_head(x)
            result_list = self.detector.bbox_head.get_bboxes(
                *outs, img_metas=img_metas, rescale=rescale)
            
            #? Check if output from probabilistic network
            #? follow the same format as ProbabilisticRetinaHead.get_bboxes
            if len(result_list[0]) == 4:
                det_bboxes, det_bbox_covs, det_labels, det_score_vars = result_list[0]
            else:
                det_bboxes, det_labels = result_list[0]
                det_bbox_covs = None
                det_score_vars = None
            num_classes = self.detector.bbox_head.num_classes
        else:
            raise TypeError('detector must have bbox_head.')

        track_results = self.tracker.track(
            img=img,
            img_metas=img_metas,
            model=self,
            feats=x,
            bboxes=det_bboxes,
            bbox_covs=det_bbox_covs,
            labels=det_labels,
            score_vars=det_score_vars,
            frame_id=frame_id,
            rescale=rescale,
            **kwargs)
        track_bboxes, track_labels, track_ids = track_results['regular']
        final_track_results = outs2results(
            bboxes=track_bboxes,
            labels=track_labels,
            ids=track_ids,
            num_classes=num_classes)
        final_det_results = outs2results(
            bboxes=det_bboxes, labels=det_labels, num_classes=num_classes)
        results =  dict(
            det_bboxes=final_det_results['bbox_results'],
            track_bboxes=final_track_results['bbox_results'])

        #? Analyze results
        if analysis_cfg:
            analysis_results = track_results.get('analysis', None)
            analysis_dict = analyze_results(analysis_cfg, analysis_results)
            results.update(analysis_dict)

        return results