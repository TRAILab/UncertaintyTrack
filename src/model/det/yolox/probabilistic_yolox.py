import numpy as np
import mmcv


from mmdet.models.builder import DETECTORS
from mmdet.models.detectors import YOLOX

from core.utils import bbox_and_cov2result
from core.visualization import imshow_det_bboxes


@DETECTORS.register_module()
class ProbabilisticYOLOX(YOLOX):
    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 input_size=(640, 640),
                 size_multiplier=32,
                 random_size_range=(15, 25),
                 random_size_interval=10,
                 init_cfg=None):
        super(ProbabilisticYOLOX, self).__init__(backbone, neck, bbox_head, train_cfg, test_cfg, 
                                                 pretrained, input_size, size_multiplier, 
                                                 random_size_range, random_size_interval, init_cfg)
    
    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[dict[str, list[np.ndarray]]]: BBox and BBox covariance results 
                of each image and class. The outer list corresponds to 
                each image. The inner dict contains two lists; 
                the first key ('bbox') corresponds to bboxes for each class, 
                the second key ('bbox_cov') corresponds to the covariance matrices 
                for each class.
        """
        feat = self.extract_feat(img)
        results_list = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale)
        bbox_with_cov_results = [
            bbox_and_cov2result(det_bboxes, det_bbox_covs, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_bbox_covs, det_labels, _ in results_list
        ]
        return bbox_with_cov_results
    
    def show_result(self,
                    img,
                    result,
                    score_thr=0.3,
                    bbox_color=(72, 101, 241),
                    text_color=(72, 101, 241),
                    mask_color=None,
                    thickness=2,
                    font_size=13,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (ndarray or dict[str, list[ndarray]]): The results to draw over `img`.
                Either result is bbox_result or a dictionary containing both bbox_result
                and bbox_cov_result with keys 'bbox' and 'bbox_cov', respectively.
            score_thr (float, optional): Minimum score of bboxes to be shown.
                Default: 0.3.
            bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
               The tuple of color should be in BGR order. Default: 'green'
            text_color (str or tuple(int) or :obj:`Color`):Color of texts.
               The tuple of color should be in BGR order. Default: 'green'
            mask_color (None or str or tuple(int) or :obj:`Color`):
               Color of masks. The tuple of color should be in BGR order.
               Default: None
            thickness (int): Thickness of lines. Default: 2
            font_size (int): Font size of texts. Default: 13
            win_name (str): The window name. Default: ''
            wait_time (float): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        img = mmcv.imread(img)
        img = img.copy()
        bbox_result, bbox_cov_result = result['bbox'], result['bbox_cov']
        bboxes = np.vstack(bbox_result)
        bbox_covs = np.vstack(bbox_cov_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False
        # draw bounding boxes
        img = imshow_det_bboxes(
            img,
            bboxes,
            labels,
            None,
            bbox_covs=bbox_covs,
            class_names=self.CLASSES,
            score_thr=score_thr,
            bbox_color=bbox_color,
            text_color=text_color,
            mask_color=mask_color,
            thickness=thickness,
            font_size=font_size,
            win_name=win_name,
            show=show,
            wait_time=wait_time,
            out_file=out_file)

        if not (show or out_file):
            return img