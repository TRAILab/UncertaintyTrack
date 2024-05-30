

from mmdet.datasets.builder import DATASETS

from .prob_coco_video_dataset import ProbabilisticCocoVideoDataset

@DATASETS.register_module()
class BDDVideoDataset(ProbabilisticCocoVideoDataset):

    CLASSES = ('pedestrian', 'rider', 'car', 'truck', 'bus', 'train',
               'motorcycle', 'bicycle')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)