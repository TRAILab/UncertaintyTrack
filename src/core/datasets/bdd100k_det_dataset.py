"""Definition of the BDD100K dataset."""

import os
import os.path as osp
from typing import List

import numpy as np
from mmdet.datasets import DATASETS
from scalabel.label.io import save
from scalabel.label.transforms import bbox_to_box2d
from scalabel.label.typing import Frame, Label

from .probabilistic_coco_dataset import ProbabilisticCocoDataset


@DATASETS.register_module()
class BDD100KDetDataset(ProbabilisticCocoDataset):  # type: ignore
    """BDD100K Dataset for detection."""

    CLASSES = ('pedestrian', 'rider', 'car', 'bus', 'truck', 'bicycle',
               'motorcycle', 'train')
    #* Removed "traffic light", "traffic sign" from CLASSES
    #* because they are not in the BDD MOT2020 dataset

    def convert_format(
        self, results: List[List[np.ndarray]], out_dir: str  # type: ignore
    ) -> None:
        """Format the results to the BDD100K prediction format."""
        assert isinstance(results, list), "results must be a list"
        assert len(results) == len(
            self
        ), f"Length of res and dset not equal: {len(results)} != {len(self)}"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        frames = []
        ann_id = 0

        for img_idx in range(len(self)):
            img_name = self.data_infos[img_idx]["file_name"]
            frame = Frame(name=img_name, labels=[])
            frames.append(frame)

            result = results[img_idx]
            for cat_idx, bboxes in enumerate(result):
                for bbox in bboxes:
                    ann_id += 1
                    label = Label(
                        id=ann_id,
                        score=bbox[-1],
                        box2d=bbox_to_box2d(self.xyxy2xywh(bbox)),
                        category=self.CLASSES[cat_idx],
                    )
                    frame.labels.append(label)  # type: ignore

        out_path = osp.join(out_dir, "det.json")
        save(out_path, frames)