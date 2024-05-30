import os
import os.path as osp
import json

import torch
import numpy as np
from scalabel.label.transforms import bbox_to_box2d
from scalabel.label.typing import Frame, Label, Dataset
from tqdm import tqdm
from typing import Any, List, Optional, Union

from core.utils import bbox_xyxy_to_xywh


BDD_CATEGORIES = [
    'pedestrian', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
    'bicycle'
]

def box_track_to_bdd100k(dataset, results, out_base):
    bdd100k = []
    track_base = osp.join(out_base, "box_track")
    if not osp.exists(track_base):
        os.makedirs(track_base)

    print(f'\nStart converting to BDD100K box tracking format')
    for idx, tracks_per_cls in tqdm(enumerate(results['track_bboxes'])):
        img_name = dataset.data_infos[idx]['file_name']
        frame_index = dataset.data_infos[idx]['frame_id']
        vid_name = os.path.split(img_name)[0]
        
        frame = Frame(
            name=img_name,
            url=None,
            videoName=vid_name,
            frameIndex=frame_index,
            labels=[])
        
        for cls, tracks in enumerate(tracks_per_cls):
            for _, track in enumerate(tracks):
                #* track: [id, x1, y1, x2, y2, score]
                id = int(track[0]) + 1  #* id starts from 1 in gt
                bbox = torch.from_numpy(track[1:5])
                score = track[-1]
                label = Label(
                    id=id,
                    score=score,
                    box2d=bbox_to_box2d(bbox_xyxy_to_xywh(bbox).tolist()),
                    category=BDD_CATEGORIES[cls],
                    box3d=None,
                    poly2d=None,
                    rle=None,
                    graph=None)
                frame.labels.append(label)
        bdd100k.append(frame)

    print(f'\nWriting the converted json')
    out_path = osp.join(out_base, "box_track.json")
    save(out_path, bdd100k)
    
def save(filepath: str, dataset: Union[List[Frame], Dataset]) -> None:
    """Save labels in Scalabel format."""
    if not isinstance(dataset, Dataset):
        dataset = Dataset(frames=dataset,
                          groups=None,
                          config=None)
    dataset_dict = dataset.dict()

    with open(filepath, mode="w", encoding="utf-8") as f:
        json.dump(dataset_dict, f, indent=2)