_base_ = "./yolox_x.py"

# model settings
model = dict(
    detector=dict(
        type='ProbabilisticYOLOX',
        bbox_head=dict(
            type='ProbabilisticYOLOXHead',
            post_process="nms",
            compute_cls_var=False,
            affinity_thr=0.9,
            loss_cls=dict(type='SampleCrossEntropyLoss',
                          use_sigmoid=True,
                          reduction='sum',
                          loss_weight=1.0),
            loss_bbox=dict(type='SampleIoULoss',
                           mode='square',
                           eps=1e-16,
                           reduction='sum',
                           loss_weight=5.0),
            loss_l1=dict(type='L1Loss', reduction='sum', loss_weight=1.0)
        )
    )
)