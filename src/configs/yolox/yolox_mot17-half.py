_base_ = [
    '../_base_/models/yolox_x.py',
    '../_base_/datasets/mot_challenge_det.py', '../_base_/default_runtime.py'
]
USE_MMDET=True

#* some hyper parameters
img_scale = (800, 1440)
samples_per_gpu = 4
exp_name = "yolox_mot17-half"
out_dir = "/home/results/" + exp_name
load_from = '/home/misc/bytetrack_yolox_x_crowdhuman_mot17-private-half_20211218_205500-1985c9f0.pth'  # noqa
resume_from = None
total_epochs = 100

model = dict(
    detector=dict(
        input_size=img_scale,
        random_size_range=(18, 32),
        bbox_head=dict(
            num_classes=1,
        ),
        test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.7)),
    ))

log_config = dict(
    _delete_=True,
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='WandbLoggerHook',
            init_kwargs={'name': exp_name,
                        'project': 'yolox_uncertainmot',
                         'dir': out_dir,
                        'notes': '',
                        'sync_tensorboard': True,
                        'resume': 'allow',   # set to must if need to resume; set id corresponding to run
                        'id': 'f1k0rxza'     # find id from wandb run
                        },
            interval=50)
    ])

evaluation = dict(metric=['bbox'])

#*----------------------------------------------------------
train_pipeline = [
    dict(
        type='Mosaic',
        img_scale=img_scale,
        pad_val=114.0,
        bbox_clip_border=False),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.1, 2),
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        bbox_clip_border=False),
    dict(
        type='MixUp',
        img_scale=img_scale,
        ratio_range=(0.8, 1.6),
        pad_val=114.0,
        bbox_clip_border=False),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Resize',
        img_scale=img_scale,
        keep_ratio=True,
        bbox_clip_border=False),
    dict(type='Pad', size_divisor=32, pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(
                type='Pad',
                size_divisor=32,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='VideoCollect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=samples_per_gpu,
    persistent_workers=True,
    train=dict(
        _delete_=True,
        type='MultiImageMixDataset',
        dataset=dict(
            type='ProbabilisticCocoDataset',
            ann_file=[
                '/home/data/MOT17/annotations/half-train_cocoformat.json',
                '/home/data/crowdhuman/annotations/crowdhuman_train.json',
                '/home/data/crowdhuman/annotations/crowdhuman_val.json'
            ],
            img_prefix=[
                '/home/data/MOT17/train', '/home/data/crowdhuman/train',
                '/home/data/crowdhuman/val'
            ],
            classes=('pedestrian', ),
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True)
            ],
            filter_empty_gt=False),
        pipeline=train_pipeline),
    val=dict(
        pipeline=test_pipeline),
    test=dict(
        pipeline=test_pipeline))

# you need to set mode='dynamic' if you are using pytorch<=1.5.0
fp16 = dict(loss_scale=dict(init_scale=512.))