_base_ = [
    '../_base_/models/prob_yolox_x.py',
    '../_base_/datasets/mot_challenge_det.py', '../_base_/default_runtime.py'
]
USE_MMDET=True

#* some hyper parameters
img_scale = (800, 1440)
samples_per_gpu = 4
total_epochs = 80
num_last_epochs = 10    #TODO: configure this (original: 10)
#* need num_last_epochs < total_epochs - 1
#* switch at (runner.max_epochs - self.num_last_epochs)
mode_switch_epoch = 10
start_eval_epoch = total_epochs - mode_switch_epoch - 1   #* start eval two epochs before switching mode
interval = 5
exp_name = "prob_yolox_x_es_mot17"
out_dir = "/home/results/" + exp_name
load_from = '/home/misc/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth'  # noqa
# resume_from = out_dir + "/latest.pth"
resume_from = None

model = dict(
    detector=dict(
        input_size=img_scale,
        random_size_range=(18, 32),
        bbox_head=dict(
            num_classes=1,
            post_process="covariance_intersection",
            separate_levels=False,
            post_process_mlvl="bayesian",
            loss_l1=dict(type='ESLoss', 
                         loss_type="L1")
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
                        # 'id': ''     # find id from wandb run
                        },
            interval=50)
    ])

evaluation = dict(metric=['bbox', 'scoring'], 
                  start=start_eval_epoch,
                  interval=2)

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
                '/home/data/MOT17/annotations/train_cocoformat.json',
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
        type='ProbabilisticCocoDataset',
        pipeline=test_pipeline),
    test=dict(
        type='ProbabilisticCocoDataset',
        ann_file='/home/data/MOT17/annotations/test_cocoformat.json',
        img_prefix='/home/data/MOT17/test',
        pipeline=test_pipeline))

# you need to set mode='dynamic' if you are using pytorch<=1.5.0
fp16 = dict(loss_scale=dict(init_scale=512.))
checkpoint_config = dict(interval=5, max_keep_ckpts=2)

# learning policy
lr_config = dict(
    policy='YOLOX',
    warmup='exp',
    by_epoch=False,
    warmup_by_epoch=True,
    warmup_ratio=1,
    warmup_iters=1,
    num_last_epochs=num_last_epochs,
    min_lr_ratio=0.05)

custom_hooks = [
    dict(
        type='YOLOXModeSwitchHook',
        num_last_epochs=mode_switch_epoch,
        priority=48),
    dict(
        type='SyncNormHook',
        num_last_epochs=mode_switch_epoch,
        interval=interval,
        priority=48),
    dict(
        type='ExpMomentumEMAHook',
        resume_from=resume_from,
        momentum=0.0001,
        priority=49)
]

optimizer_config = dict(grad_clip=None)
#* optimizer
#* default 8 gpu, 8 batch size
optimizer = dict(
    type='SGD',
    lr=0.001 * (samples_per_gpu / 8),
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True,
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0))
auto_scale_lr = dict(enable=True, base_batch_size=8*samples_per_gpu)