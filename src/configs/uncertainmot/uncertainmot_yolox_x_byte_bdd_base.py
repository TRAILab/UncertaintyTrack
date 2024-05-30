"""This config is only for tracking inference.
For detector training, please use the appropriate config for the detector."""
_base_ = [
    '../_base_/models/prob_yolox_x.py',
    '../_base_/datasets/bdd_mot.py', '../_base_/default_runtime.py'
]

#* some hyper parameters
img_scale = (800, 1440)
total_epochs = 50
weights_path = "prob_yolox_x_es_bdd"

model = dict(
    type='UncertainMOT',
    detector=dict(
        input_size=img_scale,
        random_size_range=(18, 32),
        bbox_head=dict(
            num_classes=8,
            post_process="covariance_intersection",
            loss_l1=dict(type='ESLoss', 
                         loss_type="L1")
        ),
        test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.7)),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            f"/home/results/{weights_path}/latest.pth"  # noqa: E501
        )
    ),
    motion=dict(type='KalmanFilterWithUncertainty'),
    tracker=dict(
        type='ProbabilisticByteTracker',
        obj_score_thrs=dict(high=0.6, low=0.1),
        init_track_thr=0.7,
        weight_iou_with_det_scores=True,
        match_iou_thrs=dict(high=0.1, low=0.5, tentative=0.3),
        num_frames_retain=30,
        with_covariance=False,
        det_score_mode='confidence',
        final_matching=False,
        expand_boxes=False,
        init_percent=0.65,
        final_percent=0.3,
        init_ellipse_filter=False,
        second_ellipse_filter=False,
        final_ellipse_filter=False,
        use_mahalanobis=False,
        return_expanded=False,
        return_remaining_expanded=False,
        # primary_cascade=dict(num_bins=None),
        primary_cascade=None,
        # secondary_fn=dict(type="wasserstein", threshold=7.5e3),
        secondary_fn=None,
        # secondary_cascade=dict(num_bins=None)))
        secondary_cascade=None))

log_config = dict(
    _delete_=True,
    interval=50,
    hooks=[
        dict(type='TextLoggerHook')
    ])
evaluation = dict(metric=['bbox', 'track'])

#*----------------------------------------------------------
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
    workers_per_gpu=8,
    persistent_workers=True,
    val=dict(
        pipeline=test_pipeline,),
        # interpolate_tracks_cfg=dict(min_num_frames=5, max_num_frames=20)),
    test=dict(
        pipeline=test_pipeline,))
        # interpolate_tracks_cfg=dict(min_num_frames=5, max_num_frames=20)))

search_metrics = ['MOTA', 'IDF1', 'FP', 'FN', 'IDSw']
seed = 0
eval_json = '_'