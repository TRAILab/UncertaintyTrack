_base_ = [
    '../bayesod/bayesod_cov-int.py',
    '../_base_/datasets/mot_challenge.py', '../_base_/default_runtime.py'
]

model = dict(
    type='ProbabilisticByteTrack',
    detector=dict(
        bbox_head=dict(
            num_classes=1,
            with_sampling=True,
            compute_cls_var=False,
        ),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            '/home/results/bayesod_diag_cov-int_es_L1_att_1xb2_4e_mot17-half/latest.pth')
    ),
    motion=dict(type='KalmanFilter'),
    tracker=dict(
        type='ByteTracker',
        obj_score_thrs=dict(high=0.6, low=0.1),
        init_track_thr=0.7,
        weight_iou_with_det_scores=True,
        match_iou_thrs=dict(high=0.1, low=0.5, tentative=0.3),
        num_frames_retain=30))

total_epochs = 4
evaluation = dict(metric=['bbox', 'track'], interval=1)
search_metrics = ['MOTA', 'IDF1', 'FN', 'FP', 'IDs']
seed = 0

eval_json = 'baseline'