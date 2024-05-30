"""This config is only for tracking inference.
For detector training, please use the appropriate config for the detector."""
_base_ = [
    '../bayesod/bayesod_cov-int.py',
    '../_base_/datasets/mot_challenge.py', '../_base_/default_runtime.py'
]

#* some hyper parameters
total_epochs = 4

model = dict(
    type='UncertainMOT',
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
    motion=dict(type='KalmanFilterWithUncertainty'),
    tracker=dict(
        type='ProbabilisticByteTracker',
        obj_score_thrs=dict(high=0.6, low=0.1),
        init_track_thr=0.7,
        weight_iou_with_det_scores=True,
        match_iou_thrs=dict(high=0.1, low=0.5, tentative=0.3),
        num_frames_retain=30,
        with_covariance=True,
        det_score_mode='confidence',
        final_matching=True,
        expand_boxes=True,
        init_percent=0.7,
        final_percent=0.3,
        init_ellipse_filter=True,
        second_ellipse_filter=True,
        final_ellipse_filter=True,
        use_mahalanobis=False,
        return_expanded=False,
        return_remaining_expanded=False,
        # primary_cascade=dict(num_bins=None),
        primary_cascade=None,
        # secondary_fn=dict(type="wasserstein", threshold=7.5e3),
        secondary_fn=None,
        secondary_cascade=dict(num_bins=None)))
        # secondary_cascade=None))

log_config = dict(
    _delete_=True,
    interval=50,
    hooks=[
        dict(type='TextLoggerHook')
    ])
evaluation = dict(metric=['bbox', 'track'])

#*----------------------------------------------------------
data = dict(
    persistent_workers=True,
    test=dict(
        type="ProbabilisticMOTChallengeDataset",))
        # interpolate_tracks_cfg=dict(min_num_frames=5, max_num_frames=20)))

search_metrics = ['MOTA', 'IDF1', 'FP', 'FN', 'IDs']
seed = 0
eval_json = 'final'