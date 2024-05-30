_base_ = "./retinanet_r50_fpn.py"

# model settings
model = dict(
    detector=dict(
        type="ProbabilisticRetinaNet",
        bbox_head=dict(
            type="ProbabilisticRetinaHead",
            covariance_type="diagonal",
            affinity_thr=0.9,
            loss_bbox=dict(type='NLL', 
                            loss_type="L1",
                            loss_weight=1.0),
            loss_cls=dict(
                type='SampleFocalLoss',
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0,
                num_samples=10),
            init_cfg=
                dict(type='Xavier', layer='Conv2d', std=0.01, override=[
                    dict(type='Xavier', name='retina_cls_var', layer='Conv2d', bias=-10.0, std=0.01),
                    dict(type='Xavier', name='retina_reg_cov', layer='Conv2d', bias=0.0, std=0.0001)
                    # dict(type='Normal', name='retina_reg_cov', layer='Conv2d', mean=0.0, std=0.0001)
                ])
        ),
    )
)