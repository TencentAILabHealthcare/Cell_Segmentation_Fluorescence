# The new config inherits a base config to highlight the necessary modification
_base_ = '../../configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco.py'

# custom_imports = dict(imports=['mmdet.datasets.pipelines.cell_transforms'], allow_failed_imports=False)

# use caffe img_norm
train_pipeline = [

    dict(type='Load16BitImageFromFile'),
    dict(
            type='LoadAnnotations',
            with_bbox=True,
            with_mask=True,
            poly2mask=True
    ),
    dict(
            type='Resize',
            img_scale=(256, 256),
            ratio_range=(1, 1),
            multiscale_mode='range',
            keep_ratio=True
    ),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='CellNormalizeTransform'),
    dict(type='CellRepeatTransform'),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='Load16BitImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(256, 256),
        flip=False,
        transforms=[
            dict(type='Resize',keep_ratio=True),
            dict(type='CellNormalizeTransform'),
            dict(type='CellRepeatTransform'),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ])
]

# Modify dataset related settings
dataset_type = 'CellDataset'
classes = ('cell',)
data_root = "/mnt/zihanwu/wentaopan/cell_instance_segmentation/data/ssDNA"
train_set = '/mnt/zihanwu/wentaopan/cell_instance_segmentation/data/ssDNA/train.json'
val_set = '/mnt/zihanwu/wentaopan/cell_instance_segmentation/data/ssDNA/val.json'
data = dict(
    samples_per_gpu=8,  # 单个 GPU 的 Batch size
    workers_per_gpu=8,  # 单个 GPU 分配的数据加载线程数
    train=dict(
        type=dataset_type,
        img_prefix=data_root,
        classes=classes,
        ann_file=train_set,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        img_prefix=data_root,
        classes=classes,
        ann_file=val_set,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        img_prefix=data_root,
        classes=classes,
        ann_file=val_set,
        pipeline=test_pipeline))

model = dict(
    # backbone=dict(
    #     in_channels=3
    # ),
    rpn_head=dict(
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[2],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        ),
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1)
        ),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=28,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=500,
            mask_thr_binary=0.5))
    )

# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = '/mnt/jumpyan/guy2/wentaopan/cell/pretrained_model/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
# load_from = "/mnt/zihanwu/wentaopan/cell_instance_segmentation/work_dir/base/epoch_50.pth"

evaluation = dict(  # evaluation hook 的配置，更多细节请参考 https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/evaluation/eval_hooks.py#L7。
    interval=1,  # 验证的间隔。
    iou_thrs=[0.3, 0.35, 0.4, 0.45, 0.5],
    metric=['segm'],
    proposal_nums=(300, 400, 500),
    )  # 验证期间使用的指标。

# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1,
    warmup_ratio=0.001,
    step=[110, 130]
    )
runner = dict(type='EpochBasedRunner', max_epochs=160)

log_config = dict(  # config to register logger hook
    interval=1,  # Interval to print the log
    hooks=[
        # dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')  # The Tensorboard logger is also supported
    ])  # The logger used to record the training process.
work_dir="/mnt/zihanwu/wentaopan/cell_instance_segmentation/work_dir/ssDNA_with_incomplete_annotations/baseline_110_130_500"
fp16 = dict(loss_scale='dynamic')