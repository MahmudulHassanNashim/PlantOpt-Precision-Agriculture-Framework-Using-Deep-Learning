auto_scale_lr = dict(base_batch_size=16, enable=False)
backend_args = None
classes = ('seedling', )
data_root = '/home/hasnatpie/thesis01/seedling_mm/'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(
        rule='greater', save_best='coco/bbox_mAP', type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(
        draw=True,
        test_out_dir=
        '/home/hasnatpie/thesis01/results_spikenet20/test_results/test_images',
        type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = '/home/hasnatpie/thesis01/results_spikenet20/best_coco_bbox_mAP_epoch_13.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
model = dict(
    backbone=dict(
        dcn=dict(deform_groups=1, fallback_on_stride=False, type='DCN'),
        depth=50,
        frozen_stages=1,
        init_cfg=dict(checkpoint='torchvision://resnet50', type='Pretrained'),
        norm_cfg=dict(requires_grad=True, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        stage_with_dcn=(
            False,
            True,
            True,
            True,
        ),
        style='pytorch',
        type='ResNet'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_mask=True,
        pad_size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor'),
    neck=[
        dict(
            in_channels=[
                256,
                512,
                1024,
                2048,
            ],
            num_outs=5,
            out_channels=256,
            type='FPN'),
        dict(in_channels=256, num_levels=5, type='BFP'),
    ],
    roi_head=dict(
        bbox_head=[
            dict(
                bbox_coder=dict(
                    num_buckets=14,
                    scale_factor=1.7,
                    type='BucketingBBoxCoder'),
                cls_in_channels=256,
                cls_out_channels=1024,
                loss_bbox_cls=dict(
                    loss_weight=1.0, type='CrossEntropyLoss',
                    use_sigmoid=True),
                loss_bbox_reg=dict(
                    beta=0.1, loss_weight=1.0, type='SmoothL1Loss'),
                loss_cls=dict(
                    loss_weight=1.0,
                    type='CrossEntropyLoss',
                    use_sigmoid=False),
                norm_cfg=None,
                num_classes=1,
                num_cls_fcs=1,
                num_reg_fcs=0,
                reg_class_agnostic=True,
                reg_cls_out_channels=256,
                reg_feat_up_ratio=2,
                reg_in_channels=256,
                reg_offset_out_channels=256,
                reg_post_kernel=3,
                reg_post_num=1,
                reg_pre_kernel=3,
                reg_pre_num=2,
                roi_feat_size=7,
                type='SABLHead'),
            dict(
                bbox_coder=dict(
                    num_buckets=14,
                    scale_factor=1.5,
                    type='BucketingBBoxCoder'),
                cls_in_channels=256,
                cls_out_channels=1024,
                loss_bbox_cls=dict(
                    loss_weight=1.0, type='CrossEntropyLoss',
                    use_sigmoid=True),
                loss_bbox_reg=dict(
                    beta=0.1, loss_weight=1.0, type='SmoothL1Loss'),
                loss_cls=dict(
                    loss_weight=1.0,
                    type='CrossEntropyLoss',
                    use_sigmoid=False),
                norm_cfg=None,
                num_classes=1,
                num_cls_fcs=1,
                num_reg_fcs=0,
                reg_class_agnostic=True,
                reg_cls_out_channels=256,
                reg_feat_up_ratio=2,
                reg_in_channels=256,
                reg_offset_out_channels=256,
                reg_post_kernel=3,
                reg_post_num=1,
                reg_pre_kernel=3,
                reg_pre_num=2,
                roi_feat_size=7,
                type='SABLHead'),
            dict(
                bbox_coder=dict(
                    num_buckets=14,
                    scale_factor=1.3,
                    type='BucketingBBoxCoder'),
                cls_in_channels=256,
                cls_out_channels=1024,
                loss_bbox_cls=dict(
                    loss_weight=1.0, type='CrossEntropyLoss',
                    use_sigmoid=True),
                loss_bbox_reg=dict(
                    beta=0.1, loss_weight=1.0, type='SmoothL1Loss'),
                loss_cls=dict(
                    loss_weight=1.0,
                    type='CrossEntropyLoss',
                    use_sigmoid=False),
                norm_cfg=None,
                num_classes=1,
                num_cls_fcs=1,
                num_reg_fcs=0,
                reg_class_agnostic=True,
                reg_cls_out_channels=256,
                reg_feat_up_ratio=2,
                reg_in_channels=256,
                reg_offset_out_channels=256,
                reg_post_kernel=3,
                reg_post_num=1,
                reg_pre_kernel=3,
                reg_pre_num=2,
                roi_feat_size=7,
                type='SABLHead'),
        ],
        bbox_roi_extractor=dict(
            aggregation='sum',
            featmap_strides=[
                4,
                8,
                16,
                32,
            ],
            out_channels=256,
            post_cfg=dict(
                attention_type='0100',
                in_channels=256,
                kv_stride=2,
                num_heads=6,
                spatial_range=-1,
                type='GeneralizedAttention'),
            pre_cfg=dict(
                in_channels=256,
                inplace=False,
                kernel_size=5,
                out_channels=256,
                padding=2,
                type='ConvModule'),
            roi_layer=dict(output_size=7, sampling_ratio=2, type='RoIAlign'),
            type='GenericRoIExtractor'),
        mask_head=dict(
            conv_out_channels=256,
            in_channels=256,
            loss_mask=dict(
                loss_weight=1.0, type='CrossEntropyLoss', use_mask=True),
            num_classes=1,
            num_convs=4,
            type='FCNMaskHead'),
        mask_roi_extractor=dict(
            featmap_strides=[
                4,
                8,
                16,
                32,
            ],
            out_channels=256,
            post_cfg=dict(
                attention_type='0100',
                in_channels=256,
                kv_stride=2,
                num_heads=6,
                spatial_range=-1,
                type='GeneralizedAttention'),
            pre_cfg=dict(
                in_channels=256,
                inplace=False,
                kernel_size=5,
                out_channels=256,
                padding=2,
                type='ConvModule'),
            roi_layer=dict(output_size=14, sampling_ratio=2, type='RoIAlign'),
            type='GenericRoIExtractor'),
        num_stages=3,
        stage_loss_weights=[
            1,
            0.5,
            0.25,
        ],
        type='CascadeRoIHead'),
    rpn_head=dict(
        anchor_generator=dict(
            ratios=[
                0.5,
                1.0,
                2.0,
            ],
            scales=[
                8,
            ],
            strides=[
                4,
                8,
                16,
                32,
                64,
            ],
            type='AnchorGenerator'),
        bbox_coder=dict(
            target_means=[
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            target_stds=[
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            type='DeltaXYWHBBoxCoder'),
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(
            beta=0.1111111111111111, loss_weight=1.0, type='SmoothL1Loss'),
        loss_cls=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=True),
        type='RPNHead'),
    test_cfg=dict(
        rcnn=dict(
            mask_thr_binary=0.5,
            max_per_img=100,
            nms=dict(iou_threshold=0.5, type='nms'),
            score_thr=0.5),
        rpn=dict(
            max_per_img=1000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type='nms'),
            nms_pre=1000)),
    train_cfg=dict(
        rcnn=[
            dict(
                assigner=dict(
                    ignore_iof_thr=-1,
                    match_low_quality=False,
                    min_pos_iou=0.5,
                    neg_iou_thr=0.5,
                    pos_iou_thr=0.5,
                    type='MaxIoUAssigner'),
                debug=False,
                mask_size=28,
                pos_weight=-1,
                sampler=dict(
                    add_gt_as_proposals=True,
                    neg_pos_ub=-1,
                    num=512,
                    pos_fraction=0.25,
                    type='RandomSampler')),
            dict(
                assigner=dict(
                    ignore_iof_thr=-1,
                    match_low_quality=False,
                    min_pos_iou=0.6,
                    neg_iou_thr=0.6,
                    pos_iou_thr=0.6,
                    type='MaxIoUAssigner'),
                debug=False,
                mask_size=28,
                pos_weight=-1,
                sampler=dict(
                    add_gt_as_proposals=True,
                    neg_pos_ub=-1,
                    num=512,
                    pos_fraction=0.25,
                    type='RandomSampler')),
            dict(
                assigner=dict(
                    ignore_iof_thr=-1,
                    match_low_quality=False,
                    min_pos_iou=0.7,
                    neg_iou_thr=0.7,
                    pos_iou_thr=0.7,
                    type='MaxIoUAssigner'),
                debug=False,
                mask_size=28,
                pos_weight=-1,
                sampler=dict(
                    add_gt_as_proposals=True,
                    neg_pos_ub=-1,
                    num=512,
                    pos_fraction=0.25,
                    type='RandomSampler')),
        ],
        rpn=dict(
            allowed_border=0,
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=True,
                min_pos_iou=0.5,
                neg_iou_thr=0.5,
                pos_iou_thr=0.7,
                type='MaxIoUAssigner'),
            debug=False,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=False,
                neg_pos_ub=-1,
                num=256,
                pos_fraction=0.5,
                type='RandomSampler')),
        rpn_proposal=dict(
            max_per_img=2000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type='nms'),
            nms_pre=2000)),
    type='CascadeRCNN')
optim_wrapper = dict(
    optimizer=dict(lr=0.02, momentum=0.9, type='SGD', weight_decay=0.0001),
    type='OptimWrapper')
palette = [
    (
        240,
        10,
        10,
    ),
]
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=500, start_factor=0.001, type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=12,
        gamma=0.1,
        milestones=[
            9,
            11,
        ],
        type='MultiStepLR'),
]
resume = True
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='annotations/test_coco.json',
        backend_args=None,
        data_prefix=dict(img='test/'),
        data_root='/home/hasnatpie/thesis01/seedling_mm/',
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1333,
                800,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='/home/hasnatpie/thesis01/seedling_mm/annotations/test_coco.json',
    backend_args=None,
    metric='bbox',
    type='CocoMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        1333,
        800,
    ), type='Resize'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
train_cfg = dict(max_epochs=20, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=1,
    dataset=dict(
        dataset=dict(
            ann_file='annotations/train_coco.json',
            backend_args=None,
            data_prefix=dict(img='train/'),
            data_root='/home/hasnatpie/thesis01/seedling_mm/',
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=[
                dict(backend_args=None, type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
                dict(
                    keep_ratio=True,
                    scale=[
                        (
                            1333,
                            640,
                        ),
                        (
                            1333,
                            800,
                        ),
                    ],
                    type='RandomResize'),
                dict(prob=0.5, type='RandomFlip'),
                dict(type='PackDetInputs'),
            ],
            type='CocoDataset'),
        times=3,
        type='RepeatDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        keep_ratio=True,
        scale=[
            (
                1333,
                640,
            ),
            (
                1333,
                800,
            ),
        ],
        type='RandomResize'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='annotations/val_coco.json',
        backend_args=None,
        data_prefix=dict(img='valid/'),
        data_root='/home/hasnatpie/thesis01/seedling_mm/',
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1333,
                800,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='/home/hasnatpie/thesis01/seedling_mm/annotations/val_coco.json',
    backend_args=None,
    metric='bbox',
    type='CocoMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ])
work_dir = '/home/hasnatpie/thesis01/results_spikenet20'
