model = dict(
    type='TwoStageDetector_SPJC',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        neck_names = ['backbone_neck'],
        attention = 'None',
        convtype = 'conv2d',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=[
        dict(
            type='RPNHead',
            in_channels=256,
            feat_channels=256,
            anchor_generator=dict(
                type='AnchorGenerator',
                scales=[8],
                ratios=[0.5, 1.0, 2.0],
                strides=[4, 8, 16, 32, 64]),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[1.0, 1.0, 1.0, 1.0]),
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        dict(
            type='RPNHead',
            in_channels=256,
            feat_channels=256,
            anchor_generator=dict(
                type='AnchorGenerator',
                scales=[4],
                ratios=[0.5, 1.0, 2.0],
                strides=[4, 8, 16, 32, 64]),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[1.0, 1.0, 1.0, 1.0]),
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        dict(
            type='RPNHead',
            in_channels=256,
            feat_channels=256,
            anchor_generator=dict(
                type='AnchorGenerator',
                scales=[4],
                ratios=[0.5, 1.0, 2],
                strides=[4, 8, 16, 32, 64]),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[1.0, 1.0, 1.0, 1.0]),
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    ],
    roi_head=[
        dict(#目标检测
            type='StandardRoIHead',
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32]),
            bbox_head=dict(
                type='Shared2FCBBoxHead',
                #num_shared_convs=6,
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=7,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='L1Loss', loss_weight=1.0),
                )
        ),
        dict(#人脸检测
            type='StandardRoIHead',
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32]),
            bbox_head=dict(
                type='Shared2FCBBoxHead',
                #num_shared_convs=6,
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=2,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='L1Loss', loss_weight=1.0),
                )
        ),
        dict(#人脸关键点
            type='StandardRoIHead',
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', out_size=28, sample_num=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32]),
            bbox_head=dict(
                type='Shared2FCBBoxHead',
                num_shared_convs=0,
                with_faceKp=True,
                in_channels=256,
                fc_out_channels=512,
                roi_feat_size=28,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=False,
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
            )
        ),
        dict(#人脸性别
            type='StandardRoIHead',
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', out_size=7, sample_num=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32]),
            bbox_head=dict(
                type='Shared2FCBBoxHead',
                #num_shared_convs=6,
                with_reg=False,
                with_background=True,
                in_channels=256,
                fc_out_channels=512,
                roi_feat_size=7,
                num_classes=2,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            )
        ),
        dict(#车牌检测
            type='StandardRoIHead',
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32]),
            bbox_head=dict(
                type='Shared2FCBBoxHead',
                #num_shared_convs=6,
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='L1Loss', loss_weight=1.0),
                )
        )
    ],
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
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
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
            max_per_img=100)))
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline =[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', img_scale=(1024, 576), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_keypoints', 'gt_visibles'])
]
test_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1024, 576),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
]
data = [
    dict(
        client_ip='10.10.7.201',
        client_port=5001,
        gpu_ids=[0],
        tasktype='detect',
        samples_per_gpu=2,
        workers_per_gpu=2,
        train=dict(
            type='CocoDataset',
            ann_file='/home/chase/shy/dataset/spjgh/lunwenjson/detect_train_res.json',
            img_prefix='',
            pipeline=train_pipeline),
        val=dict(
            type='CocoDataset',
            ann_file='/home/chase/shy/dataset/spjgh/lunwenjson/detect_test_res.json',
            img_prefix='',
            pipeline=test_pipeline),
        test=dict(
            type='CocoDataset',
            ann_file='/home/chase/shy/dataset/spjgh/lunwenjson/detect_test_res.json',
            img_prefix='',
            pipeline=test_pipeline)
        ),
    dict(
        client_ip='10.10.7.201',
        client_port=5001,
        tasktype='faceKp',
        gpu_ids=[1],
        samples_per_gpu=2,
        workers_per_gpu=2,
        train=dict(
            type='CocoDataset',
            ann_file='/home/chase/shy/dataset/spjgh/lunwenjson/facekp_train_res.json',
            img_prefix='',
            pipeline=train_pipeline),
        val=dict(
            type='CocoDataset',
            ann_file='/home/chase/shy/dataset/spjgh/lunwenjson/facekp_test_res.json',
            img_prefix='',
            pipeline=test_pipeline),
        test=dict(
            type='CocoDataset',
            ann_file='/home/chase/shy/dataset/spjgh/lunwenjson/facekp_test_res.json',
            img_prefix='',
            pipeline=test_pipeline)
        ),
    dict(
        client_ip='10.10.7.205',
        client_port=5003,
        tasktype='faceGender',
        gpu_ids=[0],
        samples_per_gpu=2,
        workers_per_gpu=2,
        train=dict(
            type='CocoDataset',
            ann_file='/home/chase/shy/dataset/spjgh/lunwenjson/faceGender_train_res.json',
            img_prefix='',
            pipeline=train_pipeline),
        val=dict(
            type='CocoDataset',
            ann_file='/home/chase/shy/dataset/spjgh/lunwenjson/faceGender_test_res.json',
            img_prefix='',
            pipeline=test_pipeline),
        test=dict(
            type='CocoDataset',
            ann_file='/home/chase/shy/dataset/spjgh/lunwenjson/faceGender_test_res.json',
            img_prefix='',
            pipeline=test_pipeline)
        ),
    dict(
        client_ip='10.10.7.206',
        client_port=5004,
        tasktype='faceDetect',
        gpu_ids=[0],
        samples_per_gpu=2,
        workers_per_gpu=2,
        train=dict(
            type='CocoDataset',
            ann_file='/home/chase/shy/dataset/spjgh/lunwenjson/faceDetect_train_res.json',
            img_prefix='',
            pipeline=train_pipeline),
        val=dict(
            type='CocoDataset',
            ann_file='/home/chase/shy/dataset/spjgh/lunwenjson/faceDetect_test_res.json',
            img_prefix='',
            pipeline=test_pipeline),
        test=dict(
            type='CocoDataset',
            ann_file='/home/chase/shy/dataset/spjgh/lunwenjson/faceDetect_test_res.json',
            img_prefix='',
            pipeline=test_pipeline)
        ),
    dict(
        client_ip='10.10.7.206',
        client_port=5004,
        tasktype='carplateDetect',
        gpu_ids=[1],
        samples_per_gpu=2,
        workers_per_gpu=2,
        train=dict(
            type='CocoDataset',
            ann_file='/home/chase/shy/dataset/spjgh/lunwenjson/carplateDetect_train_res.json',
            img_prefix='',
            pipeline=train_pipeline),
        val=dict(
            type='CocoDataset',
            ann_file='/home/chase/shy/dataset/spjgh/lunwenjson/carplateDetect_test_res.json',
            img_prefix='',
            pipeline=test_pipeline),
        test=dict(
            type='CocoDataset',
            ann_file='/home/chase/shy/dataset/spjgh/lunwenjson/carplateDetect_test_res.json',
            img_prefix='',
            pipeline=test_pipeline)
    )
]
# data =
evaluation = dict(interval=21, metric='bbox')
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[10, 13])
max_epochs=15
runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)
checkpoint_config = dict(interval=1)
log_config = dict(interval=10, hooks=[dict(type='TextLoggerHook')])
total_fedlw_num = 2
custom_hooks = [dict(type='FedReload', priority='LOWEST', interval=1), dict(type='NumClassCheckHook')]
#, dict(type='FedLW', priority='LOWEST', total_fedlw_num=10)
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
auto_scale_lr = dict(enable=False, base_batch_size=16)
work_dir = './work_dirs/faster_rcnn_r50_fpn_2x_coco_spjc/lunwen2000_test_res_2080'
find_unused_parameters=True
mp_start_method = 'spawn'
#联邦训练任务Id
job_root = 'job'
job_id = 'FedAvg'
test_interval=1
#联邦融合策略
fedmerge = 'FedAvg'
fedlw = False
server_ip = '10.10.5.136'
server_port = 6000
