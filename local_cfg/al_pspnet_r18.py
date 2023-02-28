_base_ = [
    '../configs/_base_/models/pspnet_r50-d8.py',
    './_base_/active_learning_runtime.py',
    './_base_/active_learning_schedule.py'
]

norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    pretrained='open-mmlab://resnet18_v1c',
    backbone=dict(
        depth=18,
        norm_cfg=norm_cfg),
    decode_head=dict(
        in_channels=512,
        channels=128,
        norm_cfg=norm_cfg
    ),
    auxiliary_head=dict(in_channels=256, channels=64))

# dataset settings
dataset_type = 'CityscapesDataset'
data_root = 'Dataset/Cityscapes'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 1024)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 1024),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

training_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    img_dir='images/train',
    ann_dir='gtFine/train',
    pipeline=train_pipeline,
    split='batch_0.txt'
)

unlabeled_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    img_dir='images/train',
    ann_dir='gtFine/train',
    pipeline=test_pipeline
)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
    train=training_dataset,
    unlabeled=unlabeled_dataset,
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/val',
        ann_dir='gtFine/val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/val',
        ann_dir='gtFine/val',
        pipeline=test_pipeline))

