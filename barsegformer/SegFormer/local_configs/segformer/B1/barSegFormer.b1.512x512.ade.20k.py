_base_ = [
    '../../_base_/models/segformer.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_160k_adamw.py'
]
dist_params = dict(backend='gloo')
norm_cfg = dict(type='BN', requires_grad=True)
find_unused_parameters = True

model = dict(
    type='EncoderDecoder',
    pretrained='pretrained/mit_b1.pth',
    backbone=dict(
        type='mit_b1',
        style='pytorch',
        norm_cfg=dict(type='BN', requires_grad=True)),
    decode_head=dict(
        type='BarSegFormerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=256,
        embed_dim=256,
        edge_loss_weight=0.4,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# optimizer
optimizer = dict(_delete_=True, type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
lr_config = dict(_delete_=True, policy='step', step=[10000, 20000])

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
dataset_type = 'CustomDataset'
data_root = 'data/my_dataset'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train_images',
        ann_dir='train_masks',
        img_suffix='.jpg',
        seg_map_suffix='.png',
        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='val_images',
        ann_dir='val_masks',
        img_suffix='.jpg',
        seg_map_suffix='.png',
        pipeline=test_pipeline
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='val_images',
        ann_dir='val_masks',
        img_suffix='.jpg',
        seg_map_suffix='.png',
        pipeline=test_pipeline
    )
)
evaluation = dict(interval=2000, metric='mIoU')
runner = dict(type='IterBasedRunner', max_iters=20000)
checkpoint_config = dict(interval=2000)
load_from = None
