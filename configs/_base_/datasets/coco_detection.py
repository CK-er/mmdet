dataset_type = 'CocoDataset'
# data_root = 'data/coco/'
# data_root = 'data/dota1.5/'
# data_root = 'data/dota-v1.5_coco_512/'
data_root = 'data/dota-v1.5_1024gap512/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    dict(type='Resize', img_scale=(1300, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    # samples_per_gpu=4,
    # samples_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        # ann_file=data_root + 'annotations/instances_train2017.json',
        # ann_file=data_root + 'annotations/voc_2007_train.json',
        ann_file=data_root + 'annotations/DOTA1_5_trainval1024.json',
        img_prefix=data_root + 'trainval1024/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        # ann_file=data_root + 'annotations/instances_val2017.json',
        # ann_file=data_root + 'annotations/voc_2007_val.json',
        ann_file=data_root + 'annotations/DOTA1_5_trainval1024.json',
        img_prefix=data_root + 'trainval1024/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        # ann_file=data_root + 'annotations/instances_val2017.json',
        # ann_file=data_root + 'annotations/voc_2007_val.json',
        ann_file=data_root + 'annotations/DOTA1_5_test1024.json',
        img_prefix=data_root + 'test1024/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
# evaluation = dict(interval=4, metric='bbox')
