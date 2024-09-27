_base_ = [
    # '../_base_/datasets/inshop_bs32_448.py',
    '../_base_/schedules/cub_bs64.py',
    '../_base_/default_runtime.py',
]

dataroot = 'path/to/dataset'
numclass = 3 # your own classes
batch_size = 32
work_dir = f'../animal_behavior_runs_result/exp1/faceid/'
# set visualizer
vis_backends = [dict(type='LocalVisBackend'),
                dict(type='TensorboardVisBackend')]
visualizer = dict(type='UniversalVisualizer', vis_backends=vis_backends)

# runtime settings
default_hooks = dict(
    # log every 20 intervals
    logger=dict(type='LoggerHook', interval=20),
    # save last three checkpoints
    checkpoint=dict(
        type='CheckpointHook',
        save_best='auto',
        interval=1,
        max_keep_ckpts=3,
        rule='greater'))

# optimizer
optim_wrapper = dict(
    optimizer=dict(
        type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0005, nesterov=True))

# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=0.01,
        by_epoch=True,
        begin=0,
        end=5,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR',
        T_max=45,
        by_epoch=True,
        begin=5,
        end=50,
    )
]

train_cfg = dict(by_epoch=True, max_epochs=50, val_interval=1)

auto_scale_lr = dict(enable=True, base_batch_size=256)

custom_hooks = [
    dict(type='PrepareProtoBeforeValLoopHook'),
    dict(type='SyncBuffersHook')
]
dataset_type = 'InShop'
data_preprocessor = dict(
    num_classes=numclass,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=512),
    dict(type='RandomCrop', crop_size=448),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=512),
    dict(type='CenterCrop', crop_size=448),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=dataroot,
        split='train',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

query_dataloader = dict(
    batch_size=batch_size,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=dataroot,
        split='query',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

gallery_dataloader = dict(
    batch_size=batch_size,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=dataroot,
        split='gallery',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_dataloader = query_dataloader
val_evaluator = [
    dict(type='RetrievalRecall', topk=1),
    dict(type='RetrievalAveragePrecision', topk=10),

]

test_dataloader = val_dataloader
test_evaluator = val_evaluator

model = dict(
    type='ImageToImageRetriever',
    image_encoder=[
        dict(
            type='ResNet',
            depth=50,
            init_cfg=None),
            # init_cfg=dict(
            #     type='Pretrained', checkpoint=pretrained, prefix='backbone')),
        dict(type='GlobalAveragePooling'),
    ],
    head=dict(
        type='ArcFaceClsHead',
        num_classes=numclass,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        init_cfg=None),
    prototype=gallery_dataloader)


