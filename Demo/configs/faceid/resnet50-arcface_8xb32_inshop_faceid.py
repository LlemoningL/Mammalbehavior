auto_scale_lr = dict(base_batch_size=256, enable=True)
batch_size = 64
custom_hooks = [
    dict(type='PrepareProtoBeforeValLoopHook'),
    dict(type='SyncBuffersHook'),
]
data_preprocessor = dict(
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    num_classes=36,
    std=[
        58.395,
        57.12,
        57.375,
    ],
    to_rgb=True)
dataroot = 'data'
dataset_type = 'InShop'
default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=3,
        rule='greater',
        save_best='auto',
        type='CheckpointHook'),
    logger=dict(interval=20, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(enable=False, type='VisualizationHook'))
default_scope = 'mmpretrain'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
gallery_dataloader = dict(
    batch_size=64,
    dataset=dict(
        data_root=dataroot,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(scale=512, type='Resize'),
            dict(crop_size=448, type='CenterCrop'),
            dict(type='PackInputs'),
        ],
        split='gallery',
        type='InShop'),
    num_workers=4,
    sampler=dict(shuffle=False, type='DefaultSampler'))
launcher = 'none'
load_from = None
log_level = 'INFO'
model = dict(
    head=dict(
        in_channels=2048,
        init_cfg=None,
        loss=dict(loss_weight=1.0, type='CrossEntropyLoss'),
        num_classes=36,
        type='ArcFaceClsHead'),
    image_encoder=[
        dict(
            depth=50,
            init_cfg=None,
            type='ResNet'),
        dict(type='GlobalAveragePooling'),
    ],
    prototype=dict(
        batch_size=64,
        dataset=dict(
            data_root=dataroot,
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(scale=512, type='Resize'),
                dict(crop_size=448, type='CenterCrop'),
                dict(type='PackInputs'),
            ],
            split='gallery',
            type='InShop'),
        num_workers=4,
        sampler=dict(shuffle=False, type='DefaultSampler')),
    type='ImageToImageRetriever')
numclass = 36
optim_wrapper = dict(
    optimizer=dict(
        lr=0.02, momentum=0.9, nesterov=True, type='SGD', weight_decay=0.0005))
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=5,
        start_factor=0.01,
        type='LinearLR'),
    dict(T_max=45, begin=5, by_epoch=True, end=50, type='CosineAnnealingLR'),
]

query_dataloader = dict(
    batch_size=64,
    dataset=dict(
        data_root=dataroot,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(scale=512, type='Resize'),
            dict(crop_size=448, type='CenterCrop'),
            dict(type='PackInputs'),
        ],
        split='query',
        type='InShop'),
    num_workers=4,
    sampler=dict(shuffle=False, type='DefaultSampler'))
randomness = dict(deterministic=False, seed=None)
resume = False
test_cfg = dict()
test_dataloader = dict(
    batch_size=64,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        data_root=dataroot,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(scale=512, type='Resize'),
            dict(crop_size=448, type='CenterCrop'),
            dict(type='PackInputs'),
        ],
        split='query',
        type='InShop'),
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = [
    dict(topk=1, type='RetrievalRecall'),
    dict(topk=10, type='RetrievalAveragePrecision'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(scale=512, type='Resize'),
    dict(crop_size=448, type='CenterCrop'),
    dict(type='PackInputs'),
]
train_cfg = dict(by_epoch=True, max_epochs=50, val_interval=1)
train_dataloader = dict(
    batch_size=64,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        data_root=dataroot,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(scale=512, type='Resize'),
            dict(crop_size=448, type='RandomCrop'),
            dict(direction='horizontal', prob=0.5, type='RandomFlip'),
            dict(type='PackInputs'),
        ],
        split='train',
        type='InShop'),
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(scale=512, type='Resize'),
    dict(crop_size=448, type='RandomCrop'),
    dict(direction='horizontal', prob=0.5, type='RandomFlip'),
    dict(type='PackInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=64,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        data_root=dataroot,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(scale=512, type='Resize'),
            dict(crop_size=448, type='CenterCrop'),
            dict(type='PackInputs'),
        ],
        split='query',
        type='InShop'),
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = [
    dict(topk=1, type='RetrievalRecall'),
    dict(topk=10, type='RetrievalAveragePrecision'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    type='UniversalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ])
work_dir = 'VideoOutput'
