_base_ = [
    '../../_base_/models/swin/swin_tiny_my_mae.py', '../../_base_/default_runtime.py'
]
model=dict(backbone=dict(patch_size=(2,4,4), drop_path_rate=0.1),
           cls_head=dict(num_classes=101), test_cfg=dict(max_testing_views=4))

# dataset settings
dataset_type = 'RawframeDataset'
data_root = 'data/ucf101/rawframes'
data_root_val = 'data/ucf101/rawframes'
split = 1  # official train/test splits. valid numbers: 1, 2, 3
ann_file_train = f'data/ucf101/ucf101_train_split_{split}_rawframes.txt'
ann_file_val = f'data/ucf101/ucf101_val_split_{split}_rawframes.txt'
ann_file_test = f'data/ucf101/ucf101_val_split_{split}_rawframes.txt'
img_norm_cfg = dict(mean=[104, 117, 128], std=[1, 1, 1], to_bgr=False)


train_pipeline = [
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(240, 320)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='SampleFrames',
         clip_len=32,
         frame_interval=2,
         num_clips=1,
         test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(240, 320)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='SampleFrames',
         clip_len=32,
         frame_interval=2,
         num_clips=10,
         test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(240, 320)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=8,
    workers_per_gpu=4,
    val_dataloader=dict(
        videos_per_gpu=1,
        workers_per_gpu=1
    ),
    test_dataloader=dict(
        videos_per_gpu=1,
        workers_per_gpu=1
    ),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline))

# optimizer
optimizer = dict(type='AdamW', lr=1e-3, betas=(0.9, 0.999), weight_decay=0.0005,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'backbone': dict(lr_mult=0.1)}))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=2.5
)
total_epochs = 300

log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
    ])

# runtime settings
checkpoint_config = dict(interval=10)
work_dir = './work_dirs/swin_tiny_patch244_window877_ucf101_mae_{}.py'.format(total_epochs)
find_unused_parameters = True


# do not use mmdet version fp16
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=2,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)


"""
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=12431 tools/train.py configs/recognition/swin/swin_tiny_patch244_window877_ucf101_mae.py --launcher pytorch
2021-11-27 13:38:36,027 - mmaction - INFO - Epoch [100][500/594]	lr: 2.467e-08, eta: 0:01:02, time: 0.678, data_time: 0.003, memory: 9050, loss: 1719.0865
2021-11-27 13:38:42,307 - mmaction - INFO - Epoch [100][510/594]	lr: 2.467e-08, eta: 0:00:55, time: 0.629, data_time: 0.002, memory: 9050, loss: 1623.7661
2021-11-27 13:38:49,062 - mmaction - INFO - Epoch [100][520/594]	lr: 2.467e-08, eta: 0:00:49, time: 0.675, data_time: 0.000, memory: 9050, loss: 1593.9346
2021-11-27 13:38:55,580 - mmaction - INFO - Epoch [100][530/594]	lr: 2.467e-08, eta: 0:00:42, time: 0.651, data_time: 0.000, memory: 9050, loss: 1589.6933
2021-11-27 13:39:01,556 - mmaction - INFO - Epoch [100][540/594]	lr: 2.467e-08, eta: 0:00:35, time: 0.598, data_time: 0.001, memory: 9050, loss: 1596.6014
2021-11-27 13:39:07,588 - mmaction - INFO - Epoch [100][550/594]	lr: 2.467e-08, eta: 0:00:29, time: 0.603, data_time: 0.000, memory: 9050, loss: 1595.5595
2021-11-27 13:39:13,625 - mmaction - INFO - Epoch [100][560/594]	lr: 2.467e-08, eta: 0:00:22, time: 0.604, data_time: 0.000, memory: 9050, loss: 1415.3443
2021-11-27 13:39:19,651 - mmaction - INFO - Epoch [100][570/594]	lr: 2.467e-08, eta: 0:00:15, time: 0.603, data_time: 0.000, memory: 9050, loss: 1649.2965
2021-11-27 13:39:25,703 - mmaction - INFO - Epoch [100][580/594]	lr: 2.467e-08, eta: 0:00:09, time: 0.605, data_time: 0.000, memory: 9050, loss: 1503.0011
2021-11-27 13:39:31,734 - mmaction - INFO - Epoch [100][590/594]	lr: 2.467e-08, eta: 0:00:02, time: 0.603, data_time: 0.000, memory: 9050, loss: 1677.2558
"""


