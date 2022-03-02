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
total_epochs = 1000

log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
    ])

# runtime settings
checkpoint_config = dict(interval=100)
work_dir = './work_dirs/3_swin_tiny_patch244_window877_ucf101_mae_new_{}.py'.format(total_epochs)
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
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=12431 tools/train.py configs/recognition/swin/swin_tiny_patch244_window877_ucf101_mae_new.py --launcher pytorch
2021-12-17 01:14:25,361 - mmaction - INFO - Epoch [300][100/594]	lr: 2.742e-09, eta: 0:03:27, time: 0.712, data_time: 0.272, memory: 9054, loss: 913.6226
2021-12-17 01:15:09,037 - mmaction - INFO - Epoch [300][200/594]	lr: 2.742e-09, eta: 0:02:45, time: 0.437, data_time: 0.000, memory: 9054, loss: 930.2529
2021-12-17 01:15:52,710 - mmaction - INFO - Epoch [300][300/594]	lr: 2.742e-09, eta: 0:02:03, time: 0.437, data_time: 0.000, memory: 9054, loss: 930.1911
2021-12-17 01:16:36,200 - mmaction - INFO - Epoch [300][400/594]	lr: 2.742e-09, eta: 0:01:21, time: 0.435, data_time: 0.000, memory: 9054, loss: 914.3142
2021-12-17 01:17:20,055 - mmaction - INFO - Epoch [300][500/594]	lr: 2.742e-09, eta: 0:00:39, time: 0.439, data_time: 0.000, memory: 9054, loss: 935.7731

/media/ubuntu/4T/ALISURE/MVAE/work_dirs/2_swin_tiny_patch244_window877_ucf101_mae_new_300.py/epoch_300.pth
2021-12-18 22:43:01,096 - mmaction - INFO - Epoch [300][100/594]	lr: 2.742e-09, eta: 0:03:24, time: 0.706, data_time: 0.268, memory: 9054, loss: 974.4400
2021-12-18 22:43:45,185 - mmaction - INFO - Epoch [300][200/594]	lr: 2.742e-09, eta: 0:02:43, time: 0.441, data_time: 0.000, memory: 9054, loss: 985.2260
2021-12-18 22:44:29,546 - mmaction - INFO - Epoch [300][300/594]	lr: 2.742e-09, eta: 0:02:01, time: 0.443, data_time: 0.000, memory: 9054, loss: 961.8657
2021-12-18 22:45:13,302 - mmaction - INFO - Epoch [300][400/594]	lr: 2.742e-09, eta: 0:01:20, time: 0.438, data_time: 0.000, memory: 9054, loss: 963.2430
2021-12-18 22:45:57,171 - mmaction - INFO - Epoch [300][500/594]	lr: 2.742e-09, eta: 0:00:38, time: 0.439, data_time: 0.000, memory: 9054, loss: 966.9057

/media/ubuntu/4T/ALISURE/MVAE/work_dirs/3_swin_tiny_patch244_window877_ucf101_mae_new_1000.py/epoch_1000.pth
2021-12-22 21:26:37,890 - mmaction - INFO - Epoch [1000][100/594]	lr: 2.467e-10, eta: 0:03:27, time: 0.704, data_time: 0.274, memory: 9061, loss: 675.9637
2021-12-22 21:27:20,539 - mmaction - INFO - Epoch [1000][200/594]	lr: 2.467e-10, eta: 0:02:45, time: 0.427, data_time: 0.000, memory: 9061, loss: 684.0396
2021-12-22 21:28:03,884 - mmaction - INFO - Epoch [1000][300/594]	lr: 2.467e-10, eta: 0:02:03, time: 0.433, data_time: 0.000, memory: 9061, loss: 674.9640
2021-12-22 21:28:47,134 - mmaction - INFO - Epoch [1000][400/594]	lr: 2.467e-10, eta: 0:01:21, time: 0.432, data_time: 0.000, memory: 9061, loss: 657.3614
2021-12-22 21:29:30,280 - mmaction - INFO - Epoch [1000][500/594]	lr: 2.467e-10, eta: 0:00:39, time: 0.432, data_time: 0.001, memory: 9061, loss: 678.4360
"""


