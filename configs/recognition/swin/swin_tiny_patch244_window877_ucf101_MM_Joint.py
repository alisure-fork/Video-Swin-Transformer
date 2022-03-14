_base_ = [
    '../../_base_/models/swin/swin_tiny_my_MM_Joint.py', '../../_base_/default_runtime.py'
]
model=dict(backbone=dict(patch_size=(2,4,4), drop_path_rate=0.1, generative=True, discriminative=True,
                         # pretrained='../'
                         ),
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
# total_epochs = 1000
total_epochs = 300

log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
    ])

# runtime settings
checkpoint_config = dict(interval=100)
work_dir = './work_dirs/Joint_2_swin_tiny_patch244_window877_ucf101_{}.py'.format(total_epochs)
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
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=12431 tools/train.py configs/recognition/swin/swin_tiny_patch244_window877_ucf101_MM_Joint.py --launcher pytorch --cfg-options model.backbone.pretrained=./checkpoints/swin/swin_tiny_patch4_224.pth

PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m torch.distributed.launch --nproc_per_node=4 --master_port=12431 tools/train.py configs/recognition/swin/swin_tiny_patch244_window877_ucf101_MM_Joint.py --launcher pytorch > ./work_dirs/3.log 2>&1 &

2022-03-06 07:08:35,916 - mmaction - INFO - Epoch [300][100/596]	lr: 2.742e-09, eta: 0:05:28, time: 0.913, data_time: 0.177, memory: 10673, loss: 930.2886, loss2: 0.0125
2022-03-06 07:09:50,226 - mmaction - INFO - Epoch [300][200/596]	lr: 2.742e-09, eta: 0:04:22, time: 0.743, data_time: 0.000, memory: 10673, loss: 883.7012, loss2: 0.0130
2022-03-06 07:11:04,499 - mmaction - INFO - Epoch [300][300/596]	lr: 2.742e-09, eta: 0:03:16, time: 0.743, data_time: 0.000, memory: 10673, loss: 908.7381, loss2: 0.0133
2022-03-06 07:12:18,912 - mmaction - INFO - Epoch [300][400/596]	lr: 2.742e-09, eta: 0:02:10, time: 0.744, data_time: 0.000, memory: 10673, loss: 889.0445, loss2: 0.0130
2022-03-06 07:13:31,653 - mmaction - INFO - Epoch [300][500/596]	lr: 2.742e-09, eta: 0:01:03, time: 0.727, data_time: 0.000, memory: 10673, loss: 889.4059, loss2: 0.0124

"""


