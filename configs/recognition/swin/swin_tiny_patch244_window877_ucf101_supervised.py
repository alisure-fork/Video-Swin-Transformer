_base_ = [
    '../../_base_/models/swin/swin_tiny_my_supervised.py', '../../_base_/default_runtime.py'
]
# pretrained = None
# pretrained = "/media/ubuntu/4T/ALISURE/MVAE/work_dirs/swin_tiny_patch244_window877_ucf101_mae.py/epoch_100.pth"
# pretrained = "/media/ubuntu/4T/ALISURE/MVAE/work_dirs/swin_tiny_patch244_window877_ucf101_mae_new_300.py/epoch_300.pth"
# pretrained = "/media/ubuntu/4T/ALISURE/MVAE/work_dirs/1_swin_tiny_patch244_window877_ucf101_mae_new_300.py/epoch_300.pth"
# pretrained = "/media/ubuntu/4T/ALISURE/MVAE/work_dirs/2_swin_tiny_patch244_window877_ucf101_mae_new_300.py/epoch_300.pth"
pretrained = "/media/ubuntu/4T/ALISURE/MVAE/work_dirs/3_swin_tiny_patch244_window877_ucf101_mae_new_1000.py/epoch_1000.pth"
model=dict(backbone=dict(patch_size=(2,4,4), drop_path_rate=0.1, pretrained=pretrained),
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
evaluation = dict(
    interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy'])

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
total_epochs = 100

log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
    ])

# runtime settings
checkpoint_config = dict(interval=50)
work_dir = './work_dirs/3_swin_tiny_patch244_window877_ucf101_shift0{}.py'.format(
    "" if pretrained is None else "_pretrained")
find_unused_parameters = False


# do not use mmdet version fp16
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=4,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)


"""
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=12431 tools/train.py configs/recognition/swin/swin_tiny_patch244_window877_ucf101_supervised.py --validate --launcher pytorch

None 1e-3
Epoch(val) [100][945]	top1_acc: 0.4911, top5_acc: 0.7388, mean_class_accuracy: 0.4921

/media/ubuntu/4T/ALISURE/MVAE/work_dirs/swin_tiny_patch244_window877_ucf101_mae.py/epoch_100.pth 1e-3
Epoch(val) [95][945]	top1_acc: 0.5221, top5_acc: 0.7801, mean_class_accuracy: 0.5230

/media/ubuntu/4T/ALISURE/MVAE/work_dirs/swin_tiny_patch244_window877_ucf101_mae_new_300.py/epoch_300.pth 1e-3
Epoch(val) [90][945]	top1_acc: 0.5491, top5_acc: 0.8126, mean_class_accuracy: 0.5472

/media/ubuntu/4T/ALISURE/MVAE/work_dirs/1_swin_tiny_patch244_window877_ucf101_mae_new_300.py/epoch_300.pth 1e-3
Epoch(val) [75][945]	top1_acc: 0.5755, top5_acc: 0.8095, mean_class_accuracy: 0.5749
Epoch(val) [100][945]	top1_acc: 0.5658, top5_acc: 0.8148, mean_class_accuracy: 0.5655

/media/ubuntu/4T/ALISURE/MVAE/work_dirs/2_swin_tiny_patch244_window877_ucf101_mae_new_300.py/epoch_300.pth 1e-4
Epoch(val) [90][945]	top1_acc: 0.5504, top5_acc: 0.8010, mean_class_accuracy: 0.5507

/media/ubuntu/4T/ALISURE/MVAE/work_dirs/3_swin_tiny_patch244_window877_ucf101_mae_new_1000.py/epoch_1000.pth 1e-3
Epoch(val) [100][945]	top1_acc: 0.5893, top5_acc: 0.8232, mean_class_accuracy: 0.5872
10 crop test: top1_acc: 0.6086, top5_acc: 0.8373, mean_class_accuracy: 0.6068
"""
