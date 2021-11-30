# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ViViT',
        image_size=224,
        patch_size=16,
        num_frames=16,
        dim=192,
        depth=4,
        heads=3,
        pool='cls',
        in_channels=3,
        dim_head=64,
        dropout=0.,
        emb_dropout=0.,
        scale_dim=4),
    cls_head=dict(
        type='I3DHead',
        in_channels=192,
        num_classes=101,
        spatial_type=None,
        dropout_ratio=0.5),
    test_cfg = dict(average_clips='prob')
)
