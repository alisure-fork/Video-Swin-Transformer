# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ViViT2',
        image_size=224,
        patch_size=16,
        num_frames=32,
        dim=512,
        depth1=4,
        depth2=2,
        heads=8,
        pool='cls',
        in_channels=3,
        dim_head=64,
        dropout=0.,
        emb_dropout=0.,
        scale_dim=4),
    cls_head=dict(
        type='I3DHead',
        in_channels=512,
        num_classes=101,
        spatial_type=None,
        dropout_ratio=0.5),
    test_cfg = dict(average_clips='prob')
)
