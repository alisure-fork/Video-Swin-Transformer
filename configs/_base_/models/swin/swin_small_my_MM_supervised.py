# model settings
_base_ = "swin_tiny_my_MM_supervised.py"
model = dict(backbone=dict(depths=[2, 2, 18, 2]))