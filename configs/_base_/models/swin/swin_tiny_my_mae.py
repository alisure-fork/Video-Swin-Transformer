# model settings
_base_ = "swin_tiny_my_supervised.py"
model = dict(type="Recognizer3DMAE", backbone=dict(is_mae=True), cls_head=dict(type='MAEHead'))
