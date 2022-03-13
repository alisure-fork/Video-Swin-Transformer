# model settings
_base_ = "swin_tiny_my_MM_supervised.py"
model = dict(type="Recognizer3DJoint", backbone=dict(generative=True, discriminative=True),
             cls_head=dict(type='JointHead'))
