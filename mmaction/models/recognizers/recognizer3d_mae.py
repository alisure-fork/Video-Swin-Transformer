import torch
from torch import nn

from ..builder import RECOGNIZERS
from .base import BaseRecognizer


@RECOGNIZERS.register_module()
class Recognizer3DMAE(BaseRecognizer):
    """3D recognizer model framework."""

    def forward_train(self, imgs, labels, **kwargs):
        """Defines the computation performed at every call when training."""
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        losses = dict()
        a, b = self.extract_feat(imgs)
        loss = self.cls_head.forward(a, b)
        losses["loss"] = loss
        return losses

    def _do_test(self, imgs):
        raise Exception("no _do_test function")

    def forward_test(self, imgs):
        raise Exception("no forward_test function")

    def forward_dummy(self, imgs, softmax=False):
        raise Exception("no forward_dummy function")

    def forward_gradcam(self, imgs):
        raise Exception("no forward_gradcam function")

    pass
