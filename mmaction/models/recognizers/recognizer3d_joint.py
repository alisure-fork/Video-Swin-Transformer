import torch
from torch import nn

from ..builder import RECOGNIZERS
from .base import BaseRecognizer


@RECOGNIZERS.register_module()
class Recognizer3DJoint(BaseRecognizer):
    """3D recognizer model framework."""

    def forward_train(self, imgs, labels, **kwargs):
        """Defines the computation performed at every call when training."""
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        result_dict = self.extract_feat(imgs)

        a, b, c, d, z1, z2, p1, p2, has_predictor = None, None, None, None, None, None, None, None, True
        if "generative_output" in result_dict:
            a, b = result_dict["generative_output"]
            c, d = result_dict["generative_output2"]
        if "discriminative_output" in result_dict:
            z1, z2, p1, p2, has_predictor = result_dict["discriminative_output"]

        losses = self.cls_head.forward(a, b, c, d, z1, z2, p1, p2, has_predictor)
        return losses

    def _do_test(self, imgs):
        raise Exception("no _do_test function")

    def forward_test(self, imgs):
        result_dict = self.extract_feat(imgs[0])
        return result_dict

    def forward_dummy(self, imgs, softmax=False):
        raise Exception("no forward_dummy function")

    def forward_gradcam(self, imgs):
        raise Exception("no forward_gradcam function")

    pass
