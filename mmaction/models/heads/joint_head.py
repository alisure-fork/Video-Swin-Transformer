import torch.nn as nn
from mmcv.cnn import normal_init

import torch.nn.functional as F
from ..builder import HEADS
from .base import AvgConsensus, BaseHead


@HEADS.register_module()
class JointHead(object):

    def __init__(self, **kwargs):
        self.criterion = nn.CrossEntropyLoss()
        pass

    def init_weights(self):
        pass

    def forward(self, a, b, c, d, z1, z2, p1, p2, has_predictor=True):
        losses = dict()
        if a is not None and b is not None and c is not None and d is not None:
            recon_loss = 0.5 * (F.mse_loss(a, b) + F.mse_loss(c, d))
            losses["loss"] = recon_loss
            pass
        if z1 is not None and z2 is not None and p1 is not None and p2 is not None:
            if has_predictor:
                discriminative_loss = 0.5 * (self.asymmetric_loss(p1, z2) + self.asymmetric_loss(p2, z1))
            else:
                discriminative_loss = 0.5 * (self.criterion(z1, p2) + self.criterion(z2, p1))
            losses["loss2"] = discriminative_loss
            pass
        return losses

    @staticmethod
    def asymmetric_loss(p, z):
        z = z.detach()  # stop gradient
        # p = nn.functional.normalize(p, dim=1)
        # z = nn.functional.normalize(z, dim=1)
        # return -(p * z).sum(dim=1).mean()
        return 1 - nn.functional.cosine_similarity(p, z, dim=-1).mean()
    pass
