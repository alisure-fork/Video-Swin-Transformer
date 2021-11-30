import torch.nn as nn
from mmcv.cnn import normal_init

import torch.nn.functional as F
from ..builder import HEADS
from .base import AvgConsensus, BaseHead


@HEADS.register_module()
class MAEHead(object):

    def __init__(self, **kwargs):
        pass

    def init_weights(self):
        pass

    def forward(self, x, y):
        recon_loss = F.mse_loss(x, y)
        return recon_loss

    pass
