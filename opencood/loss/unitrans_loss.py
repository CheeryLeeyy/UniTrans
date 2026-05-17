# -*- coding: utf-8 -*-
# Author: Xiangbo Gao <xiangbogaobarry@gmail.com>
# License: MIT License

from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.loss.moe_converter_loss import MoEConverterLoss
from opencood.loss.point_pillar_pyramid_loss import PointPillarPyramidLoss


class _InitStopperMixin:
    """Stopper to swallow super().__init__ in the MoEConverterLoss chain.
    接受任意参数但不继续向下调用，以阻断 super 链。
    """
    def __init__(self, *args, **kwargs):
        # 不调用 super()，什么都不做
        pass




class UniTransLoss(MoEConverterLoss, _InitStopperMixin, PointPillarPyramidLoss):
    def __init__(self, args):
        PointPillarPyramidLoss.__init__(self, args)
        MoEConverterLoss.__init__(self, args["converter"])

        self.w_detection_total = args.get("w_detection_total", 2.0)


        self.loss_dict = {}

    def forward(self, output_dict, target_dict, suffix="", val=False):
        total_loss = 0
        if suffix == "" and val == False:
            converter_loss = MoEConverterLoss.forward(self, output_dict, target_dict, suffix, val)
            total_loss += converter_loss

        detection_loss = PointPillarPyramidLoss.forward(self, output_dict, target_dict, suffix)

        total_loss += detection_loss * self.w_detection_total

        self.loss_dict.update({"total_loss": total_loss.item()})

        return total_loss

    def logging(self, epoch, batch_id, batch_len, writer=None, suffix=""):
        PointPillarPyramidLoss.logging(self, epoch, batch_id, batch_len, writer, suffix)
        MoEConverterLoss.logging(self, epoch, batch_id, batch_len, writer, suffix)
        self.loss_dict = {}











