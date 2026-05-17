# -*- coding: utf-8 -*-
# Author: Xiangbo Gao <xiangbogaobarry@gmail.com>
# License: MIT License

from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.loss.point_pillar_pyramid_loss import PointPillarPyramidLoss


class ConverterLoss(PointPillarPyramidLoss):
    def __init__(self, args):
        super().__init__(args)
        self.alpha_N2E = args["N2E"]["weight"]
        self.l2loss = nn.MSELoss()

    def forward(self, output_dict, target_dict, suffix="", val=False):
        """
        Compute loss for pixor network
        Parameters
        ----------
        output_dict : dict
           The dictionary that contains the output.

        target_dict : dict
           The dictionary that contains the target.

        Returns
        -------
        total_loss : torch.Tensor
            Total loss.

        """


        total_loss = super().forward(output_dict, target_dict, suffix)

        # if suffix == "":
        if val == False and suffix == "":
            FE = output_dict["FE"]
            FN2E = output_dict["FN2E"]
            N2E = self.l2loss(FE, FN2E)
            N2E = self.alpha_N2E * N2E
            total_loss += N2E

            self.loss_dict.update({"N2E": N2E.item()})

        self.loss_dict.update({"total_loss": total_loss})

        return total_loss

    def logging(self, epoch, batch_id, batch_len, writer=None, suffix=""):
        """
        Print out  the loss function for current iteration.

        Parameters
        ----------
        epoch : int
            Current epoch for training.
        batch_id : int
            The current batch.
        batch_len : int
            Total batch length in one iteration of training,
        writer : SummaryWriter
            Used to visualize on tensorboard
        """
        total_loss = self.loss_dict.get('total_loss', 0)
        reg_loss = self.loss_dict.get('reg_loss', 0)
        cls_loss = self.loss_dict.get('cls_loss', 0)
        dir_loss = self.loss_dict.get('dir_loss', 0)
        iou_loss = self.loss_dict.get('iou_loss', 0)
        depth_loss = self.loss_dict.get('depth_loss', 0)
        pyramid_loss = self.loss_dict.get('pyramid_loss', 0)
        N2E_loss = self.loss_dict.get('N2E', 0)

        self.loss_dict = {}

        print("[epoch %d][%d/%d]%s || Loss: %.4f || Conf Loss: %.4f"
              " || Loc Loss: %.4f || Dir Loss: %.4f || IoU Loss: %.4f || Depth Loss: %.4f || Pyramid Loss: %.4f || N2E Loss: %.4f" % (
                  epoch, batch_id + 1, batch_len, suffix,
                  total_loss, cls_loss, reg_loss, dir_loss, iou_loss, depth_loss, pyramid_loss, N2E_loss))


        if not writer is None:
            writer.add_scalar('Regression_loss' + suffix, reg_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Confidence_loss' + suffix, cls_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Dir_loss' + suffix, dir_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Iou_loss' + suffix, iou_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Depth_loss' + suffix, depth_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Pyramid_loss' + suffix, pyramid_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('N2E_loss' + suffix, N2E_loss,
                            epoch*batch_len + batch_id)


