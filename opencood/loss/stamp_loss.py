# -*- coding: utf-8 -*-
# Author: Xiangbo Gao <xiangbogaobarry@gmail.com>
# License: MIT License

from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.loss.point_pillar_pyramid_loss import PointPillarPyramidLoss

class StampLoss(PointPillarPyramidLoss):
    def __init__(self, args):
        super().__init__(args)
        self.alpha_P2M = args["alpha_P2M"]
        self.alpha_M2P2M = args["alpha_M2P2M"]
        self.alpha_M2P = args["alpha_M2P"]
        self.l2loss = nn.MSELoss()
        self.loss_dict = {}

    def forward(self, output_dict, target_dict, suffix="", val=False):
        total_loss = 0

        detection_loss = super().forward(output_dict, target_dict, suffix)
        total_loss += detection_loss


        if suffix == "":
            FE = output_dict["FE"]
            FP = output_dict["FP"]
            FN2P2E = output_dict["FN2P2E"]
            FP2E = output_dict["FP2E"]
            FN2P = output_dict["FN2P"]

            P2E_loss = self.l2loss(FE, FP2E)
            N2P2E_loss = self.l2loss(FE, FN2P2E)
            N2P_loss = self.l2loss(FP, FN2P)

            total_loss_ar = self.alpha_P2M * P2E_loss + self.alpha_M2P2M * N2P2E_loss + self.alpha_M2P * N2P_loss
            total_loss = total_loss + total_loss_ar
            self.loss_dict.update({"total_loss": total_loss, "P2M": P2E_loss, "M2P2M": N2P2E_loss, "M2P": N2P_loss})



        # P2M = self.l2loss(FM, FP2M)
        # M2P2M = self.l2loss(FM, FM2P2M)
        # M2P = self.l2loss(FP, FM2P)
        #
        # total_loss = self.alpha_P2M * P2M + self.alpha_M2P2M * M2P2M + self.alpha_M2P * M2P
        #
        # self.loss_dict.update({"total_loss": total_loss, "P2M": P2M, "M2P2M": M2P2M, "M2P": M2P})

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

        PointPillarPyramidLoss.logging(self, epoch, batch_id, batch_len, writer, suffix)

        if suffix == "":
            total_loss = self.loss_dict["total_loss"]
            P2M_loss = self.loss_dict["P2M"]
            M2P2M_loss = self.loss_dict["M2P2M"]
            M2P_loss = self.loss_dict["M2P"]

            print(
                "[epoch %d][%d/%d], || Loss: %.6f || P2E Loss: %.6f"
                " || N2P2E Loss: %.6f || N2P Loss: %.6f"
                % (epoch, batch_id + 1, batch_len, total_loss.item(), P2M_loss.item(), M2P2M_loss.item(), M2P_loss.item())
            )

            if not writer is None:
                writer.add_scalar("P2E_loss", P2M_loss.item(), epoch * batch_len + batch_id)
                writer.add_scalar("N2P2E_loss", M2P2M_loss.item(), epoch * batch_len + batch_id)
                writer.add_scalar("N2P_loss", M2P_loss.item(), epoch * batch_len + batch_id)

        self.loss_dict = {}


def test():
    torch.manual_seed(0)
    loss = PixorLoss(None)
    pred = torch.sigmoid(torch.randn(1, 7, 2, 3))
    label = torch.zeros(1, 7, 2, 3)
    loss = loss(pred, label)
    print(loss)


if __name__ == "__main__":
    test()
