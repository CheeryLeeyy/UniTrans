# -*- coding: utf-8 -*-
# Author: Yifan Lu
# Add direction classification loss
# The originally point_pillar_loss.py, can not determine if the box heading is opposite to the GT.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from opencood.utils.common_utils import limit_period
from opencood.data_utils.post_processor.voxel_postprocessor import VoxelPostprocessor
from icecream import ic

from opencood.loss.point_pillar_loss import PointPillarLoss
from opencood.loss.point_pillar_pyramid_loss import PointPillarPyramidLoss


class PointPillarMPDALoss(nn.Module):
    def __init__(self, args):
        super(PointPillarMPDALoss, self).__init__()

        if 'pyramid' in args:
            self.detection_loss_fun = PointPillarPyramidLoss(args)
        else:
            self.detection_loss_fun = PointPillarLoss(args)

        self.da = False
        if 'da' in args:
            self.da = True
            self.w_da = args['da'].get('weight', 1.0)

        self.loss_dict = {}

    def forward(self, output_dict, target_dict, suffix="", val=False):
        total_loss = 0

        total_loss += self.detection_loss_fun(output_dict, target_dict, suffix)
        self.loss_dict.update(self.detection_loss_fun.loss_dict)


        # domain adaption loss
        record_len = output_dict['record_len_da']
        if suffix=="" and self.da and not val:
            da_feature = output_dict['da_feature']
            source_index = self.return_index(record_len)

            N, C, H, W = da_feature.shape
            da_feature = da_feature.permute(0, 2, 3, 1)
            da_targets = torch.zeros_like(da_feature, dtype=torch.float32)
            for i in source_index:
                da_targets[i, :] = 1

            da_feature = da_feature.reshape(N, -1)
            da_targets = da_targets.reshape(N, -1)

            da_loss = F.binary_cross_entropy_with_logits(da_feature, da_targets)

            total_loss += da_loss * self.w_da
            self.loss_dict.update({
                'da_loss': da_loss.item(),
            })


        self.loss_dict.update({
            'total_loss': total_loss.item(),
        })

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
        da_loss = self.loss_dict.get('da_loss', 0)

        self.loss_dict = {}

        print("[epoch %d][%d/%d]%s || Loss: %.4f || Conf Loss: %.4f"
              " || Loc Loss: %.4f || Dir Loss: %.4f || IoU Loss: %.4f || Depth Loss: %.4f || Pyramid Loss: %.4f || DA Loss: %.4f" % (
                  epoch, batch_id + 1, batch_len, suffix,
                  total_loss, cls_loss, reg_loss, dir_loss, iou_loss, depth_loss, pyramid_loss, da_loss))

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
            writer.add_scalar('DA_loss' + suffix, da_loss,
                            epoch*batch_len + batch_id)



    def return_index(self, record_len):
        index = []
        cum_sum_len = list(np.cumsum(self.torch_tensor_to_numpy(record_len)))
        index.append(0)

        for i in range(len(cum_sum_len) - 1):
            index.append(cum_sum_len[i])

        return index

    def torch_tensor_to_numpy(self, torch_tensor):
        """
        Convert a torch tensor to numpy.

        Parameters
        ----------
        torch_tensor : torch.Tensor

        Returns
        -------
        A numpy array.
        """
        return torch_tensor.numpy() if not torch_tensor.is_cuda else \
            torch_tensor.cpu().detach().numpy()

