# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import torch
import torch.nn as nn
import torch.nn.functional as F
from opencood.loss.point_pillar_depth_loss import PointPillarDepthLoss
from opencood.loss.point_pillar_loss import sigmoid_focal_loss


class PointPillarPyramidLoss(PointPillarDepthLoss):
    def __init__(self, args):
        super().__init__(args)
        self.pyramid = args['pyramid']

        # relative downsampled GT cls map from fused labels.
        self.relative_downsample = self.pyramid['relative_downsample']
        self.pyramid_weight = self.pyramid['weight']
        self.num_levels = len(self.relative_downsample)

    def forward(self, output_dict, target_dict, suffix=""):
        if output_dict['pyramid'] == 'collab':  # intermediate fusion, pyramid collab.
            return self.forward_collab(output_dict, target_dict, suffix)

        elif output_dict['pyramid'] == 'single':  # late fusion, pyramid single
            return self.forward_single(output_dict, target_dict, suffix)
        else:
            return self.forward_collab(output_dict, target_dict, suffix)

    def forward_single(self, output_dict, target_dict, suffix):
        """
        for heter_pyramid_single
        """
        batch_size = target_dict['pos_equal_one'].shape[0]
        total_loss = super().forward(output_dict, target_dict, suffix)

        occ_single_list = output_dict['occ_single_list']
        occ_loss = self.calc_occ_loss(occ_single_list, target_dict['pos_equal_one'], target_dict['neg_equal_one'],
                                      batch_size)
        total_loss += occ_loss
        self.loss_dict.update({
            'pyramid_loss': occ_loss.item(),
            'total_loss': total_loss.item()
        })
        return total_loss

    def forward_collab(self, output_dict, target_dict, suffix):
        """
        for heter_pyramid_collab
        """
        if suffix == "":
            return super().forward(output_dict, target_dict)
        assert suffix == "_single"
        batch_size = target_dict['pos_equal_one'].shape[0]

        positives = target_dict['pos_equal_one']
        negatives = target_dict['neg_equal_one']

        occ_single_list = output_dict['occ_single_list']
        occ_loss = self.calc_occ_loss(occ_single_list, positives, negatives, batch_size)
        total_loss = occ_loss
        self.loss_dict = {
            'pyramid_loss': occ_loss.item(),
            'total_loss': total_loss.item()
        }

        return total_loss

    def forward_collab_moe(self, output_dict, target_dict, suffix):
        """
        for heter_pyramid_collab
        """
        if suffix == "":
            return super().forward(output_dict, target_dict)
        assert suffix == "_single"
        batch_size = target_dict['pos_equal_one'].shape[0]

        positives = target_dict['pos_equal_one']
        negatives = target_dict['neg_equal_one']

        occ_single_list = output_dict['occ_single_list']
        occ_loss = self.calc_occ_loss(occ_single_list, positives, negatives, batch_size)
        total_loss = occ_loss
        self.loss_dict = {
            'pyramid_loss': occ_loss.item(),
            'total_loss': total_loss.item()
        }

        return total_loss

    def compute_z_loss(self, all_logits):
        """
        计算全局Router z-loss

        :param all_logits: 所有模态的logits列表
        :return: z_loss
        """
        # 拼接所有logits
        global_logits = torch.cat(all_logits, dim=0)  # [total_B, num_experts]

        # 计算log-sum-exp
        log_z = torch.logsumexp(global_logits, dim=-1)  # [total_B]

        # 计算z-loss
        z_loss = 0.5 * torch.mean(log_z ** 2)  # 整个全局batch平均

        return z_loss

    def compute_load_balance_loss(self, all_logits, all_topk_indices):
        """
        计算全局负载均衡损失

        :param all_logits: 所有模态的logits列表
        :param all_topk_indices: 所有模态的topk_indices列表
        :return: load_balance_loss
        """
        # 拼接所有logits和topk_indices
        global_logits = torch.cat(all_logits, dim=0)  # [total_B, num_experts]
        global_topk_indices = torch.cat(all_topk_indices, dim=0)  # [total_B, top_k]

        # 1. 计算每个专家的重要性（门控概率之和）
        probs = F.softmax(global_logits, dim=-1)  # [total_B, num_experts]
        importance = probs.sum(dim=0)  # [num_experts]

        # 2. 计算每个专家的负载（被选择的次数）
        num_experts = importance.size(0)
        load = torch.zeros(num_experts, device=global_logits.device)

        # 展平所有选择
        indices_flat = global_topk_indices.view(-1)  # [total_B * top_k]
        load.scatter_add_(0, indices_flat, torch.ones_like(indices_flat, dtype=torch.float))

        # 3. 添加平滑项避免除零
        importance += 1e-6
        load += 1e-6

        # 4. 计算重要性分布和负载分布
        importance_dist = importance / importance.sum()
        load_dist = load / load.sum()

        # 5. 使用KL散度计算分布差异
        load_balance_loss = F.kl_div(
            importance_dist.log(),
            load_dist,
            reduction='batchmean'
        )

        return load_balance_loss

    def calc_occ_loss(self, occ_single_list, positives, negatives, batch_size):
        total_occ_loss = 0
        occ_positives = torch.logical_or(positives[..., 0], positives[..., 1]).unsqueeze(-1).float()  # N, H, W
        occ_negatives = torch.logical_and(negatives[..., 0], negatives[..., 1]).unsqueeze(-1).float()  # N, H, W

        for i, occ_preds_single in enumerate(occ_single_list):
            """
            occ_preds_single: N, 1, H, W

            occ_positives: N, H, W, 1
            occ_negatives: N, H, W, 1

            """

            positives_level = F.max_pool2d(occ_positives.permute(0, 3, 1, 2),
                                           kernel_size=self.relative_downsample[i]).permute(0, 2, 3, 1)
            negatives_level = 1 - F.max_pool2d((1 - occ_negatives).permute(0, 3, 1, 2),
                                               kernel_size=self.relative_downsample[i]).permute(0, 2, 3, 1)

            occ_labls = positives_level.view(batch_size, -1, 1)
            positives_level = occ_labls
            negatives_level = negatives_level.view(batch_size, -1, 1)

            pos_normalizer = positives_level.sum(1, keepdim=True).float()

            occ_preds = occ_preds_single.permute(0, 2, 3, 1).contiguous() \
                .view(batch_size, -1, 1)
            occ_weights = positives_level * self.pos_cls_weight + negatives_level * 1.0
            occ_weights /= torch.clamp(pos_normalizer, min=1.0)
            occ_loss = sigmoid_focal_loss(occ_preds, occ_labls, weights=occ_weights, **self.cls)
            occ_loss = occ_loss.sum() / batch_size
            occ_loss *= self.pyramid_weight[i]

            total_occ_loss += occ_loss

        return total_occ_loss

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

        print("[epoch %d][%d/%d]%s || Loss: %.4f || Conf Loss: %.4f"
              " || Loc Loss: %.4f || Dir Loss: %.4f || IoU Loss: %.4f || Depth Loss: %.4f || Pyramid Loss: %.4f " % (
                  epoch, batch_id + 1, batch_len, suffix,
                  total_loss, cls_loss, reg_loss, dir_loss, iou_loss, depth_loss, pyramid_loss))

        if not writer is None:
            writer.add_scalar('Regression_loss' + suffix, reg_loss,
                              epoch * batch_len + batch_id)
            writer.add_scalar('Confidence_loss' + suffix, cls_loss,
                              epoch * batch_len + batch_id)
            writer.add_scalar('Dir_loss' + suffix, dir_loss,
                              epoch * batch_len + batch_id)
            writer.add_scalar('Iou_loss' + suffix, iou_loss,
                              epoch * batch_len + batch_id)
            writer.add_scalar('Depth_loss' + suffix, depth_loss,
                              epoch * batch_len + batch_id)
            writer.add_scalar('Pyramid_loss' + suffix, pyramid_loss,
                              epoch * batch_len + batch_id)


