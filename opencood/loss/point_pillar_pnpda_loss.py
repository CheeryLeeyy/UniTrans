# -*- coding: utf-8 -*-
# Author: Based on point_pillar_mpda_loss.py
# PnPDA: Plug-and-Play Domain Adaptation with Contrastive Learning Loss

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from opencood.utils.common_utils import limit_period
from opencood.data_utils.post_processor.voxel_postprocessor import VoxelPostprocessor
from icecream import ic

# Import wandb/swanlab for logging
try:
    import swanlab as wandb  # Use swanlab if available
except ImportError:
    try:
        import wandb  # Fall back to wandb
    except ImportError:
        wandb = None  # Neither available


class PointPillarPnPDALoss(nn.Module):
    def __init__(self, args):
        super(PointPillarPnPDALoss, self).__init__()
        self.pos_cls_weight = args['pos_cls_weight']

        self.cls = args['cls']
        self.reg = args['reg']

        if 'dir' in args:
            self.dir = args['dir']
        else:
            self.dir = None

        if 'iou' in args:
            from opencood.pcdet_utils.iou3d_nms.iou3d_nms_utils import aligned_boxes_iou3d_gpu
            self.iou_loss_func = aligned_boxes_iou3d_gpu
            self.iou = args['iou']
        else:
            self.iou = None
        
        # PnPDA: Contrastive Learning Loss parameters
        self.pnpda = False
        if 'pnpda' in args:
            self.pnpda = args['pnpda']
            self.tau = args.get('tau', 0.1)
            self.max_voxel = args.get('max_voxel', 40)
            self.pnpda_weight = args.get('pnpda_weight', 1.0)
        
        self.loss_dict = {}

    def forward(self, output_dict, target_dict, suffix="", val=False):
        """
        Parameters
        ----------
        output_dict : dict
        target_dict : dict
        """
        total_loss = 0
        # device = next(iter(output_dict.values())).device if output_dict else torch.device('cuda')
        device = output_dict['cls_preds'].device

        # Check if detection outputs are available
        has_detection_output = (f'psm{suffix}' in output_dict or 
                                f'cls_preds{suffix}' in output_dict)
        
        if has_detection_output:
            batch_size = target_dict['pos_equal_one'].shape[0]

            cls_labls = target_dict['pos_equal_one'].view(batch_size, -1,  1)
            positives = cls_labls > 0
            negatives = target_dict['neg_equal_one'].view(batch_size, -1,  1) > 0
            pos_normalizer = positives.sum(1, keepdim=True).float()

            # rename variable 
            if f'psm{suffix}' in output_dict:
                output_dict[f'cls_preds{suffix}'] = output_dict[f'psm{suffix}']
            if f'rm{suffix}' in output_dict:
                output_dict[f'reg_preds{suffix}'] = output_dict[f'rm{suffix}']
            if f'dm{suffix}' in output_dict:
                output_dict[f'dir_preds{suffix}'] = output_dict[f'dm{suffix}']

            # cls loss
            cls_preds = output_dict[f'cls_preds{suffix}'].permute(0, 2, 3, 1).contiguous() \
                        .view(batch_size, -1,  1)
            cls_weights = positives * self.pos_cls_weight + negatives * 1.0
            cls_weights /= torch.clamp(pos_normalizer, min=1.0)
            cls_loss = sigmoid_focal_loss(cls_preds, cls_labls, weights=cls_weights, **self.cls)
            cls_loss = cls_loss.sum() * self.cls['weight'] / batch_size

            # reg loss
            reg_weights = positives / torch.clamp(pos_normalizer, min=1.0)
            reg_preds = output_dict[f'reg_preds{suffix}'].permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 7)
            reg_targets = target_dict['targets'].view(batch_size, -1, 7)
            reg_preds, reg_targets = self.add_sin_difference(reg_preds, reg_targets)
            reg_loss = weighted_smooth_l1_loss(reg_preds, reg_targets, weights=reg_weights, sigma=self.reg['sigma'])
            reg_loss = reg_loss.sum() * self.reg['weight'] / batch_size

            ######## direction ##########
            if self.dir:
                dir_targets = self.get_direction_target(target_dict['targets'].view(batch_size, -1, 7))
                dir_logits = output_dict[f"dir_preds{suffix}"].permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)

                dir_loss = softmax_cross_entropy_with_logits(dir_logits.view(-1, self.anchor_num), dir_targets.view(-1, self.anchor_num)) 
                dir_loss = dir_loss.flatten() * reg_weights.flatten() 
                dir_loss = dir_loss.sum() * self.dir['weight'] / batch_size
                total_loss += dir_loss
                self.loss_dict.update({'dir_loss': dir_loss.item()})

            ######## IoU ###########
            if self.iou:
                iou_preds = output_dict["iou_preds{suffix}"].permute(0, 2, 3, 1).contiguous()
                pos_pred_mask = reg_weights.squeeze(dim=-1) > 0
                iou_pos_preds = iou_preds.view(batch_size, -1)[pos_pred_mask]
                boxes3d_pred = VoxelPostprocessor.delta_to_boxes3d(
                    output_dict[f'reg_preds{suffix}'].permute(0, 2, 3, 1).contiguous().detach(),
                    output_dict['anchor_box'])[pos_pred_mask]
                boxes3d_tgt = VoxelPostprocessor.delta_to_boxes3d(
                    target_dict['targets'],
                    output_dict['anchor_box'])[pos_pred_mask]
                iou_weights = reg_weights[pos_pred_mask].view(-1)
                iou_pos_targets = self.iou_loss_func(
                    boxes3d_pred.float()[:, [0, 1, 2, 5, 4, 3, 6]],
                    boxes3d_tgt.float()[:, [0, 1, 2, 5, 4, 3, 6]]).detach().squeeze()
                iou_pos_targets = 2 * iou_pos_targets.view(-1) - 1
                iou_loss = weighted_smooth_l1_loss(iou_pos_preds, iou_pos_targets, weights=iou_weights, sigma=self.iou['sigma'])

                iou_loss = iou_loss.sum() * self.iou['weight'] / batch_size
                total_loss += iou_loss
                self.loss_dict.update({'iou_loss': iou_loss.item()})

            total_loss += reg_loss + cls_loss
            self.loss_dict.update({
                'reg_loss': reg_loss.item(),
                'cls_loss': cls_loss.item()
            })
        else:
            # No detection output, initialize detection loss as 0
            self.loss_dict.update({
                'reg_loss': 0,
                'cls_loss': 0
            })

        # PnPDA: Contrastive Learning Loss
        pnpda_loss = torch.zeros(1, device=device)
        
        if self.pnpda and not val:
            if 'features_q' in output_dict and 'features_k' in output_dict:
                features_q = output_dict["features_q"]
                features_k = output_dict["features_k"]
                mask = target_dict.get("pos_region_ranges", None)
                
                if mask is not None:
                    pnpda_loss = self.compute_contrastive_loss(features_q, features_k, mask)
                    pnpda_loss = pnpda_loss * self.pnpda_weight
                    total_loss = total_loss + pnpda_loss

        self.loss_dict.update({
            'total_loss': total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss,
            'pnpda_loss': pnpda_loss.item()
        })

        return total_loss

    def compute_contrastive_loss(self, features_q, features_k, mask):
        """
        Compute contrastive learning loss for PnPDA (following original implementation)
        
        Parameters
        ----------
        features_q : torch.Tensor
            Query features (predicted features), shape (B, C, H, W)
            B = total_cav (treat different CAVs as batch dimension)
        features_k : torch.Tensor  
            Key features (ground truth features), shape (B, C, H, W)
        mask : torch.Tensor
            Positive region mask, shape (B, max_num, H, W)
            
        Note: Since features are already in ego coordinate system, we can directly
        apply the original PnPDA logic: contrast at object level, not CAV level.
        """
        device = features_q.device
        B = features_q.shape[0]  # Treat total_cav as batch dimension
        
        # Expand mask to match batch size if needed
        if mask.shape[0] == 1 and B > 1:
            mask = mask.expand(B, -1, -1, -1)  # (B, max_num, H, W)
        
        # ========== Following original PnPDA implementation ==========
        # Transpose mask: (B, max_num, H, W) → (max_num, B, H, W)
        pos_mask = mask.transpose(0, 1).contiguous().unsqueeze(2)  # (max_num, B, 1, H, W)
        
        # Apply mask to features: (B, C, H, W) * (max_num, B, 1, H, W) → (max_num, B, C, H, W)
        masked_features_q = features_q * pos_mask.float()
        masked_features_k = features_k * pos_mask.float()
        
        # Sample voxels for each object
        # Output: (n_objects, 1, C) and (n_objects, p, C)
        sampled_features_q, _ = self.sample_voxel(masked_features_q, mask, is_avg=True)
        sampled_features_k, pad_mask = self.sample_voxel(masked_features_k, mask, is_avg=True)
        
        if sampled_features_q.shape[0] == 0 or sampled_features_k.shape[0] == 0:
            return torch.zeros(1, device=device, requires_grad=True)
        
        # Transpose for matrix multiplication
        sampled_features_q = sampled_features_q.transpose(0, 1)  # (1, n_objects, C)
        
        # Normalize features
        norm_features_q = F.normalize(sampled_features_q, p=2, dim=-1)
        norm_features_k = F.normalize(sampled_features_k, p=2, dim=-1)
        
        # Compute similarity: (n_objects, p, C) @ (n_objects, 1, C).T → (n_objects, p, n_objects)
        sim = norm_features_k @ norm_features_q.transpose(-1, -2)
        
        # Temperature scaling
        logits = sim.clone()
        logits /= self.tau
        
        # Labels: expect diagonal to be maximum
        labels = (
            torch.arange(logits.shape[0], device=device)
            .unsqueeze(-1)
            .expand(logits.shape[0], logits.shape[1])
        )
        
        # Cross entropy loss
        loss = F.cross_entropy(logits[pad_mask], labels[pad_mask])
        
        # Compute similarity metrics for logging
        target_idx = [*range(len(sim))]
        target_idx_ = torch.zeros_like(sim)
        target_idx_[target_idx, :, target_idx] = 1.0
        target_idx = target_idx_.bool()
        
        pos_cos_sim = sim[target_idx].mean()
        neg_cos_sim = (
            sim[~target_idx].mean()
            if sampled_features_k.shape[0] > 1
            else torch.tensor(0).to(device)
        )
        
        sim_softmax = sim.softmax(-1)
        pos_softmax_sim = sim_softmax[target_idx].mean()
        neg_softmax_sim = (
            sim_softmax[~target_idx].mean()
            if sampled_features_k.shape[0] > 1
            else torch.tensor(0).to(device)
        )
        
        self.loss_dict.update({
            'pos_cos_sim': pos_cos_sim.item(),
            'neg_cos_sim': neg_cos_sim.item(),
            'pos_softmax_sim': pos_softmax_sim.item(),
            'neg_softmax_sim': neg_softmax_sim.item(),
        })
        
        return loss
    
    def sample_voxel(self, feature, mask, is_avg):
        """
        Sample voxels from feature maps based on mask (original PnPDA implementation)
        
        Parameters
        ----------
        feature : torch.Tensor
            Feature maps, shape (max_num, B, C, H, W)
        mask : torch.Tensor
            Binary mask, shape (B, max_num, H, W)
        is_avg : bool
            Whether to average sampled voxels
            
        Returns
        -------
        sampled_features : torch.Tensor
            Sampled features, shape (n_objects, 1, C) if is_avg else (n_objects, max_voxel, C)
        pad_mask : torch.Tensor
            Padding mask, shape (n_objects, 1) if is_avg else (n_objects, max_voxel)
        """
        # Flatten: (max_num, B, C, H, W) → (max_num * B, C, H, W)
        # Flatten mask: (B, max_num, H, W) → (max_num * B, H, W)
        mask = mask.flatten(0, 1)  # (B * max_num, H, W) → transpose → (max_num * B, H, W)
        feature = feature.flatten(0, 1)  # (max_num * B, C, H, W)
        
        N = feature.shape[0]
        
        f_list = []
        pad_list = []
        
        for i in range(N):
            # Find positive voxel locations
            index = torch.stack(torch.where(mask[i] == True))
            if index.shape[1] == 0:
                continue
            
            # Random permutation for sampling
            idx = torch.randperm(index.shape[1])
            index = index[:, idx].view(index.size())
            
            # Sample positive region voxels: (C, num_pos) → (num_pos, C) → (max_voxel, C)
            sampled_voxel = feature[i, :, index[0], index[1]].transpose(0, 1)[
                : self.max_voxel
            ]
            
            if is_avg:
                # Average all sampled voxels to get one representative vector
                sampled_voxel = torch.mean(sampled_voxel, dim=(0))  # (C,)
                pad = sampled_voxel[0].bool()  # Check if valid
                pad = pad.unsqueeze(0)  # (1,)
                sampled_voxel = sampled_voxel.unsqueeze(0)  # (1, C)
            else:
                # Pad to fixed size
                sampled_voxel = F.pad(
                    sampled_voxel, (0, 0, 0, self.max_voxel - sampled_voxel.shape[0])
                )  # (max_voxel, C)
                pad = sampled_voxel[:, 0].bool()  # (max_voxel,)
            
            f_list.extend([sampled_voxel])
            pad_list.extend([pad])
        
        if len(f_list) == 0:
            # Return empty tensors with correct device
            device = feature.device
            return torch.zeros(0, 1 if is_avg else self.max_voxel, feature.shape[1], device=device), \
                   torch.zeros(0, 1 if is_avg else self.max_voxel, dtype=torch.bool, device=device)
        
        return torch.stack(f_list), torch.stack(pad_list)

    @staticmethod
    def add_sin_difference(boxes1, boxes2, dim=6):
        assert dim != -1
        rad_pred_encoding = torch.sin(boxes1[..., dim:dim + 1]) * \
                            torch.cos(boxes2[..., dim:dim + 1])
        rad_tg_encoding = torch.cos(boxes1[..., dim:dim + 1]) * \
                          torch.sin(boxes2[..., dim:dim + 1])

        boxes1 = torch.cat([boxes1[..., :dim], rad_pred_encoding,
                            boxes1[..., dim + 1:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :dim], rad_tg_encoding,
                            boxes2[..., dim + 1:]], dim=-1)
        return boxes1, boxes2

    def get_direction_target(self, reg_targets):
        """
        Args:
            reg_targets:  [N, H * W * #anchor_num, 7]
                The last term is (theta_gt - theta_a)
        
        Returns:
            dir_targets:
                theta_gt: [N, H * W * #anchor_num, NUM_BIN] 
                NUM_BIN = 2
        """
        num_bins = self.dir['args']['num_bins']
        dir_offset = self.dir['args']['dir_offset']
        anchor_yaw = np.deg2rad(np.array(self.dir['args']['anchor_yaw']))
        self.anchor_yaw_map = torch.from_numpy(anchor_yaw).view(1,-1,1)
        self.anchor_num = self.anchor_yaw_map.shape[1]

        H_times_W_times_anchor_num = reg_targets.shape[1]
        anchor_map = self.anchor_yaw_map.repeat(1, H_times_W_times_anchor_num//self.anchor_num, 1).to(reg_targets.device)
        rot_gt = reg_targets[..., -1] + anchor_map[..., -1]
        offset_rot = limit_period(rot_gt - dir_offset, 0, 2 * np.pi)
        dir_cls_targets = torch.floor(offset_rot / (2 * np.pi / num_bins)).long()
        dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=num_bins - 1)
        dir_cls_targets = one_hot_f(dir_cls_targets, num_bins)
        return dir_cls_targets

    def logging(self, epoch, batch_id, batch_len, writer=None, suffix="", pbar=None, iter=None):
        """
        Print out the loss function for current iteration.

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
        pnpda_loss = self.loss_dict.get('pnpda_loss', 0)
        pos_cos_sim = self.loss_dict.get('pos_cos_sim', 0)
        neg_cos_sim = self.loss_dict.get('neg_cos_sim', 0)

        if pbar is None:
            print("[epoch %d][%d/%d]%s || Loss: %.4f || Conf Loss: %.4f"
                  " || Loc Loss: %.4f || Dir Loss: %.4f || IoU Loss: %.4f || PnPDA Loss: %.4f || pos_sim: %.4f || neg_sim: %.4f" % (
                      epoch, batch_id + 1, batch_len, suffix,
                      total_loss, cls_loss, reg_loss, dir_loss, iou_loss, pnpda_loss, pos_cos_sim, neg_cos_sim))
        else:
            pbar.set_description(
                "[epoch %d] || Loss: %.4f || PnPDA Loss: %.4f || pos_sim: %.4f || neg_sim: %.4f"
                % (epoch, total_loss, pnpda_loss, pos_cos_sim, neg_cos_sim)
            )

        if writer is not None:
            writer.add_scalar('Regression_loss'+suffix, reg_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Confidence_loss'+suffix, cls_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Dir_loss'+suffix, dir_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Iou_loss'+suffix, iou_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('PnPDA_loss'+suffix, pnpda_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('pos_cos_sim'+suffix, pos_cos_sim,
                            epoch*batch_len + batch_id)
            writer.add_scalar('neg_cos_sim'+suffix, neg_cos_sim,
                            epoch*batch_len + batch_id)

        # Log to wandb/swanlab
        if iter is not None and wandb is not None:
            try:
                wandb.log({
                    'Loss'+ suffix: total_loss,
                    'Reg_loss'+suffix: reg_loss,
                    'Conf_loss'+suffix: cls_loss,
                    'Dir_loss'+suffix: dir_loss,
                    'Iou_loss'+suffix: iou_loss,
                    'PnPDA_loss'+suffix: pnpda_loss,
                    'pos_cos_sim': pos_cos_sim,
                    'neg_cos_sim': neg_cos_sim,
                    'pos_softmax_sim': self.loss_dict.get('pos_softmax_sim', 0),
                    'neg_softmax_sim': self.loss_dict.get('neg_softmax_sim', 0),
                }, step=iter)
            except Exception as e:
                # swanlab/wandb not available or not initialized
                pass


def one_hot_f(tensor, num_bins, dim=-1, on_value=1.0, dtype=torch.float32):
    tensor_onehot = torch.zeros(*list(tensor.shape), num_bins, dtype=dtype, device=tensor.device) 
    tensor_onehot.scatter_(dim, tensor.unsqueeze(dim).long(), on_value)                    
    return tensor_onehot


def softmax_cross_entropy_with_logits(logits, labels):
    param = list(range(len(logits.shape)))
    transpose_param = [0] + [param[-1]] + param[1:-1]
    logits = logits.permute(*transpose_param)
    loss_ftor = torch.nn.CrossEntropyLoss(reduction="none")
    loss = loss_ftor(logits, labels.max(dim=-1)[1])
    return loss


def weighted_smooth_l1_loss(preds, targets, sigma=3.0, weights=None):
    diff = preds - targets
    abs_diff = torch.abs(diff)
    abs_diff_lt_1 = torch.le(abs_diff, 1 / (sigma ** 2)).type_as(abs_diff)
    loss = abs_diff_lt_1 * 0.5 * torch.pow(abs_diff * sigma, 2) + \
               (abs_diff - 0.5 / (sigma ** 2)) * (1.0 - abs_diff_lt_1)
    if weights is not None:
        loss *= weights
    return loss


def sigmoid_focal_loss(preds, targets, weights=None, **kwargs):
    assert 'gamma' in kwargs and 'alpha' in kwargs
    per_entry_cross_ent = torch.clamp(preds, min=0) - preds * targets.type_as(preds)
    per_entry_cross_ent += torch.log1p(torch.exp(-torch.abs(preds)))
    prediction_probabilities = torch.sigmoid(preds)
    p_t = (targets * prediction_probabilities) + ((1 - targets) * (1 - prediction_probabilities))
    modulating_factor = torch.pow(1.0 - p_t, kwargs['gamma'])
    alpha_weight_factor = targets * kwargs['alpha'] + (1 - targets) * (1 - kwargs['alpha'])

    loss = modulating_factor * alpha_weight_factor * per_entry_cross_ent
    if weights is not None:
        loss *= weights
    return loss
