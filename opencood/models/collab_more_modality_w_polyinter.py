# -*- coding: utf-8 -*-
# Author: Xiangbo Gao <xiangbogaobarry@gmail.com>
# License: MIT License

import torch
import cv2
import torch.nn as nn
import torch.nn.functional as F


import numpy as np
from icecream import ic
from collections import OrderedDict, Counter
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.feature_alignnet import AlignNet
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.pyramid_fuse import PyramidFusion
from opencood.models.fuse_modules.adapter import Adapter, Reverter
from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple
from opencood.utils.transformation_utils import normalize_pairwise_tfm
from opencood.models.sub_modules.translators import IntrinsicModalEncoder, HeteroFeatureConverter, build_feature_converter
from opencood.models.fuse_modules.fusion_in_one import (
    MaxFusion,
    AttFusion,
    DiscoFusion,
    V2VNetFusion,
    V2XViTFusion,
    CoBEVT,
    Where2commFusion,
    Who2comFusion,
)

from opencood.models.fuse_modules.fusion_in_one import regroup

from opencood.models.sub_modules.naive_decoder import NaiveDecoder
from opencood.models.sub_modules.bev_seg_head import BevSegHead
from opencood.utils.model_utils import check_trainable_module, fix_bn, unfix_bn

import importlib
import torchvision




import math
from opencood.models.polyinter_modules.adapter import TransformerDecoder
from opencood.models.polyinter_modules.gradient_layer import GradientScalarLayer
from opencood.models.polyinter_modules.multihead_attention import MultiheadAttention
from opencood.models.polyinter_modules.compressor import simple_align


class CollabMoreModalityWPolyInter(nn.Module):

    def __init__(self, args):
        super(CollabMoreModalityWPolyInter, self).__init__()




        self.general_prompt_ng = nn.Parameter(torch.FloatTensor(1, args["general_prompt_ng"]["c"], args["general_prompt_ng"]["h"], args["general_prompt_ng"]["w"]).to('cuda'), requires_grad=True)
        torch.nn.init.xavier_normal_(self.general_prompt_ng, gain=1.0)

        # 只一阶段训练，二阶段加载参数
        self.transformer = TransformerDecoder(args["transformer"])
        self.domain_classifier = DomainClassifier(args['domain_classifier'])
        self.channel_cross_attention = CrossAttention(args["channel_cross_attention"]['h']*args["channel_cross_attention"]['w'], args["channel_cross_attention"]['c'])

        # 一、二阶段均训练，不用加载参数
        self.compressor_ng = None
        if 'compressor_ng' in args:
            self.compressor_ng = simple_align(args['compressor_ng'])
        self.specific_prompt_ng_temp = torch.zeros(args["specific_prompt_ng"]["c"], args["specific_prompt_ng"]["h"], args["specific_prompt_ng"]["w"])
        self.specific_prompt_ng = nn.Parameter(self.specific_prompt_ng_temp, requires_grad=True)

        torch.nn.init.xavier_normal_(self.specific_prompt_ng, gain=1.0)

        self.count = 0




        self.unfix_modules = [self.general_prompt_ng, self.transformer, self.domain_classifier, self.channel_cross_attention, self.compressor_ng,  self.specific_prompt_ng ]


        self.args = args
        self.crop_to_visible = args.get("crop_to_visible", False)


        self.testing = args.get("testing", False)

        inference_modality = set(args.get("ignored_modality", []))
        ignored_modality = []
        if not self.testing:
            ignored_modality = inference_modality

        mods = [
            k for k in args.keys() if k not in ignored_modality and k.startswith("m") and k[1:].isdigit()
        ]
        # stable sort (for ddp)：m0, m1, m2, ...
        mods = sorted(mods, key=lambda s: int(s[1:]))
        self.modality_name_list = mods



        all_mods = [
            k for k in args.keys() if k.startswith("m") and k[1:].isdigit()
        ]
        # stable sort (for ddp)：m0, m1, m2, ...
        all_mods = sorted(all_mods, key=lambda s: int(s[1:]))
        self.all_modality_name_list = all_mods



        self.ego_modality_name = args["ego_modality"]


        self.default_modality = args.get("default_modality", "m4")
        self.default_modality_setting = args[self.default_modality]
        self.sensor_type_dict = OrderedDict()
        self.cam_crop_info = {}

        # setup each modality model
        for modality_name in self.all_modality_name_list:
            model_setting = args[modality_name]
            setattr(self, f"cav_range_{modality_name}", model_setting["lidar_range"])
            setattr(
                self, f"visible_range_{modality_name}", model_setting.get("visible_range", model_setting["lidar_range"])
            )

            self.build_encoder(modality_name, model_setting)
            self.build_backbone(modality_name, model_setting)
            self.build_aligner(modality_name, model_setting)

            """For feature transformation"""
            setattr(
                self,
                f"H_{modality_name}",
                (eval(f"self.cav_range_{modality_name}")[4] - eval(f"self.cav_range_{modality_name}")[1]),
            )
            setattr(
                self,
                f"W_{modality_name}",
                (eval(f"self.cav_range_{modality_name}")[3] - eval(f"self.cav_range_{modality_name}")[0]),
            )
            self.fake_voxel_size = 1

            # # only build for ego modality
            # if modality_name == self.ego_modality_name:
            #     self.build_fusion(modality_name, model_setting)
            #     self.build_shrink_header(modality_name, model_setting)
            #     self.build_head(modality_name, model_setting)

            try:
                self.build_fusion(modality_name, model_setting)
                self.build_shrink_header(modality_name, model_setting)
                self.build_head(modality_name, model_setting)
            except:
                self.build_fusion(modality_name, self.default_modality_setting)
                self.build_shrink_header(modality_name, self.default_modality_setting)
                self.build_head(modality_name, self.default_modality_setting)






        self.model_train_init()
        # check again which module is not fixed.
        check_trainable_module(self)

        self.testing = False



    def model_train_init(self):
        # 先全冻结
        for p in self.parameters():
            p.requires_grad_(False)
        self.apply(fix_bn)

        # 再解冻指定模块/参数
        for mm in self.unfix_modules:
            if isinstance(mm, nn.Module):
                for p in mm.parameters():
                    p.requires_grad_(True)
                mm.apply(unfix_bn)
            elif isinstance(mm, nn.Parameter):
                mm.requires_grad_(True)
            else:
                # raise TypeError(f"unfix_modules element must be nn.Module or nn.Parameter, got {type(mm)}: {mm}")
                pass


    def proj_neb2ego(self, neb_feat: torch.Tensor,
                     ego2neb_t_matrix: torch.Tensor) -> torch.Tensor:
        """
        将neb feature投射到自车坐标下。

        Args:
            neb_feat:           [num_cav, C, H, W]
            ego2neb_t_matrix:   [1, max_cav, 2, 3] normalize后的t_matrix
        Returns:
            projected_neb_feat:  投影后的neb feature.
        """
        N, C, H, W = neb_feat.shape
        ego2neb_t_matrix = ego2neb_t_matrix[0, :N, :, :]
        neb_feat = warp_affine_simple(neb_feat, ego2neb_t_matrix, (H, W))

        return neb_feat


    def regroup_any(self, x, record_len):
        """
        Split a tensor/list by record_len along the first dimension.

        x:
          - torch.Tensor with shape [N, ...], or
          - python list with length N
        record_len: 1D torch.Tensor / list[int], sum(record_len) == N
        """
        lens = record_len.tolist() if torch.is_tensor(record_len) else list(record_len)

        if torch.is_tensor(x):
            return list(torch.split(x, lens, dim=0))
        # python list
        out, s = [], 0
        for L in lens:
            out.append(x[s:s + int(L)])
            s += int(L)
        return out



    def forward(self, data_dict, show_bev=False):
        agent_modality_list = data_dict["agent_modality_list"]
        print(f'\nagent_modality_list:{agent_modality_list}')
        # print(f'self.modality_name_list:{self.modality_name_list}')
        record_len = data_dict["record_len"]
        print(f"{sum(record_len)=}")

        # Filter out the modality that is not ready for inference
        pairwise_t_matrix = data_dict["pairwise_t_matrix"]
        pairwise_t_matrix_new = torch.zeros_like(pairwise_t_matrix)
        agent_modality_list_filtered = []
        record_len_filtered = []
        cur = 0
        count = 0
        ptr = 0
        indices = []
        for m in agent_modality_list:
            if m in self.modality_name_list:
                agent_modality_list_filtered.append(m)
                count += 1
                indices.append(cur)
            cur += 1
            if record_len[ptr] == cur:
                record_len_filtered.append(count)
                if len(indices) > 0:
                    for i in range(len(indices)):
                        for j in range(len(indices)):
                            pairwise_t_matrix_new[ptr][i][j] = pairwise_t_matrix[ptr][indices[i]][indices[j]]
                cur = 0
                count = 0
                ptr += 1
                indices = []

        pairwise_t_matrix = pairwise_t_matrix_new
        record_len = torch.tensor(record_len_filtered, device=record_len.device)
        agent_modality_list = agent_modality_list_filtered

        used_modalities = []
        for modality in agent_modality_list:
            if modality in self.modality_name_list and modality not in used_modalities:
                used_modalities.append(modality)
        self.used_modality_name_list = used_modalities
        # print(f'self.used_modality_name_list:{self.used_modality_name_list}')
        output_dict = {}
        output_dict.update({"pyramid": "collab"})



        batch_ego_modality_list = []
        start = 0
        for mini_batch_idx, mini_batch_size in enumerate(record_len):
            batch_ego_modality_list.append(agent_modality_list[start])
            start += mini_batch_size



        modality_count_dict = Counter(agent_modality_list)
        modality_feature_dict = {}
        with torch.no_grad():  #
            # setup each modality model
            for modality_name in self.used_modality_name_list:
                # print(f'modality_name{modality_name}')
                if modality_name not in modality_count_dict:
                    continue

                feature = self.forward_encoder(data_dict, modality_name, output_dict)
                feature = self.forward_backbone(feature, modality_name)
                feature = self.forward_aligner(feature, modality_name)

                modality_feature_dict[modality_name] = feature


                if not eval(f"self.multi_sensor_{modality_name}"):
                    """
                    Crop/Padd camera feature map.
                    """
                    if "camera" in self.sensor_type_dict[modality_name]:
                        feature = modality_feature_dict[modality_name]
                        _, _, H, W = feature.shape
                        target_H = int(H * eval(f"self.crop_ratio_H_{modality_name}"))
                        target_W = int(W * eval(f"self.crop_ratio_W_{modality_name}"))

                        crop_func = torchvision.transforms.CenterCrop((target_H, target_W))
                        modality_feature_dict[modality_name] = crop_func(feature)
                        if eval(f"self.depth_supervision_{modality_name}"):
                            output_dict[modality_name].update(
                                {f"depth_items_{modality_name}": eval(f"self.encoder_{modality_name}").depth_items}
                            )






        """
        Assemble heter features
        """

        modality_name = self.ego_modality_name

        affine_matrix = normalize_pairwise_tfm(
            pairwise_t_matrix,
            eval(f"self.H_{modality_name}"),
            eval(f"self.W_{modality_name}"),
            self.fake_voxel_size,
        )

        counting_dict = {modality_name:0 for modality_name in self.modality_name_list}
        heter_feature_2d_list = []
        for modality_name in agent_modality_list:
            feat_idx = counting_dict[modality_name]
            heter_feature_2d_list.append(modality_feature_dict[modality_name][feat_idx])
            counting_dict[modality_name] += 1

        heter_feature_2d = torch.stack(heter_feature_2d_list)

        ######PolyInter########
        heter_feature_split = regroup(heter_feature_2d, record_len)



        ego_feature_list = []          # First agent of each scene (for final fusion)
        ego_feature_expanded_list = [] # Ego features expanded to match neighbors (for attention)
        neighbor_feature_list = []     # Remaining agents of each scene (neighbors)
        neighbor_feature_gt_list = []  # Ego features as ground truth for neighbors

        for i in range(len(heter_feature_split)):
            # First agent in each scene is ego
            ego_feat = heter_feature_split[i][0:1]
            ego_feature_list.append(ego_feat)

            # Remaining agents are neighbors
            if heter_feature_split[i].shape[0] > 1:
                neighbor_feats = heter_feature_split[i][1:]
                num_neighbors = neighbor_feats.shape[0]
                neighbor_feature_list.append(neighbor_feats)
                # Expand ego feature to match number of neighbors (for batch attention)
                ego_expanded = ego_feat.expand(num_neighbors, -1, -1, -1)
                ego_feature_expanded_list.append(ego_expanded)
                # Ground truth: repeat ego feature for each neighbor
                for j in range(num_neighbors):
                    neighbor_feature_gt_list.append(ego_feat)

        # Concatenate features
        ego_feature_2d = torch.cat(ego_feature_list, dim=0)  # shape[0] = len(record_len) = batch_size

        if neighbor_feature_list:
            neighbor_feature_2d = torch.cat(neighbor_feature_list, dim=0)  # shape[0] = sum(record_len) - len(record_len)
            ego_feature_expanded = torch.cat(ego_feature_expanded_list, dim=0)  # same shape as neighbor_feature_2d
            neighbor_feature_2d_gt = torch.cat(neighbor_feature_gt_list, dim=0)

            # Use expanded ego features as query (same size as key/neighbors)
            query = ego_feature_expanded
            key = neighbor_feature_2d

            # for i in range(ego_feature_2d.shape[0]):
            #     plot_feature_map(ego_feature_2d[i], f"/home/bayunqi/HEAL/A_Visualization/vis_temp/ployinter_ego_feature_2d_{i}_stage1.png")
            # for i in range(neighbor_feature_2d.shape[0]):
            #     plot_feature_map(neighbor_feature_2d[i], f"/home/bayunqi/HEAL/A_Visualization/vis_temp/ployinter_neighbor_feature_2d_{i}_stage1.png")

            specific_prompt = self.specific_prompt_ng.expand(key.size(0),-1,-1,-1)
            if self.compressor_ng:
                key = self.compressor_ng(key)
                specific_prompt = self.compressor_ng.conv_downsampling(specific_prompt)

            # 因为下面融合网络中会投射的
            # ## 将key的特征投射到自车坐标下：
            # i = 0
            # projected_key = []

            # for b, cav_num in enumerate(record_len):
            #     neb_num = cav_num - 1
            #     neb_num = 1 if neb_num == 0 else neb_num # 邻车数量为0的时候补充了自车数据进去
            #     temp =  self.proj_neb2ego(key[i:i+neb_num], ego2neb_t_matrix[b])
            #     i += neb_num
            #     projected_key.append(temp)
            # key = torch.cat(projected_key)

            general_prompt = self.general_prompt_ng.expand(key.size(0),-1,-1,-1)
            prompts = torch.cat([specific_prompt, general_prompt],dim=1)
            key = self.channel_cross_attention(prompts, query, key)
            specific_prompt = key[:,:specific_prompt.size(1),:,:]
            general_prompt = key[:,specific_prompt.size(1):,:,:]

            # spatial
            specific_prompt_out = self.transformer((specific_prompt+general_prompt)/2, query)

            # for style loss
            mean_k = torch.mean(specific_prompt_out.contiguous().view(specific_prompt_out.size(0), -1), dim=1, keepdim=False)
            mean_k2 = torch.mean(specific_prompt.contiguous().view(specific_prompt.size(0), -1), dim=1, keepdim=False)
            mean_k3 = torch.mean(general_prompt.contiguous().view(general_prompt.size(0), -1), dim=1, keepdim=False)
            mean_q = torch.mean(query.contiguous().view(query.size(0), -1), dim=1, keepdim=False)

            std_k = torch.var(specific_prompt_out.contiguous().view(specific_prompt_out.size(0), -1), dim=1, keepdim=False, unbiased=False)
            std_k2 = torch.var(specific_prompt.contiguous().view(specific_prompt.size(0), -1), dim=1, keepdim=False, unbiased=False)
            std_k3 = torch.var(general_prompt.contiguous().view(general_prompt.size(0), -1), dim=1, keepdim=False, unbiased=False)
            std_q = torch.var(query.contiguous().view(query.size(0), -1), dim=1, keepdim=False, unbiased=False)


            # for adv loss
            out_q = self.domain_classifier(query) #[2,1,50,176]
            out_k = self.domain_classifier(general_prompt)


            # for fusion
            spatial_features_2d=[]
            i = 0  # Index for neighbors in specific_prompt_out
            for data_idx in range(len(record_len)):
                # Add ego feature (use ego_feature_2d which has one feature per scene)
                spatial_features_2d.append(ego_feature_2d[data_idx:data_idx+1])
                neb_num = record_len[data_idx] - 1
                if (neb_num != 0):
                    # Add neighbor features (transformed by PolyInter)
                    spatial_features_2d.append(specific_prompt_out[i:i+neb_num])
                    i = i + neb_num
            spatial_features_2d = torch.cat(spatial_features_2d,dim=0)
            assert spatial_features_2d.shape[0] == sum(record_len)
        else:
            # No neighbor features available: use ego features directly for fusion
            spatial_features_2d = ego_feature_2d
            # Initialize dummy values for loss (will be skipped in loss computation)
            out_q = None
            out_k = None
            mean_q = mean_k = std_q = std_k = None
            mean_k2 = mean_k3 = std_k2 = std_k3 = None
        # for fusion
        # spatial_features_2d = specific_prompt_out

        # for i in range(spatial_features_2d.shape[0]):
        #     plot_feature_map(spatial_features_2d[i], f"/home/bayunqi/HEAL/A_Visualization/vis_temp/ployinter_transformed_feature_{i}_stage1.png")
        # update output dict
        # Only add these when there are neighbors (variables defined in else block)

        if neighbor_feature_list:
            output_dict.update({"out_q":out_q, "out_k":out_k})
            output_dict.update({"mean_q":mean_q, "mean_k":mean_k, "std_q":std_q, "std_k":std_k})
            output_dict.update({"mean_k2": mean_k2, "mean_k3": mean_k3, "std_k2": std_k2, "std_k3":std_k3})
            output_dict['neighbor_feature_gt'] = neighbor_feature_2d_gt  # GT features
            output_dict['neighbor_feature_pred'] = specific_prompt_out   # Predicted transformed features





        all_processed_features = spatial_features_2d
        if len(set(batch_ego_modality_list)) == 1:
            ego_modality_name = batch_ego_modality_list[0]
            fused_feature = self.forward_fusion(
                all_processed_features,
                pairwise_t_matrix,
                ego_modality_name,
                record_len,
                agent_modality_list,
                output_dict,
            )
            fused_feature = self.forward_shrink(fused_feature, ego_modality_name)
            self.forward_head(fused_feature, ego_modality_name, output_dict)
        else:
            print("batch_ego_modality_list", batch_ego_modality_list)
            occ_outputs_all = []
            cls_preds_all = []
            reg_preds_all = []
            dir_preds_all = []

            heter_feature_2d_mini_batch_list = self.regroup_any(all_processed_features, record_len)
            agent_modality_mini_batch_list = self.regroup_any(agent_modality_list, record_len)
            for mini_batch_idx, mini_batch_size in enumerate(record_len):
                ego_modality_name = batch_ego_modality_list[mini_batch_idx]



                affine_matrix = normalize_pairwise_tfm(
                    pairwise_t_matrix[mini_batch_idx].unsqueeze(0),
                    eval(f"self.H_{ego_modality_name}"),
                    eval(f"self.W_{ego_modality_name}"),
                    self.fake_voxel_size,
                )

                fused_feature, occ_outputs = eval(f"self.pyramid_backbone_{ego_modality_name}").forward_collab(
                    heter_feature_2d_mini_batch_list[mini_batch_idx],
                    record_len[mini_batch_idx:mini_batch_idx+1],
                    affine_matrix,
                    agent_modality_mini_batch_list[mini_batch_idx],
                    self.cam_crop_info,
                    # transform_idx=0,
                )
                occ_outputs_all.append(occ_outputs)

                feature = self.forward_shrink(fused_feature, ego_modality_name)
                if eval(f"self.head_method_{ego_modality_name}") in ["bev_seg_head", "seg_head"]:
                    output_dict.update(eval(f"self.head_{ego_modality_name}")(feature))             #### todo
                else:
                    cls_preds = eval(f"self.cls_head_{ego_modality_name}")(feature)
                    reg_preds = eval(f"self.reg_head_{ego_modality_name}")(feature)
                    cls_preds_all.append(cls_preds)
                    reg_preds_all.append(reg_preds)

                if hasattr(self, f"dir_head_{ego_modality_name}"):
                    dir_preds = eval(f"self.dir_head_{ego_modality_name}")(feature)
                else:
                    dir_preds = None
                dir_preds_all.append(dir_preds)

            occ_outputs_by_scale = [
                torch.cat(tensors_this_scale, dim=0)
                for tensors_this_scale in zip(*occ_outputs_all)  # 每个元素是该尺度下的 N 个 tensor
            ]
            cls_preds_all = torch.cat(cls_preds_all, dim=0)
            reg_preds_all = torch.cat(reg_preds_all, dim=0)
            try:
                dir_preds_all = torch.cat(dir_preds_all, dim=0)
            except:
                dir_preds_all = None
            output_dict.update({
                "occ_single_list": occ_outputs_by_scale,
                "cls_preds": cls_preds_all,
                "reg_preds": reg_preds_all,
                "dir_preds": dir_preds_all
            })

        return output_dict



    def build_encoder(self, modality_name, model_setting):
        """
        Builds the encoder for a given modality.

        Parameters:
        - modality_name (str): The name of the modality (e.g., 'camera', 'lidar').
        - model_setting (dict): Configuration settings for the model.

        The function dynamically imports the encoder module, determines the type of encoder
        (single sensor or multi-sensor), and sets appropriate attributes for the encoder.
        """

        encoder_filename = "opencood.models.heter_encoders"
        encoder_lib = importlib.import_module(encoder_filename)
        setattr(self, f"multi_sensor_{modality_name}", False)
        if isinstance(model_setting["core_method"], str):
            setattr(self, f"multi_sensor_{modality_name}", False)
            target_model_name = model_setting["core_method"].replace("_", "")
            for name, cls in encoder_lib.__dict__.items():
                if name.lower() == target_model_name.lower():
                    encoder_class = cls

            assert model_setting.get("encoder_args", None), "encoder_args should be provided"
            setattr(
                self,
                f"encoder_{modality_name}",
                encoder_class(model_setting["encoder_args"]),
            )
            if model_setting["encoder_args"].get("depth_supervision", False):
                setattr(self, f"depth_supervision_{modality_name}", True)
            else:
                setattr(self, f"depth_supervision_{modality_name}", False)

        elif isinstance(model_setting["core_method"], dict):
            setattr(self, f"multi_sensor_{modality_name}", True)
            target_model_name_camera = model_setting["core_method"]["camera"].replace("_", "")
            target_model_name_lidar = model_setting["core_method"]["lidar"].replace("_", "")
            for name, cls in encoder_lib.__dict__.items():
                if name.lower() == target_model_name_camera.lower():
                    encoder_class_camera = cls
                if name.lower() == target_model_name_lidar.lower():
                    encoder_class_lidar = cls

            assert model_setting.get("encoder_args_camera", None) and model_setting.get(
                "encoder_args_lidar", None
            ), "for multi_sensor, encoder_args_camera and encoder_args_lidar should be provided"
            setattr(
                self,
                f"encoder_{modality_name}_camera",
                encoder_class_camera(model_setting["encoder_args_camera"]),
            )
            setattr(
                self,
                f"encoder_{modality_name}_lidar",
                encoder_class_lidar(model_setting["encoder_args_lidar"]),
            )
            if model_setting["encoder_args_camera"].get("depth_supervision", False):
                setattr(self, f"depth_supervision_{modality_name}", True)
            else:
                setattr(self, f"depth_supervision_{modality_name}", False)

    def build_backbone(self, modality_name, model_setting):
        """
        Builds the backbone for a given modality.

        Parameters:
        - modality_name (str): The name of the modality (e.g., 'camera', 'lidar').
        - model_setting (dict): Configuration settings for the model.

        This function sets up the backbone network if the necessary backbone arguments are provided.
        """

        self.backbone_flag = False
        if model_setting.get("backbone_args", None):
            self.backbone_flag = True
            setattr(
                self,
                f"backbone_{modality_name}",
                ResNetBEVBackbone(model_setting["backbone_args"]),
            )

    def build_aligner(self, modality_name, model_setting):
        """
        Builds the aligner for a given modality.

        Parameters:
        - modality_name (str): The name of the modality (e.g., 'camera', 'lidar').
        - model_setting (dict): Configuration settings for the model.

        This function sets up the aligner network and computes cropping ratios if the sensor type is a camera.
        """

        sensor_name = model_setting["sensor_type"]
        self.sensor_type_dict[modality_name] = sensor_name
        setattr(self, f"aligner_{modality_name}", AlignNet(model_setting["aligner_args"]))

        if "camera" in sensor_name:
            camera_mask_args = model_setting["camera_mask_args"]
            setattr(
                self,
                f"crop_ratio_W_{modality_name}",
                (eval(f"self.cav_range_{modality_name}")[3]) / (camera_mask_args["grid_conf"]["xbound"][1]),
            )
            setattr(
                self,
                f"crop_ratio_H_{modality_name}",
                (eval(f"self.cav_range_{modality_name}")[4]) / (camera_mask_args["grid_conf"]["ybound"][1]),
            )
            setattr(
                self,
                f"xdist_{modality_name}",
                (camera_mask_args["grid_conf"]["xbound"][1] - camera_mask_args["grid_conf"]["xbound"][0]),
            )
            setattr(
                self,
                f"ydist_{modality_name}",
                (camera_mask_args["grid_conf"]["ybound"][1] - camera_mask_args["grid_conf"]["ybound"][0]),
            )
            self.cam_crop_info[modality_name] = {
                f"crop_ratio_W_{modality_name}": eval(f"self.crop_ratio_W_{modality_name}"),
                f"crop_ratio_H_{modality_name}": eval(f"self.crop_ratio_H_{modality_name}"),
            }

    def build_fusion(self, modality_name, model_setting):
        """
        Builds the fusion module for a given modality.

        Parameters:
        - modality_name (str): The name of the modality (e.g., 'camera', 'lidar').
        - model_setting (dict): Configuration settings for the model.

        This function sets up the fusion method based on the specified fusion method in the model settings.
        """

        """
        Fusion, by default multiscale fusion:
        Note the input of PyramidFusion has downsampled 2x. (SECOND required)
        """

        if model_setting["fusion_method"] == "max":
            setattr(self, f"pyramid_backbone_{modality_name}", MaxFusion())
        elif model_setting["fusion_method"] == "att":
            setattr(
                self,
                f"pyramid_backbone_{modality_name}",
                AttFusion(model_setting["fusion_backbone"]),
            )
        elif model_setting["fusion_method"] == "disconet":
            setattr(
                self,
                f"pyramid_backbone_{modality_name}",
                DiscoFusion(model_setting["fusion_backbone"]),
            )
        elif model_setting["fusion_method"] == "v2vnet":
            setattr(
                self,
                f"pyramid_backbone_{modality_name}",
                V2VNetFusion(model_setting["fusion_backbone"]),
            )
        elif model_setting["fusion_method"] == "v2xvit":
            setattr(
                self,
                f"pyramid_backbone_{modality_name}",
                V2XViTFusion(model_setting["fusion_backbone"]),
            )
        elif model_setting["fusion_method"] == "cobevt":
            setattr(
                self,
                f"pyramid_backbone_{modality_name}",
                CoBEVT(model_setting["fusion_backbone"]),
            )
        elif model_setting["fusion_method"] == "where2comm":
            setattr(
                self,
                f"pyramid_backbone_{modality_name}",
                Where2commFusion(model_setting["fusion_backbone"]),
            )
        elif model_setting["fusion_method"] == "who2com":
            setattr(
                self,
                f"pyramid_backbone_{modality_name}",
                Who2comFusion(model_setting["fusion_backbone"]),
            )
        elif model_setting["fusion_method"] == "pyramid":
            setattr(
                self,
                f"pyramid_backbone_{modality_name}",
                PyramidFusion(model_setting["fusion_backbone"]),
            )
        else:
            raise NotImplementedError(f"Method {model_setting['fusion_method']} not implemented.")

        if model_setting["fusion_method"] != "pyramid":
            # other method does not have agent_modality_list and cam_crop_info, neither returning occ_single_list
            pyramid_backbone = getattr(self, f"pyramid_backbone_{modality_name}")
            pyramid_backbone.forward_collab = lambda *args: (
                pyramid_backbone.forward(*args[:3]),
                [],
            )

    def build_shrink_header(self, modality_name, model_setting):
        """
        Builds the shrink header for a given modality.

        Parameters:
        - modality_name (str): The name of the modality (e.g., 'camera', 'lidar').
        - model_setting (dict): Configuration settings for the model.

        This function sets up a downsample convolutional layer if the shrink header is specified.
        """

        setattr(self, f"shrink_flag_{modality_name}", False)
        if "shrink_header" in model_setting:
            setattr(self, f"shrink_flag_{modality_name}", True)
            setattr(
                self,
                f"shrink_conv_{modality_name}",
                DownsampleConv(model_setting["shrink_header"]),
            )

    def build_head(self, modality_name, model_setting):
        """
        Builds the head for a given modality.

        Parameters:
        - modality_name (str): The name of the modality (e.g., 'camera', 'lidar').
        - model_setting (dict): Configuration settings for the model.

        This function sets up the head network for various head methods such as object detection, segmentation, etc.
        """

        # By default, point pillar pyramid object detection head
        head_method = model_setting.get("head_method", "point_pillar_pyramid_object_detection_head")
        downsample_rate = model_setting.get("downsample_rate", 1)
        setattr(self, f"head_method_{modality_name}", head_method)
        setattr(self, f"downsample_rate_{modality_name}", downsample_rate)
        # self.head_method = model_setting.get("head_method", "point_pillar_pyramid_object_detection_head")
        # self.downsample_rate = model_setting.get("downsample_rate", 1)
        if head_method == "point_pillar_pyramid_object_detection_head":

            setattr(
                self,
                f"cls_head_{modality_name}",
                nn.Conv2d(
                    model_setting["in_head"],
                    model_setting["anchor_number"],
                    kernel_size=1,
                ),
            )
            setattr(
                self,
                f"reg_head_{modality_name}",
                nn.Conv2d(
                    model_setting["in_head"],
                    7 * model_setting["anchor_number"],
                    kernel_size=1,
                ),
            )
            if model_setting.get("dir_args", None):
                setattr(
                    self,
                    f"dir_head_{modality_name}",
                    nn.Conv2d(
                        model_setting["in_head"],
                        model_setting["dir_args"]["num_bins"] * model_setting["anchor_number"],
                        kernel_size=1,
                    ),
                )

        elif head_method == "point_pillar_object_detection_head":
            setattr(
                self,
                f"cls_head_{modality_name}",
                nn.Conv2d(model_setting["in_head"], 1, kernel_size=1),
            )
            setattr(
                self,
                f"reg_head_{modality_name}",
                nn.Conv2d(model_setting["in_head"], 7, kernel_size=1),
            )
            if model_setting.get("dir_args", None):
                setattr(
                    self,
                    f"dir_head_{modality_name}",
                    nn.Conv2d(
                        model_setting["in_head"],
                        model_setting["dir_args"]["num_bins"],
                        kernel_size=1,
                    ),
                )

        elif head_method == "bev_seg_head":
            setattr(
                self,
                f"head_{modality_name}",
                nn.Sequential(
                    NaiveDecoder(model_setting["decoder_args"]),
                    BevSegHead(
                        model_setting["target"],
                        model_setting["seg_head_dim"],
                        model_setting["output_class_dynamic"],
                        model_setting["output_class_static"],
                    ),
                ),
            )

        elif head_method == "seg_head":
            setattr(
                self,
                f"head_{modality_name}",
                nn.Sequential(
                    BevSegHead(
                        model_setting["target"],
                        model_setting["seg_head_dim"],
                        model_setting["output_class_dynamic"],
                        model_setting["output_class_static"],
                    ),
                ),
            )

        elif head_method == "pixor_head":

            setattr(
                self,
                f"cls_head_{modality_name}",
                nn.Conv2d(model_setting["in_head"], 1, kernel_size=1),
            )
            setattr(
                self,
                f"reg_head_{modality_name}",
                nn.Conv2d(model_setting["in_head"], 6, kernel_size=1),
            )

        else:
            raise NotImplementedError(f"Head method {head_method} not implemented.")

    def build_compressor(self, modality_name, model_setting):
        """
        Builds the compressor for a given modality.
        # compressor will be only trainable

        Parameters:
        - modality_name (str): The name of the modality (e.g., 'camera', 'lidar').
        - model_setting (dict): Configuration settings for the model.

        This function sets up the compressor module if the compressor settings are provided.
        """

        setattr(self, f"compress_{modality_name}", False)
        if "compressor" in model_setting:
            setattr(self, f"compress_{modality_name}", True)
            setattr(
                self,
                f"compressor_{modality_name}",
                NaiveCompressor(
                    model_setting["compressor"]["input_dim"],
                    model_setting["compressor"]["compress_ratio"],
                ),
            )



    def forward_encoder(self, data_dict, modality_name, output_dict):
        """
        Forwards the input data through the encoder.
        """
        if modality_name == "ego":
            feature = eval(f"self.encoder_{self.ego_modality_name}")(
                data_dict, modality_name, False
            )
            return feature

        if eval(f"self.multi_sensor_{modality_name}"):
            feature_camera = eval(f"self.encoder_{modality_name}_camera")(
                data_dict, modality_name, eval(f"self.multi_sensor_{modality_name}")
            )

            """
            Crop/Padd camera feature map.

            Parameters:
            - data_dict (dict): Input data dictionary.
            - modality_name (str): The name of the modality (e.g., 'camera', 'lidar').
            - output_dict (dict): Output data dictionary.

            Returns:
            - feature (Tensor): Encoded features.
            """

            if "camera" in self.sensor_type_dict[modality_name]:
                # should be padding. Instead of masking
                _, _, H, W = feature_camera.shape
                target_H = int(H * eval(f"self.crop_ratio_H_{modality_name}"))
                target_W = int(W * eval(f"self.crop_ratio_W_{modality_name}"))

                crop_func = torchvision.transforms.CenterCrop((target_H, target_W))
                feature_camera = crop_func(feature_camera)
                if eval(f"self.depth_supervision_{modality_name}"):
                    output_dict.update(
                        {f"depth_items_{modality_name}": eval(f"self.encoder_{modality_name}_camera").depth_items}
                    )

            feature_lidar = eval(f"self.encoder_{modality_name}_lidar")(
                data_dict, modality_name, eval(f"self.multi_sensor_{modality_name}")
            )

            feature = feature_camera + feature_lidar
        else:
            feature = eval(f"self.encoder_{modality_name}")(
                data_dict, modality_name, eval(f"self.multi_sensor_{modality_name}")
            )
        return feature

    def forward_backbone(self, feature, modality_name):
        """
        Forwards the encoded feature through the backbone.

        Parameters:
        - feature (Tensor): Encoded features.
        - modality_name (str): The name of the modality (e.g., 'camera', 'lidar').

        Returns:
        - feature (Tensor): Backbone features.
        """

        if self.backbone_flag:
            feature = eval(f"self.backbone_{modality_name}")({"spatial_features": feature})["spatial_features_2d"]
        return feature

    def forward_aligner(self, feature, modality_name):
        """
        Forwards the feature through the aligner.

        Parameters:
        - feature (Tensor): Backbone features.
        - modality_name (str): The name of the modality (e.g., 'camera', 'lidar').

        Returns:
        - feature (Tensor): Aligned features.
        """

        feature = eval(f"self.aligner_{modality_name}")(feature)
        return feature

    def forward_shrink(self, feature, modality_name):
        """
        Forwards the feature through the shrink header if available.

        Parameters:
        - feature (Tensor): Aligned features.
        - modality_name (str): The name of the modality (e.g., 'camera', 'lidar').

        Returns:
        - feature (Tensor): Shrunken features.
        """

        if getattr(self, f"shrink_flag_{modality_name}"):
            feature = eval(f"self.shrink_conv_{modality_name}")(feature)
        return feature

    def forward_compress(self, feature, modality_name):
        """
        Forwards the feature through the compressor if available.

        Parameters:
        - feature (Tensor): Shrunken features.
        - modality_name (str): The name of the modality (e.g., 'camera', 'lidar').

        Returns:
        - feature (Tensor): Compressed features.
        """

        if getattr(self, f"compress_{modality_name}"):
            feature = eval(f"self.compressor_{modality_name}")(feature)
        return feature



    def forward_fusion(
            self,
            feature,
            pairwise_t_matrix,
            modality_name,
            record_len,
            agent_modality_list,
            output_dict,
    ):
        """
        Forwards the feature through the fusion module.

        Parameters:
        - feature (Tensor): Compressed features.
        - data_dict (dict): Input data dictionary.
        - modality_name (str): The name of the modality (e.g., 'camera', 'lidar').
        - record_len (int): Length of the record.
        - agent_modality_list (list): List of agent modalities.
        - output_dict (dict): Output data dictionary.

        Returns:
        - fused_feature (Tensor): Fused features.
        """

        affine_matrix = normalize_pairwise_tfm(
            pairwise_t_matrix,
            eval(f"self.H_{modality_name}"),
            eval(f"self.W_{modality_name}"),
            self.fake_voxel_size,
        )

        fused_feature, occ_outputs = eval(f"self.pyramid_backbone_{modality_name}").forward_collab(
            feature,
            record_len,
            affine_matrix,
            agent_modality_list,
            self.cam_crop_info,
            # transform_idx=0,
        )

        output_dict.update({"occ_single_list": occ_outputs})

        return fused_feature

    def forward_head(self, feature, modality_name, output_dict):
        """
        Forwards the feature through the head network.

        Parameters:
        - feature (Tensor): Fused features.
        - modality_name (str): The name of the modality (e.g., 'camera', 'lidar').
        - output_dict (dict): Output data dictionary.

        This function handles the forward pass for different head methods such as object detection, segmentation, etc.
        """

        if eval(f"self.head_method_{modality_name}") in ["bev_seg_head", "seg_head"]:
            output_dict.update(eval(f"self.head_{modality_name}")(feature))
        else:
            cls_preds = eval(f"self.cls_head_{modality_name}")(feature)
            reg_preds = eval(f"self.reg_head_{modality_name}")(feature)
            if hasattr(self, f"dir_head_{modality_name}"):
                dir_preds = eval(f"self.dir_head_{modality_name}")(feature)
            else:
                dir_preds = None

            output_dict.update({"cls_preds": cls_preds, "reg_preds": reg_preds, "dir_preds": dir_preds})







class CrossAttention(nn.Module):
    def __init__(self, in_features, in_channels, embed_dim=256, num_heads=1, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.multihead_attn1 = MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.multihead_attn2 = MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.mlp1 = nn.Linear(in_features=in_features, out_features=embed_dim)
        self.mlp2 = nn.Linear(in_features=in_features, out_features=embed_dim)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.layer_norm3 = nn.LayerNorm(in_features)
        self.layer_norm4 = nn.LayerNorm(in_features)
        self.layer_norm5 = nn.LayerNorm(in_features)
        self.conv_k1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1)
        self.conv_k2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1)

    def forward(self, prompts, query, key, selected_char='k', mask=None):
        # query shape: (num_ego, channels, height, width)
        # key shape: (num_neighbor, channels, height, width)
        # prompts shape: (num_neighbor, prompts_channel, height, width)
        # Note: num_ego and num_neighbor may be different!

        num_ego, channels, height, width = query.shape
        num_neighbor = key.size(0)
        prompts_channel = prompts.size(1)

        prompts1_conv = self.conv_k1(prompts[:,:query.size(1),:,:])
        prompts2_conv = self.conv_k1(prompts[:,query.size(1):,:,:])

        prompts = torch.cat([prompts1_conv, prompts2_conv], 1)

        key_conv = self.conv_k2(key)

        # Use correct batch sizes for each tensor
        key_conv = key_conv.view(num_neighbor, channels, height * width)

        # Reshape to (height*width, batch_size, channels) for multihead attention
        prompts = prompts.view(num_neighbor, prompts_channel, height * width)
        query = query.view(num_ego, channels, height * width)
        key = key.reshape(num_neighbor, channels, height * width)
        query = self.layer_norm1(self.mlp1(query))
        key_k = self.layer_norm2(self.mlp2(key))
        # prompts = self.layer_norm6(self.mlp3(prompts))

        # Perform cross-attention
        # When query and key have different batch_sizes, we need to handle them differently
        # Option 1: If num_ego == 1, expand query to match num_neighbor for batch processing
        # Option 2: Loop over each neighbor (slower but more flexible)

        if num_ego == 1 and num_neighbor > 1:
            # Expand ego query to match neighbor batch size for parallel processing
            query_expanded = query.expand(num_neighbor, -1, -1)  # (num_neighbor, channels, H*W)

            prompts_output1, _ = self.multihead_attn1(query_expanded, key_k, prompts[:,:query.size(1),:])
            prompts_output2, _ = self.multihead_attn1(query_expanded, key_k, prompts[:,query.size(1):,:])
            prompts_output = torch.cat([prompts_output1, prompts_output2],1)
            prompts_output = self.layer_norm3(prompts + prompts_output)

            key_output = self.layer_norm4(key_conv + self.multihead_attn2(query_expanded, key_k, key_conv)[0])

            prompts_output[:,:query.size(1),:] += key_output
            prompts_output[:,query.size(1):,:] += key_output
            prompts_output = self.layer_norm5(prompts_output)

            # Output shape: (num_neighbor, prompts_channel, H, W)
            prompts_output = prompts_output.view(num_neighbor, prompts_channel, height, width)
        else:
            # num_ego == num_neighbor or handle other cases
            # Original logic assumes equal batch sizes
            prompts_output1, _ = self.multihead_attn1(query, key_k, prompts[:,:query.size(1),:])
            prompts_output2, _ = self.multihead_attn1(query, key_k, prompts[:,query.size(1):,:])
            prompts_output = torch.cat([prompts_output1, prompts_output2],1)
            prompts_output = self.layer_norm3(prompts + prompts_output)

            key_output = self.layer_norm4(key_conv + self.multihead_attn2(query, key_k, key_conv)[0])

            prompts_output[:,:query.size(1),:] += key_output
            prompts_output[:,query.size(1):,:] += key_output
            prompts_output = self.layer_norm5(prompts_output)

            prompts_output = prompts_output.view(num_neighbor, prompts_channel, height, width)

        return prompts_output


class detect_head(nn.Module):
    def __init__(self, args) -> None:
        super(detect_head, self).__init__()
        in_channel=args['channel']
        self.cls_head = nn.Conv2d(in_channel, args["anchor_number"], kernel_size=1)
        self.reg_head = nn.Conv2d(
            in_channel, 7 * args["anchor_number"], kernel_size=1
        )

    def forward(self, x):
        psm = self.cls_head(x)
        rm = self.reg_head(x)
        return {"psm": psm, "rm": rm}

class DomainClassifier(nn.Module):
    def __init__(self, args) -> None:
        super(DomainClassifier, self).__init__()
        self.conv_layer1 = nn.Conv2d(args['in_channel'], 64, kernel_size=3)
        self.conv_layer2 = nn.Conv2d(64, 32, kernel_size=3)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=6, stride=2, padding=0)
        temp = (math.floor((args['in_size'][0]-10)/2)+1)*(math.floor((args['in_size'][1]-10)/2)+1)
        self.linear_layer = nn.Linear(temp, args['out_size'])

        self.rgl = GradientScalarLayer(-9.1)

    def forward(self, feature):
        feature = self.rgl(feature)
        feature = self.conv_layer1(feature)
        feature = torch.relu(feature)
        feature = self.conv_layer2(feature)
        feature = torch.relu(feature)
        feature = self.maxpool(feature)
        feature = feature.max(dim=1)[0]
        feature = feature.reshape(feature.size(0), -1)
        out = self.linear_layer(feature)
        return out
