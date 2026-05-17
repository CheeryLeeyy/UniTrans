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
from opencood.utils.transformation_utils import normalize_pairwise_tfm
from opencood.models.sub_modules.moe2 import build_feature_converter
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
from opencood.models.sub_modules.naive_decoder import NaiveDecoder
from opencood.models.sub_modules.bev_seg_head import BevSegHead
from opencood.utils.model_utils import check_trainable_module, fix_bn, unfix_bn

import importlib
import torchvision

from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple




class CollabMoreModalityWMoE(nn.Module):

    def __init__(self, args):
        super(CollabMoreModalityWMoE, self).__init__()

        self.args = args
        self.stage = args.get("stage", None)
        self.crop_to_visible = args.get("crop_to_visible", False)


        self.testing = args.get("testing", False)

        inference_modality = set(args.get("ignored_modality", []))
        ignored_modality = []
        if not self.testing:
            ignored_modality = inference_modality
            inference_modality = []

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
        self.ego_modalitsy_setting= args[self.ego_modality_name]
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


            try:
                self.build_fusion(modality_name, model_setting)
                self.build_shrink_header(modality_name, model_setting)
                self.build_head(modality_name, model_setting)
            except:
                print(modality_name)
                print("self.build_fusion(modality_name, self.ego_modality_setting)  self.build_shrink_header(modality_name, self.ego_modality_setting)  self.build_head(modality_name, self.ego_modality_setting)")
                self.build_fusion(modality_name, self.ego_modality_setting)
                self.build_shrink_header(modality_name, self.ego_modality_setting)
                self.build_head(modality_name, self.ego_modality_setting)





        self.converter = build_feature_converter(args["converter"])


        self.num_modalities = len(self.modality_name_list) - len(inference_modality)
        self.modality_to_id = {m: i for i, m in enumerate(self.modality_name_list)}


        self.model_train_init()
        # check again which module is not fixed.
        check_trainable_module(self)


    def init_parameters(self):
        try:
            self.converter.init_parameters()
            print("self.converter.init_parameters()")
        except Exception as  e:
            print(e)





    def model_train_init(self):
        print("collab_moe.model_train_init!!!")
        #
        for p in self.parameters():
            p.requires_grad_(False)
        self.apply(fix_bn)


        # for p in self.intrinsic_modal_encoder.parameters():
        #     p.requires_grad_(True)
        # self.intrinsic_modal_encoder.apply(unfix_bn)

        for p in self.converter.parameters():
            p.requires_grad_(True)
        self.converter.apply(unfix_bn)


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
        output_dict.update({"pyramid": "collab_moe"})




        # batch_ego_modality_list = []
        # agent_modality_mini_batch_list = self.regroup_any(agent_modality_list, record_len)
        # for mini_batch_idx, mini_batch_size in enumerate(record_len):
        #     ego_modality_mb = agent_modality_mini_batch_list[mini_batch_idx][0]
        #     batch_ego_modality_list.append(ego_modality_mb)


        batch_ego_modality_list = []
        start = 0
        for mini_batch_idx, mini_batch_size in enumerate(record_len):
            batch_ego_modality_list.append(agent_modality_list[start])
            start += mini_batch_size




        modality_count_dict = Counter(agent_modality_list)
        modality_feature_dict = {}
        with torch.no_grad():  # 禁用梯度计算
            # setup each modality model
            for modality_name in self.used_modality_name_list:
                # print(f'modality_name{modality_name}')
                if self.stage not in ["train_moe"] and modality_name not in modality_count_dict:
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

            if not self.testing:
            # if self.training or self.converter.training:
                # For ego modality

                if isinstance(data_dict["inputs_ego"], list):
                    all_processed_features_gt = []
                    inputs_ego_list = data_dict["inputs_ego"]

                    for i in range (len(inputs_ego_list)):
                        batch_inputs = inputs_ego_list[i]
                        batch_modality_gt = batch_ego_modality_list[i]
                        data_dict_tmp = {f"inputs_{batch_modality_gt}": batch_inputs}

                        feature_ego = self.forward_encoder(data_dict_tmp, batch_modality_gt, {})
                        # import pdb; pdb.set_trace()
                        # print(f'spatial_feature,shape:{feature.shape}')
                        feature_ego = self.forward_backbone(feature_ego, batch_modality_gt)
                        # max_val = torch.max(feature)
                        # min_val = torch.min(feature)
                        # cv2.imwrite("debug/feature.png", ((feature[0].abs().max(0)[0].cpu().numpy() - feature[0].abs().max(0)[0].cpu().numpy().max()) / (feature[0].abs().max(0)[0].cpu().numpy().max() - feature[0].abs().max(0)[0].cpu().numpy().min()) * 255).astype(np.uint8))
                        # print(f'spatial_feature_2d.shape:{feature.shape}')
                        FE = self.forward_aligner(feature_ego, batch_modality_gt)

                        if not eval(f"self.multi_sensor_{batch_modality_gt}"):
                            """
                            Crop/Padd camera feature map.
                            """
                            if "camera" in self.sensor_type_dict[batch_modality_gt]:
                                feature = FE
                                _, _, H, W = feature.shape
                                target_H = int(H * eval(f"self.crop_ratio_H_{batch_modality_gt}"))
                                target_W = int(W * eval(f"self.crop_ratio_W_{batch_modality_gt}"))

                                crop_func = torchvision.transforms.CenterCrop((target_H, target_W))
                                FE = crop_func(feature)
                                if eval(f"self.depth_supervision_{batch_modality_gt}"):
                                    output_dict[modality_name].update(
                                        {f"depth_items_{batch_modality_gt}": eval(
                                            f"self.encoder_{batch_modality_gt}").depth_items}
                                    )

                        all_processed_features_gt.append(FE)
                    FE = torch.cat(all_processed_features_gt, dim=0)

                else:
                    feature_ego = self.forward_encoder(data_dict, "ego", output_dict)
                    # import pdb; pdb.set_trace()
                    # print(f'spatial_feature,shape:{feature.shape}')
                    feature_ego = self.forward_backbone(feature_ego, self.ego_modality_name)
                    # max_val = torch.max(feature)
                    # min_val = torch.min(feature)
                    # cv2.imwrite("debug/feature.png", ((feature[0].abs().max(0)[0].cpu().numpy() - feature[0].abs().max(0)[0].cpu().numpy().max()) / (feature[0].abs().max(0)[0].cpu().numpy().max() - feature[0].abs().max(0)[0].cpu().numpy().min()) * 255).astype(np.uint8))
                    # print(f'spatial_feature_2d.shape:{feature.shape}')
                    FE = self.forward_aligner(feature_ego, self.ego_modality_name)

                    if not eval(f"self.multi_sensor_{self.ego_modality_name}"):
                        """
                        Crop/Padd camera feature map.
                        """
                        if "camera" in self.sensor_type_dict[self.ego_modality_name]:
                            feature = FE
                            _, _, H, W = feature.shape
                            target_H = int(H * eval(f"self.crop_ratio_H_{self.ego_modality_name}"))
                            target_W = int(W * eval(f"self.crop_ratio_W_{self.ego_modality_name}"))

                            crop_func = torchvision.transforms.CenterCrop((target_H, target_W))
                            FE = crop_func(feature)
                            if eval(f"self.depth_supervision_{self.ego_modality_name}"):
                                output_dict[self.ego_modality_name].update(
                                    {f"depth_items_{self.ego_modality_name}": eval(
                                        f"self.encoder_{self.ego_modality_name}").depth_items}
                                )

                output_dict.update({
                    "FE": FE,
                })




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

        counting_dict = {modality_name: 0 for modality_name in self.modality_name_list}
        heter_feature_2d_list = []
        for modality_name in agent_modality_list:
            feat_idx = counting_dict[modality_name]
            heter_feature_2d_list.append(modality_feature_dict[modality_name][feat_idx])
            counting_dict[modality_name] += 1

        heter_feature_2d_list = torch.stack(heter_feature_2d_list)

        ## minibatch split
        heter_feature_2d_mini_batch_list = self.regroup_any(heter_feature_2d_list, record_len)
        agent_modality_mini_batch_list = self.regroup_any(agent_modality_list, record_len)

        # 存储所有处理后的特征（按原始车辆顺序）
        all_processed_features = []
        all_alphas = []
        all_modality_pairs = []
        batch_ego_modality_list = []  ### 每个batch的0号agent的模态，gt模态

        for mini_batch_idx, mini_batch_size in enumerate(record_len):
            batch_feature = heter_feature_2d_mini_batch_list[mini_batch_idx]
            batch_agent_modality = agent_modality_mini_batch_list[mini_batch_idx]

            _, _, H, W = batch_feature.shape

            ego_feat = batch_feature[0:1]
            ego_modality = batch_agent_modality[0]
            batch_ego_modality_list.append(ego_modality)

            for mm in batch_agent_modality:
                all_modality_pairs.append((self.modality_to_id[ego_modality], self.modality_to_id[mm]))

            ## 将ego特征转换到neb坐标系下
            t_matrix = affine_matrix[mini_batch_idx][:mini_batch_size, :mini_batch_size, :, :]
            ego_repeat = ego_feat.repeat(batch_feature.shape[0], 1, 1, 1)
            ego_in_nebcoord = warp_affine_simple(ego_repeat, t_matrix[:, 0, :, :], (H, W), align_corners=True)

            # 通过moe转换neb特征到ego视角
            moe_output, alphas = self.converter(
                ego_in_nebcoord,
                batch_feature,
                return_alpha=True
            )  # [b, C, H, W] - 转换后的neb特征
            # print(f"[{neb_modality=}] --> [{ego_modality=}]")

            if self.testing:
                moe_output[0:1] = ego_feat

            all_processed_features.append(moe_output)
            all_alphas.append(alphas)

        all_processed_features = torch.cat(all_processed_features, dim=0)
        all_alphas = torch.cat(all_alphas, dim=1)



        agent_modality_ids = torch.tensor([self.modality_to_id[m] for m in agent_modality_list], dtype=torch.long, device=all_alphas.device)          # [M]

        moe_output_dict = {
            "agent_modality_ids": agent_modality_ids,
            "FN2E": all_processed_features,
            "all_alphas": all_alphas,
            "all_modality_pairs": all_modality_pairs,
            "feat_vis": {
                "record_len": record_len,
                "raw": heter_feature_2d_list,
                "processed": all_processed_features,
            }
        }



        # if all_alphas:
        #       # n_layers, batch, d
        # all_alphas = F.softmax(all_alphas, dim=-1)
        eg = all_alphas[:2, -3:].detach()
        order = eg.argsort(dim=-1, descending=True)  # 每行从大到小的索引
        ranks = order.argsort(dim=-1)  # 每个元素的名次(0=最大)
        print(ranks)
        print(eg)

        _all_alphas = F.softmax(all_alphas, dim=-1)
        eg = _all_alphas[:2, -3:].detach()
        order = eg.argsort(dim=-1, descending=True)  # 每行从大到小的索引
        print(eg)


        # heter_feature_2d_list, moe_output_dict = self.forward_moe(
        #     modality_feature_dict, intrinsic_code_dict, pairwise_t_matrix, record_len, agent_modality_list
        # )
        # print(f"heter_feature_2d_list: {heter_feature_2d_list.shape}")
        fused_feature_dict = dict()
        # for calculating feature loss
        # todo: 这里需要cropFN2E参考STAMP
        # FN2E = heter_feature_2d_list
        # has_neb = not (record_len == 1).all()
        # output_dict.update({
        #     "has_neb": has_neb,
        # })

        output_dict.update(moe_output_dict)

        # if output_dict.get("FE") is not None:
        #     assert output_dict["FE"].shape == output_dict["FN2E"].shape

        # FE, FN2E = self.postprocess_feature(FE, FN2E) #todo:函数还没修改完



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







        # print(f"FE:{FE.shape}")
        # print(f"FN2E:{FN2E.shape}")
        return output_dict

    # def set_collab_train(self, is_train):
    #     """在collab_moe中的train属性"""
    #     self.train = is_train
    #     print(f"已将collab_moe的train设置为: {is_train}")

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

    def build_adapter_and_reverter(self, modality_name, model_setting):
        """
        Builds the adapter and reverter for a given modality.

        Parameters:
        - modality_name (str): The name of the modality (e.g., 'camera', 'lidar').
        - model_setting (dict): Configuration settings for the model.

        This function sets up the adapter and reverter modules for modalities other than 'm0'.
        """

        if modality_name != "m0":  # Never equip adapter and reverter for m0
            setattr(self, f"adapter_{modality_name}", Adapter(model_setting["adapter"]))
            setattr(self, f"reverter_{modality_name}", Reverter(model_setting["reverter"]))

    def build_moe(self, modality_name, model_setting):
        """
        Builds the adapter and reverter for a given modality.

        Parameters:
        - modality_name (str): The name of the modality (e.g., 'camera', 'lidar').
        - model_setting (dict): Configuration settings for the model.

        This function sets up the adapter and reverter modules for modalities other than 'm0'.
        """

        if modality_name == self.ego_modality_name:  # Never equip adapter and reverter for m0
            setattr(self, f"moe_{modality_name}", MoE(model_setting["moe"]))

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

    def forward_moe(self, modality_feature_dict, intrinsic_code_dict, pairwise_t_matrix, record_len, agent_modality_list):
        modality_name = self.ego_modality_name
        affine_matrix = normalize_pairwise_tfm(
            pairwise_t_matrix,
            eval(f"self.H_{modality_name}"),
            eval(f"self.W_{modality_name}"),
            self.fake_voxel_size,
        )

        counting_dict = {modality_name:0 for modality_name in self.modality_name_list}
        heter_feature_2d_list = []
        intrinsic_code_list = []
        for modality_name in agent_modality_list:
            feat_idx = counting_dict[modality_name]
            heter_feature_2d_list.append(modality_feature_dict[modality_name][feat_idx])
            intrinsic_code_list.append(intrinsic_code_dict[modality_name][feat_idx])
            counting_dict[modality_name] += 1

        heter_feature_2d_list = torch.stack(heter_feature_2d_list)
        intrinsic_code_list = torch.stack(intrinsic_code_list)

        ## minibatch split
        heter_feature_2d_mini_batch_list = self.regroup_any(heter_feature_2d_list, record_len)
        intrinsic_code_mini_batch_list = self.regroup_any(intrinsic_code_list, record_len)
        agent_modality_mini_batch_list = self.regroup_any(agent_modality_list, record_len)


        # 存储所有处理后的特征（按原始车辆顺序）
        all_processed_features = []
        all_alphas = []
        all_modality_pairs = []
        batch_ego_modality_list = []   ### 每个batch的0号agent的模态，gt模态


        for mini_batch_idx, mini_batch_size in enumerate(record_len):
            batch_feature = heter_feature_2d_mini_batch_list[mini_batch_idx]
            batch_code = intrinsic_code_mini_batch_list[mini_batch_idx]
            batch_agent_modality = agent_modality_mini_batch_list[mini_batch_idx]

            _, _, H, W = batch_feature.shape

            ego_feat = batch_feature[0:1]
            ego_code = batch_code[0:1]
            ego_modality = batch_agent_modality[0]
            batch_ego_modality_list.append(ego_modality)

            for mm in batch_agent_modality:
                all_modality_pairs.append((self.modality_to_id[ego_modality], self.modality_to_id[mm]))

            ## 将ego特征转换到neb坐标系下
            t_matrix = affine_matrix[mini_batch_idx][:mini_batch_size, :mini_batch_size, :, :]
            ego_repeat = ego_feat.repeat(batch_feature.shape[0], 1, 1, 1)
            ego_in_nebcoord = warp_affine_simple(ego_repeat, t_matrix[:, 0, :, :], (H, W), align_corners=True)

            # 通过moe转换neb特征到ego视角
            moe_output, alphas = self.converter(
                ego_in_nebcoord,
                batch_feature,
                ego_code.repeat(batch_code.shape[0], 1),
                batch_code,
                return_alpha=True
            )  # [b, C, H, W] - 转换后的neb特征
            # print(f"[{neb_modality=}] --> [{ego_modality=}]")


            if self.testing:
                moe_output[0:1] = ego_feat

            all_processed_features.append(moe_output)
            all_alphas.append(alphas)


        all_processed_features = torch.cat(all_processed_features, dim=0)


        if all_alphas:
            all_alphas = torch.cat(all_alphas, dim=1)               # n_layers, batch, d
            all_alphas = F.softmax(all_alphas, dim=-1)
            eg = all_alphas[:2, -3:].detach()
            order = eg.argsort(dim=-1, descending=True)  # 每行从大到小的索引
            ranks = order.argsort(dim=-1)  # 每个元素的名次(0=最大)
            print(ranks)
            print(eg)


        output_dict = {
            "FN2E": all_processed_features,
            # "all_alphas": all_alphas,
            "feat_vis": {
                "record_len": record_len,
                "raw": heter_feature_2d_list,
                "processed": all_processed_features,
            }
        }


        return all_processed_features, output_dict

    # def forward_moe(
    #         self, modality_count_dict, modality_feature_dict, ego_moe_features
    # ):
    #     """
    #     Forwards the features through the moe.
    #
    #     Parameters:
    #     - modality_count_dict (dict): Dictionary of modality counts.
    #     - modality_feature_dict (dict): Dictionary of modality features.
    #     - ego_moe_features (dict): Dictionary of moe features.
    #
    #     """
    #     #todo: 这里有问题，不应该区分模态，所有模态的数据都应该输入到moe中
    #     for modality_name in self.used_modality_name_list:
    #         if modality_name in modality_count_dict:
    #
    #             # if modality_name != self.ego_modality_name:
    #             #     ego_moe_features[f"moe_feature_{modality_name}"] = eval(f"self.moe_{self.ego_modality_name}")(modality_feature_dict[modality_name])
    #             # else:
    #             #     ego_moe_features[f"moe_feature_{modality_name}"] = modality_feature_dict[modality_name]
    #             # print(f"{modality_name} input into MoE")
    #             if self.train:
    #                 moe_feature = eval(f"self.moe_{self.ego_modality_name}")(modality_feature_dict[modality_name])
    #                 ego_moe_features[f"moe_feature_{modality_name}"] = moe_feature
    #             else:
    #                 if modality_name != self.ego_modality_name:
    #                     moe_feature = eval(f"self.moe_{self.ego_modality_name}")(
    #                         modality_feature_dict[modality_name])
    #                     ego_moe_features[f"moe_feature_{modality_name}"] = moe_feature
    #                 else:
    #                     print(f"ego:{modality_feature_dict[modality_name].shape}")
    #                     ego_moe_features[f"moe_feature_{modality_name}"] = modality_feature_dict[modality_name]
    #     return

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

    def postprocess_feature(self, FE, FN2E):
        """
        Post-processes the features for training the adapter.

        Parameters:
        - modality_feature_dict (dict): Dictionary of modality features.
        - protocol_features (dict): Dictionary of protocol features.
        - cur_feature_dict (dict): Dictionary of current features.
        - agent_modality_list (list): List of agent modalities.
        - output_dict (dict): Output data dictionary.

        This function handles the post-processing of features for training the adapter.
        """

        used_modality_list = self.used_modality_name_list.copy()
        ego_modality = self.ego_modality_name

        # todo:其实这里做的crop不是很重要，重要的是如果有camera模态，需要将feature统一到一个shape
        if self.crop_to_visible:
            # calculate the minimum cav_range to crop the feature
            min_cav_range = np.array([-np.inf, -np.inf, -np.inf, np.inf, np.inf, np.inf])
            for modality_name in used_modality_list:
                cav_range = eval(f"self.visible_range_{modality_name}")
                min_cav_range = np.concatenate(
                    [np.maximum(min_cav_range[:3], cav_range[:3]), np.minimum(min_cav_range[3:], cav_range[3:])]
                )

            # Crop the feature in ego domain
            B, C_FE, H_FE, W_FE = FE.shape
            # feature-lidar range ratio
            X = eval(f"self.cav_range_{ego_modality}")[3] - eval(f"self.cav_range_{ego_modality}")[0]
            Y = eval(f"self.cav_range_{ego_modality}")[4] - eval(f"self.cav_range_{ego_modality}")[1]
            fl_ratio = np.array([X / W_FM, Y / H_FM])

            left_diff = (eval(f"self.cav_range_{ego_modality}")[0] - min_cav_range[0]) / fl_ratio[0]
            right_diff = (min_cav_range[3] - eval(f"self.cav_range_{ego_modality}")[3]) / fl_ratio[0]
            top_diff = (eval(f"self.cav_range_{ego_modality}")[1] - min_cav_range[1]) / fl_ratio[1]
            bottom_diff = (min_cav_range[4] - eval(f"self.cav_range_{ego_modality}")[4]) / fl_ratio[1]

            pad_ego = nn.ZeroPad2d((round(left_diff), round(right_diff), round(top_diff), round(bottom_diff)))
            FE = pad_ego(FE)

            # Crop the feature in protocol domain
            protocol_modality = "m0"
            B, C_FP, H_FP, W_FP = FP.shape
            # feature-lidar range ratio
            X = eval(f"self.cav_range_{protocol_modality}")[3] - eval(f"self.cav_range_{protocol_modality}")[0]
            Y = eval(f"self.cav_range_{protocol_modality}")[4] - eval(f"self.cav_range_{protocol_modality}")[1]
            fl_ratio = np.array([X / W_FP, Y / H_FP])
            left_diff = (eval(f"self.cav_range_{protocol_modality}")[0] - min_cav_range[0]) / fl_ratio[0]
            right_diff = (min_cav_range[3] - eval(f"self.cav_range_{protocol_modality}")[3]) / fl_ratio[0]
            top_diff = (eval(f"self.cav_range_{protocol_modality}")[1] - min_cav_range[1]) / fl_ratio[1]
            bottom_diff = (min_cav_range[4] - eval(f"self.cav_range_{protocol_modality}")[4]) / fl_ratio[1]
            pad_protocol = nn.ZeroPad2d((round(left_diff), round(right_diff), round(top_diff), round(bottom_diff)))
            FP = pad_protocol(FP)
            FM2P = pad_protocol(FM2P)

        return FE, FN2E
