# -*- coding: utf-8 -*-
# Author: Xiangbo Gao <xiangbogaobarry@gmail.com>
# License: MIT License

import torch
import cv2
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import numpy as np
from icecream import ic
from collections import OrderedDict, Counter
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.feature_alignnet import AlignNet
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.translators import IntrinsicModalEncoder, HeteroFeatureConverter
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.pyramid_fuse import PyramidFusion
from opencood.models.fuse_modules.adapter import Adapter, Reverter
from opencood.utils.transformation_utils import normalize_pairwise_tfm
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



class CollabModalityIntrinsic(nn.Module):

    def __init__(self, args):
        super(CollabModalityIntrinsic, self).__init__()
        self.args = args
        self.stage = args["stage"]
        self.crop_to_visible = args.get("crop_to_visible", False)
        ignored_modality = set(args.get("ignored_modality", []))
        mods = [
            k for k in args.keys()
            if k not in ignored_modality and k.startswith("m") and k[1:].isdigit()
        ]
        # 稳定排序：m0, m1, m2, ...
        mods = sorted(mods, key=lambda s: int(s[1:]))
        self.modality_name_list = mods
        print(f"modality_num: {len(self.modality_name_list)}")


        all_mods = [
            k for k in args.keys() if k.startswith("m") and k[1:].isdigit()
        ]
        # stable sort (for ddp)：m0, m1, m2, ...
        all_mods = sorted(all_mods, key=lambda s: int(s[1:]))
        self.all_modality_name_list = all_mods

        self.ego_modality_name = args["ego_modality"]
        self.ego_modality_setting= args[self.ego_modality_name]
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
            # # if self.stage == "train_moe" and modality_name == self.ego_modality_name:
            #     # self.build_moe(modality_name, model_setting)
            #     self.build_fusion(modality_name, model_setting)
            #     self.build_shrink_header(modality_name, model_setting)
            #     self.build_head(modality_name, model_setting)

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







        self.intrinsic_modal_encoder = IntrinsicModalEncoder(args["intrinsic"])
        # self.converter = HeteroFeatureConverter(args["converter"])


        self.num_modalities = len(self.modality_name_list)
        self.embed_dim = args["intrinsic"]["embed_dim"]
        self.center_momentum = args["intrinsic"]["center_momentum"]

        self.modality_to_id = {m: i for i, m in enumerate(self.modality_name_list)}
        # 注册为 buffer（随模型保存/加载、但不参与梯度）
        self.register_buffer(
            "modality_centers",
            F.normalize(torch.randn(self.num_modalities, self.embed_dim), p=2, dim=1)
        )
        self.register_buffer(
            "center_inited",
            torch.zeros(self.num_modalities, dtype=torch.bool)
        )



        self.modality_predictor = nn.Sequential(
            nn.Linear(self.embed_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.num_modalities),
        )

        # self.modality_predictor = nn.Linear(self.embed_dim,  self.num_modalities)


        self.model_train_init()
        # check again which module is not fixed.
        # todo:check_trainable_module有问题，“encoder_m10” 的前缀是 “encoder_m1”，会跳过m10的判断
        check_trainable_module(self)



        self.model_dir = args.get("model_dir", None)
        self.update_ct = 0
        self.update_start = 30




    def load_pretrained(self, model_dir=None):
        self.load_pretrained_modality_centers(model_dir)

    def load_pretrained_modality_centers(self, model_dir=None):
        """
        Load centers from a checkpoint in model_dir.
        - Match files whose name contains both 'modality' and 'centers' (case-insensitive).
        - If multiple matches, pick the one with the lexicographically largest filename.
        - Expect:
            centers = ckpt["modality_centers"]            # Tensor [N, D]
            names   = ckpt["meta"]["modality_names"]      # List[str], e.g., "m38_lift_splat_shoot"
        - Reorder rows by the numeric id of "mXX" (ascending), then copy into self.modality_centers.
        - Return True on success
        """
        from pathlib import Path

        if model_dir is None:
            model_dir = self.model_dir

        model_dir = Path(model_dir)

        # find candidates (recursive), pick lexicographically largest filename
        cands = []
        for p in model_dir.rglob("*"):
            if p.is_file():
                name = p.name.lower()
                if ("modality" in name or "modalities" in name) and ("centers" in name):
                    cands.append(p)

        cands.sort(key=lambda x: x.name, reverse=True)
        ckpt_path = cands[0]

        # load
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        centers = ckpt["modality_centers"]  # [N, D]
        names = ckpt["meta"]["modality_names"]  # list[str]


        N, D = centers.shape
        core = []
        for s in names:
            t = s.split("_", 1)[0]  # "m38_lss" -> "m38"
            core.append(t)


        order = sorted(range(N), key=lambda i: int(core[i][1:]))  # by numeric id
        centers_sorted = centers[order].contiguous()


        # normalize & load to buffer device/dtype
        # centers_sorted = torch.nan_to_num(centers_sorted, nan=0.0, posinf=0.0, neginf=0.0)
        centers_sorted = F.normalize(centers_sorted, p=2, dim=1, eps=1e-6)
        centers_sorted = centers_sorted.to(device=self.modality_centers.device, dtype=self.modality_centers.dtype)
        self.modality_centers.copy_(centers_sorted)

        # mark inited if buffer exists and size matches
        if hasattr(self, "center_inited") and self.center_inited.numel() == N:
            self.center_inited.fill_(True)

        print("=="*50)
        print("self.modality_centers")
        print(self.modality_centers)
        print("==" * 50)

        return True



    def model_train_init(self):
        # if train adapter, then all modules are fixed except adapter and reverter

        for p in self.parameters():
            p.requires_grad_(False)
        self.apply(fix_bn)

        # self.intrinsic_modal_encoder.train()
        for p in self.intrinsic_modal_encoder.parameters():
            p.requires_grad_(True)
        self.intrinsic_modal_encoder.apply(unfix_bn)

        # self.modality_predictor.train()
        for p in self.modality_predictor.parameters():
            p.requires_grad_(True)
        self.modality_predictor.apply(unfix_bn)



    def forward(self, data_dict, show_bev=False):
        agent_modality_list = data_dict["agent_modality_list"]
        # print(f'agent_modality_list:{agent_modality_list}')
        # print(f'self.modality_name_list:{self.modality_name_list}')
        record_len = data_dict["record_len"]

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

        # output_dict = {self.ego_modality_name : {"pyramid": "collab_moe_intrinsic"} }
        output_dict = {}
        modality_count_dict = Counter(agent_modality_list)
        modality_feature_dict = {}

        with torch.no_grad():  # 禁用梯度计算
            # setup each modality model
            for modality_name in self.used_modality_name_list:
                # print(f'modality_name{modality_name}')
                if self.stage not in ["train_moe", "train_intrinsic"] and modality_name not in modality_count_dict:
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
        # -------- intrinsic encoder (trainable) & build labels --------
        emb_list = []
        m_pred_list = []
        lab_list = []

        for m in self.used_modality_name_list:
            if m not in modality_feature_dict:
                continue
            feat = modality_feature_dict[m]  # [Ni, C, H, W]
            code = self.intrinsic_modal_encoder(feat)  # [Ni, D],
            m_pred = self.modality_predictor(code)

            emb_list.append(code)
            m_pred_list.append(m_pred)

            global_id = self.modality_to_id[m]  # stable global id
            lab_list.append(torch.full((code.size(0),), global_id, dtype=torch.long, device=code.device))

        if len(emb_list) == 0:
            output_dict["intrinsic_loss"] = torch.tensor(0.0, device=record_len.device, requires_grad=True)
            return output_dict

        emb_all = torch.cat(emb_list, dim=0)  # [N, D]
        m_pred_all = torch.cat(m_pred_list, dim=0)
        lab_all = torch.cat(lab_list, dim=0)  # [N]

        # print(lab_all)
        print(len(emb_all))
        # print(emb_all[:5])


        output_dict.update({
            "emb_all": emb_all,
            "m_pred_all": m_pred_all,
            "lab_all": lab_all,
            "modality_centers": self.modality_centers.detach(),
        })


        # -------- EMA update centers (no grad) --------
        if self.training:
            if self.update_ct > self.update_start:
                self.update_modality_centers(emb_all.detach(), lab_all.detach())
            else:
                self.update_ct += 1


        return output_dict



    @torch.no_grad()
    def update_modality_centers(self, emb: torch.Tensor, labels: torch.Tensor):
        """
        Rank-0 update + broadcast (DDP-safe, minimal change).
        - Each rank computes local class sums & counts.
        - Reduce to rank-0 (SUM), rank-0 does EMA update.
        - Broadcast updated centers & flags to all ranks.
        Works for single-GPU (dist not initialized) as well.
        """
        if emb.numel() == 0:
            return

        emb = F.normalize(emb, p=2, dim=1, eps=1e-6)  # unit-norm for cosine similarity

        M = self.num_modalities
        D = emb.size(1)
        device = emb.device
        mom = float(self.center_momentum)

        # ensure buffers on same device
        if self.modality_centers.device != device:
            self.modality_centers = self.modality_centers.to(device)
        if self.center_inited.device != device:
            self.center_inited = self.center_inited.to(device)

        # --- per-rank local stats (float32 + contiguous) ---
        one_hot = F.one_hot(labels.to(torch.long), num_classes=M).to(device=device, dtype=torch.float32).contiguous()  # [N,M]
        emb32 = emb.to(torch.float32).contiguous()  # [N,D]
        class_sums = (emb32.t().contiguous() @ one_hot).t().contiguous()  # [M,D]
        class_counts = one_hot.sum(dim=0).contiguous()  # [M]

        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()

            # reduce to rank-0
            dist.reduce(class_sums, dst=0, op=dist.ReduceOp.SUM)
            dist.reduce(class_counts, dst=0, op=dist.ReduceOp.SUM)

            if rank == 0:
                # global means (only for appeared classes)
                counts = class_counts.clamp_min(1.0)  # [M]
                means = class_sums / counts.unsqueeze(1)  # [M,D]
                # means = torch.nan_to_num(means, nan=0.0, posinf=0.0, neginf=0.0)
                means = F.normalize(means, p=2, dim=1, eps=1e-6).to(self.modality_centers.dtype)

                appear = (class_counts > 0)
                idxs = torch.nonzero(appear, as_tuple=False).flatten().tolist()
                for idx in idxs:
                    if not bool(self.center_inited[idx]):
                        self.modality_centers[idx] = means[idx]
                        self.center_inited[idx] = True
                    else:
                        c_old = self.modality_centers[idx]
                        c_new = F.normalize(c_old * mom + means[idx] * (1.0 - mom), p=2, dim=0, eps=1e-6)
                        c_new = torch.nan_to_num(c_new, nan=0.0, posinf=0.0, neginf=0.0)
                        self.modality_centers[idx] = c_new

            # broadcast updated buffers to all ranks
            dist.broadcast(self.modality_centers, src=0)
            dist.broadcast(self.center_inited, src=0)

        else:
            # --- single process / single GPU fallback ---
            counts = class_counts.clamp_min(1.0)
            means = class_sums / counts.unsqueeze(1)
            # means = torch.nan_to_num(means, nan=0.0, posinf=0.0, neginf=0.0)
            means = F.normalize(means, p=2, dim=1, eps=1e-6).to(self.modality_centers.dtype)

            appear = (class_counts > 0)
            idxs = torch.nonzero(appear, as_tuple=False).flatten().tolist()
            for idx in idxs:
                if not bool(self.center_inited[idx]):
                    self.modality_centers[idx] = means[idx]
                    self.center_inited[idx] = True
                else:
                    c_old = self.modality_centers[idx]
                    c_new = F.normalize(c_old * mom + means[idx] * (1.0 - mom), p=2, dim=0, eps=1e-6)
                    # c_new = torch.nan_to_num(c_new, nan=0.0, posinf=0.0, neginf=0.0)
                    self.modality_centers[idx] = c_new




    def set_collab_train(self, is_train):
        """在collab_moe中的train属性"""
        self.train = is_train
        print(f"已将collab_moe的train设置为: {is_train}")

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

            # print(f"cls_preds梯度状态: {cls_preds.requires_grad}")
            # print(f"reg_preds梯度状态: {reg_preds.requires_grad}")
            # print(f"dir_preds梯度状态: {dir_preds.requires_grad}")
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
