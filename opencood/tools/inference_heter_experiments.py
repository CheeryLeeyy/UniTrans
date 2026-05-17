# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>, Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>,
# License: TDG-Attribution-NonCommercial-NoDistrib

# Modifications by Xiangbo Gao <xiangbogaobarry@gmail.com>
# New License for modifications: MIT License

import argparse
import os
import json
import statistics
import time
from typing import OrderedDict
import importlib
import torch
import torchvision
import open3d as o3d
from torch.utils.data import DataLoader, Subset
import numpy as np
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.visualization import vis_utils, my_vis, simple_vis
from opencood.utils.common_utils import update_dict
from opencood.utils.seg_iou import mean_IU
from matplotlib import pyplot as plt
from tqdm import tqdm
import cv2
import numpy as np
import ast

torch.multiprocessing.set_sharing_strategy("file_system")


def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--model_dir", type=str,
                        default="",
                        help="Continued training path")
    parser.add_argument("--fusion_method", type=str, default="intermediate", help="no, no_w_uncertainty, late, early or intermediate")
    parser.add_argument("--save_vis_interval", type=int, default=400, help="interval of saving visualization")
    parser.add_argument("--save_npy", action="store_true", help="whether to save prediction and gt result" "in npy file")
    parser.add_argument(
        "--range", type=str,
        default="102.4,102.4",
        # default="51.2,51.2",
        # default="102.4,51.2",
        help="detection range is [-102.4, +102.4, -51.2, +51.2]"
    )
    parser.add_argument("--no_score", action="store_true", help="whether print the score of prediction")
    parser.add_argument("--note", default="", type=str, help="any other thing?")
    parser.add_argument("--noise", type=float, default=0.0, help="add noise to pose")
    parser.add_argument("--all", action="store_true", help="evaluate all the agents instead of the first one.")
    parser.add_argument("--show_bev", action="store_true", help="Visualize the BEV feature")
    parser.add_argument("--protocol_result", action="store_true", help="plot the protocol result instead of the ego result.")
    parser.add_argument("--data_only", action="store_true", help="Only visualize the data")
    parser.add_argument("--score_threshold", type=float, default=0.2, help="score threshold for visualization")
    parser.add_argument("--aggregation", default="", choices=["", "nms", "psa"], help="post process method")
    parser.add_argument("--task", default="detection", choices=["detection", "segmentation"], help="task type")
    parser.add_argument("--other_model_dirs", type=str, help="Other models to load eg.[xxx/xxx.pth,xxx/xxx.pth]")
    parser.add_argument('--use_cav', type=str, default="[5]", help="evaluate with real collaborator number")
    parser.add_argument('--ego_modality', type=str, default="", help="ego modality name")
    parser.add_argument("--assignment_path", type=str,
                        default="opencood/logs/heter_modality_assign/opv2v_4modality_in_order_new.json",
                        help="Assignment path")
    parser.add_argument('--assignment_mode', type=str, default='random', choices=['specified', 'random'], help='Assignment mode: specified or random')
    parser.add_argument('--assignment', type=str, help="Assignment list for specified mode, eg.'m7,m7,m7'")
    parser.add_argument('--save_features', action='store_true', help="Whether to save intermediate feature")
    parser.add_argument("--save_feat_interval", type=int, default=40, help="interval of saving visualization")


    opt = parser.parse_args()

    # if opt.protocol_result:
    #     # No need to plot BEV feature when plotting protocol result, the BEV feature is plotted in the ego mode.
    #     opt.show_bev = False
    #     return opt

    return opt




import os
from PIL import Image


def _to_uint8_img(x_2d, cmap="viridis") -> Image.Image:
    x = x_2d.detach().float().cpu().numpy()

    x = x - x.min()
    x = x / (x.max() + 1e-12)

    if cmap is None:
        x_u8 = (x * 255.0).clip(0, 255).astype(np.uint8)
        return Image.fromarray(x_u8, mode="L")

    import matplotlib
    cm = matplotlib.colormaps.get_cmap(cmap)
    rgba = cm(x)  # numpy [H,W,4]
    rgb_u8 = (rgba[..., :3] * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(rgb_u8, mode="RGB")


def _hconcat(img_left: Image.Image, img_right: Image.Image) -> Image.Image:
    """Horizontally concatenate two PIL images with the same height."""
    if img_left.size[1] != img_right.size[1]:
        img_right = img_right.resize((img_right.size[0], img_left.size[1]))

    # Ensure both are RGB so colors are preserved
    img_left = img_left.convert("RGB")
    img_right = img_right.convert("RGB")

    w1, h1 = img_left.size
    w2, _ = img_right.size

    out = Image.new("RGB", (w1 + w2, h1))
    out.paste(img_left, (0, 0))
    out.paste(img_right, (w1, 0))
    return out



def save_feature_images(feat_in, feat_out, vis_id, vis_save_dir, prefix=""):
    """
    Save per-agent feature visualization for one mini-batch.

    Inputs:
      feat_in:  [M, C, H, W] features before conversion
      feat_out: [M, C, H, W] features after conversion
      vis_id:   global counter id (self.vis_ct)
      prefix:   optional string to separate different modules/experiments

    Naming:
      {prefix}{vis_id}_{i}_mean.png   (left=before_mean, right=after_mean)
      {prefix}{vis_id}_{i}_max.png    (left=before_max,  right=after_max)
    """

    assert feat_in.dim() == 4 and feat_out.dim() == 4
    assert feat_in.shape == feat_out.shape
    M, C, H, W = feat_in.shape

    if M <= 1:
        print("minibatch=1")
        return

    # Ensure on CPU for saving
    fin = feat_in.detach()
    fout = feat_out.detach()

    for i in range(M):
        # Channel-wise reductions -> [H, W]
        in_mean = fin[i].mean(dim=0)
        out_mean = fout[i].mean(dim=0)

        in_max = fin[i].amax(dim=0)
        out_max = fout[i].amax(dim=0)

        # Build side-by-side images for easier comparison
        img_mean = _hconcat(_to_uint8_img(in_mean), _to_uint8_img(out_mean))
        img_max = _hconcat(_to_uint8_img(in_max), _to_uint8_img(out_max))

        base = f"{prefix}{vis_id}_{i}"
        mean_path = os.path.join(vis_save_dir, f"{base}_mean.png")
        max_path = os.path.join(vis_save_dir, f"{base}_max.png")

        img_mean.save(mean_path)
        img_max.save(max_path)
        print(mean_path)



def _hconcat_many(imgs, *, align_height_to="first", gap=0) -> Image.Image:
    """
    Horizontally concatenate a list of PIL images.
    align_height_to: "first" | "max"
    gap: pixels between images
    """
    assert len(imgs) > 0

    if align_height_to == "max":
        target_h = max(im.size[1] for im in imgs)
    else:
        target_h = imgs[0].size[1]

    imgs_rgb = []
    for im in imgs:
        if im.size[1] != target_h:
            # keep aspect ratio when resizing by height
            new_w = max(1, int(round(im.size[0] * (target_h / float(im.size[1])))))
            im = im.resize((new_w, target_h))
        imgs_rgb.append(im.convert("RGB"))

    total_w = sum(im.size[0] for im in imgs_rgb) + gap * (len(imgs_rgb) - 1)
    out = Image.new("RGB", (total_w, target_h))

    x = 0
    for idx, im in enumerate(imgs_rgb):
        out.paste(im, (x, 0))
        x += im.size[0] + (gap if idx < len(imgs_rgb) - 1 else 0)
    return out


def save_feature_images(feat_list, vis_id, vis_save_dir, prefix="", cmap="viridis", gap=0):
    """
    Save per-agent feature visualization for one mini-batch.

    Inputs:
      feat_list: list of features. Each element can be:
                - Tensor [M, C, H, W]
                - (name: str, Tensor [M, C, H, W])  for optional labeling/debug
      vis_id: global counter id
      prefix: optional string to separate different modules/experiments

    Output naming:
      {prefix}{vis_id}_{i}_mean.png   (concat of mean maps in feat_list order)
      {prefix}{vis_id}_{i}_max.png    (concat of max  maps in feat_list order)
    """
    if not isinstance(feat_list, (list, tuple)) or len(feat_list) == 0:
        print("feat_list is empty")
        return

    parsed = []
    for k, item in enumerate(feat_list):
        if (
                isinstance(item, (list, tuple))
                and len(item) == 2
                and isinstance(item[0], str)
                and torch.is_tensor(item[1])
        ):
            name, feat = item[0], item[1]
        else:
            name, feat = f"f{k}", item
        assert torch.is_tensor(feat), f"feat_list[{k}] is not a Tensor"
        assert feat.dim() == 4, f"feat_list[{k}] must be [M,C,H,W], got {tuple(feat.shape)}"
        parsed.append((name, feat.detach()))

    # Validate M,H,W are consistent (C can differ)
    M, _, H, W = parsed[0][1].shape
    for k, (name, feat) in enumerate(parsed[1:], start=1):
        assert feat.shape[0] == M and feat.shape[2] == H and feat.shape[3] == W, \
            f"feat_list[{k}] ({name}) shape mismatch: expect [M,*,H,W]=[{M},*,{H},{W}], got {tuple(feat.shape)}"

    if M <= 1:
        print("minibatch=1")
        return

    for i in range(M):
        mean_imgs = []
        max_imgs = []
        for name, feat in parsed:
            # Channel-wise reductions -> [H, W]
            fmap_mean = feat[i].mean(dim=0)
            fmap_max = feat[i].amax(dim=0)

            mean_imgs.append(_to_uint8_img(fmap_mean, cmap=cmap))
            max_imgs.append(_to_uint8_img(fmap_max, cmap=cmap))

        img_mean = _hconcat_many(mean_imgs, gap=gap)
        img_max = _hconcat_many(max_imgs, gap=gap)

        base = f"{prefix}{vis_id}_{i}"
        mean_path = os.path.join(vis_save_dir, f"{base}_mean.png")
        max_path = os.path.join(vis_save_dir, f"{base}_max.png")

        img_mean.save(mean_path)
        img_max.save(max_path)
        print(mean_path)
        print(max_path)






def main():
    opt = test_parser()

    assert opt.fusion_method in ["late", "late_heter", "early", "intermediate", "no", "no_w_uncertainty", "single"]
    # if opt.all:
    #     assert not opt.show_bev

    hypes = yaml_utils.load_yaml(None, opt)



    hypes = update_dict(
        hypes,
        {
            "score_threshold": opt.score_threshold,
        },
    )
    # print(f"opt.all:{opt.all}") # default False
    if "heter" in hypes:
        # hypes['heter']['lidar_channels'] = 16
        # opt.note += "_16ch"

        x_min, x_max = -eval(opt.range.split(",")[0]), eval(opt.range.split(",")[0])
        y_min, y_max = -eval(opt.range.split(",")[1]), eval(opt.range.split(",")[1])
        opt.note += f"_{x_max}_{y_max}"

        new_cav_range = [x_min, y_min, hypes["cav_lidar_range"][2], x_max, y_max, hypes["cav_lidar_range"][5]]
        # replace all appearance
        hypes = update_dict(
            hypes, {"cav_lidar_range": new_cav_range, "lidar_range": new_cav_range, "gt_range": new_cav_range}
        )

        # reload anchor
        hypes = yaml_utils.update_yaml(hypes, opt)

    if opt.aggregation:
        hypes = update_dict(hypes, {"aggretation": opt.aggregation})

    hypes["validate_dir"] = hypes["test_dir"]
    if "OPV2V" in hypes["test_dir"] or "v2xsim" in hypes["test_dir"]:
        assert "test" in hypes["validate_dir"]

    # This is used in visualization
    # left hand: OPV2V, V2XSet
    # right hand: V2X-Sim 2.0 and DAIR-V2X
    opt.left_hand = True if ("OPV2V" in hypes["test_dir"] or "V2XSET" in hypes["test_dir"]) else False

    print(f"Left hand visualizing: {opt.left_hand}")

    if "box_align" in hypes.keys():
        hypes["box_align"]["val_result"] = hypes["box_align"]["test_result"]


# =============================================================================================================
# =============================================================================================================
# =============================================================================================================
    hypes["fusion"]["args"]["mode"] = "train_moe"
    hypes["model"]["args"]["stage"] = "train_moe"

    hypes['model']['args']['testing'] = True

    # if "intrinsic" in hypes["model"]["core_method"] or "converter" in hypes["model"]["core_method"]:
    #     hypes["model"]["core_method"] = "collab_moe"

    if "moe_converter" in hypes["model"]["core_method"] or ("moe" in hypes["model"]["core_method"] and "finetune" in hypes["model"]["core_method"]):
        hypes["model"]["core_method"] = "collab_moe"

    hypes["model"]["args"]["default_modality"] = "m4"

    if opt.ego_modality:
        hypes["heter"]["ego_modality"] = opt.ego_modality
        hypes["model"]["args"]["ego_modality"] = opt.ego_modality
        print(opt.ego_modality)

    ego_modality = hypes["model"]["args"]["ego_modality"]

    # if opt.save_features:
    #     hypes["model"]["core_method"] = "collab_moe_vis"
    #     hypes['model']['args']['save_features'] = True
    #
    #     base = opt.fusion_method + opt.note + ("_all" if opt.all else "") + "_noise" + str(opt.noise)
    #
    #     vis_feat_dir = os.path.join(opt.model_dir, "vis_feat_"+base)
    #     if not os.path.exists(vis_feat_dir):
    #         os.makedirs(vis_feat_dir)
    #     hypes['model']['args']["vis_save_dir"] = vis_feat_dir
# =============================================================================================================
# =============================================================================================================
# =============================================================================================================


    print("Creating Model")
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device:{device}")

    print("Loading Model from checkpoint")
    saved_path = opt.model_dir
    resume_epoch, model = train_utils.load_saved_model(saved_path, model)
    print(f"resume from {resume_epoch} epoch.")
    opt.note += f"_epoch{resume_epoch}"
    """
    other_model_dirs
    """
    if opt.other_model_dirs:
        # 安全解析模型路径列表
        try:
            # 方法1: 使用ast.literal_eval (最安全)
            model_paths = ast.literal_eval(opt.other_model_dirs)
        except (SyntaxError, ValueError):
            try:
                # 方法2: 使用正则表达式解析
                model_paths = re.findall(r'[^,\[\]]+\.pth', opt.other_model_dirs)
            except:
                # 方法3: 简单分割作为备选
                model_paths = [path.strip(" []'\"") for path in opt.other_model_dirs.split(',')]

        # 验证路径列表
        if not isinstance(model_paths, list) or not all(isinstance(p, str) for p in model_paths):
            raise ValueError(f"Invalid model paths format: {opt.other_model_dirs}")

        # 加载每个模型
        for model_path in model_paths:
            # 清理路径中的引号和空格
            cleaned_path = model_path.strip(" '\"")

            print(f"Loading Model from {cleaned_path}")
            try:
                # 加载状态字典
                loaded_state_dict = torch.load(cleaned_path, map_location='cpu')

                # 处理不同格式的模型文件
                if 'state_dict' in loaded_state_dict:  # 完整检查点格式
                    state_dict = loaded_state_dict['state_dict']
                elif 'model' in loaded_state_dict:  # 另一种常见格式
                    state_dict = loaded_state_dict['model']
                else:  # 纯状态字典
                    state_dict = loaded_state_dict

                # 加载到当前模型
                model.load_state_dict(state_dict, strict=False)

                # 打印加载结果
                print(f"✓ Successfully loaded model from {cleaned_path}")
                print(f"  - Loaded parameters: {len(state_dict)} layers")

            except Exception as e:
                print(f"× Failed to load model from {cleaned_path}: {str(e)}")
                print("Skipping this model and continuing...")
    # print(model)


    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    # # # 推理模式下设置gate.train=False
    # if hypes['model']['core_method'] == 'collab_moe':
    #     # model.set_collab_train(False)
    model.testing = True


    if opt.noise:
        # add noise to pose.
        pos_std = opt.noise
        rot_std = opt.noise
        pos_mean = 0
        rot_mean = 0

        # setting noise
        np.random.seed(303)
        noise_setting = OrderedDict()
        noise_args = {"pos_std": pos_std, "rot_std": rot_std, "pos_mean": pos_mean, "rot_mean": rot_mean}

        noise_setting["add_noise"] = True
        noise_setting["args"] = noise_args

        # build dataset for each noise setting
        print("Dataset Building")
        print(f"Noise Added: {pos_std}/{rot_std}/{pos_mean}/{rot_mean}.")
        hypes.update({"noise_setting": noise_setting})

    if opt.fusion_method == 'intermediate':
        hypes['fusion']['core_method'] += 'infer'

    ####  for stamp
    if opt.fusion_method == 'intermediate' and 'moeadapter' in hypes['fusion']['core_method']:
        hypes['fusion']['core_method'] = "intermediatehetermoeinfer"


    if opt.assignment_path and os.path.exists(opt.assignment_path):
        hypes['heter']['assignment_path'] = opt.assignment_path
        print(f"Assignment path change to {hypes['heter']['assignment_path']}")

    if opt.assignment_mode == 'specified':
        hypes['fusion']['args']['assignment_mode'] = opt.assignment_mode
        assignment_list = [modality.strip() for modality in opt.assignment.split(',')]
        # import ast
        # # 使用 ast.literal_eval 替代 eval
        # try:
        #     assignment_list = ast.literal_eval(opt.assignment)
        # except (ValueError, SyntaxError) as e:
        #     raise ValueError(f"Invalid assignment format: {opt.assignment}. Error`: {e}")
        if not isinstance(assignment_list, list) or len(assignment_list) != 3:
            raise ValueError(f"Assignment must be a list of 3 modalities, got: {assignment_list}")
        hypes['heter']['mapping_dict']['m101'] = hypes['heter']['ego_modality']
        hypes['heter']['mapping_dict']['m102'] = assignment_list[0]
        hypes['heter']['mapping_dict']['m103'] = assignment_list[1]
        hypes['heter']['mapping_dict']['m104'] = assignment_list[2]
    elif opt.assignment_mode == 'random':
        hypes['fusion']['args']['assignment_mode'] = opt.assignment_mode
        hypes['heter']['mapping_dict']['m101'] = 'm101'
        hypes['heter']['mapping_dict']['m102'] = 'm102'
        hypes['heter']['mapping_dict']['m103'] = 'm103'
        hypes['heter']['mapping_dict']['m104'] = 'm104'
    else:
        raise ValueError(f"Invalid assignment mode: {opt.assignment_mode}")


    use_cav_list = eval(opt.use_cav)

    for use_cav in use_cav_list:
        hypes['use_cav'] = use_cav
        # build dataset for each noise setting
        print("Dataset Building")
        opencood_dataset = build_dataset(hypes, visualize=True, train=False)
        # opencood_dataset_subset = Subset(opencood_dataset, range(640,2100))
        # data_loader = DataLoader(opencood_dataset_subset,
        data_loader = DataLoader(
            opencood_dataset,
            batch_size=1,
            num_workers=8,
            collate_fn=opencood_dataset.collate_batch_test,
            shuffle=False,
            pin_memory=False,
            drop_last=False,
        )

        modality_list = opencood_dataset.modality_name_list
        # Create the dictionary for evaluation
        if opt.all:
            result_stat = dict()  # for detection
            ave_ious = dict()  # for segmentation

            for modality_name in modality_list:
                assert 'task' in hypes['heter']['modality_setting'][modality_name]
                if hypes['heter']['modality_setting'][modality_name]['task'] == 'detection':
                    result_stat[modality_name] = {
                        0.3: {"tp": [], "fp": [], "gt": 0, "score": []},
                        0.5: {"tp": [], "fp": [], "gt": 0, "score": []},
                        0.7: {"tp": [], "fp": [], "gt": 0, "score": []},
                    }
                elif hypes['heter']['modality_setting'][modality_name]['task'] == 'segmentation':
                    ave_ious[modality_name] = {
                        'static_ave_iou': [],
                        'dynamic_ave_iou': [],
                        'lane_ave_iou': []
                    }
                else:
                    raise NotImplementedError("Only detection and segmentation task is supported.")
        else:
            if opt.task == "detection":
                result_stat = {
                    0.3: {"tp": [], "fp": [], "gt": 0, "score": []},
                    0.5: {"tp": [], "fp": [], "gt": 0, "score": []},
                    0.7: {"tp": [], "fp": [], "gt": 0, "score": []},
                }
            elif opt.task == "segmentation":
                ave_ious = {
                    'static_ave_iou': [],
                    'dynamic_ave_iou': [],
                    'lane_ave_iou': []
                }
            else:
                raise NotImplementedError("Only detection and segmentation task is supported.")

        opt.infer_info = opt.fusion_method + opt.note + ("_all" if opt.all else "") + "_noise" + str(opt.noise) + f"_use_cav{use_cav}"

        opt.infer_info += f"_ego{ego_modality}"

        infer_len = len(data_loader)
        pbar = tqdm(enumerate(data_loader))
        for i, batch_data in pbar:
            pbar.set_description(f"{opt.infer_info}_{i}/{infer_len}")
            if batch_data is None:
                continue

            if opt.data_only:
                os.makedirs(os.path.join(opt.model_dir, "data"), exist_ok=True)
                simple_vis.visualize(
                    None,
                    batch_data["ego"]["origin_lidar"][0],
                    new_cav_range,
                    os.path.join(opt.model_dir, "data", f"lidar_{i}.png"),
                    method="bev",
                    left_hand=opt.left_hand,
                )
                continue

            with torch.no_grad():
                batch_data = train_utils.to_device(batch_data, device)

                if opt.fusion_method == "late":
                    infer_result = inference_utils.inference_late_fusion(batch_data, model, opencood_dataset)
                elif opt.fusion_method == "early":
                    infer_result = inference_utils.inference_early_fusion(batch_data, model, opencood_dataset)
                elif opt.fusion_method == "intermediate":
                    infer_result = inference_utils.inference_intermediate_fusion(
                        batch_data,
                        model,
                        opencood_dataset,
                        infer_all=opt.all,
                        show_bev=opt.show_bev,
                        protocol_result=opt.protocol_result,
                    )
                elif opt.fusion_method == "no":
                    infer_result = inference_utils.inference_no_fusion(batch_data, model, opencood_dataset)
                elif opt.fusion_method == "no_w_uncertainty":
                    infer_result = inference_utils.inference_no_fusion_w_uncertainty(batch_data, model, opencood_dataset)
                elif opt.fusion_method == "single":
                    infer_result = inference_utils.inference_no_fusion(batch_data, model, opencood_dataset, single_gt=True)
                elif opt.fusion_method == "late_heter":
                    infer_result = inference_utils.inference_heter_late(
                        batch_data,
                        model,
                        opencood_dataset,
                        show_bev=opt.show_bev,
                        infer_all=opt.all,
                    )
                else:
                    raise NotImplementedError(
                        "Only single, no, no_w_uncertainty, early, late and intermediate" "fusion is supported."
                    )

                agent_modality_list = batch_data["ego"]["agent_modality_list"]
                if not opt.all:
                    infer_result = [infer_result]

                for idx, infer_result_single in enumerate(infer_result):
                    if opt.all:
                        work_dir = os.path.join(opt.model_dir, f"modality_{agent_modality_list[idx]}")
                        os.makedirs(work_dir, exist_ok=True)
                        if hypes['heter']['modality_setting'][agent_modality_list[idx]]['task'] == 'detection':
                            eval_detection_result(
                                opt,
                                agent_modality_list,
                                opencood_dataset,
                                infer_result_single,
                                result_stat,
                                batch_data,
                                idx,
                                work_dir,
                                hypes,
                                i,
                            )
                        elif hypes['heter']['modality_setting'][agent_modality_list[idx]]['task'] == "segmentation":
                            iou_static, iou_dynamic = eval_segmentation_result(opt, infer_result_single, idx, work_dir, i)
                            if iou_static is not None:
                                ave_ious[agent_modality_list[idx]]["static_ave_iou"].append(iou_static[1])
                                ave_ious[agent_modality_list[idx]]["lane_ave_iou"].append(iou_static[2])
                            if iou_dynamic is not None:
                                ave_ious[agent_modality_list[idx]]["dynamic_ave_iou"].append(iou_dynamic[1])

                        else:
                            raise NotImplementedError("Only detection and segmentation task is supported.")
                    else:
                        work_dir = opt.model_dir
                        if opt.task == 'detection':
                            eval_detection_result(
                                opt,
                                agent_modality_list,
                                opencood_dataset,
                                infer_result_single,
                                result_stat,
                                batch_data,
                                idx,
                                work_dir,
                                hypes,
                                i,
                            )
                        elif opt.task == "segmentation":
                            iou_static, iou_dynamic = eval_segmentation_result(opt, infer_result_single, idx, work_dir, i)
                            if iou_static is not None:
                                ave_ious["static_ave_iou"].append(iou_static[1])
                                ave_ious["lane_ave_iou"].append(iou_static[2])
                            if iou_dynamic is not None:
                                ave_ious["dynamic_ave_iou"].append(iou_dynamic[1])

                        else:
                            raise NotImplementedError("Only detection and segmentation task is supported.")

            torch.cuda.empty_cache()
        if opt.all:
            # detection
            result_stat_all = {
                0.3: {"tp": [], "fp": [], "gt": 0, "score": []},
                0.5: {"tp": [], "fp": [], "gt": 0, "score": []},
                0.7: {"tp": [], "fp": [], "gt": 0, "score": []},
            }
            for modality_name in result_stat:
                for iou in [0.3, 0.5, 0.7]:
                    result_stat_all[iou]["tp"] += result_stat[modality_name][iou]["tp"]
                    result_stat_all[iou]["fp"] += result_stat[modality_name][iou]["fp"]
                    result_stat_all[iou]["gt"] += result_stat[modality_name][iou]["gt"]
                    result_stat_all[iou]["score"] += result_stat[modality_name][iou]["score"]
                if result_stat[modality_name][iou]["tp"]:
                    os.makedirs(f"{opt.model_dir}/{modality_name}", exist_ok=True)
                    _, ap50, ap70 = eval_utils.eval_final_results(
                        result_stat[modality_name], f"{opt.model_dir}/{modality_name}", opt.infer_info
                    )
            _, ap50, ap70 = eval_utils.eval_final_results(result_stat_all, opt.model_dir, opt.infer_info)

            # segmentation
            for modality in ave_ious:
                if not ave_ious[modality]["static_ave_iou"] or not ave_ious[modality]["dynamic_ave_iou"]:
                    continue
                static_ave_iou = statistics.mean(ave_ious[modality]["static_ave_iou"])
                dynamic_ave_iou = statistics.mean(ave_ious[modality]["dynamic_ave_iou"])
                lane_ave_iou = statistics.mean(ave_ious[modality]["lane_ave_iou"])

                print(f"Modality: {modality}")
                print("Road IoU: %f" % static_ave_iou)
                print("Lane IoU: %f" % lane_ave_iou)
                print("Dynamic IoU: %f" % dynamic_ave_iou)
                if not os.path.exists(os.path.join(opt.model_dir, modality)):
                    os.mkdir(os.path.join(opt.model_dir, modality))

                with open(os.path.join(opt.model_dir, modality, f"{opt.infer_info}_ave_iou.json"), "w") as f:
                    json.dump(
                        {"static_ave_iou": static_ave_iou, "dynamic_ave_iou": dynamic_ave_iou,
                         "lane_ave_iou": lane_ave_iou}, f)
        else:
            if opt.task == "detection":
                _, ap50, ap70 = eval_utils.eval_final_results(result_stat, opt.model_dir, opt.infer_info)
            elif opt.task == "segmentation":
                static_ave_iou = statistics.mean(static_ave_iou)
                dynamic_ave_iou = statistics.mean(dynamic_ave_iou)
                lane_ave_iou = statistics.mean(lane_ave_iou)

                print("Road IoU: %f" % static_ave_iou)
                print("Lane IoU: %f" % lane_ave_iou)
                print("Dynamic IoU: %f" % dynamic_ave_iou)
                with open(os.path.join(opt.model_dir, f"{opt.infer_info}_ave_iou.json"), "w") as f:
                    json.dump(
                        {"static_ave_iou": static_ave_iou, "dynamic_ave_iou": dynamic_ave_iou,
                         "lane_ave_iou": lane_ave_iou}, f
                    )


def eval_detection_result(
        opt, agent_modality_list, opencood_dataset, infer_result_single, result_stat, batch_data, idx, work_dir, hypes,
        i
):
    pred_box_tensor = infer_result_single["pred_box_tensor"]
    gt_box_tensor = infer_result_single["gt_box_tensor"]
    pred_score = infer_result_single["pred_score"]
    if pred_box_tensor is None or gt_box_tensor is None or pred_score is None:
        return
    eval_utils.caluclate_tp_fp(
        pred_box_tensor,
        pred_score,
        gt_box_tensor,
        result_stat[agent_modality_list[idx]] if opt.all else result_stat,
        0.3,
    )
    eval_utils.caluclate_tp_fp(
        pred_box_tensor,
        pred_score,
        gt_box_tensor,
        result_stat[agent_modality_list[idx]] if opt.all else result_stat,
        0.5,
    )
    eval_utils.caluclate_tp_fp(
        pred_box_tensor,
        pred_score,
        gt_box_tensor,
        result_stat[agent_modality_list[idx]] if opt.all else result_stat,
        0.7,
    )
    if opt.save_npy:
        npy_save_path = os.path.join(work_dir, "npy")
        if not os.path.exists(npy_save_path):
            os.makedirs(npy_save_path)
        inference_utils.save_prediction_gt(
            pred_box_tensor, gt_box_tensor, batch_data["ego"]["origin_lidar"][0], i, npy_save_path
        )

    if not opt.no_score:
        infer_result_single.update({"score_tensor": pred_score})

    if getattr(opencood_dataset, "heterogeneous", False):
        cav_box_np, agent_modality_list = inference_utils.get_cav_box(batch_data)
        infer_result_single.update({"cav_box_np": cav_box_np, "agent_modality_list": agent_modality_list})

    if (i % opt.save_vis_interval == 0) and (pred_box_tensor is not None or gt_box_tensor is not None):
        vis_save_path_root = os.path.join(work_dir, f'vis_{opt.infer_info}{"_protocol" if opt.protocol_result else ""}')
        if not os.path.exists(vis_save_path_root):
            os.makedirs(vis_save_path_root)

        # vis_save_path = os.path.join(vis_save_path_root, '3d_%05d.png' % i)
        # simple_vis.visualize(infer_result_single,
        #                     batch_data['ego'][
        #                         'origin_lidar'][0],
        #                     hypes['postprocess']['gt_range'],
        #                     vis_save_path,
        #                     method='3d',
        #                     left_hand=left_hand)
        vis_save_path = os.path.join(vis_save_path_root, "bev_%05d.png" % i)
        try:
            # new version considering various gt ranges
            gt_range = hypes["heter"]["modality_setting"][infer_result_single["ego_modality"]]["postprocess"][
                "gt_range"
            ]
        except:
            gt_range = hypes["postprocess"]["gt_range"]
        try:
            simple_vis.visualize(
                infer_result_single,
                batch_data["ego"]["origin_lidar"][0],
                gt_range,
                vis_save_path,
                method="bev",
                transformation_matrix_clean=torch.inverse(batch_data["ego"]["transformation_matrix_clean"][idx]),
                transformation_matrix=torch.inverse(batch_data["ego"]["transformation_matrix"][idx]),
                left_hand=opt.left_hand,
                show_bev=opt.show_bev,
                pcd_modality=batch_data["ego"]["origin_lidar_modality"][0],
                color_method="cav"
            )
        except:
            pass


    # if i % opt.save_feat_interval == 0 :
    #     feat_vis = infer_result_single.get("feat_vis", None)
    #     if feat_vis:
    #         vis_save_path_root = os.path.join(work_dir, f'vis_feat_{opt.infer_info}')
    #         if not os.path.exists(vis_save_path_root):
    #             os.makedirs(vis_save_path_root)
    #         save_feature_images(feat_vis["raw"], feat_vis["processed"], i, vis_save_path_root)

    if i % opt.save_feat_interval == 0 :
        feat_vis = infer_result_single.get("feat_vis", None)
        if feat_vis:
            vis_save_path_root = os.path.join(work_dir, f'vis_feat_{opt.infer_info}')
            if not os.path.exists(vis_save_path_root):
                os.makedirs(vis_save_path_root)
            feat_list = [ feat_vis["raw"], feat_vis["processed"], ]
            if "processed_gt" in feat_vis.keys():
                feat_list.append(feat_vis["processed_gt"])
            save_feature_images(feat_list, i, vis_save_path_root)





def eval_segmentation_result(opt, infer_result_single, idx, work_dir, i):
    """
    Calculate IoU during training.

    Parameters
    ----------
    batch_dict: dict
        The data that contains the gt.

    output_dict : dict
        The output directory with predictions.

    Returns
    -------
    The iou for static and dynamic bev map.
    """
    pred_dict = infer_result_single["pred_box_tensor"]
    gt_dict = infer_result_single["gt_box_tensor"]
    if pred_dict is None or gt_dict is None:
        return None, None
    # score_dict = infer_result_single['pred_score']
    batch_size = gt_dict["static_bev"].shape[0]
    assert batch_size == 1, "Only support batch size 1 for now."

    gt_static = gt_dict["static_bev"].detach().cpu().data.numpy()[0]
    gt_static = np.array(gt_static, dtype=int)

    gt_dynamic = gt_dict["dynamic_bev"].detach().cpu().data.numpy()[0]
    gt_dynamic = np.array(gt_dynamic, dtype=int)

    pred_static = pred_dict["static_map"]
    pred_static = torchvision.transforms.CenterCrop(gt_static.shape)(pred_static[0]).detach().cpu().data.numpy()
    pred_static = np.array(pred_static, dtype=int)

    pred_dynamic = pred_dict["dynamic_map"]
    pred_dynamic = torchvision.transforms.CenterCrop(gt_dynamic.shape)(pred_dynamic[0]).detach().cpu().data.numpy()
    pred_dynamic = np.array(pred_dynamic, dtype=int)

    iou_dynamic = mean_IU(pred_dynamic, gt_dynamic)
    iou_static = mean_IU(pred_static, gt_static)

    if i % opt.save_vis_interval == 0:
        vis_save_path_root = os.path.join(work_dir, f'vis_{opt.infer_info}{"_protocol" if opt.protocol_result else ""}')
        if not os.path.exists(vis_save_path_root):
            os.makedirs(vis_save_path_root)

        save_path = os.path.join(vis_save_path_root, "%05d_bev_seg.png" % i)
        static_save_path = os.path.join(vis_save_path_root, "%05d_bev_static.png" % i)
        dynamic_save_path = os.path.join(vis_save_path_root, "%05d_bev_dynamic.png" % i)

        static_gt_save_path = os.path.join(vis_save_path_root, "%05d_gt_static.png" % i)
        dynamic_gt_save_path = os.path.join(vis_save_path_root, "%05d_gt_dynamic.png" % i)

        colors = [(255, 255, 255), (255, 200, 200), (20, 20, 220), (80, 40, 40)]
        seg_image = np.ones((256, 256, 3), dtype=np.uint8) * 255
        dynamic_image = np.ones((256, 256, 3), dtype=np.uint8) * 255
        static_image = np.ones((256, 256, 3), dtype=np.uint8) * 255
        static_gt = np.ones((256, 256, 3), dtype=np.uint8) * 255
        dynamic_gt = np.ones((256, 256, 3), dtype=np.uint8) * 255

        for j in range(3):
            seg_image[pred_static == j] = colors[j]
            static_image[pred_static == j] = colors[j]
            static_gt[gt_static == j] = colors[j]
        seg_image[pred_dynamic == 1] = colors[3]
        dynamic_image[pred_dynamic == 1] = colors[3]
        dynamic_gt[gt_dynamic == 1] = colors[3]
        cv2.imwrite(save_path, seg_image)
        cv2.imwrite(static_save_path, static_image)
        cv2.imwrite(dynamic_save_path, dynamic_image)
        cv2.imwrite(static_gt_save_path, static_gt)
        cv2.imwrite(dynamic_gt_save_path, dynamic_gt)

        if opt.show_bev:
            simple_vis.visualize_bev(infer_result_single, os.path.join(vis_save_path_root, "%05d_bev.png" % i))

    return iou_static, iou_dynamic


if __name__ == "__main__":
    main()
