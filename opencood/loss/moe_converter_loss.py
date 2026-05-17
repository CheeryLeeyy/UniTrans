# -*- coding: utf-8 -*-
# Author: Xiangbo Gao <xiangbogaobarry@gmail.com>
# License: MIT License

from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Literal, Optional, Tuple





import torch.distributed as dist

def ddp_is_on():
    return dist.is_available() and dist.is_initialized()

def all_reduce_sum(x: torch.Tensor):
    if ddp_is_on():
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x

def all_gather_cat(x: torch.Tensor):
    if not ddp_is_on():
        return x
    world = dist.get_world_size()
    xs = [torch.zeros_like(x) for _ in range(world)]
    dist.all_gather(xs, x)
    return torch.cat(xs, dim=0)




class MoEConverterLoss(nn.Module):
    def __init__(self, args: dict):
        super(MoEConverterLoss, self).__init__()
        self.l2loss = nn.MSELoss()

        # existing losses
        self.w_N2E = args.get("w_N2E", 10.0)

        self.gram_w_sched = RampUpWeight(
            start=0.0,
            target=args.get("w_gram", 5.0),
            warmup_steps=args.get("w_gram_warmup_steps", 200),
            mode="linear",
        )

        # alpha regularization
        self.top_k = args.get("top_k", 3)

        self.alpha_apply_softmax = args.get("alpha_apply_softmax", True)
        self.alpha_softmax_tau = args.get("alpha_softmax_tau", 1.0)

        self.alpha_mass_target = args.get("alpha_mass_target", 0.85)
        self.alpha_cap_max = args.get("alpha_cap_max", 0.70)
        self.alpha_tau = args.get("alpha_tau", 0.2)

        # weights
        self.w_alpha_mass = args.get("w_alpha_mass", 1.0)
        self.w_alpha_effnum = args.get("w_alpha_effnum", 0.01)
        self.w_alpha_maxcap = args.get("w_alpha_maxcap", 1.0)
        self.w_alpha_lb = args.get("w_alpha_lb", 1.0)                 # added: importance x load
        self.w_alpha_z = args.get("w_alpha_z", 0.1)
        self.w_alpha_contrast = args.get("w_alpha_contrast", 0.5)

        self.w_alpha_total = args.get("w_alpha_total", 0.1)
        self.w_alpha_stop_ep = args.get("w_alpha_stop_ep", 500)

        self.eps = 1e-8
        self.loss_dict = {}


    @staticmethod
    def _labels_from_pairs(all_modality_pairs, device):
        """
        DDP-aware label assignment:
        1) gather (ego,neb) pairs across all ranks
        2) build a global, stable label id for each unique pair
        3) return local labels and global labels in the gathered order

        Returns:
            labels_local: [B_local]
            labels_all:   [B_global]
            meta: dict with keys {ddp, rank, world, sizes, offset, B_local, B_global}
        """
        ddp = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if ddp else 0
        world = dist.get_world_size() if ddp else 1

        pairs_local = torch.tensor(all_modality_pairs, dtype=torch.long, device=device)  # [B_local, 2]
        B_local = int(pairs_local.shape[0])

        if not ddp:
            keys = (pairs_local[:, 0].to(torch.int64) << 32) + pairs_local[:, 1].to(torch.int64)  # [B]
            uniq = torch.unique(keys, sorted=True)  # [U]
            labels_local = torch.searchsorted(uniq, keys).to(torch.long)
            meta = {
                "ddp": False, "rank": 0, "world": 1,
                "sizes": [B_local], "offset": 0,
                "B_local": B_local, "B_global": B_local,
            }
            return labels_local, labels_local, meta

        # gather sizes (handle last batch with uneven per-rank batch sizes)
        size_t = torch.tensor([B_local], dtype=torch.long, device=device)
        size_list = [torch.zeros_like(size_t) for _ in range(world)]
        dist.all_gather(size_list, size_t)
        sizes = [int(x.item()) for x in size_list]
        max_B = max(sizes)
        offset = sum(sizes[:rank])
        B_global = sum(sizes)

        # pad pairs to max_B, then all_gather
        if B_local < max_B:
            pad = torch.full((max_B - B_local, 2), -1, dtype=torch.long, device=device)
            pairs_pad = torch.cat([pairs_local, pad], dim=0)
        else:
            pairs_pad = pairs_local

        gathered = [torch.empty((max_B, 2), dtype=torch.long, device=device) for _ in range(world)]
        dist.all_gather(gathered, pairs_pad)

        # unpad and concat in rank order to define the global sample order
        pairs_all = []
        for r in range(world):
            if sizes[r] > 0:
                pairs_all.append(gathered[r][:sizes[r]])
        pairs_all = torch.cat(pairs_all, dim=0)  # [B_global, 2]

        # numeric key for stable mapping
        keys_all = (pairs_all[:, 0].to(torch.int64) << 32) + pairs_all[:, 1].to(torch.int64)  # [B_global]
        uniq = torch.unique(keys_all, sorted=True)  # [U]
        labels_all = torch.searchsorted(uniq, keys_all).to(torch.long)  # [B_global]
        labels_local = labels_all[offset: offset + B_local]

        meta = {
            "ddp": True, "rank": rank, "world": world,
            "sizes": sizes, "offset": offset,
            "B_local": B_local, "B_global": B_global,
        }
        return labels_local, labels_all, meta


    def _alpha_contrast_loss(self, A_local, labels_local, A_all, labels_all, offset):
        """
        DDP-aware InfoNCE on alpha vectors.
        Anchors: A_local (with grad)
        Bank:    A_all (recommended to be detached)
        Positives: same label id, excluding self position (offset + i)
        """
        B_local = int(A_local.shape[0])
        B_all = int(A_all.shape[0])
        if B_local <= 2 or B_all <= 2:
            return A_local.new_zeros(())

        An_local = F.normalize(A_local, p=2, dim=1)  # [B_local, N]
        An_all = F.normalize(A_all, p=2, dim=1)      # [B_all,   N]
        sim = An_local @ An_all.t()                  # [B_local, B_all]

        logits = sim / max(self.alpha_tau, 1e-6)
        logits = logits - logits.max(dim=1, keepdim=True).values

        pos_mask = labels_local.view(-1, 1).eq(labels_all.view(1, -1))  # [B_local, B_all]

        # exclude self (the same sample in the global bank)
        self_idx = torch.arange(B_local, device=A_local.device) + int(offset)  # [B_local]
        self_mask = torch.zeros_like(pos_mask, dtype=torch.bool)
        self_mask.scatter_(1, self_idx.view(-1, 1), True)
        pos_mask = pos_mask & (~self_mask)

        valid = pos_mask.any(dim=1)
        if not valid.any():
            return A_local.new_zeros(())

        exp_logits = torch.exp(logits) * (~self_mask)
        denom = exp_logits.sum(dim=1) + self.eps
        pos_sum = (exp_logits * pos_mask).sum(dim=1)

        loss_i = -(torch.log(pos_sum[valid] + self.eps) - torch.log(denom[valid]))
        return loss_i.mean()


    def _alpha_losses(self, all_alphas: torch.Tensor, all_modality_pairs):
        """
        DDP-aware alpha regularization:
          - global labels are built by gathering modality pairs first
          - contrastive uses global bank (all_gather)
          - importance x load uses global statistics (all_reduce)
        """
        n_blocks, B_local, Nexp = all_alphas.shape
        K = self.top_k
        device = all_alphas.device
        ddp = dist.is_available() and dist.is_initialized()

        labels_local, labels_all, meta = self._labels_from_pairs(all_modality_pairs, device)
        offset = meta["offset"]
        sizes = meta["sizes"]
        world = meta["world"]
        max_B = max(sizes)
        B_global = meta["B_global"]

        def _gather_feat_with_padding(x_local):
            if not ddp:
                return x_local
            # x_local: [B_local, D]
            B = x_local.shape[0]
            D = x_local.shape[1]
            if B < max_B:
                pad = torch.zeros((max_B - B, D), dtype=x_local.dtype, device=x_local.device)
                x_pad = torch.cat([x_local, pad], dim=0)
            else:
                x_pad = x_local
            buf = [torch.empty((max_B, D), dtype=x_local.dtype, device=x_local.device) for _ in range(world)]
            dist.all_gather(buf, x_pad)
            xs = []
            for r in range(world):
                if sizes[r] > 0:
                    xs.append(buf[r][:sizes[r]])
            return torch.cat(xs, dim=0)  # [B_global, D]

        L_total_sum = all_alphas.new_zeros(())
        L_mass_sum = all_alphas.new_zeros(())
        L_eff_sum = all_alphas.new_zeros(())
        L_cap_sum = all_alphas.new_zeros(())
        L_lb_sum = all_alphas.new_zeros(())
        L_z_sum = all_alphas.new_zeros(())
        L_ctr_sum = all_alphas.new_zeros(())
        topk_mass_mean_sum = all_alphas.new_zeros(())
        max_alpha_mean_sum = all_alphas.new_zeros(())
        sel_std_sum = all_alphas.new_zeros(())

        for i in range(n_blocks):
            A_logits = all_alphas[i]  # [B_local, Nexp]

            if self.alpha_apply_softmax:
                A_prob = F.softmax(A_logits / max(self.alpha_softmax_tau, 1e-6), dim=-1)
                route_scores = A_logits
            else:
                A_prob = A_logits.clamp_min(0.0)
                A_prob = A_prob / (A_prob.sum(dim=1, keepdim=True) + self.eps)
                route_scores = A_prob

            topk_idx = torch.topk(route_scores, k=K, dim=1).indices
            topk_mask = torch.zeros_like(A_prob).scatter_(1, topk_idx, 1.0)

            # 1) top-k mass target
            topk_vals = torch.gather(A_prob, dim=1, index=topk_idx)   # [B_local, K]
            topk_mass = topk_vals.sum(dim=1)                          # [B_local]
            L_mass = ((topk_mass - self.alpha_mass_target) ** 2).mean()

            # 2) effective number target (~K)
            sum2 = (A_prob ** 2).sum(dim=1) + self.eps
            N_eff = 1.0 / sum2
            L_eff = ((N_eff - float(K)) ** 2).mean()

            # 3) max cap
            max_alpha = A_prob.max(dim=1).values
            L_cap = F.relu(max_alpha - self.alpha_cap_max).pow(2).mean()

            # 4) Switch-style importance x load, use global statistics
            imp_local = A_prob.sum(dim=0)       # [Nexp]
            load_local = topk_mask.sum(dim=0)   # [Nexp]

            if ddp:
                imp_g = imp_local.clone()
                load_g = load_local.clone()
                dist.all_reduce(imp_g, op=dist.ReduceOp.SUM)
                dist.all_reduce(load_g, op=dist.ReduceOp.SUM)
                imp_frac = imp_g / float(B_global)
                load_frac = load_g / float(B_global * K)
            else:
                imp_frac = imp_local / float(B_local)
                load_frac = load_local / float(B_local * K)

            L_lb = Nexp * (imp_frac * load_frac).sum()
            sel_std = load_frac.std()

            # 5) z-loss (only meaningful for logits)
            if self.alpha_apply_softmax:
                L_z = (torch.logsumexp(A_logits, dim=-1).pow(2)).mean()
            else:
                L_z = A_prob.new_zeros(())

            # 6) contrastive with global bank
            if ddp:
                A_all = _gather_feat_with_padding(A_prob.detach())
                L_ctr = self._alpha_contrast_loss(A_prob, labels_local, A_all, labels_all, offset)
            else:
                L_ctr = self._alpha_contrast_loss(A_prob, labels_local, A_prob.detach(), labels_local, 0)

            L_block = (
                    self.w_alpha_mass * L_mass
                    + self.w_alpha_effnum * L_eff
                    + self.w_alpha_maxcap * L_cap
                    + self.w_alpha_lb * L_lb
                    + self.w_alpha_z * L_z
                    + self.w_alpha_contrast * L_ctr
            )
            L_total_sum = L_total_sum + L_block

            # logging (detach)
            L_mass_sum += L_mass.detach()
            L_eff_sum += L_eff.detach()
            L_cap_sum += L_cap.detach()
            L_lb_sum += L_lb.detach()
            L_z_sum += L_z.detach()
            L_ctr_sum += L_ctr.detach()
            topk_mass_mean_sum += topk_mass.mean().detach()
            max_alpha_mean_sum += max_alpha.mean().detach()
            sel_std_sum += sel_std.detach()

        denom = float(max(n_blocks, 1))
        stats = {
            "L_alpha_total": L_total_sum / denom,
            "L_alpha_mass": L_mass_sum / denom,
            "L_alpha_effnum": L_eff_sum / denom,
            "L_alpha_maxcap": L_cap_sum / denom,
            "L_alpha_lb": L_lb_sum / denom,
            "L_alpha_z": L_z_sum / denom,
            "L_alpha_contrast": L_ctr_sum / denom,
            "alpha_topk_mass_mean": topk_mass_mean_sum / denom,
            "alpha_max_mean": max_alpha_mean_sum / denom,
            "alpha_select_rate_std": sel_std_sum / denom,
        }

        return stats

    def forward(self, output_dict, target_dict=None, suffix="", val=False):
        if output_dict.get("skip_flag", False):
            return output_dict["loss"]

        FE = output_dict["FE"]
        FN2E = output_dict["FN2E"]
        all_alphas = output_dict.get("all_alphas", None)
        all_modality_pairs = output_dict.get("all_modality_pairs", None)

        total_loss = FE.new_zeros(())
        has_neb = output_dict.get("has_neb", True) or True

        if has_neb:
            N2E = self.l2loss(FE, FN2E) * self.w_N2E
            w_gram_anchor = self.gram_w_sched.step()
            gram_anchor_loss = gram_anchoring_loss(FN2E, FE) * w_gram_anchor

            total_loss = N2E + gram_anchor_loss

            self.loss_dict.update({
                "N2E": float(N2E.item()),
                "gram_anchor_loss": float(gram_anchor_loss.item()),
            })

            if (
                    all_alphas is not None
                    and all_modality_pairs is not None
                    and self.w_alpha_total > 0
            ):
                alpha_stats = self._alpha_losses(all_alphas, all_modality_pairs)
                L_alpha_total = alpha_stats.pop("L_alpha_total")

                total_loss = total_loss + self.w_alpha_total * L_alpha_total

                self.loss_dict["L_alpha_total"] = float(L_alpha_total.item())
                for k, v in alpha_stats.items():
                    self.loss_dict[k] = float(v.item()) if torch.is_tensor(v) else float(v)

        self.loss_dict["total_loss"] = float(total_loss.item())
        return total_loss

    def logging(self, epoch, batch_id, batch_len, writer=None, suffix=""):
        total_loss = self.loss_dict.get("total_loss", 0.0)
        N2E_loss = self.loss_dict.get("N2E", 0.0)
        gram_anchor_loss = self.loss_dict.get("gram_anchor_loss", 0.0)

        L_alpha_total = self.loss_dict.get("L_alpha_total", 0.0)
        L_alpha_mass = self.loss_dict.get("L_alpha_mass", 0.0)
        L_alpha_effnum = self.loss_dict.get("L_alpha_effnum", 0.0)
        L_alpha_maxcap = self.loss_dict.get("L_alpha_maxcap", 0.0)
        L_alpha_lb = self.loss_dict.get("L_alpha_lb", 0.0)
        L_alpha_z = self.loss_dict.get("L_alpha_z", 0.0)
        L_alpha_contrast = self.loss_dict.get("L_alpha_contrast", 0.0)
        alpha_topk_mass_mean = self.loss_dict.get("alpha_topk_mass_mean", 0.0)
        alpha_max_mean = self.loss_dict.get("alpha_max_mean", 0.0)
        alpha_select_rate_std = self.loss_dict.get("alpha_select_rate_std", 0.0)

        self.loss_dict = {}

        print(
            "[epoch %d][%d/%d]%s || Loss: %.4f || N2E: %.4f || Gram: %.4f || "
            "AlphaTot: %.4f (mass: %.4f, eff: %.4f, cap: %.4f, lb: %.4f, z: %.4f, ctr: %.4f, "
            "topk_mass: %.4f, max_a: %.4f, sel_std: %.4f)"
            % (
                epoch, batch_id + 1, batch_len, suffix,
                total_loss, N2E_loss, gram_anchor_loss,
                L_alpha_total, L_alpha_mass, L_alpha_effnum, L_alpha_maxcap,
                L_alpha_lb, L_alpha_z, L_alpha_contrast,
                alpha_topk_mass_mean, alpha_max_mean, alpha_select_rate_std
            )
        )

        if writer is not None:
            gid = epoch * batch_len + batch_id
            writer.add_scalar("N2E_loss" + suffix, N2E_loss, gid)
            writer.add_scalar("gram_anchor_loss" + suffix, gram_anchor_loss, gid)

            writer.add_scalar("alpha/L_total" + suffix, L_alpha_total, gid)
            writer.add_scalar("alpha/L_mass" + suffix, L_alpha_mass, gid)
            writer.add_scalar("alpha/L_effnum" + suffix, L_alpha_effnum, gid)
            writer.add_scalar("alpha/L_maxcap" + suffix, L_alpha_maxcap, gid)
            writer.add_scalar("alpha/L_lb" + suffix, L_alpha_lb, gid)
            writer.add_scalar("alpha/L_z" + suffix, L_alpha_z, gid)
            writer.add_scalar("alpha/L_contrast" + suffix, L_alpha_contrast, gid)
            writer.add_scalar("alpha/topk_mass_mean" + suffix, alpha_topk_mass_mean, gid)
            writer.add_scalar("alpha/max_alpha_mean" + suffix, alpha_max_mean, gid)
            writer.add_scalar("alpha/select_rate_std" + suffix, alpha_select_rate_std, gid)

        self.gram_w_sched.ensure_aligned(epoch, batch_id, batch_len, tolerance=100, for_next_step=True)

        if epoch > self.w_alpha_stop_ep:
            self.w_alpha_total = 0.001



# class MoEConverterLoss(nn.Module):
#     def __init__(self, args):
#         super(MoEConverterLoss, self).__init__()
#         self.l2loss = nn.MSELoss()
#
#
#
#         self.gram_w_sched = RampUpWeight(
#             start=0.0,
#             target=10,
#             # warmup_steps=1,
#             warmup_steps=800*20,
#             mode="linear",  # "linear" | "cosine"
#         )
#
#         self.loss_dict = {}
#
#     def forward(self, output_dict, target_dict, suffix="", val=False):
#         """
#         output_dict
#             "FE": FE,
#             "FN2E": FN2E,
#             "has_neb": has_neb,
#             "all_alphas": all_alphas,
#             "all_modality_pairs": all_modality_pairs,
#
#
#         """
#
#         total_loss = 0
#         FE = output_dict["FE"]
#         FN2E = output_dict["FN2E"]
#         all_alphas = output_dict["all_alphas"]      # n_blocks, minibatch, n_experts
#         all_modality_pairs = output_dict["all_modality_pairs"]      # [ minibatch ]       [ (ego_modality, neb_modality), ...  ]
#
#
#         has_neb = output_dict["has_neb"] or True
#         if has_neb:     ## 存在neb
#             N2E = self.l2loss(FE, FN2E) * 10
#
#
#             w_gram_anchor = self.gram_w_sched.step()
#             gram_anchor_loss = gram_anchoring_loss(FN2E, FE) * w_gram_anchor
#
#
#
#
#
#             total_loss = N2E + gram_anchor_loss
#
#             self.loss_dict.update({
#                 "N2E": N2E.item(),
#                 "gram_anchor_loss": gram_anchor_loss.item(),
#             })
#
#         self.loss_dict.update({"total_loss": total_loss.item() if torch.is_tensor(total_loss) else total_loss})
#
#         return total_loss
#
#
#
#
#
#
#
#
#     def logging(self, epoch, batch_id, batch_len, writer=None, suffix=""):
#         """
#         Print out  the loss function for current iteration.
#
#         Parameters
#         ----------
#         epoch : int
#             Current epoch for training.
#         batch_id : int
#             The current batch.
#         batch_len : int
#             Total batch length in one iteration of training,
#         writer : SummaryWriter
#             Used to visualize on tensorboard
#         """
#         total_loss = self.loss_dict.get('total_loss', 0)
#         N2E_loss = self.loss_dict.get('N2E', 0)
#         gram_anchor_loss = self.loss_dict.get('gram_anchor_loss', 0)
#
#         self.loss_dict = {}
#
#         print("[epoch %d][%d/%d]%s || Loss: %.4f ||  N2E Loss: %.4f || || Gram Loss: %.4f" % (
#                   epoch, batch_id + 1, batch_len, suffix,  total_loss,  N2E_loss, gram_anchor_loss))
#
#
#         if not writer is None:
#
#             writer.add_scalar('N2E_loss' + suffix, N2E_loss, epoch*batch_len + batch_id)
#             writer.add_scalar('gram_anchor_loss' + suffix, gram_anchor_loss, epoch*batch_len + batch_id)
#
#
#         self.gram_w_sched.ensure_aligned(epoch, batch_id, batch_len, tolerance=100, for_next_step=True)


class EntropyWeightSchedule:
    """
    3-phase schedule for alpha-entropy weight:
      warmup:   start -> peak
      hold:     keep at peak
      cooldown: peak -> final

    epoch/batch are 0-based. Call `step()` once per training step.
    Use `ensure_aligned(...)` to realign internal step if resuming.
    """
    def __init__(self, start, peak, final, warmup_steps, hold_steps, cooldown_steps, mode="linear"):
        self.start = float(start)
        self.peak = float(peak)
        self.final = float(final)
        self.warm = max(0, int(warmup_steps))
        self.hold = max(0, int(hold_steps))
        self.cool = max(0, int(cooldown_steps))
        self.mode = mode
        self._step = 0
        if (self.warm + self.hold + self.cool) == 0:
            raise ValueError("At least one of warm/hold/cool must be > 0")

    # ---- core API ----
    def step(self) -> float:
        """Advance one step and return the current weight."""
        self._step += 1
        return self._weight_at(self._step)

    def ensure_aligned(self, epoch: int, batch_id: int, batch_len: int,
                       tolerance: int = 0, for_next_step: bool = False) -> bool:
        """
        Realign internal step to s = epoch*batch_len + batch_id (0-based)
        only if |current - s| > tolerance. If for_next_step=True, s -= 1 so that
        the next `step()` call returns the weight for (epoch, batch_id).
        Returns True iff an adjustment is made.
        """
        s = epoch * batch_len + batch_id
        if for_next_step:
            s -= 1
        s = max(0, s)
        if abs(self._step - s) > int(tolerance):
            self._step = s
            return True
        return False

    # ---- internals ----
    def _interp(self, a: float, b: float, t: float) -> float:
        t = max(0.0, min(1.0, t))
        if self.mode == "cosine":
            return a + (b - a) * 0.5 * (1.0 - math.cos(math.pi * t))
        return a + (b - a) * t

    def _weight_at(self, s: int) -> float:
        if s <= self.warm and self.warm > 0:
            return self._interp(self.start, self.peak, s / float(self.warm))
        if s <= self.warm + self.hold and self.hold > 0:
            return self.peak
        if self.cool > 0:
            sc = s - (self.warm + self.hold)
            return self._interp(self.peak, self.final, min(1.0, sc / float(self.cool)))
        return self.final


class RampUpWeight:
    """
    Simple ramp: start -> target over warmup_steps (linear/cosine).
    epoch/batch are 0-based. Call `step()` once per training step.
    """
    def __init__(self, start, target, warmup_steps, mode: str = "linear"):
        self.start = float(start)
        self.target = float(target)
        self.warm = max(1, int(warmup_steps))
        self.mode = mode
        self._step = 0

    # ---- core API ----
    def step(self) -> float:
        """Advance one step and return the current weight."""
        self._step += 1
        return self._weight_at(self._step)

    def ensure_aligned(self, epoch: int, batch_id: int, batch_len: int,
                       tolerance: int = 0, for_next_step: bool = False) -> bool:
        """
        Realign internal step to s = epoch*batch_len + batch_id (0-based)
        only if |current - s| > tolerance. If for_next_step=True, s -= 1 so that
        the next `step()` call returns the weight for (epoch, batch_id).
        Returns True iff an adjustment is made.
        """
        s = epoch * batch_len + batch_id
        if for_next_step:
            s -= 1
        s = max(0, s)
        if abs(self._step - s) > int(tolerance):
            self._step = s
            return True
        return False

    # ---- internals ----
    def _ratio(self, t: float) -> float:
        t = max(0.0, min(1.0, t))
        if self.mode == "cosine":
            return 0.5 * (1.0 - math.cos(math.pi * t))
        return t

    def _weight_at(self, s: int) -> float:
        r = self._ratio(s / float(self.warm))
        return self.start + (self.target - self.start) * r






def _ensure_batch(x: torch.Tensor) -> torch.Tensor:
    """Ensure input has batch dim: accept (C,H,W) or (B,C,H,W)."""
    if x.dim() == 3:
        return x.unsqueeze(0)
    return x

def _avgpool_feat(feat: torch.Tensor, out_size: Tuple[int, int]) -> torch.Tensor:
    """
    Apply spatial average pooling to reduce HxW -> out_size.
    Input: feat (B, C, H, W)
    Output: pooled (B, C, H2, W2)
    """
    B, C, H, W = feat.shape
    target_h, target_w = out_size
    # use adaptive avg pool to allow arbitrary input sizes
    return F.adaptive_avg_pool2d(feat, output_size=(target_h, target_w))

def _normalize_vectors(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """L2 normalize tensor along given dim."""
    norm = x.norm(dim=dim, keepdim=True).clamp_min(eps)
    return x / norm

def gram_matrix_spatial(patches: torch.Tensor) -> torch.Tensor:
    """
    Compute spatial Gram (patch x patch) for a batch.
    Input: patches (B, P, C) where P = H*W (patch vectors, last dim = channel)
    Output: gram (B, P, P)
    """
    # patches assumed already normalized along channel dim if desired
    return torch.matmul(patches, patches.transpose(1, 2))

def gram_matrix_channel(feat: torch.Tensor) -> torch.Tensor:
    """
    Compute channel Gram (channel x channel) for a batch.
    Input: feat (B, C, P) where P = H*W
    Output: gram (B, C, C)
    """
    return torch.matmul(feat, feat.transpose(1, 2))

def gram_anchoring_loss(
    feat: torch.Tensor,
    gt: torch.Tensor,
    use_spatial: bool = True,
    use_channel: bool = False,
    pool_size: Tuple[int, int] = (32, 32),
    spatial_weight: float = 1.0,
    channel_weight: float = 1.0,
    normalize_patches: bool = False,
    normalize_channels: bool = False,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute Gram Anchoring loss with options:
      - spatial Gram computed after avgpool to pool_size (W*H -> pool_size[0]*pool_size[1])
      - channel Gram computed on channels (C x C)
      - both can be used and weighted by spatial_weight / channel_weight

    Args:
      feat, gt: tensors of shape (B,C,H,W) or (C,H,W)
      use_spatial: whether to compute spatial Gram (patch x patch)
      use_channel: whether to compute channel Gram (C x C)
      pool_size: target spatial size for avgpool before spatial Gram (H2, W2)
      spatial_weight: scalar weight for spatial Gram loss
      channel_weight: scalar weight for channel Gram loss
      normalize_patches: L2-normalize each patch vector across channel dim before spatial Gram
      normalize_channels: L2-normalize each channel vector across spatial dim before channel Gram
      reduction: "mean" or "sum" (over batch)
    Returns:
      scalar loss Tensor
    """
    feat = _ensure_batch(feat)
    gt = _ensure_batch(gt)
    assert feat.shape == gt.shape, "feat and gt must have same shape"

    B, C, H, W = feat.shape
    loss_terms = []
    total_weight = 0.0

    # --- spatial gram (after avgpool) ---
    if use_spatial:
        # pool features
        feat_p = _avgpool_feat(feat, pool_size)  # (B, C, H2, W2)
        gt_p = _avgpool_feat(gt, pool_size)
        B, C, H2, W2 = feat_p.shape
        P = H2 * W2

        # reshape to (B, P, C)
        patches_f = feat_p.permute(0, 2, 3, 1).contiguous().view(B, P, C)
        patches_g = gt_p.permute(0, 2, 3, 1).contiguous().view(B, P, C)

        if normalize_patches:
            # normalize each patch vector across channel dim
            patches_f = _normalize_vectors(patches_f, dim=-1)
            patches_g = _normalize_vectors(patches_g, dim=-1)

        # compute Gram (B, P, P)
        gram_f = gram_matrix_spatial(patches_f)
        gram_g = gram_matrix_spatial(patches_g)

        # optionally normalize gram by P to keep scale consistent across pool sizes
        gram_f = gram_f / float(P)
        gram_g = gram_g / float(P)

        diff = gram_f - gram_g
        spatial_loss = (diff * diff)
        if reduction == "mean":
            spatial_loss = spatial_loss.mean()
        else:
            spatial_loss = spatial_loss.sum()

        loss_terms.append(spatial_weight * spatial_loss)
        total_weight += spatial_weight

    # --- channel gram (C x C) ---
    if use_channel:
        # reshape to (B, C, P)
        P_full = H * W
        f_ch = feat.view(B, C, P_full)
        g_ch = gt.view(B, C, P_full)

        if normalize_channels:
            # normalize each channel vector across spatial dim
            f_ch = _normalize_vectors(f_ch, dim=-1)
            g_ch = _normalize_vectors(g_ch, dim=-1)

        gram_fc = gram_matrix_channel(f_ch)  # (B, C, C)
        gram_gc = gram_matrix_channel(g_ch)  # (B, C, C)

        # optionally normalize gram by P_full
        gram_fc = gram_fc / float(P_full)
        gram_gc = gram_gc / float(P_full)

        diff_c = gram_fc - gram_gc
        channel_loss = (diff_c * diff_c)
        if reduction == "mean":
            channel_loss = channel_loss.mean()
        else:
            channel_loss = channel_loss.sum()

        loss_terms.append(channel_weight * channel_loss)
        total_weight += channel_weight

    if len(loss_terms) == 0:
        raise ValueError("At least one of use_spatial or use_channel must be True.")

    # combine and normalize by sum of weights so absolute scale is independent
    loss = sum(loss_terms) / float(total_weight)
    return loss






