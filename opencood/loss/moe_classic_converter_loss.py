# -*- coding: utf-8 -*-
# Author: Xiangbo Gao <xiangbogaobarry@gmail.com>
# License: MIT License

from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Literal, Optional, Tuple

from opencood.loss.point_pillar_pyramid_loss import PointPillarPyramidLoss




class MoEClassicConverterLoss(nn.Module):
    def __init__(self, args):
        super(MoEClassicConverterLoss, self).__init__()
        self.l2loss = nn.MSELoss()

        self.w_N2E = args.get("w_N2E", 10.0)


        self.gram_w_sched = RampUpWeight(
            start=0.0,
            target=args.get("w_gram", 5),
            warmup_steps=200 * 20,
            mode="linear",
        )

        # -------- SoE alpha constraints (configurable via args) --------
        # Participation target: about one-third experts, mass ~= 0.9
        self.alpha_mass_target = args.get("alpha_mass_target", 0.85)
        self.alpha_cap_max = args.get("alpha_cap_max", 0.55)  # per-expert max alpha
        self.alpha_tau = args.get("alpha_tau", 0.2)  # InfoNCE temperature
        self.alpha_apply_softmax = args.get("alpha_apply_softmax", True)  # if alphas are logits


        self.top_k = getattr(args, "top_k", 3)

        # Weights for each term of alpha loss
        self.w_alpha_mass = args.get("w_alpha_mass", 1.0)
        self.w_alpha_effnum = args.get("w_alpha_effnum", 0.5)
        self.w_alpha_maxcap = args.get("w_alpha_maxcap", 1.0)
        self.w_alpha_balance = args.get("w_alpha_balance", 1.0)
        self.w_alpha_z = args.get("w_alpha_z", 0.6)
        self.w_alpha_contrast = args.get("w_alpha_contrast", 2.0)

        # Overall weight for SoE alpha regularization
        self.w_alpha_total = args.get("w_alpha_total", 2)
        self.w_alpha_stop_ep = args.get("w_alpha_stop_ep", 50)


        self.eps = 1e-8
        self.loss_dict = {}



    @staticmethod
    def _labels_from_pairs(all_modality_pairs):
        """
        Build stable string keys from list of tuples (ego, neb).
        Example input: [(8,8), (8,19), ...] -> ["8->8", "8->19", ...]
        """
        keys = []
        for t in all_modality_pairs:
            ego, neb = t[0], t[1]
            keys.append(f"{ego},{neb}")

        uniq = {k: i for i, k in enumerate(sorted(set(keys)))}
        labels = torch.tensor([uniq[k] for k in keys], dtype=torch.long)  # [B]

        return labels

    def _alpha_contrast_loss(self, A, labels):
        """
        Simple InfoNCE on alpha vectors:
          - positives: same (ego,neb) mapping
          - negatives: different mappings
        A: [B, Nexp], each row on simplex (sum to 1, non-negative)
        """
        B, N = A.shape
        if B <= 2:
            return A.new_zeros(())

        # Cosine similarity after L2 normalization
        An = F.normalize(A, p=2, dim=1)          # [B, N]
        sim = An @ An.t()                        # [B, B]
        eye = torch.eye(B, dtype=torch.bool, device=A.device)

        # Build positive mask by labels
        pos_mask = labels.unsqueeze(0).eq(labels.unsqueeze(1)) & (~eye)

        # InfoNCE
        logits = sim / max(self.alpha_tau, 1e-6)
        logits = logits - logits.max(dim=1, keepdim=True).values      # numerical stability
        exp_logits = torch.exp(logits) * (~eye)                       # zero diagonal
        denom = exp_logits.sum(dim=1) + self.eps
        pos_sum = (exp_logits * pos_mask).sum(dim=1)

        valid = pos_mask.any(dim=1)
        if valid.any():
            loss_i = -(torch.log(pos_sum[valid] + self.eps) - torch.log(denom[valid]))
            return loss_i.mean()
        else:
            return A.new_zeros(())

    def _alpha_losses(self, all_alphas: torch.Tensor, all_modality_pairs):
        """
        Compute MoE alpha regularization terms per block, accumulate into stats.

        Args:
            all_alphas: [n_blocks, B, Nexp], logits or probabilities
            all_modality_pairs: list of tuples (ego, neb), len=B

        Returns:
            stats (dict): initialized-at-zero accumulators + derived means.
        """
        n_blocks, B, Nexp = all_alphas.shape
        K = self.top_k

        # Labels are shared across blocks: same (ego,neb) → same class
        alpha_labels = self._labels_from_pairs(all_modality_pairs).to(all_alphas.device)

        print(alpha_labels.tolist())

        # ---- initialize stats dict with zeros on the correct device ----
        stats = {
            "L_alpha_total": all_alphas.new_zeros(()),  # sum of per-block alpha loss (kept with grad)
            "L_alpha_mass": all_alphas.new_zeros(()),  # Σ L_mass (detached)
            "L_alpha_effnum": all_alphas.new_zeros(()),  # Σ L_effnum (detached)
            "L_alpha_maxcap": all_alphas.new_zeros(()),  # Σ L_maxcap (detached)
            "L_alpha_balance": all_alphas.new_zeros(()),  # Σ L_balance (detached)
            "L_alpha_z": all_alphas.new_zeros(()),
            "L_alpha_contrast": all_alphas.new_zeros(()),  # Σ L_contrast (detached)
            "alpha_topk_mass_mean": all_alphas.new_zeros(()),  # Σ mean(topk_mass) for logging
            "alpha_max_mean": all_alphas.new_zeros(()),  # Σ mean(max_alpha)
            "alpha_select_rate_std": all_alphas.new_zeros(()),  # Σ std(select_rate)
        }

        # ---- per-block computation and accumulation ----
        for block_idx in range(n_blocks):
            A = all_alphas[block_idx]  # [B, Nexp]
            A_logits = A

            # Map logits to simplex if needed
            if self.alpha_apply_softmax:
                A = F.softmax(A, dim=-1)

            # # Safety renormalization
            # A = A / (A.sum(dim=1, keepdim=True) + self.eps)

            #  Importance Loss (P_j) & Load Loss (C_j)
            # Mean probability per expert across batch (Importance)
            # P_j = (1/B) * sum_b(p_bj)
            prob_mean_per_expert = A.mean(dim=0)  # [Nexp]

            # Soft Count / Usage (Load)
            # L_balance = Nexp * sum(mean_P_i ^ 2) -> minimal when all P_i = 1/N
            L_balance = Nexp * torch.sum(prob_mean_per_expert ** 2)


            #  z-loss on pre-softmax logits (stabilizes router)
            L_z = A_logits.pow(2).mean()

            #  Contrastive consistency on (ego,neb)
            L_contrast = self._alpha_contrast_loss(A_logits, alpha_labels)
            # L_contrast = self._alpha_contrast_loss(A, alpha_labels)

            # Per-block weighted sum (kept with grad)
            L_alpha_block = (
                    + self.w_alpha_balance * L_balance
                    + self.w_alpha_z * L_z
                    + self.w_alpha_contrast * L_contrast
            )

            # ---- accumulate into stats ----
            stats["L_alpha_total"] += L_alpha_block
            stats["L_alpha_balance"] += L_balance.detach()
            stats["L_alpha_z"] += L_z.detach()
            stats["L_alpha_contrast"] += L_contrast.detach()

            # stats["alpha_topk_mass_mean"] += topk_mass.mean().detach()
            # stats["alpha_max_mean"] += max_alpha.mean().detach()
            # stats["alpha_select_rate_std"] += select_rate.std().detach()



        return stats

    # -------------------- main forward --------------------
    def forward(self, output_dict, target_dict, suffix="", val=False):
        """
        output_dict:
            "FE": FE,
            "FN2E": FN2E,
            "has_neb": has_neb,
            "all_alphas": all_alphas,                    # [n_blocks, B, Nexp]
            "all_modality_pairs": all_modality_pairs,    # python list of tuples [(ego, neb), ...]
        """


        if output_dict.get("skip_flag", False):
            return output_dict["loss"]


        total_loss = 0.0

        FE = output_dict["FE"]
        FN2E = output_dict["FN2E"]
        all_alphas = output_dict["all_alphas"]
        all_modality_pairs = output_dict["all_modality_pairs"]

        has_neb = output_dict.get("has_neb", True) or True
        if has_neb:
            # -------- existing terms --------
            N2E = self.l2loss(FE, FN2E) * self.w_N2E
            w_gram_anchor = self.gram_w_sched.step()
            gram_anchor_loss = gram_anchoring_loss(FN2E, FE) * 10 * w_gram_anchor

            total_loss = N2E + gram_anchor_loss

            self.loss_dict.update({
                "N2E": float(N2E.item()),
                "gram_anchor_loss": float(gram_anchor_loss.item()),
            })

            # 2. MoE / Alpha Regularization
            # Only calculate if we have alpha logits
            if all_alphas is not None and self.w_alpha_total > 0:
                alpha_stats = self._alpha_losses(all_alphas, all_modality_pairs)
                L_alpha_total = alpha_stats.pop("L_alpha_total")

                total_loss = total_loss + self.w_alpha_total * L_alpha_total

                # log scalars
                self.loss_dict.update({k: (float(v.item()) if torch.is_tensor(v) else float(v)) for k, v in alpha_stats.items()})
                self.loss_dict["L_alpha_total"] = float(L_alpha_total.item())

        # finalize
        self.loss_dict.update({
            "total_loss": float(total_loss.item() if torch.is_tensor(total_loss) else total_loss)
        })
        return total_loss

    def logging(self, epoch, batch_id, batch_len, writer=None, suffix=""):
        """
        Print out the loss terms for current iteration.
        """
        total_loss = self.loss_dict.get('total_loss', 0.0)
        N2E_loss = self.loss_dict.get('N2E', 0.0)
        gram_anchor_loss = self.loss_dict.get('gram_anchor_loss', 0.0)

        # alpha-related (may be absent if not computed)
        L_alpha_total = self.loss_dict.get('L_alpha_total', 0.0)
        L_alpha_balance = self.loss_dict.get('L_alpha_balance', 0.0)
        L_alpha_z = self.loss_dict.get('L_alpha_z', 0.0)
        L_alpha_contrast = self.loss_dict.get('L_alpha_contrast', 0.0)

        self.loss_dict = {}

        print("[epoch %d][%d/%d]%s || Loss: %.4f || N2E: %.4f || Gram: %.4f || "
              "AlphaTot: %.4f (bal: %.4f, z: %.4f, ctr: %.4f)" % (
                  epoch, batch_id + 1, batch_len, suffix,
                  total_loss, N2E_loss, gram_anchor_loss,
                  L_alpha_total, L_alpha_balance, L_alpha_z, L_alpha_contrast))

        if writer is not None:
            gid = epoch * batch_len + batch_id
            writer.add_scalar('N2E_loss' + suffix, N2E_loss, gid)
            writer.add_scalar('gram_anchor_loss' + suffix, gram_anchor_loss, gid)

            writer.add_scalar('alpha/L_total' + suffix, L_alpha_total, gid)
            writer.add_scalar('alpha/L_balance' + suffix, L_alpha_balance, gid)
            writer.add_scalar('alpha/L_z' + suffix, L_alpha_z, gid)
            writer.add_scalar('alpha/L_contrast' + suffix, L_alpha_contrast, gid)


        # keep schedules aligned
        self.gram_w_sched.ensure_aligned(epoch, batch_id, batch_len, tolerance=100, for_next_step=True)

        if epoch > self.w_alpha_stop_ep:
            self.w_alpha_total = 0.01




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






