import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import hypsecant
import math
import torch.distributed as dist


class ModalityIntrinsicLoss(nn.Module):
    def __init__(self, args):
        super(ModalityIntrinsicLoss, self).__init__()
        # ---- loss weight schedulers (independent objects) ----
        if "intrinsic" in args.keys():
            self.intrinsic = args["intrinsic"]
        else:
            self.intrinsic =args

        self.center_w_sched = RampUpWeight(
            start=self.intrinsic.get("center_loss_w_start", 0.0),
            target=self.intrinsic.get("center_loss_w_max", 0.4),
            warmup_steps=self.intrinsic.get("loss_warmup_steps", 2000),
            mode=self.intrinsic.get("loss_warmup_mode", "linear"),   # "linear" | "cosine"
        )
        self.arc_w_sched = RampUpWeight(
            start=self.intrinsic.get("arcface_weight_start", 0.0),
            target=self.intrinsic.get("arcface_weight_max", 0.1),
            warmup_steps=self.intrinsic.get("loss_warmup_steps", 2050),
            mode=self.intrinsic.get("loss_warmup_mode", "linear"),
        )

        self.m_cls_loss = nn.CrossEntropyLoss()
        self.cls_weight = self.intrinsic.get("cls_weight", 0.0)
        self.z_weight = self.intrinsic.get("regular_weight", 0.01)

        self.arcface_scale = self.intrinsic.get("arcface_scale", 0.0)
        self.arcface_margin = self.intrinsic.get("arcface_margin", 0.0)
        self.loss_dict = {}

    def forward(self, output_dict, target_dict, suffix="", val=True):
        # emb_all = output_dict["emb_all"].float()
        emb_all = output_dict["emb_all"]
        lab_all = output_dict["lab_all"]

        m_pred_all = output_dict.get("m_pred_all", None)
        modality_centers = output_dict.get("modality_centers", None)


        loss_supcon = emb_all.new_zeros(())
        loss_cls = emb_all.new_zeros(())
        loss_center = emb_all.new_zeros(())
        loss_arc = emb_all.new_zeros(())
        loss_z = emb_all.new_zeros(())



        # -------- regularization loss term --------
        loss_z = emb_all.pow(2).mean() * self.z_weight


        emb_all = F.normalize(emb_all, dim=1)

        # -------- SupCon (same-modality positives, different-modality negatives) --------
        loss_supcon = self.supcon_infonce(emb_all, lab_all, temp=0.12)
        # loss_supcon = self.supcon_infonce(emb_all, lab_all, temp=0.07) * 5

        if modality_centers is not None:
            # -------- ramp-up weights for center & arc (small -> large) --------
            w_center = self.center_w_sched.step()
            w_arc = self.arc_w_sched.step()

            # -------- cross-batch center pull (shrink intra-modality variance) --------
            loss_center = self.intra_modality_center_pull(emb_all, lab_all, modality_centers) * w_center

            # -------- angular margin (increase inter-modality angular margin) --------
            loss_arc = self.angular_margin_loss(emb_all, lab_all, modality_centers, mode="arcface") * w_arc

        # -------- Sup modality-cls loss --------
        if m_pred_all is not None:
            loss_cls = self.m_cls_loss(m_pred_all, lab_all) * self.cls_weight

        intrinsic_loss = loss_supcon + loss_center + loss_arc + loss_z + loss_cls
        total_loss = intrinsic_loss

        self.loss_dict.update({
            "total_loss": total_loss.item(),
            "intrinsic_loss": intrinsic_loss.item(),
            "contrastive_loss": loss_supcon.item(),
            "m_pred_loss": loss_cls.item(),
            "center_loss": loss_center.item(),
            "arc_loss": loss_arc.item(),
            "z_loss": loss_z.item(),
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
        intrinsic_loss = self.loss_dict.get('intrinsic_loss', 0)
        contrastive_loss = self.loss_dict.get('contrastive_loss', 0)
        m_pred_loss = self.loss_dict.get('m_pred_loss', 0)
        center_loss = self.loss_dict.get('center_loss', 0)
        arc_loss = self.loss_dict.get('arc_loss', 0)
        z_loss = self.loss_dict.get('z_loss', 0)

        # self.loss_dict = {}


        print(
            "[epoch %d][%d/%d] || Loss: %.6f || Intrinsic Loss: %.6f || Contrastive Loss: %.6f || M-Pred Loss: %.6f || Center Loss: %.6f || Arc Loss: %.6f || z Loss: %.6f"
            % (epoch, batch_id + 1, batch_len, total_loss, intrinsic_loss, contrastive_loss, m_pred_loss, center_loss, arc_loss, z_loss)
        )

        if not writer is None:
            writer.add_scalar("intrinsic_loss" + suffix, intrinsic_loss, epoch * batch_len + batch_id)
            writer.add_scalar("contrastive_loss" + suffix, contrastive_loss, epoch * batch_len + batch_id)
            writer.add_scalar("m_pred_loss" + suffix, m_pred_loss, epoch * batch_len + batch_id)
            writer.add_scalar("center_loss" + suffix, center_loss, epoch * batch_len + batch_id)
            writer.add_scalar("arc_loss" + suffix, arc_loss, epoch * batch_len + batch_id)
            writer.add_scalar("z_loss" + suffix, z_loss, epoch * batch_len + batch_id)

        self.center_w_sched.ensure_aligned(epoch, batch_id, batch_len, tolerance=10, for_next_step=True)
        self.arc_w_sched.ensure_aligned(epoch, batch_id, batch_len, tolerance=10, for_next_step=True)




        # ---------- losses ----------
    def supcon_infonce(
            self,
            emb: torch.Tensor,
            labels: torch.Tensor,
            temp: float = 0.07,
            alpha: float = 0.0,
    ) -> torch.Tensor:
        """
        Stable supervised contrastive loss (SupCon / supervised InfoNCE), DDP-aware.

        Args:
            emb:    [B_local, D] L2-normalized embeddings (float16/float32 ok)
            labels: [B_local] integer labels
            temp:   temperature
            alpha:  class-balanced exponent (0 disables).
                    weight_i ∝ (1 / count[label_i])^alpha, normalized by mean.

        Returns:
            scalar loss
        """
        device = emb.device
        B_local = int(emb.size(0))
        if B_local <= 1:
            return emb.new_tensor(0.0, requires_grad=True)

        ddp = dist.is_available() and dist.is_initialized()

        # -------------------- DDP gather (supports uneven last batch) --------------------
        if ddp:
            world = dist.get_world_size()
            rank = dist.get_rank()

            # gather local batch sizes
            size_t = torch.tensor([B_local], dtype=torch.long, device=device)
            size_list = [torch.zeros_like(size_t) for _ in range(world)]
            dist.all_gather(size_list, size_t)
            sizes = [int(x.item()) for x in size_list]
            max_B = max(sizes)
            offset = sum(sizes[:rank])
            B_global = sum(sizes)

            D = int(emb.size(1))

            # pad to max_B so all_gather shapes match
            if B_local < max_B:
                pad_emb = torch.zeros((max_B - B_local, D), dtype=emb.dtype, device=device)
                emb_pad = torch.cat([emb, pad_emb], dim=0)

                pad_lab = torch.full((max_B - B_local,), -1, dtype=labels.dtype, device=device)
                lab_pad = torch.cat([labels, pad_lab], dim=0)
            else:
                emb_pad = emb
                lab_pad = labels

            # all_gather padded tensors
            emb_buf = [torch.empty((max_B, D), dtype=emb.dtype, device=device) for _ in range(world)]
            lab_buf = [torch.empty((max_B,), dtype=labels.dtype, device=device) for _ in range(world)]
            dist.all_gather(emb_buf, emb_pad)
            dist.all_gather(lab_buf, lab_pad)

            # unpad + concat in rank order to form global bank
            emb_all = []
            lab_all = []
            for r in range(world):
                if sizes[r] > 0:
                    emb_all.append(emb_buf[r][:sizes[r]])
                    lab_all.append(lab_buf[r][:sizes[r]])
            emb_all = torch.cat(emb_all, dim=0)    # [B_global, D]
            lab_all = torch.cat(lab_all, dim=0)    # [B_global]
        else:
            offset = 0
            B_global = B_local
            emb_all = emb
            lab_all = labels

        # -------------------- compute logits (anchors: local, bank: global) --------------------
        # Use float32 for stability (especially under AMP)
        emb_local_f = emb.float()
        emb_bank_f = emb_all.detach().float()  # stop-grad bank
        logits = (emb_local_f @ emb_bank_f.t()) / max(float(temp), 1e-6)   # [B_local, B_global]

        # stabilize softmax (logsumexp) per row
        logits = logits - logits.max(dim=1, keepdim=True).values

        # -------------------- masks: positives + self exclusion --------------------
        # positives: same label
        pos_mask = labels.view(B_local, 1).eq(lab_all.view(1, B_global))   # [B_local, B_global]

        # exclude self position in global bank: column = offset + i
        self_idx = torch.arange(B_local, device=device) + int(offset)
        self_mask = torch.zeros_like(pos_mask, dtype=torch.bool)
        self_mask.scatter_(1, self_idx.view(-1, 1), True)

        pos_mask = pos_mask & (~self_mask)  # remove self from positives

        pos_cnt = pos_mask.sum(dim=1)       # [B_local]
        valid = pos_cnt > 0
        if not valid.any():
            return emb.new_tensor(0.0, requires_grad=True)

        # -------------------- standard SupCon: logsumexp(denom) - mean(pos_logits) --------------------
        # denom excludes self by masking to -inf
        logits_den = logits.masked_fill(self_mask, float("-inf"))
        log_denom = torch.logsumexp(logits_den, dim=1)  # [B_local]

        # mean over positives (on logits scale)
        pos_sum = logits.masked_fill(~pos_mask, 0.0).sum(dim=1)            # [B_local]
        pos_mean = pos_sum / pos_cnt.clamp_min(1).to(pos_sum.dtype)        # [B_local]

        loss_i = (log_denom - pos_mean)[valid]                             # [B_valid]

        # -------------------- optional class-balanced anchor weighting --------------------
        if alpha and float(alpha) > 0.0:
            with torch.no_grad():
                # ignore padded labels (-1) in bincount
                lab_valid_all = lab_all[lab_all >= 0]
                n_cls = int(lab_valid_all.max().item()) + 1 if lab_valid_all.numel() > 0 else 0
                counts = torch.bincount(lab_valid_all, minlength=max(n_cls, 1)).float().clamp_min(1.0)
                w = (1.0 / counts[labels[valid]].float()).pow(float(alpha))
                w = w / w.mean().clamp_min(1e-12)
            loss = (loss_i * w).mean()
        else:
            loss = loss_i.mean()

        return loss





    def intra_modality_center_pull(self, emb: torch.Tensor, labels: torch.Tensor, modality_centers) -> torch.Tensor:
        """
        Pull each embedding towards its modality's *global* center (cross-batch).
        emb    : [N, D], L2-normalized
        labels : [N],    global class ids [0..M-1]
        """
        if emb.numel() == 0:
            return emb.new_tensor(0.0, requires_grad=True)
        centers = modality_centers[labels]          # [N, D]
        loss = ((emb - centers) ** 2).sum(dim=1).mean()  # on unit sphere
        return loss

    def angular_margin_loss(self, emb: torch.Tensor, labels: torch.Tensor, modality_centers, mode: str = "arcface") -> torch.Tensor:
        """
        ArcFace/CosFace with global centers as classifier weights (no grad to centers).
        emb    : [N, D] (L2-normalized)
        labels : [N]
        """
        centers = F.normalize(modality_centers, p=2, dim=1)  # freeze centers for CE
        cos = emb @ centers.t()                                            # [N, M]
        s = self.arcface_scale
        m = self.arcface_margin

        if mode == "cosface":
            logits = cos.clone()
            logits[torch.arange(cos.size(0)), labels] = logits[torch.arange(cos.size(0)), labels] - m
            logits = s * logits
        else:  # arcface
            cm = torch.cos(torch.tensor(m, device=emb.device))
            sm = torch.sin(torch.tensor(m, device=emb.device))
            t = cos[torch.arange(cos.size(0)), labels].clamp(-1 + 1e-7, 1 - 1e-7)
            sin_t = torch.sqrt(1.0 - t * t)
            cos_mt = t * cm - sin_t * sm
            logits = cos.clone()
            logits[torch.arange(cos.size(0)), labels] = cos_mt
            logits = s * logits

        return F.cross_entropy(logits, labels)




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