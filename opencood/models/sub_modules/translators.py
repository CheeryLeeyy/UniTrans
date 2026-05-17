

# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange



def init_linear_(m: nn.Module, mode: str = "xavier_uniform", bias: float = 0.0) -> None:
    if not isinstance(m, nn.Linear):
        return
    if mode == "xavier_uniform":
        nn.init.xavier_uniform_(m.weight)
    elif mode == "xavier_normal":
        nn.init.xavier_normal_(m.weight)
    elif mode == "kaiming_uniform":
        nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
    elif mode == "kaiming_normal":
        nn.init.kaiming_normal_(m.weight, a=math.sqrt(5))
    else:
        raise ValueError(f"Unknown init mode: {mode}")
    if m.bias is not None:
        nn.init.constant_(m.bias, bias)

def init_conv2d_(m: nn.Module, mode: str = "kaiming_uniform", bias: float = 0.0, a: float = None,) -> None:
    if not isinstance(m, nn.Conv2d):
        return

    if a is None:
        a = math.sqrt(5)

    if mode == "kaiming_uniform":
        nn.init.kaiming_uniform_(m.weight, a=a)
    elif mode == "kaiming_normal":
        nn.init.kaiming_normal_(m.weight, a=a)
    elif mode == "xavier_uniform":
        nn.init.xavier_uniform_(m.weight)
    elif mode == "xavier_normal":
        nn.init.xavier_normal_(m.weight)
    else:
        raise ValueError(f"Unknown init mode: {mode}")

    if m.bias is not None:
        nn.init.constant_(m.bias, bias)


def init_layernorm_(m: nn.Module, weight: float = 1.0, bias: float = 0.0) -> None:
    if not isinstance(m, nn.LayerNorm):
        return
    if m.elementwise_affine:
        nn.init.constant_(m.weight, weight)
        nn.init.constant_(m.bias, bias)


def init_linear_normal_(m: nn.Module, std: float = 0.02, bias: float = 0.0) -> None:
    """Small normal init for stability (common in Transformer-style models)."""
    if not isinstance(m, nn.Linear):
        return
    nn.init.normal_(m.weight, mean=0.0, std=std)
    if m.bias is not None:
        nn.init.constant_(m.bias, bias)


def zero_linear_(m: nn.Module) -> None:
    if not isinstance(m, nn.Linear):
        return
    nn.init.zeros_(m.weight)
    if m.bias is not None:
        nn.init.zeros_(m.bias)




# -------------------------
# ConvNeXt-like lightweight block
# -------------------------
class ConvNeXtLiteBlock(nn.Module):
    def __init__(self, dim: int, drop: float = 0.0):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.ln = nn.LayerNorm(dim, eps=1e-6)
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4 * dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        residual = x
        x = self.dw(x)
        x = x.permute(0, 2, 3, 1)  # -> [B, H, W, C]
        x = self.ln(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = x.permute(0, 3, 1, 2)  # -> [B, C, H, W]
        return x + residual


# -------------------------
# Lightweight conv projector branch (global response)
# -------------------------
class ConvProj(nn.Module):
    def __init__(self, in_ch: int = 64, hid_ch: int = 128, drop: float = 0.1):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_ch, hid_ch, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            ConvNeXtLiteBlock(hid_ch, drop),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(hid_ch, hid_ch, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            ConvNeXtLiteBlock(hid_ch, drop),
        )
        self.block = ConvNeXtLiteBlock(hid_ch, drop)
        self.pool = nn.AdaptiveAvgPool2d(1)  # -> [B, hid_ch, 1, 1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)  # H/2, W/2
        x = self.stage2(x)  # H/4, W/4
        x = self.block(x)
        x = self.pool(x).flatten(1)  # [B, hid_ch]
        return x


# -------------------------
# Intrinsic modal/style encoder
# Input feature: [B, 64, 128, 256]
# Output embedding: [B, D] (L2-normalized)
# -------------------------
class IntrinsicModalEncoder(nn.Module):
    def __init__(self, args):
        super(IntrinsicModalEncoder, self).__init__()

        in_ch = args.get("in_ch", 64)
        embed_dim = args.get("embed_dim", 64)
        gram_dim = args.get("gram_dim", 128)
        dropout = args.get("dropout", 0.1)
        gram_down = args.get("gram_down", 4)  # spatial downsample factor before Gram

        # Global response branch
        self.conv_proj = ConvProj(in_ch=in_ch, hid_ch=128, drop=dropout)

        self.conv_proj2 = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256)
        )

        # Gram branch: downsample then compute channel Gram
        self.gram_pool = nn.AvgPool2d(kernel_size=gram_down, stride=gram_down)
        tri_size = in_ch * (in_ch + 1) // 2  # upper-triangular size of CxC
        self.gram_fc = nn.Sequential(
            nn.Linear(tri_size, gram_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(gram_dim, gram_dim),
            nn.ReLU(inplace=True),
        )

        # Fusion projector
        fusion_in = in_ch * 2 + 256 + 128 + gram_dim  # mu(C) + sigma(C) + pool(128) + gram(gram_dim)
        self.projector = nn.Sequential(
            nn.Linear(fusion_in, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, embed_dim),
        )

    @staticmethod
    def _tri_upper_flat(mat: torch.Tensor) -> torch.Tensor:
        """Flatten upper triangular (including diagonal) of [B, C, C] -> [B, C*(C+1)/2]."""
        _, c, _ = mat.shape
        idx = torch.triu_indices(c, c, offset=0, device=mat.device)
        return mat[:, idx[0], idx[1]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 64, 128, 256]
        return: L2-normalized embedding [B, D]
        """
        b, c, h, w = x.shape

        # 1) First-order stats (scene/view-invariant)
        mu = x.mean(dim=(2, 3))  # [B, C]
        sigma = x.std(dim=(2, 3), unbiased=False)  # [B, C]

        # 2) Global response branch
        channel_feat = self.conv_proj(x)  # [B, 128]

        channel_mu = x.mean(dim=(1), keepdim=True)
        channel_sigma = x.std(dim=(1), unbiased=False, keepdim=True)
        channel_max = x.max(dim=(1), keepdim=True)[0]
        channel_min = x.min(dim=(1), keepdim=True)[0]

        y = torch.cat([channel_mu, channel_sigma, channel_max, channel_min], dim=1)
        spatial_feat = self.conv_proj2(y)

        # 3) Gram branch (second-order channel correlation)
        xs = self.gram_pool(x)  # [B, C, H', W']
        b2, c2, hs, ws = xs.shape
        xs = xs.view(b2, c2, hs * ws)  # [B, C, N]
        gram = torch.bmm(xs, xs.transpose(1, 2)) / (hs * ws + 1e-6)  # [B, C, C]
        gram_ut = self._tri_upper_flat(gram)  # [B, C*(C+1)/2]
        gram_vec = self.gram_fc(gram_ut)  # [B, gram_dim]

        # 4) Fusion -> embedding
        feat = torch.cat([mu, sigma, channel_feat, spatial_feat, gram_vec], dim=1)  # [B, fusion_in]
        d = self.projector(feat)  # [B, D]
        # d = F.normalize(d, p=2, dim=1, eps=1e-6)  # unit-norm for cosine similarity
        return d







######################################################################################################################################################





# ============ Plain FeedForward ============

class PlainFeedForward(nn.Module):
    """Token-wise MLP used inside each expert ."""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=True)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=True)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, M, C]
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


# ============ Plain Cross-Attention over 2D windows ============

class PlainCrossAttention2D(nn.Module):
    """
    Standard multi-head cross-attention over 2D windows.
    Q from neb tokens, K/V from ego tokens.
    All projections are standard Linear.
    """
    def __init__(self, dim, n_heads, dim_head, dropout=0.0):
        super().__init__()
        assert dim == n_heads * dim_head
        self.n_heads = n_heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, dim, bias=True)
        self.to_k = nn.Linear(dim, dim, bias=True)
        self.to_v = nn.Linear(dim, dim, bias=True)
        self.to_o = nn.Linear(dim, dim, bias=True)

        self.dropout = nn.Dropout(dropout)

    def forward(self, neb_tokens, ego_tokens):
        """
        neb_tokens, ego_tokens: [B, nW, N, C]
        return: [B, nW, N, C]
        """
        B, nW, N, C = neb_tokens.shape
        H, Dh = self.n_heads, self.dim_head

        # Project
        q = self.to_q(neb_tokens)  # [B,nW,N,C]
        k = self.to_k(ego_tokens)
        v = self.to_v(ego_tokens)

        # Split heads
        q = rearrange(q, 'b nw n (h d) -> b h nw n d', h=H, d=Dh)
        k = rearrange(k, 'b nw n (h d) -> b h nw n d', h=H, d=Dh)
        v = rearrange(v, 'b nw n (h d) -> b h nw n d', h=H, d=Dh)

        # Attention
        attn = torch.einsum('b h w i d, b h w j d -> b h w i j',
                            q * self.scale, k)          # [B,H,nW,N,N]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum('b h w i j, b h w j d -> b h w i d', attn, v)
        out = rearrange(out, 'b h w n d -> b w n (h d)')  # [B,nW,N,C]
        out = self.to_o(out)
        return out


# --------- Helpers for windows ---------
def window_partition(x, win):
    """[B,C,H,W] -> [B, nW, N, C] where N=win*win"""
    B, C, H, W = x.shape
    assert H % win == 0 and W % win == 0
    x = rearrange(x, 'b c (x w1) (y w2) -> b x y w1 w2 c', w1=win, w2=win)
    tokens = rearrange(x, 'b x y w1 w2 c -> b (x y) (w1 w2) c')
    return tokens


def window_merge(tokens, win, H, W):
    """[B, nW, N, C] -> [B, C, H, W]"""
    B, nW, N, C = tokens.shape
    X = H // win
    Y = W // win
    x = rearrange(tokens, 'b (x y) (w1 w2) c -> b x y w1 w2 c', x=X, y=Y, w1=win, w2=win)
    x = rearrange(x, 'b x y w1 w2 c -> b c (x w1) (y w2)')
    return x


def swap_grid(tokens, win, H, W):
    """MaxViT-style swapped windows: local<->grid repartition."""
    x = window_merge(tokens, win, H, W)
    x = rearrange(x, 'b c (w1 x) (w2 y) -> b x y w1 w2 c', w1=win, w2=win)
    tokens_swapped = rearrange(x, 'b x y w1 w2 c -> b (x y) (w1 w2) c')
    return tokens_swapped





def to_grid_tokens(local_tokens, win, H, W):
    """
    local_tokens: [B, X*Y, win*win, C]
    return: [B, win*win, X*Y, C]  (grid partition)
    """
    feat = window_merge(local_tokens, win, H, W)  # [B, C, H, W]
    x = rearrange(feat, 'b c (w1 x) (w2 y) -> b x y w1 w2 c', w1=win, w2=win)  # swap
    grid_tokens = rearrange(x, 'b x y w1 w2 c -> b (w1 w2) (x y) c')           # [B, win^2, X*Y, C]
    return grid_tokens


def to_local_tokens(grid_tokens, win, H, W):
    """
    grid_tokens: [B, win*win, X*Y, C]
    return: [B, X*Y, win*win, C]  (back to local windows)
    """
    X, Y = H // win, W // win
    x = rearrange(grid_tokens, 'b (w1 w2) (x y) c -> b x y w1 w2 c',
                  w1=win, w2=win, x=X, y=Y)
    feat = rearrange(x, 'b x y w1 w2 c -> b c (w1 x) (w2 y)')  # inverse swap back to [B,C,H,W]
    local_tokens = window_partition(feat, win)                 # [B, X*Y, win*win, C]
    return local_tokens





# def to_grid_tokens(local_tokens: torch.Tensor, win: int, H: int, W: int) -> torch.Tensor:
#     """
#     local_tokens: [B, X*Y, win*win, C]
#     return grid_tokens: [B, win*win, X*Y, C]
#     """
#     B, nW, N, C = local_tokens.shape
#     X, Y = H // win, W // win
#     assert nW == X * Y and N == win * win
#
#     x = rearrange(local_tokens, 'b (x y) (w1 w2) c -> b x y w1 w2 c', x=X, y=Y, w1=win, w2=win)
#     x = x.permute(0, 3, 4, 1, 2, 5).contiguous()  # b w1 w2 x y c
#     grid_tokens = rearrange(x, 'b w1 w2 x y c -> b (w1 w2) (x y) c')
#     return grid_tokens
#
#
# def to_local_tokens(grid_tokens: torch.Tensor, win: int, H: int, W: int) -> torch.Tensor:
#     """
#     grid_tokens: [B, win*win, X*Y, C]
#     return local_tokens: [B, X*Y, win*win, C]
#     """
#     B, nWg, Ng, C = grid_tokens.shape
#     X, Y = H // win, W // win
#     assert nWg == win * win and Ng == X * Y
#
#     x = rearrange(grid_tokens, 'b (w1 w2) (x y) c -> b w1 w2 x y c', w1=win, w2=win, x=X, y=Y)
#     x = x.permute(0, 3, 4, 1, 2, 5).contiguous()  # b x y w1 w2 c
#     local_tokens = rearrange(x, 'b x y w1 w2 c -> b (x y) (w1 w2) c')
#     return local_tokens



# --------- One expert layer: local + global cross-attention ---------
class HeteroExpertLayer(nn.Module):
    """
    One expert layer with:
      - Local window cross-attention
      - Global (grid) cross-attention (MaxViT-style)
      - Pre-LN + residual + plain FFN
    """

    def __init__(self, in_dim, n_heads, dim_head, window_size,
                 mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        assert in_dim == n_heads * dim_head
        self.window = window_size
        hidden = int(in_dim * mlp_ratio)

        # Local
        self.local_attn_ln = nn.LayerNorm(in_dim)
        self.local_attn = PlainCrossAttention2D(in_dim, n_heads, dim_head, dropout)
        self.local_ffn_ln = nn.LayerNorm(in_dim)
        self.local_ffn = PlainFeedForward(in_dim, hidden, dropout)

        # Global (grid)
        self.global_attn_ln = nn.LayerNorm(in_dim)
        self.global_attn = PlainCrossAttention2D(in_dim, n_heads, dim_head, dropout)
        self.global_ffn_ln = nn.LayerNorm(in_dim)
        self.global_ffn = PlainFeedForward(in_dim, hidden, dropout)

    def forward(self, ego_feat, neb_feat):
        """
        ego_feat, neb_feat: [B,C,H,W]
        return: updated neb_feat in ego domain: [B,C,H,W]
        """
        B, C, H, W = neb_feat.shape
        win = self.window

        # ---- Local window tokens ----
        neb_tokens = window_partition(neb_feat, win)  # [B, X*Y, win*win, C]
        ego_tokens = window_partition(ego_feat, win)  # [B, X*Y, win*win, C]

        # ---- Local cross-attention ----
        x = self.local_attn_ln(neb_tokens)
        neb_tokens = neb_tokens + self.local_attn(x, ego_tokens)

        y = self.local_ffn_ln(neb_tokens)
        neb_tokens = neb_tokens + self.local_ffn(y)

        # ---- Global cross-attention (grid tokens) ----
        neb_grid = to_grid_tokens(neb_tokens, win, H, W)  # [B, win*win, X*Y, C]
        ego_grid = to_grid_tokens(ego_tokens, win, H, W)  # [B, win*win, X*Y, C]

        xg = self.global_attn_ln(neb_grid)
        neb_grid = neb_grid + self.global_attn(xg, ego_grid)

        yg = self.global_ffn_ln(neb_grid)
        neb_grid = neb_grid + self.global_ffn(yg)

        # ---- Back to local tokens then merge ----
        neb_tokens = to_local_tokens(neb_grid, win, H, W)  # [B, X*Y, win*win, C]
        neb_feat_out = window_merge(neb_tokens, win, H, W) # [B, C, H, W]
        return neb_feat_out


class HeteroExpertNet(nn.Module):
    """
    One *independent* expert network:
      - Stack of `depth` HeteroExpertLayer
      - All parameters are private to this expert
    """

    def __init__(self, in_dim, n_heads, dim_head, window_size,
                 depth=2, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            HeteroExpertLayer(in_dim, n_heads, dim_head, window_size,
                              mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(depth)
        ])

    def forward(self, ego_feat, neb_feat):
        x = neb_feat
        for layer in self.layers:
            x = layer(ego_feat, x)
        return x



class ClassicMoEFeatureConverter(nn.Module):
    """
    Classic block-level MoE converter:
      - Train n_experts *independent* expert networks (HeteroExpertNet).
      - A router (CodeConditioner) produces per-sample gates α[b, e].
      - Optionally apply top-k sparse gating.
      - Final output is the weighted sum of expert outputs.

    Interface is kept consistent with HeteroFeatureConverter:
      forward(ego_feat, neb_feat, ego_code, neb_code, return_gates=False)
    """

    def __init__(self, args):
        super().__init__()
        depth = args.get("depth", 2)          # depth *inside each expert*
        in_dim = args.get("in_dim", 64)
        code_dim = args.get("code_dim", 64)
        n_experts = args.get("n_experts", 8)
        n_heads = args.get("n_heads", 8)
        window_size = args.get("window_size", 8)
        mlp_ratio = args.get("mlp_ratio", 4.0)
        dropout = args.get("dropout", 0.1)

        # MoE-specific
        self.top_k = args.get("top_k", None)  # if None -> use all experts
        router_embed_dim = args.get("router_embed_dim", 256)

        assert in_dim % n_heads == 0
        dim_head = in_dim // n_heads
        self.n_experts = n_experts

        # Router: reuse CodeConditioner but only use alpha / alpha_logits
        self.router = CodeConditioner(
            code_dim=code_dim,
            n_experts=n_experts,
            d_model=None,                 # no AdaLN needed here
            embed_dim=router_embed_dim,
            tau_min=0.5,
            tau_max=2.0,
            noise_scale=args.get("router_noise_scale", 0.1),
            use_adaLN_modulation=False
        )

        # Independent expert networks
        self.experts = nn.ModuleList([
            HeteroExpertNet(
                in_dim=in_dim,
                n_heads=n_heads,
                dim_head=dim_head,
                window_size=window_size,
                depth=depth,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(n_experts)
        ])

        # Final head: simple token-wise MLP (no MoE here)
        self.final_head = FinalMLP(in_dim, mlp_ratio=4.0, dropout=dropout)

    def forward(self, ego_feat, neb_feat, ego_code, neb_code, return_alpha=False):
        """
        ego_feat, neb_feat: [B,C,H,W]
        ego_code, neb_code: [B,L]
        """
        B, C, H, W = neb_feat.shape

        # 1) Routing: get per-sample gates α[b,e]
        router_out = self.router(ego_code, neb_code)
        gates_full = router_out["alpha"]  # [B, E]  (softmax over experts)
        logits = router_out["alpha_logits"].unsqueeze(0)  # [B, E]

        # --------- Case 1: no top-k (dense MoE) -> keep原行为 ----------
        if self.top_k is None or self.top_k >= self.n_experts:
            # Run all experts once on full batch
            expert_outs = []
            for e, expert in enumerate(self.experts):
                out_e = expert(ego_feat, neb_feat)  # [B,C,H,W]
                expert_outs.append(out_e)
            expert_outs = torch.stack(expert_outs, dim=1)  # [B,E,C,H,W]

            # Weighted sum by gates α
            gates_reshaped = gates_full.view(B, self.n_experts, 1, 1, 1)  # [B,E,1,1,1]
            mixed = (expert_outs * gates_reshaped).sum(dim=1)  # [B,C,H,W]

            # Final projection head (shared, non-MoE)
            out = self.final_head(mixed)

            if return_alpha:
                return out,  logits
            return out

        # --------- Case 2: top-k 稀疏 MoE，只计算 top-k 专家 ----------
        k = self.top_k

        # top-k experts per sample
        topk_vals, topk_idx = torch.topk(gates_full, k, dim=-1)  # [B,k], [B,k]
        # renormalize so that sum_e α[b,e] = 1 over selected experts
        weights = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + 1e-9)  # [B,k]

        # 构造稀疏 gate 矩阵（便于返回观察）：非 top-k 的位置为 0
        gates_sparse = gates_full.new_zeros(gates_full.shape)  # [B,E]
        gates_sparse.scatter_(1, topk_idx, weights)

        # 对每个样本单独只跑它的 top-k 专家
        y_list = []  # collect per-sample outputs

        for b in range(B):
            ego_b = ego_feat[b:b + 1]  # [1,C,H,W]
            neb_b = neb_feat[b:b + 1]  # [1,C,H,W]
            y_b = neb_b.new_zeros(1, C, H, W)

            idx_b = topk_idx[b]  # [k]
            w_b = weights[b]  # [k]

            # only selected experts for this sample
            for j in range(k):
                e_idx = idx_b[j].item()
                out_e = self.experts[e_idx](ego_b, neb_b)  # [1,C,H,W]
                y_b = y_b + w_b[j] * out_e

            y_list.append(y_b)

        # 拼回 batch 维度
        mixed = torch.cat(y_list, dim=0)  # [B,C,H,W]

        # Final projection head
        out = self.final_head(mixed)

        if return_alpha:
            return out, logits
        return out

    def conditioner_fix(self, requires_grad=False):
        def params_fix(blk, requires_grad):
            for p in blk.parameters():
                p.requires_grad_(requires_grad)

        params_fix(self.router, requires_grad)

    def conditioner_apply(self, func):
        self.router.apply(func)

    def set_conditioner_eval(self):

        self.router.eval()

    def get_last_alpha_entropy(self):
        """Expose router entropy (for regularization / schedule)."""
        return self.router.last_alpha_entropy



#############################################################################################################################
class LinearMoEFFN(nn.Module):
    """
    MoE FFN over token dimension:
      - Routed experts (PlainFeedForward) mixed by per-sample gates
      - Optional shared experts that are always evaluated and added to output

    x:     [B, M, C]
    gates: [B, E] where E = n_experts (routed experts only)
    """

    def __init__(self, dim, hidden_dim, n_experts, dropout=0.0, top_k=None, n_shared_experts: int = 0):
        super().__init__()
        self.n_experts = int(n_experts)
        self.top_k = top_k

        # routed experts
        self.experts = nn.ModuleList([
            PlainFeedForward(dim, hidden_dim, dropout)
            for _ in range(self.n_experts)
        ])

        # shared experts (always active)
        self.n_shared_experts = int(n_shared_experts)
        self.shared_experts = nn.ModuleList([
            PlainFeedForward(dim, hidden_dim, dropout)
            for _ in range(self.n_shared_experts)
        ])

    def _shared_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, M, C]
        return: [B, M, C]
        """
        if self.n_shared_experts <= 0:
            return x.new_zeros(x.shape)

        # Sum or average are both acceptable; average keeps magnitude stable
        y = x.new_zeros(x.shape)
        for ff in self.shared_experts:
            y = y + ff(x)
        y = y / float(self.n_shared_experts)
        return y

    def forward(self, x, gates):
        """
        x:     [B, M, C]
        gates: [B, E]
        return: [B, M, C]
        """
        B, M, C = x.shape
        E = self.n_experts
        assert gates.shape == (B, E), f"gates shape {gates.shape} != (B, E={E})"

        # Shared branch (always computed)
        y_shared = self._shared_forward(x)  # [B, M, C]

        # Dense case: no top-k or top_k >= E -> compute all experts once
        if self.top_k is None or self.top_k >= E:
            y = x.new_zeros(B, M, C)
            for e, ff in enumerate(self.experts):
                out_e = ff(x)  # [B, M, C]
                coef = gates[:, e].view(B, 1, 1)  # [B,1,1]
                y = y + out_e * coef
            return y + y_shared

        # Sparse top-k: loop over batch, only evaluate selected experts for each sample
        k = self.top_k
        y_list = []

        for b in range(B):
            g_b = gates[b]  # [E]
            topk_vals, topk_idx = torch.topk(g_b, k)  # [k], [k]
            w_b = topk_vals / (topk_vals.sum() + 1e-9)  # [k]

            x_b = x[b:b+1]  # [1, M, C]
            y_b = x_b.new_zeros(1, M, C)

            for j in range(k):
                e = topk_idx[j].item()
                out_e = self.experts[e](x_b)  # [1, M, C]
                y_b = y_b + w_b[j] * out_e

            # add shared experts for this sample
            if self.n_shared_experts > 0:
                y_b = y_b + self._shared_forward(x_b)

            y_list.append(y_b)

        y = torch.cat(y_list, dim=0)  # [B, M, C]
        return y


class LinearMoETransformerBlock(nn.Module):
    """
    One Transformer block with:
      - local window cross-attention
      - global (grid) cross-attention (MaxViT-style)
      - MoE FFN (LinearMoEFFN) using CodeConditioner as router

    Router produces per-sample gates g[b,e], shared by local/global FFN.
    """

    def __init__(self, in_dim, code_dim, n_experts, n_heads, dim_head, window_size,
                 mlp_ratio=2.0, dropout=0.1, top_k=None, router_embed_dim=256,
                 router_noise_scale=0.1, n_shared_experts: int = 0):
        super().__init__()
        assert in_dim == n_heads * dim_head
        self.window = window_size
        hidden = int(in_dim * mlp_ratio)

        # Router: CodeConditioner, only use alpha / alpha_logits
        self.router = CodeConditioner(
            code_dim=code_dim,
            n_experts=n_experts,
            d_model=None,
            embed_dim=router_embed_dim,
            tau_min=0.5,
            tau_max=2.0,
            noise_scale=router_noise_scale,
            use_adaLN_modulation=False
        )

        # Local attention + MoE FFN
        self.local_attn_ln = nn.LayerNorm(in_dim)
        self.local_attn = PlainCrossAttention2D(in_dim, n_heads, dim_head, dropout)
        self.local_ffn_ln = nn.LayerNorm(in_dim)
        self.local_ffn = LinearMoEFFN(
            in_dim, hidden, n_experts, dropout,
            top_k=top_k, n_shared_experts=n_shared_experts
        )

        # Global (grid) attention + MoE FFN
        self.global_attn_ln = nn.LayerNorm(in_dim)
        self.global_attn = PlainCrossAttention2D(in_dim, n_heads, dim_head, dropout)
        self.global_ffn_ln = nn.LayerNorm(in_dim)
        self.global_ffn = LinearMoEFFN(
            in_dim, hidden, n_experts, dropout,
            top_k=top_k, n_shared_experts=n_shared_experts
        )

    @staticmethod
    def _apply_moe_ffn(ffn, x_tokens, gates):
        """
        Apply MoE FFN on token tensor with shape [B, nW, N, C] by flattening tokens.

        x_tokens: [B, nW, N, C]
        gates:    [B, E]
        return:   [B, nW, N, C]
        """
        B, nW, N, C = x_tokens.shape
        x_flat = rearrange(x_tokens, 'b nw n c -> b (nw n) c')     # [B, M, C]
        x_flat = ffn(x_flat, gates)                                # [B, M, C]
        x_out = rearrange(x_flat, 'b (nw n) c -> b nw n c', nw=nW, n=N)
        return x_out

    def forward(self, ego_feat, neb_feat, ego_code, neb_code, return_gates=False):
        """
        ego_feat, neb_feat: [B,C,H,W]
        ego_code, neb_code: [B,L]
        return:
          neb_feat_out: [B,C,H,W]
          optional gate_info: {"gates": [B,E], "logits": [B,E]}
        """
        B, C, H, W = neb_feat.shape
        win = self.window

        # 1) Router: per-sample gates
        router_out = self.router(ego_code, neb_code)
        gates = router_out["alpha"]          # [B,E]
        logits = router_out["alpha_logits"]  # [B,E]

        # 2) Local window tokens: [B, X*Y, win*win, C]
        neb_tokens = window_partition(neb_feat, win)
        ego_tokens = window_partition(ego_feat, win)

        # -------- Local cross-attention + MoE FFN --------
        x = self.local_attn_ln(neb_tokens)
        neb_tokens = neb_tokens + self.local_attn(x, ego_tokens)                 # residual

        y = self.local_ffn_ln(neb_tokens)
        neb_tokens = neb_tokens + self._apply_moe_ffn(self.local_ffn, y, gates)  # residual

        # -------- Global (grid) cross-attention + MoE FFN --------
        # Convert to grid tokens: [B, win*win, X*Y, C]
        neb_grid = to_grid_tokens(neb_tokens, win, H, W)
        ego_grid = to_grid_tokens(ego_tokens, win, H, W)

        xg = self.global_attn_ln(neb_grid)
        neb_grid = neb_grid + self.global_attn(xg, ego_grid)                     # residual

        yg = self.global_ffn_ln(neb_grid)
        neb_grid = neb_grid + self._apply_moe_ffn(self.global_ffn, yg, gates)    # residual

        # Back to local then merge
        neb_tokens = to_local_tokens(neb_grid, win, H, W)                         # [B, X*Y, win*win, C]
        neb_feat_out = window_merge(neb_tokens, win, H, W)                        # [B, C, H, W]

        if return_gates:
            return neb_feat_out, {"gates": gates, "logits": logits}
        return neb_feat_out


class LinearMoEFeatureConverter(nn.Module):
    """
    Feature converter with:
      - shared cross-attention in each block
      - MoE FFN (LinearMoEFFN) per block
      - shared FinalMLP as tail

    Interface:
      forward(ego_feat, neb_feat, ego_code, neb_code, return_gates=False)
    """

    def __init__(self, args):
        super().__init__()
        depth = args.get("depth", 2)
        in_dim = args.get("in_dim", 64)
        code_dim = args.get("code_dim", 64)
        n_experts = args.get("n_experts", 8)
        n_shared_experts = int(args.get("n_shared_experts", 0))  # default 0 for backward compatibility
        n_heads = args.get("n_heads", 8)
        window_size = args.get("window_size", 8)
        mlp_ratio = args.get("mlp_ratio", 4.0)
        dropout = args.get("dropout", 0.1)

        top_k = args.get("top_k", None)
        router_embed_dim = args.get("router_embed_dim", 256)
        router_noise_scale = args.get("router_noise_scale", 0.1)

        # init options
        init_cfg = args.get("init_cfg", {})
        self.init_std = float(init_cfg.get("init_std", 0.02))              # default for Linear
        self.ffn_w_init = init_cfg.get("ffn_w_init", "xavier")             # xavier | kaiming
        self.router_last_zero = bool(init_cfg.get("router_last_zero", False))

        assert in_dim % n_heads == 0
        dim_head = in_dim // n_heads

        self.blocks = nn.ModuleList([
            LinearMoETransformerBlock(
                in_dim=in_dim,
                code_dim=code_dim,
                n_experts=n_experts,
                n_shared_experts=n_shared_experts,
                n_heads=n_heads,
                dim_head=dim_head,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                top_k=top_k,
                router_embed_dim=router_embed_dim,
                router_noise_scale=router_noise_scale,
            )
            for _ in range(depth)
        ])

        # Final shared MLP head (no MoE here)
        self.final_head = FinalMLP(in_dim, mlp_ratio=4.0, dropout=dropout)

        # Apply parameter initialization
        self.init_parameters()

    @torch.no_grad()
    def init_parameters(self):
        """
        Initialize parameters for LinearMoEFeatureConverter.

        Principles:
          1) LayerNorm: weight=1, bias=0.
          2) Attention projections (PlainCrossAttention2D): small normal init (std=self.init_std).
          3) Final MLP head: small normal init (std=self.init_std) for stability.
          4) MoE FFN experts: Xavier (default) or Kaiming for FFN Linear layers.
          5) Router:
             - trunk: small normal init (std=self.init_std)
             - head_alpha: optionally zero-init to start from uniform gates.
        """

        # 1) LayerNorm everywhere
        self.apply(init_layernorm_)

        # Helper: choose FFN init mode
        if self.ffn_w_init.lower() == "kaiming":
            ffn_init = lambda m: init_linear_(m, mode="kaiming_uniform", bias=0.0)
        else:
            # default "xavier"
            ffn_init = lambda m: init_linear_(m, mode="xavier_uniform", bias=0.0)

        # 2) Final head (plain MLP): small normal
        self.final_head.apply(lambda m: init_linear_normal_(m, std=self.init_std, bias=0.0))

        # 3) Each block
        for blk in self.blocks:
            # ---- attention projections (PlainCrossAttention2D uses nn.Linear) ----
            blk.local_attn.apply(lambda m: init_linear_normal_(m, std=self.init_std, bias=0.0))
            blk.global_attn.apply(lambda m: init_linear_normal_(m, std=self.init_std, bias=0.0))

            # ---- MoE FFN (LinearMoEFFN): init all experts' Linear with FFN init ----
            # LinearMoEFFN contains PlainFeedForward experts (Linear layers).
            blk.local_ffn.apply(ffn_init)
            blk.global_ffn.apply(ffn_init)

            # ---- router trunk: small normal ----
            # CodeConditioner.trunk is an nn.Sequential of Linear/ReLU layers.
            if hasattr(blk, "router") and hasattr(blk.router, "trunk"):
                blk.router.trunk.apply(lambda m: init_linear_normal_(m, std=self.init_std, bias=0.0))

            # ---- router head_alpha: zero-init (uniform gates) or small normal ----
            if hasattr(blk, "router") and hasattr(blk.router, "head_alpha") and isinstance(blk.router.head_alpha, nn.Linear):
                if self.router_last_zero:
                    zero_linear_(blk.router.head_alpha)
                else:
                    init_linear_normal_(blk.router.head_alpha, std=self.init_std, bias=0.0)

            print("init", blk.named_children())




    def forward(self, ego_feat, neb_feat, ego_code, neb_code, return_alpha=False):
        """
        ego_feat / neb_feat: [B,C,H,W]
        ego_code / neb_code: [B,L]
        """
        x = neb_feat
        gates_per_layer = []
        return_gates = return_alpha

        for blk in self.blocks:
            if return_gates:
                x, gate_info = blk(ego_feat, x, ego_code, neb_code, return_gates=True)
                gates_per_layer.append(gate_info["logits"])  # [B,E]
            else:
                x = blk(ego_feat, x, ego_code, neb_code, return_gates=False)

        x = self.final_head(x)  # [B,C,H,W]

        if return_gates:
            # [depth, B, E]
            gates_stack = torch.stack(gates_per_layer, dim=0)
            return x, gates_stack
        return x

    def conditioner_fix(self, requires_grad=False):
        def params_fix(blk, requires_grad):
            for p in blk.parameters():
                p.requires_grad_(requires_grad)

        for blk in self.blocks:
            params_fix(blk.router, requires_grad)


    def conditioner_apply(self, func):
        for blk in self.blocks:
            blk.router.apply(func)


    def set_conditioner_eval(self):
        for blk in self.blocks:
            blk.router.eval()


    def get_last_alpha_entropy(self):
        """Average router entropy over all blocks."""
        ents = [blk.last_alpha_entropy for blk in self.blocks]
        return torch.stack(ents).mean()






######################################################################################################################################################


# --------- MCT Linear ---------
class SoEExpertLinear(nn.Module):
    """Per-sample MCT (Mapping-Conditioned Translator) linear: y[b] = x[b] @ W(α_b)^T + b(α_b)"""

    def __init__(self, in_features, out_features, n_experts, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_experts = n_experts
        self.weight_S = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_E = nn.Parameter(torch.zeros(n_experts, out_features, in_features))
        self.bias_S = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.bias_E = nn.Parameter(torch.zeros(n_experts, out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight_S, a=math.sqrt(5))
        if bias:
            bound = 1 / math.sqrt(in_features)
            nn.init.uniform_(self.bias_S, -bound, bound)

    def forward(self, x, alpha):
        """
        x: [B, ..., I], alpha: [B, E]  (per-sample)
        return: [B, ..., O]
        """
        B, I, O = x.shape[0], self.in_features, self.out_features
        x_flat = x.view(B, -1, I)  # [B,M,I]
        # W_eff[b] = W_S + sum_e α[b,e] * W_E[e]
        W_eff = self.weight_S.unsqueeze(0) + torch.einsum('be,eoi->boi', alpha, self.weight_E)  # [B,O,I]
        y = torch.einsum('bmi,boi->bmo', x_flat, W_eff)  # [B,M,O]
        if self.bias_S is not None:
            b_eff = self.bias_S.unsqueeze(0) + torch.einsum('be,eo->bo', alpha, self.bias_E)  # [B,O]
            y = y + b_eff.unsqueeze(1)
        return y.view(*x.shape[:-1], O)


# --------- Expert FFN ---------
class SoEExpertFeedForward(nn.Module):
    """Token-wise MLP using ExpertLinear"""

    def __init__(self, dim, hidden_dim, n_experts, dropout=0.0):
        super().__init__()
        self.fc1 = SoEExpertLinear(dim, hidden_dim, n_experts, bias=True)
        self.fc2 = SoEExpertLinear(hidden_dim, dim, n_experts, bias=True)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x, alpha):
        x = self.fc1(x, alpha)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x, alpha)
        x = self.drop(x)
        return x


class CodeConditioner(nn.Module):
    """
    Predicts:
      - alpha: [B, n_experts] with similarity-adaptive temperature
      - FiLM params for Q and K/V: (gamma, beta) each [B, 1, 1, C]
      - per-head attention bias: [B, H]

    During training we optionally add noise (gaussian / gumbel / dirichlet) to logits
    so that multiple experts get exercised. The last batch alpha entropy is stored
    in self.last_alpha_entropy for optional regularization.
    """
    def __init__(self, code_dim, n_experts, d_model=None, embed_dim=256,
                 tau_min=0.5, tau_max=2.0, noise_scale=0.1, use_adaLN_modulation=True):
        super().__init__()
        self.n_experts = n_experts
        self.tau_min, self.tau_max = tau_min, tau_max
        self.noise_scale = float(noise_scale)
        self.use_adaLN_modulation = use_adaLN_modulation

        # Neural network to condition on codes and output expert weights
        fused_in = code_dim * 4 + 2  # ego, neb, |diff|, prod, [cos, l2]
        self.trunk = nn.Sequential(
            nn.Linear(fused_in, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )
        self.head_alpha = nn.Linear(embed_dim, n_experts)

        # AdaLN modulation parameters (gamma, beta)
        if self.use_adaLN_modulation and d_model:
            self.gamma_fc = nn.Linear(embed_dim, d_model)  # Generates gamma (scaling)
            self.beta_fc  = nn.Linear(embed_dim, d_model)  # Generates beta (shifting)

        self.register_buffer('last_alpha_entropy', torch.tensor(0.0), persistent=False)

    def forward(self, ego_code, neb_code):
        """
        ego_code, neb_code: [B, L]
        return:
          alpha:    [B, n_experts]
          adaLN:    dict with gamma and beta for modulation if enabled
        """
        # Compute the difference and cosine similarity
        diff = neb_code - ego_code
        prod = neb_code * ego_code
        cos  = F.cosine_similarity(neb_code, ego_code, dim=-1, eps=1e-6)  # [B]
        l2   = torch.norm(diff, dim=-1)                                   # [B]

        fused = torch.cat([ego_code, neb_code, diff.abs(), prod, cos.unsqueeze(-1), l2.unsqueeze(-1)], dim=-1)   # [B, *]
        h = self.trunk(fused)
        logits = self.head_alpha(h)                                        # [B, E]

        # similarity-adaptive temperature
        sim01 = (cos + 1.0) * 0.5
        tau   = self.tau_min + (1.0 - sim01) * (self.tau_max - self.tau_min)  # [B]

        # --- stochasticization during training ---
        if self.training and self.noise_scale > 0.0:
            logits = logits + torch.randn_like(logits) * self.noise_scale


        # Compute alpha (expert mixing weights)
        # alpha = F.softmax(logits / tau.unsqueeze(-1), dim=-1)              # [B, E]
        alpha = F.softmax(logits, dim=-1)              # [B, E]

        # Compute AdaLN parameters (optional)
        adaLN_params = {}
        if self.use_adaLN_modulation:
            gamma = self.gamma_fc(h)     # [B, C]
            beta = self.beta_fc(h)          # [B, C]
            # Reshape to fit LayerNorm broadcast
            gamma = gamma.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, C]
            beta  = beta.unsqueeze(1).unsqueeze(2)    # [B, 1, 1, C]
            adaLN_params = {'gamma': gamma, 'beta': beta}

        # Store average entropy for external regularization
        entropy = -(alpha * (alpha + 1e-12).log()).sum(dim=-1).mean()
        self.last_alpha_entropy = entropy.detach()

        return {"alpha": alpha, "alpha_logits": logits, "adaLN_params": adaLN_params}





# --------- MCT Cross-Attention (tokens) ---------
class SoECrossAttention2D(nn.Module):
    """Q from neb tokens, K/V from ego tokens; projections are ExpertLinear with AdaLN."""
    def __init__(self, dim, n_heads, dim_head, n_experts, dropout=0.0):
        super().__init__()
        assert dim == n_heads * dim_head
        self.n_heads = n_heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.to_q = SoEExpertLinear(dim, dim, n_experts, True)
        self.to_k = SoEExpertLinear(dim, dim, n_experts, True)
        self.to_v = SoEExpertLinear(dim, dim, n_experts, True)
        self.to_o = SoEExpertLinear(dim, dim, n_experts, True)

        self.dropout = nn.Dropout(dropout)

    def forward(self, neb_tokens, ego_tokens, alpha, adaLN_params=None):
        """
        neb_tokens, ego_tokens: [B, nW, N, C]
        alpha: [B, E]
        adaLN_params: dict with 'gamma' and 'beta' [B, 1, 1, C]
        return: [B, nW, N, C]
        """
        B, nW, N, C = neb_tokens.shape
        H, Dh = self.n_heads, self.dim_head

        if adaLN_params:
            gamma, beta = adaLN_params['gamma'], adaLN_params['beta']
            q_in = neb_tokens * (1 + gamma) + beta
            k_in = ego_tokens * (1 + gamma) + beta
            v_in = ego_tokens * (1 + gamma) + beta
        else:
            q_in = neb_tokens
            k_in = ego_tokens
            v_in = ego_tokens


        q = self.to_q(q_in, alpha)  # [B,nW,N,C]
        k = self.to_k(k_in, alpha)
        v = self.to_v(v_in, alpha)

        # Split heads
        q = rearrange(q, 'b nw n (h d) -> b h nw n d', h=H, d=Dh)
        k = rearrange(k, 'b nw n (h d) -> b h nw n d', h=H, d=Dh)
        v = rearrange(v, 'b nw n (h d) -> b h nw n d', h=H, d=Dh)

        # Attention logits
        attn = torch.einsum('b h w i d, b h w j d -> b h w i j', q*self.scale, k)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out  = torch.einsum('b h w i j, b h w j d -> b h w i d', attn, v)
        out  = rearrange(out, 'b h w n d -> b w n (h d)')
        out  = self.to_o(out, alpha)
        return out



# --------- Transformer Block (everything inside) ---------
class HeteroTransformerBlock(nn.Module):
    """
    Transformer block without an internal router.

    The block consumes shared expert gates `alpha` produced once by the top-level router.
    Optionally, shared AdaLN (gamma,beta) can modulate the pre-norm activations.
    """

    def __init__(
            self,
            in_dim,
            n_experts,
            n_heads,
            dim_head,
            window_size,
            mlp_ratio=2.0,
            dropout=0.1,
            adaLN_modulation=False,
    ):
        super().__init__()
        self.window = window_size
        hidden = int(in_dim * mlp_ratio)
        assert in_dim == n_heads * dim_head

        # Local sublayers
        self.local_attn = PlainCrossAttention2D(in_dim, n_heads, dim_head, dropout=dropout)
        self.local_attn_ln = nn.LayerNorm(in_dim)
        self.local_ffn = SoEExpertFeedForward(in_dim, hidden, n_experts, dropout)
        self.local_ffn_ln = nn.LayerNorm(in_dim)

        # Global sublayers
        self.global_attn = PlainCrossAttention2D(in_dim, n_heads, dim_head, dropout=dropout)
        self.global_attn_ln = nn.LayerNorm(in_dim)
        self.global_ffn = SoEExpertFeedForward(in_dim, hidden, n_experts, dropout)
        self.global_ffn_ln = nn.LayerNorm(in_dim)

        self.adaLN_modulation = adaLN_modulation

    @staticmethod
    def _apply_adaln(x_ln: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """Apply adaptive LayerNorm modulation: x' = x*(1+gamma) + beta."""
        while gamma.dim() < x_ln.dim():
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
        return x_ln * (1.0 + gamma) + beta

    def forward(self, ego_feat, neb_feat, alpha, adaLN_params=None):
        """
        ego_feat, neb_feat: [B,C,H,W]
        alpha: [B,E] shared expert gates
        adaLN_params: dict or None (shared across blocks)
        """
        B, C, H, W = neb_feat.shape

        # Local window tokens: [B, X*Y, win*win, C]
        neb_tokens = window_partition(neb_feat, self.window)
        ego_tokens = window_partition(ego_feat, self.window)

        x = self.local_attn_ln(neb_tokens)
        if self.adaLN_modulation and adaLN_params is not None:
            g, b = adaLN_params["local_attn"]
            x = self._apply_adaln(x, g, b)
        neb_tokens = neb_tokens + self.local_attn(x, ego_tokens)

        y = self.local_ffn_ln(neb_tokens)
        if self.adaLN_modulation and adaLN_params is not None:
            g, b = adaLN_params["local_ffn"]
            y = self._apply_adaln(y, g, b)
        neb_tokens = neb_tokens + self.local_ffn(y, alpha)

        # Grid tokens: [B, win*win, X*Y, C]
        neb_grid = to_grid_tokens(neb_tokens, self.window, H, W)
        ego_grid = to_grid_tokens(ego_tokens, self.window, H, W)

        xg = self.global_attn_ln(neb_grid)
        if self.adaLN_modulation and adaLN_params is not None:
            g, b = adaLN_params["global_attn"]
            xg = self._apply_adaln(xg, g, b)
        neb_grid = neb_grid + self.global_attn(xg, ego_grid)

        yg = self.global_ffn_ln(neb_grid)
        if self.adaLN_modulation and adaLN_params is not None:
            g, b = adaLN_params["global_ffn"]
            yg = self._apply_adaln(yg, g, b)
        neb_grid = neb_grid + self.global_ffn(yg, alpha)

        # Back to feature map
        neb_tokens = to_local_tokens(neb_grid, self.window, H, W)
        neb_feat_out = window_merge(neb_tokens, self.window, H, W)
        return neb_feat_out



# --------- Final Expert MLP Head (inside converter tail) ---------
class FinalProjector(nn.Module):
    """
    Final token-wise Expert MLP with its own conditioner.
    This refines converted features before output.
    """

    def __init__(self, in_dim, n_experts, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        hidden = int(in_dim * mlp_ratio)
        self.ln = nn.LayerNorm(in_dim)
        self.ffn = SoEExpertFeedForward(in_dim, hidden, n_experts, dropout)

    def forward(self, x_feat, alpha):
        # x_feat: [B,C,H,W] -> tokens -> apply -> back
        B, C, H, W = x_feat.shape
        x = x_feat.permute(0, 2, 3, 1).contiguous()   # [B,H,W,C]
        x = x.view(B, H * W, C)                       # [B,HW,C]
        x = x + self.ffn(self.ln(x), alpha)           # residual
        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return x


# ---------- Plain MLP head  ----------
class FinalMLP(nn.Module):
    """
    Plain token-wise MLP head:
      LayerNorm -> Linear -> GELU -> Dropout -> Linear -> Dropout
    """

    def __init__(self, in_dim: int, mlp_ratio: float = 2.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(in_dim * mlp_ratio)
        self.ln = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden, bias=True)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, in_dim, bias=True)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        feat: [B, C, H, W]
        returns: [B, C, H, W]
        """
        B, C, H, W = feat.shape
        x = rearrange(feat, 'b c h w -> b (h w) c')
        x = self.ln(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop2(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        return x




# --------- Converter (now only stacks blocks + final MLP) ---------
class HeteroFeatureConverter(nn.Module):
    def __init__(self, args):
        super().__init__()
        depth = args.get("depth", 2)
        in_dim = args.get("in_dim", 64)
        code_dim = args.get("code_dim", 64)
        n_experts = args.get("n_experts", 8)
        n_heads = args.get("n_heads", 8)
        window_size = args.get("window_size", 8)
        mlp_ratio = args.get("mlp_ratio", 4)
        dropout = args.get("dropout", 0.1)
        adaLN_modulation = args.get("use_adaLN_modulation", False)
        out_conv = args.get("out_conv", False)


        # Router configs
        router_embed_dim = args.get("router_embed_dim", 256)
        router_noise_scale = args.get("router_noise_scale", 0.05)


        # init options
        init_cfg = args.get("init_cfg", {})
        self.init_std = float(init_cfg.get("init_std", 0.02))          # Linear normal std
        self.conv_w_init = init_cfg.get("conv_w_init", "kaiming_uniform")
        self.conv_bias = float(init_cfg.get("conv_bias", 0.0))
        self.conv_kaiming_a = init_cfg.get("conv_kaiming_a", None)     # None -> sqrt(5)


        assert in_dim % n_heads == 0
        dim_head = in_dim // n_heads


        self.router = CodeConditioner(
            code_dim=code_dim,
            n_experts=n_experts,
            d_model=256,
            embed_dim=router_embed_dim,
            tau_min=0.5,
            tau_max=2.0,
            noise_scale=router_noise_scale,
            use_adaLN_modulation=adaLN_modulation
        )

        self.blocks = nn.ModuleList([
            HeteroTransformerBlock(
                in_dim=in_dim,
                n_experts=n_experts,
                n_heads=n_heads,
                dim_head=dim_head,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                adaLN_modulation=adaLN_modulation,
            )
            for _ in range(depth)
        ])
        self.final_head = FinalProjector(in_dim, n_experts=n_experts, mlp_ratio=2.0, dropout=dropout)


        self.use_out_conv = out_conv
        if self.use_out_conv:
            self.out_conv = ConvAdapter(in_dim)

    @torch.no_grad()
    def init_parameters(self):
        """
        Unified init using external helper funcs.

        - LayerNorm: (1,0)
        - Plain Linear: small normal (std=self.init_std)
        - SoEExpertLinear: weight_S kaiming, weight_E zero; bias_S uniform; bias_E zero
        - Router head_alpha: optional zero-init
        - AdaLN heads: zero-init (identity modulation at start)
        """

        # 1) LayerNorm everywhere
        self.apply(init_layernorm_)

        # 2) Default Linear init everywhere (Transformer-style)
        self.apply(lambda m: init_linear_normal_(m, std=self.init_std, bias=0.0))
        self.apply(lambda m: init_conv2d_(m, mode=self.conv_w_init, bias=self.conv_bias, a=self.conv_kaiming_a))

        # 3) Override SoEExpertLinear explicitly (do not rely on its internal init)
        def _init_soe(m: nn.Module):
            if m.__class__.__name__ != "SoEExpertLinear":
                return
            nn.init.kaiming_uniform_(m.weight_S, a=math.sqrt(5))
            nn.init.zeros_(m.weight_E)
            if m.bias_S is not None:
                bound = 1.0 / math.sqrt(m.in_features)
                nn.init.uniform_(m.bias_S, -bound, bound)
            if m.bias_E is not None:
                nn.init.zeros_(m.bias_E)

        self.apply(_init_soe)

        # 4) Router: head_alpha special handling (uniform gates at start if desired)
        if hasattr(self, "router") and hasattr(self.router, "head_alpha"):
            if getattr(self, "router_last_zero", False):
                zero_linear_(self.router.head_alpha)
            else:
                init_linear_normal_(self.router.head_alpha, std=self.init_std, bias=0.0)

        # 5) AdaLN heads: start from identity modulation (gamma=0, beta=0)
        if hasattr(self, "router") and getattr(self.router, "use_adaLN_modulation", False):
            if hasattr(self.router, "gamma_fc"):
                zero_linear_(self.router.gamma_fc)
            if hasattr(self.router, "beta_fc"):
                zero_linear_(self.router.beta_fc)



    def forward(self, ego_feat, neb_feat, ego_code, neb_code, return_alpha=False):
        # Shared routing
        cond = self.router(ego_code, neb_code)
        alpha = cond["alpha"]                     # [B,E]
        alpha_logits = cond["alpha_logits"]       # [B,E]
        adaLN_params = cond.get("adaLN_params", None)

        x = neb_feat
        for blk in self.blocks:
            x = blk(ego_feat, x, alpha, adaLN_params=adaLN_params)

        # Final head uses the same alpha
        x = self.final_head(x, alpha)

        if self.use_out_conv:
            x = self.out_conv(x)

        if return_alpha:
            alpha_stack = alpha_logits.unsqueeze(0)
            return x, alpha_stack

        return x

    def conditioner_fix(self, requires_grad=False):
        for p in self.router.parameters():
            p.requires_grad_(requires_grad)



    def conditioner_apply(self, func):
        self.router.apply(func)


    def set_conditioner_eval(self):
        self.router.eval()



    def get_last_alpha_entropy(self):
        alpha_entropy_all = []
        for blk in self.blocks:
            alpha_entropy_all.append(blk.conditioner.last_alpha_entropy)

        return torch.stack(alpha_entropy_all).mean()




class ConvAdapter(nn.Module):
    """Small spatial refinement head to reduce block artifacts."""
    def __init__(self, channels: int, dropout: float = 0.0, use_norm: bool = True):
        super().__init__()
        self.conv5 = nn.Conv2d(channels, channels, kernel_size=5, padding=2, bias=True)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.act = nn.GELU()
        self.drop = nn.Dropout2d(dropout) if dropout and dropout > 0 else nn.Identity()

        # Optional norm, keeps feature scale stable for detector
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=channels) if use_norm else nn.Identity()
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=channels) if use_norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv5(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.conv3(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.drop(x)

        return residual + x



# class ResizeNet(nn.Module):
#     def __init__(self, input_dim, out_dim, method = 'conv'):
#         super().__init__()
#
#         if method == 'conv':
#             self.channel_unify = nn.Sequential(
#                 nn.Conv2d(input_dim, out_dim, kernel_size=1),
#             )
#         elif method == 'conv3':
#             self.channel_unify = nn.Sequential(
#                 nn.Conv2d(input_dim, out_dim, kernel_size=3),
#             )
#         elif method == 'cbam':
#             downsample = nn.Sequential(nn.Conv2d(input_dim, out_dim, stride=1, kernel_size=1), nn.ReLU(nn.ReLU(inplace=True)))
#             self.channel_unify = CbamBasicBlock(input_dim, out_dim, downsample=downsample)
#
#
#     def forward(self, x, target_size):
#         _, C, H, W = x.size()
#
#         if (H, W) != target_size[1:]:
#             x = F.interpolate(x, size=target_size[1:], mode='bilinear')
#
#         if C != target_size[0]:
#             x = self.channel_unify(x)
#
#         return x







# =================================================================================================================
# =================================================================================================================
# =================================================================================================================



class CrossDomainTransformerConverter(nn.Module):
    def __init__(self, args):
        super().__init__()
        depth = args.get("depth", 2)
        in_dim = args.get("in_dim", 64)
        n_heads = args.get("n_heads", 8)
        window_size = args.get("window_size", 8)
        mlp_ratio = args.get("mlp_ratio", 4.0)
        dropout = args.get("dropout", 0.01)

        dim_head = in_dim // n_heads


        self.transformer_blocks = HeteroExpertNet(in_dim, n_heads, dim_head, window_size, depth, mlp_ratio, dropout)
        self.final_head = FinalMLP(in_dim, mlp_ratio=4.0, dropout=dropout)


    def forward(self, ego_feat, neb_feat):
        x = self.transformer_blocks(ego_feat, neb_feat)
        x = self.final_head(x)
        return x




# ---------------------------
# ConvNeXt blocks
# ---------------------------

class LayerNorm(nn.Module):
    """
    LayerNorm that supports channels_last inputs: [B,H,W,C]
    (matching the ConvNeXt implementation style you used)
    """
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x):
        # x: [B,H,W,C]
        mean = x.mean(dim=-1, keepdim=True)
        var  = (x - mean).pow(2).mean(dim=-1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight + self.bias


class ConvNeXtBlock(nn.Module):
    """
    DwConv -> (N,H,W,C) LN -> Linear -> GELU -> Linear -> (N,C,H,W) + residual
    """
    def __init__(self, dim, kernel_size=7, layer_scale_init_value=1e-6, drop_path=0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim))  if layer_scale_init_value > 0 else None
        self.drop_path = nn.Identity()  # keep it simple

    def forward(self, x):
        identity = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)         # [B,C,H,W] -> [B,H,W,C]
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)         # [B,H,W,C] -> [B,C,H,W]
        x = identity + self.drop_path(x)
        return x


class ConvNeXtStage(nn.Module):
    """A stack of ConvNeXtBlock."""
    def __init__(self, dim, depth, kernel_size=7, layer_scale_init_value=1e-6, drop_path=0.0):
        super().__init__()
        self.blocks = nn.Sequential(*[
            ConvNeXtBlock(dim, kernel_size=kernel_size, layer_scale_init_value=layer_scale_init_value, drop_path=drop_path)
            for _ in range(depth)
        ])

    def forward(self, x):
        return self.blocks(x)


# ---------------------------
# Cross-domain converter based on ConvNeXt
# ---------------------------

class CrossDomainConvNeXtConverter(nn.Module):
    """
    Cross-domain converter:
      - takes (ego_feat, neb_feat) in [B,C,H,W]
      - outputs neb_feat converted toward ego domain, [B,C,H,W]

    Design:
      1) Build a cross-domain context from ego_feat and neb_feat
         (concat + difference) -> 1x1 projection
      2) Use ConvNeXt to predict a residual delta to add onto neb_feat
      3) Optional cross-gating from ego context to modulate the residual magnitude

    This is a fully convolutional alternative to transformer cross-attention.
    """
    def __init__(self, args):
        super().__init__()
        self.in_dim = args.get("in_dim", 64)          # C
        self.hidden_dim = args.get("hidden_dim", self.in_dim)
        self.depth = args.get("depth", 2)
        self.kernel_size = args.get("kernel_size", 7)
        self.use_cross_gate = args.get("use_cross_gate", True)

        # Input fusion: [neb, ego, |neb-ego|] -> hidden
        fusion_in = 3 * self.in_dim
        self.fuse = nn.Sequential(
            nn.Conv2d(fusion_in, self.hidden_dim, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=1, bias=True),
        )

        # ConvNeXt trunk operating on fused context
        self.trunk = ConvNeXtStage(
            dim=self.hidden_dim,
            depth=self.depth,
            kernel_size=self.kernel_size,
            layer_scale_init_value=args.get("layer_scale_init_value", 1e-6),
            drop_path=args.get("drop_path", 0.0),
        )

        # Predict residual in neb feature space
        self.to_delta = nn.Conv2d(self.hidden_dim, self.in_dim, kernel_size=1, bias=True)

        # Optional cross-gate derived mainly from ego (helps stabilize)
        if self.use_cross_gate:
            self.gate = nn.Sequential(
                nn.Conv2d(self.in_dim, self.in_dim, kernel_size=1, bias=True),
                nn.Sigmoid()
            )

        # Final smoothing (optional)
        self.smoothing = nn.Conv2d(self.in_dim, self.in_dim, kernel_size=3, padding=1, bias=True)

        # Residual scaling for stability
        self.res_scale = float(args.get("res_scale", 1.0))

    def forward(self, ego_feat, neb_feat):
        """
        ego_feat, neb_feat: [B,C,H,W]
        return: converted_neb: [B,C,H,W]
        """
        assert ego_feat.shape == neb_feat.shape, "ego_feat and neb_feat must have same shape [B,C,H,W]"
        neb = neb_feat
        ego = ego_feat

        # Cross-domain context
        diff = (neb - ego).abs()
        x = torch.cat([neb, ego, diff], dim=1)          # [B, 3C, H, W]
        x = self.fuse(x)                                 # [B, hidden, H, W]
        x = self.trunk(x)                                # [B, hidden, H, W]

        delta = self.to_delta(x)                         # [B, C, H, W]

        if self.use_cross_gate:
            g = self.gate(ego)                           # [B, C, H, W]
            delta = delta * g

        out = neb + self.res_scale * delta               # residual conversion
        out = self.smoothing(out)
        return out


class ConvResBlock(nn.Module):
    """
    A simple residual conv block:
      Conv -> Norm -> GELU -> Conv -> Norm -> Residual
    """
    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.0, gn_groups: int = 8):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=pad, bias=False)
        self.norm1 = nn.GroupNorm(num_groups=min(gn_groups, channels), num_channels=channels, eps=1e-6)
        self.act = nn.GELU()
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=pad, bias=False)
        self.norm2 = nn.GroupNorm(num_groups=min(gn_groups, channels), num_channels=channels, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return x + res


class CrossDomainConvConverter(nn.Module):
    """
    Cross-domain feature converter using plain convolutions.

    Input:
      ego_feat: [B, C, H, W]
      neb_feat: [B, C, H, W]
    Output:
      neb_feat_aligned: [B, C, H, W]

    Design:
      - Fuse ego and neb via channel concatenation: [neb, ego, neb-ego, neb*ego]
      - Project back to C channels
      - Several residual conv blocks
      - Final 1x1 projection + residual with original neb_feat
    """
    def __init__(self, args: dict):
        super().__init__()
        in_dim = args.get("in_dim", 64)                 # C
        hidden_dim = args.get("hidden_dim", in_dim)     # internal channels
        depth = args.get("depth", 4)                    # number of residual blocks
        kernel_size = args.get("kernel_size", 3)
        dropout = args.get("dropout", 0.0)
        gn_groups = args.get("gn_groups", 8)

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        # 4-way fusion: neb, ego, diff, prod
        fusion_in = 4 * in_dim

        self.fuse = nn.Sequential(
            nn.Conv2d(fusion_in, hidden_dim, kernel_size=1, padding=0, bias=False),
            nn.GroupNorm(num_groups=min(gn_groups, hidden_dim), num_channels=hidden_dim, eps=1e-6),
            nn.GELU(),
        )

        self.blocks = nn.ModuleList([
            ConvResBlock(hidden_dim, kernel_size=kernel_size, dropout=dropout, gn_groups=gn_groups)
            for _ in range(depth)
        ])

        self.out_proj = nn.Sequential(
            nn.Conv2d(hidden_dim, in_dim, kernel_size=1, padding=0, bias=True),
        )

        # Optional smoothing (often helps stabilize outputs)
        self.smoothing = nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, bias=True)

    def forward(self, ego_feat: torch.Tensor, neb_feat: torch.Tensor) -> torch.Tensor:
        """
        ego_feat, neb_feat: [B, C, H, W]
        """
        assert ego_feat.shape == neb_feat.shape, "ego_feat and neb_feat must have the same shape"
        B, C, H, W = neb_feat.shape
        assert C == self.in_dim, f"Expected C={self.in_dim}, got C={C}"

        diff = neb_feat - ego_feat
        prod = neb_feat * ego_feat
        x = torch.cat([neb_feat, ego_feat, diff, prod], dim=1)  # [B, 4C, H, W]

        x = self.fuse(x)  # [B, hidden, H, W]
        for blk in self.blocks:
            x = blk(x)

        x = self.out_proj(x)             # [B, C, H, W]
        x = x + neb_feat                 # residual: keep original content
        x = self.smoothing(x)            # light refinement
        return x









def build_feature_converter(args, name=None) -> nn.Module:
    """
    Factory for feature converters.

    name: one of
      - "unitrans" : parameter-combination Mapping-conditioned converter
      - "block_moe" / "classic_moe": expert-net MoE (each expert is a full HeteroExpertNet)
      - "linear_moe" / "ffn_moe": Transformer-Linear-style MoE (MoE only in FFN)
    args: dict of hyper-parameters, passed to underlying constructor.
    """
    if name is None:
        name = args.get("type", "unitrans")
    name_l = name.lower()
    if name_l in ["unitrans",]:
        return HeteroFeatureConverter(args)

    if name_l in ["modular_moe", "block_moe", "classic_moe", "expert_net_moe"]:
        # Classic block-level MoE from previous answer
        return ClassicMoEFeatureConverter(args)

    if name_l in ["linear_moe", "ffn_moe", "transformer_linear_moe"]:
        # New Linear-MoE converter from this answer
        return LinearMoEFeatureConverter(args)

    if name_l in ["transformer_converter"]:
        return CrossDomainTransformerConverter(args)

    if name_l in ["convnext_converter"]:
        return CrossDomainConvNeXtConverter(args)

    raise ValueError(f"Unknown converter type: {name}")





# ---------------------------
# Smoke test
# ---------------------------


# -------------------------
# Tests
# -------------------------
@torch.no_grad()
def test_window_partition_merge(B=2, C=64, H=128, W=128, win=8, device="cuda", dtype=torch.float32):
    x = torch.randn(B, C, H, W, device=device, dtype=dtype)
    t = window_partition(x, win)          # [B, X*Y, win^2, C]
    x_rec = window_merge(t, win, H, W)    # [B, C, H, W]
    max_err = (x - x_rec).abs().max().item()
    print(f"[window_partition -> window_merge] max_err={max_err:.6e}")
    assert max_err == 0.0, "window_partition + window_merge is NOT perfectly invertible."

@torch.no_grad()
def test_local_grid_roundtrip_tokens(B=2, C=64, H=128, W=128, win=8, device="cuda", dtype=torch.float32):
    x = torch.randn(B, C, H, W, device=device, dtype=dtype)
    local = window_partition(x, win)                  # [B, X*Y, win^2, C]
    grid  = to_grid_tokens(local, win, H, W)          # [B, win^2, X*Y, C]
    local_rec = to_local_tokens(grid, win, H, W)      # [B, X*Y, win^2, C]
    max_err = (local - local_rec).abs().max().item()
    print(f"[local_tokens -> grid_tokens -> local_tokens] max_err={max_err:.6e}")
    assert max_err == 0.0, "to_grid_tokens + to_local_tokens is NOT perfectly invertible on tokens."

@torch.no_grad()
def test_full_roundtrip_feature(B=2, C=64, H=128, W=128, win=8, device="cuda", dtype=torch.float32):
    x = torch.randn(B, C, H, W, device=device, dtype=dtype)

    local = window_partition(x, win)                  # [B, X*Y, win^2, C]
    grid  = to_grid_tokens(local, win, H, W)          # [B, win^2, X*Y, C]
    local2 = to_local_tokens(grid, win, H, W)         # [B, X*Y, win^2, C]
    x_rec = window_merge(local2, win, H, W)           # [B, C, H, W]

    max_err = (x - x_rec).abs().max().item()
    print(f"[FULL x -> local -> grid -> local -> x] max_err={max_err:.6e}")
    assert max_err == 0.0, "FULL roundtrip is NOT perfectly invertible."

@torch.no_grad()
def sanity_check_shapes(H=128, W=128, win=8, C=64, B=2, device="cuda"):
    x = torch.randn(B, C, H, W, device=device)
    local = window_partition(x, win)
    X, Y = H // win, W // win
    assert local.shape == (B, X * Y, win * win, C), f"local shape wrong: {local.shape}"

    grid = to_grid_tokens(local, win, H, W)
    assert grid.shape == (B, win * win, X * Y, C), f"grid shape wrong: {grid.shape}"

    local2 = to_local_tokens(grid, win, H, W)
    assert local2.shape == local.shape, f"local2 shape wrong: {local2.shape}"

    x2 = window_merge(local2, win, H, W)
    assert x2.shape == x.shape, f"x2 shape wrong: {x2.shape}"

    print("[shape sanity] passed.")

def run_all_tests(device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    sanity_check_shapes(device=device)
    test_window_partition_merge(device=device)
    test_local_grid_roundtrip_tokens(device=device)
    test_full_roundtrip_feature(device=device)
    print("All tests passed.")



if __name__ == "__main__":
    torch.manual_seed(0)
    B, C, H, W = 4, 64, 256, 128
    L = 64
    n_heads = 8
    dim_head = C // n_heads
    win = 8

    ego_feat = torch.randn(B, C, H, W)
    neb_feat = torch.randn(B, C, H, W)
    ego_code = torch.randn(B, L)
    neb_code = torch.randn(B, L)

    # model = HeteroFeatureConverter(
    #     depth=2,
    #     in_dim=C,
    #     code_dim=L,
    #     n_experts=4,
    #     n_heads=n_heads,
    #     window_size=win,
    #     mlp_ratio=4,
    #     dropout=0.1
    # )
    #
    # y = model(ego_feat, neb_feat, ego_code, neb_code)
    # print("Output shape:", y.shape)  # [B, C, H, W]

    args = dict(
        depth=2,
        in_dim=64,
        code_dim=64,
        n_experts=4,
        n_heads=8,
        window_size=8,
        mlp_ratio=4.0,
        dropout=0.1,
        top_k=2,  # for MoE variants
        router_embed_dim=256,
        router_noise_scale=0.1,
    )

    # converter_soe = build_feature_converter(args, "unitrans")
    # converter_block = build_feature_converter(args, "block_moe")
    # converter_lin = build_feature_converter(args, "linear_moe")
    #
    # out_soe = converter_soe(ego_feat, neb_feat, ego_code, neb_code)
    # out_block = converter_block(ego_feat, neb_feat, ego_code, neb_code)
    # out_lin = converter_lin(ego_feat, neb_feat, ego_code, neb_code)


    run_all_tests()










