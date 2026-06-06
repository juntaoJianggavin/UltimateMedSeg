"""MISSFormer Bridge skip — per-pair adaptation of MISSFormer's BridgeLayer_4.

Original (MISSFormer, MICCAI 2022): operates on a LIST of 4 encoder skip
features at all scales. Flattens each to tokens, concatenates into one long
sequence, applies efficient transformer self-attention (with spatial
reduction by reduction_ratios per scale), then splits back and applies
per-scale MixFFN with depthwise-conv local mixing. Output is added to the
flattened residual.

This module adapts the same idea to the framework's per-pair (decoder_feat,
skip_feat) skip interface so it can be used as a drop-in skip. With 2
features instead of 4, joint self-attention becomes a cross-scale token-mix
between decoder and skip tokens.

Distinct from:
- `cross_attn_skip` (Q from decoder, K/V from skip — one-direction cross-attn)
- `transformer_fusion_skip` (single transformer block with Q=decoder, K/V=skip)
- `sc_att_bridge_skip` (spatial conv-attention + channel 1D-conv mixing, no
  transformer)

The MISSFormer style is "joint self-attention on concatenated tokens" — it
mixes decoder and skip tokens symmetrically via standard MHSA with efficient
spatial reduction, then applies per-feature DWConv-based MixFFN.
"""
# Source: https://github.com/ZhifangDeng/MISSFormer

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from medseg.registry import SKIP_REGISTRY


class _EfficientSelfAttention(nn.Module):
    """Efficient self-attention with optional spatial-reduction (per MISSFormer)."""

    def __init__(self, dim: int, num_heads: int = 4, reduction_ratio: int = 1):
        super().__init__()
        assert dim % num_heads == 0, f"dim={dim} not divisible by num_heads={num_heads}"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.reduction_ratio = reduction_ratio

        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, 2 * dim, bias=True)
        self.proj = nn.Linear(dim, dim)

        # Spatial-reduction conv for K/V (MISSFormer SR-attn style)
        if reduction_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=reduction_ratio,
                                stride=reduction_ratio)
            self.sr_norm = nn.LayerNorm(dim)
        else:
            self.sr = None

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """x: (B, N, C) where N = h*w; returns (B, N, C)."""
        B, N, C = x.shape
        assert N == h * w, f"N={N} != h*w={h*w}"

        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.sr is not None:
            x_spatial = x.permute(0, 2, 1).reshape(B, C, h, w)
            x_red = self.sr(x_spatial)
            x_red = x_red.reshape(B, C, -1).permute(0, 2, 1)
            x_red = self.sr_norm(x_red)
            kv = self.kv(x_red).reshape(B, -1, 2, self.num_heads, self.head_dim)\
                .permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim)\
                .permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class _MixFFN(nn.Module):
    """DWConv + MLP — MISSFormer's MixFFN_skip adapted to per-pair use.

    Matches official MixFFN_skip: act(norm1(dwconv(fc1(x)) + fc1(x))) -> fc2
    """

    def __init__(self, dim: int, mlp_ratio: int = 4):
        super().__init__()
        hidden_dim = dim * mlp_ratio
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3,
                                padding=1, groups=hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """x: (B, N, C). Returns (B, N, C)."""
        B, N, C = x.shape
        fx = self.fc1(x)
        fx_2d = fx.transpose(1, 2).reshape(B, -1, h, w)
        dw_out = self.dwconv(fx_2d).flatten(2).transpose(1, 2)  # (B, N, hidden)
        # Residual + norm (matching official MixFFN_skip)
        ax = self.act(self.norm1(dw_out + fx))
        return self.fc2(ax)


@SKIP_REGISTRY.register("missformer_bridge")
class MISSFormerBridgeSkip(nn.Module):
    """MISSFormer-style joint self-attention bridge skip (per-pair).

    Procedure (paraphrasing MISSFormer's BridgeLayer_4 for 2 features):
        1. Project decoder_feat and skip_feat to a common channel C via 1x1 convs
           (default C = min(decoder_ch, skip_ch)).
        2. Flatten both to tokens, concatenate along sequence: (B, N_d + N_s, C).
        3. Efficient self-attention (with reduction_ratio) + residual.
        4. LayerNorm.
        5. Split back to per-scale tokens; apply DWConv-based MixFFN per scale.
        6. Concatenate decoder+skip tokens back, reshape to (B, 2C, H, W) and
           return.

    get_out_channels(d_ch, s_ch) -> 2 * common_dim where common_dim = min(d_ch, s_ch).

    Kwargs:
        num_heads (int): attention heads (default 4).
        reduction_ratio (int): spatial-reduction stride for K/V (default 1).
        mlp_ratio (int): MixFFN expansion (default 4).
    """

    _MAX_TOKENS = 8192  # max tokens for joint self-attention

    def __init__(self, num_heads: int = 4, reduction_ratio: int = 1,
                 mlp_ratio: int = 4, max_tokens: int = None, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.reduction_ratio = reduction_ratio
        self.mlp_ratio = mlp_ratio
        if max_tokens is not None:
            self._MAX_TOKENS = max_tokens
        self._cache: dict = {}  # (d_ch, s_ch, common_dim, device) -> nn.ModuleDict

    def get_out_channels(self, decoder_ch: int, skip_ch: int) -> int:
        common_dim = self._pick_common_dim(decoder_ch, skip_ch)
        return 2 * common_dim

    def _pick_common_dim(self, d_ch: int, s_ch: int) -> int:
        # Use min and round to multiple of num_heads so attention head-dim is integer
        c = min(d_ch, s_ch)
        c = max(self.num_heads, (c // self.num_heads) * self.num_heads)
        return c

    def _build(self, d_ch: int, s_ch: int, device) -> nn.ModuleDict:
        common_dim = self._pick_common_dim(d_ch, s_ch)
        key = (d_ch, s_ch, common_dim, str(device))
        if key in self._cache:
            return self._cache[key]

        mod = nn.ModuleDict({
            "proj_d": nn.Conv2d(d_ch, common_dim, kernel_size=1).to(device),
            "proj_s": nn.Conv2d(s_ch, common_dim, kernel_size=1).to(device),
            "norm1": nn.LayerNorm(common_dim).to(device),
            "attn": _EfficientSelfAttention(common_dim, self.num_heads,
                                            self.reduction_ratio).to(device),
            "norm2": nn.LayerNorm(common_dim).to(device),
            "mixffn_d": _MixFFN(common_dim, self.mlp_ratio).to(device),
            "mixffn_s": _MixFFN(common_dim, self.mlp_ratio).to(device),
        })
        safe = f"_mfb_{d_ch}_{s_ch}_{common_dim}_{str(device).replace(':', '_')}"
        setattr(self, safe, mod)
        self._cache[key] = mod
        return mod

    def forward(self, decoder_feat: torch.Tensor, skip_feat: torch.Tensor) -> torch.Tensor:
        # Align skip spatial to decoder
        if skip_feat.shape[2:] != decoder_feat.shape[2:]:
            skip_feat = F.interpolate(skip_feat, size=decoder_feat.shape[2:],
                                      mode="bilinear", align_corners=False)

        B, d_ch, H, W = decoder_feat.shape
        s_ch = skip_feat.shape[1]
        N = H * W

        # If total token count is too large for joint self-attention,
        # fall back to projected concatenation (identity skip)
        if 2 * N > self._MAX_TOKENS:
            mod = self._build(d_ch, s_ch, decoder_feat.device)
            d_proj = mod["proj_d"](decoder_feat)
            s_proj = mod["proj_s"](skip_feat)
            return torch.cat([d_proj, s_proj], dim=1)

        mod = self._build(d_ch, s_ch, decoder_feat.device)
        common_dim = self._pick_common_dim(d_ch, s_ch)

        # 1. Project to common channels
        d = mod["proj_d"](decoder_feat)   # (B, C, H, W)
        s = mod["proj_s"](skip_feat)      # (B, C, H, W)

        B, C, H, W = d.shape
        N = H * W

        # 2. Flatten and concat — both at the same spatial resolution
        d_tok = d.flatten(2).transpose(1, 2)  # (B, N, C)
        s_tok = s.flatten(2).transpose(1, 2)  # (B, N, C)
        # For efficient SR-attn we need to view the concatenated tokens as a
        # "2H x W" grid so the SR conv has well-defined input shape.
        concat = torch.cat([d_tok, s_tok], dim=1)  # (B, 2N, C)

        # 3. Joint self-attention + residual (treat as 2H x W spatial grid)
        # Pad to make reduction_ratio divisible if needed
        joint_h, joint_w = 2 * H, W
        if self.reduction_ratio > 1:
            pad_h = (-joint_h) % self.reduction_ratio
            pad_w = (-joint_w) % self.reduction_ratio
            if pad_h or pad_w:
                # Pad tokens in the spatial grid
                concat_spatial = concat.transpose(1, 2).reshape(B, C, joint_h, joint_w)
                concat_spatial = F.pad(concat_spatial, (0, pad_w, 0, pad_h),
                                       mode="replicate")
                joint_h += pad_h
                joint_w += pad_w
                concat = concat_spatial.flatten(2).transpose(1, 2)
        attn_out = mod["attn"](mod["norm1"](concat), joint_h, joint_w)
        tx1 = concat + attn_out  # (B, 2N_padded, C)

        # Crop back the padding if any
        if self.reduction_ratio > 1 and (joint_h != 2 * H or joint_w != W):
            tx1_spatial = tx1.transpose(1, 2).reshape(B, C, joint_h, joint_w)
            tx1_spatial = tx1_spatial[:, :, :2 * H, :W]
            tx1 = tx1_spatial.flatten(2).transpose(1, 2)

        # 4. Norm
        tx2 = mod["norm2"](tx1)

        # 5. Split back to per-scale tokens; per-scale MixFFN
        d_split = tx2[:, :N, :]   # decoder tokens
        s_split = tx2[:, N:, :]   # skip tokens
        d_ffn = mod["mixffn_d"](d_split, H, W)
        s_ffn = mod["mixffn_s"](s_split, H, W)

        # 6. Add residual (per scale) and reshape back to BCHW
        d_out = (tx1[:, :N, :] + d_ffn).transpose(1, 2).reshape(B, C, H, W)
        s_out = (tx1[:, N:, :] + s_ffn).transpose(1, 2).reshape(B, C, H, W)

        return torch.cat([d_out, s_out], dim=1)  # (B, 2C, H, W)
