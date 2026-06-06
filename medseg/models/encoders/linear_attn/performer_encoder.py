"""Performer (FAVOR+) linear-attention encoder.

Extracted from `medseg.models.networks.other.performer_unet`. Follows the standard
encoder interface used by the rest of the codebase:

    forward(x: (B, in_channels, H, W)) -> List[Tensor]
        - 4 multi-scale feature maps in BCHW layout
        - shallowest first, deepest last
        - strides 4, 8, 16, 32

Reference: Choromanski et al., "Rethinking Attention with Performers" (2021).
"""
# Source: UNCHECKED — please verify

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.registry import ENCODER_REGISTRY


# ---------------------------------------------------------------------------
# FAVOR+ random-feature linear attention
# ---------------------------------------------------------------------------
class _FavorAttention(nn.Module):
    """Multi-head FAVOR+ linear attention with fixed random projection."""

    def __init__(self, dim: int, num_heads: int, num_features: int = 64,
                 qkv_bias: bool = True, proj_drop: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must divide num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_features = num_features

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Fixed random Gaussian projection W: (num_heads, m, head_dim)
        w = torch.randn(num_heads, num_features, self.head_dim)
        for h in range(num_heads):
            q, _ = torch.linalg.qr(torch.randn(max(num_features, self.head_dim),
                                               self.head_dim))
            w[h] = q[:num_features]
        self.register_buffer("proj_w", w, persistent=True)

    def _phi(self, x: torch.Tensor) -> torch.Tensor:
        sq = (x * x).sum(dim=-1, keepdim=True) * 0.5
        wx = torch.einsum("bhnd,hmd->bhnm", x, self.proj_w)
        stabilizer = wx.amax(dim=-1, keepdim=True).detach()
        out = torch.exp(wx - sq - stabilizer) / math.sqrt(self.num_features)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q / math.sqrt(math.sqrt(self.head_dim))
        k = k / math.sqrt(math.sqrt(self.head_dim))

        phi_q = self._phi(q)
        phi_k = self._phi(k)

        kv = torch.einsum("bhnm,bhnd->bhmd", phi_k, v)
        numer = torch.einsum("bhnm,bhmd->bhnd", phi_q, kv)
        k_sum = phi_k.sum(dim=2)
        denom = torch.einsum("bhnm,bhm->bhn", phi_q, k_sum).clamp(min=1e-6)
        out = numer / denom.unsqueeze(-1)

        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj_drop(self.proj(out))
        return out


# ---------------------------------------------------------------------------
# MLP and Performer block
# ---------------------------------------------------------------------------
class _Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class _PerformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_features: int = 64,
                 mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _FavorAttention(dim, num_heads=num_heads,
                                    num_features=num_features, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _Mlp(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Patch embed / patch merging
# ---------------------------------------------------------------------------
class _PatchEmbed(nn.Module):
    """Stride-4 patch embedding (Conv stem)."""

    def __init__(self, in_channels: int, embed_dim: int, patch_size: int = 4):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class _PatchMerging(nn.Module):
    """Halves spatial resolution, doubles channels (controlled by out_dim)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(4 * in_dim)
        self.reduction = nn.Linear(4 * in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, int, int]:
        B, N, C = x.shape
        assert N == H * W
        x = x.view(B, H, W, C)
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
            H = H + pad_h
            W = W + pad_w
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        Hn, Wn = H // 2, W // 2
        x = x.view(B, Hn * Wn, 4 * C)
        x = self.reduction(self.norm(x))
        return x, Hn, Wn


# ---------------------------------------------------------------------------
# Stage = stack of Performer blocks
# ---------------------------------------------------------------------------
class _PerformerStage(nn.Module):
    def __init__(self, dim: int, depth: int, num_heads: int,
                 num_features: int = 64, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            _PerformerBlock(dim, num_heads=num_heads,
                            num_features=num_features,
                            mlp_ratio=mlp_ratio, drop=drop)
            for _ in range(depth)
        ])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------
@ENCODER_REGISTRY.register("performer")
class PerformerEncoder(nn.Module):
    """Performer / FAVOR+ kernel linear-attention encoder (Choromanski 2021).

    4-stage hierarchical encoder mirroring the PerformerUNet wrapper. Returns
    multi-scale BCHW feature maps at strides 4, 8, 16, 32, shallowest first
    and deepest last.

    out_channels: [64, 128, 256, 512]
    """

    def __init__(self, in_channels: int = 3, img_size: int = 224,
                 pretrained: bool = False, num_features: int = 64, **kwargs):
        super().__init__()
        # pretrained is a no-op for this from-scratch encoder.
        del pretrained

        self.in_channels = in_channels
        self.img_size = img_size
        self.num_features = num_features

        depths: List[int] = [2, 2, 4, 2]
        dims: List[int] = [64, 128, 256, 512]
        num_heads: List[int] = [2, 4, 8, 16]
        self.dims = dims
        self.depths = depths
        self.out_channels = list(dims)

        # 1x1 conv to map arbitrary input channel count to 3 before the
        # standard patch-embed stem.
        if in_channels != 3:
            self.input_proj = nn.Conv2d(in_channels, 3, kernel_size=1, bias=True)
            stem_in_channels = 3
        else:
            self.input_proj = nn.Identity()
            stem_in_channels = in_channels

        self.patch_embed = _PatchEmbed(stem_in_channels, dims[0], patch_size=4)

        self.enc_stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i in range(4):
            self.enc_stages.append(_PerformerStage(
                dim=dims[i], depth=depths[i], num_heads=num_heads[i],
                num_features=num_features,
            ))
            if i < 3:
                self.downsamples.append(_PatchMerging(dims[i], dims[i + 1]))

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        B, _, H_in, W_in = x.shape

        x = self.input_proj(x)

        # Pad to a multiple of 32 so 3 patch-merging steps (over a stride-4
        # patch embed) divide evenly.
        pad_mult = 32
        pad_h = (pad_mult - H_in % pad_mult) % pad_mult
        pad_w = (pad_mult - W_in % pad_mult) % pad_mult
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        x, H, W = self.patch_embed(x)

        feats: List[torch.Tensor] = []
        for i in range(4):
            x = self.enc_stages[i](x)
            # Reshape token sequence (B, N, C) -> (B, C, H, W) for the
            # standard encoder interface.
            feat = x.transpose(1, 2).reshape(B, self.dims[i], H, W).contiguous()
            feats.append(feat)
            if i < 3:
                x, H, W = self.downsamples[i](x, H, W)

        return feats


__all__ = ["PerformerEncoder"]
