"""Linformer encoder.

Hierarchical 4-stage encoder using Linformer (Wang 2020) low-rank linear
self-attention. Extracted from
``medseg.models.networks.other.linformer_unet.LinformerUNet`` so it can be reused
as a standalone backbone behind any decoder.

Reference:
    Wang et al. "Linformer: Self-Attention with Linear Complexity" (2020),
    https://arxiv.org/abs/2006.04768

Topology (matches the source UNet wrapper):
    patch-embed (stride 4, kernel 7) -> 64
    Stage 0: 2 x LinformerBlock @ 64  (2 heads)         stride 4
    Down 2x ->
    Stage 1: 2 x LinformerBlock @ 128 (4 heads)         stride 8
    Down 2x ->
    Stage 2: 4 x LinformerBlock @ 256 (8 heads)         stride 16
    Down 2x ->
    Stage 3: 2 x LinformerBlock @ 512 (16 heads)        stride 32

Resolution-friendly:
    The Linformer projection (E, F) is replaced with a deterministic
    ``adaptive_avg_pool1d`` along the sequence axis, so the encoder accepts
    any input HxW (no resolution-specific parameters).
"""
# Source: UNCHECKED — please verify

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.registry import ENCODER_REGISTRY


# ---------------------------------------------------------------------------
# Deterministic sequence-axis projection (replaces learned E, F of Linformer)
# ---------------------------------------------------------------------------

def _seq_project(x_seq: torch.Tensor, k: int) -> torch.Tensor:
    """Project (B, N, D) -> (B, k_eff, D) by 1D adaptive average pooling."""
    B, N, D = x_seq.shape
    k_eff = min(k, N)
    if k_eff == N:
        return x_seq
    xt = x_seq.transpose(1, 2)              # (B, D, N)
    xp = F.adaptive_avg_pool1d(xt, k_eff)   # (B, D, k_eff)
    return xp.transpose(1, 2).contiguous()  # (B, k_eff, D)


# ---------------------------------------------------------------------------
# Linformer attention / MLP / block
# ---------------------------------------------------------------------------

class _LinformerAttention(nn.Module):
    """Linformer attention: project K,V along sequence dim from N to k_eff."""

    def __init__(self, dim, num_heads, k=256, qkv_bias=True,
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} not divisible by num_heads {num_heads}"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.k = k

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        H = self.num_heads
        Dh = self.head_dim

        qkv = self.qkv(x)                                   # (B, N, 3C)
        qkv = qkv.reshape(B, N, 3, C).permute(2, 0, 1, 3)   # (3, B, N, C)
        q, k_full, v_full = qkv[0], qkv[1], qkv[2]

        k_proj = _seq_project(k_full, self.k)               # (B, k_eff, C)
        v_proj = _seq_project(v_full, self.k)               # (B, k_eff, C)
        k_eff = k_proj.shape[1]

        q = q.reshape(B, N, H, Dh).transpose(1, 2)              # (B, H, N, Dh)
        k_proj = k_proj.reshape(B, k_eff, H, Dh).transpose(1, 2)  # (B, H, k_eff, Dh)
        v_proj = v_proj.reshape(B, k_eff, H, Dh).transpose(1, 2)  # (B, H, k_eff, Dh)

        attn = (q @ k_proj.transpose(-2, -1)) * self.scale  # (B, H, N, k_eff)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ v_proj                                 # (B, H, N, Dh)
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class _Mlp(nn.Module):
    def __init__(self, dim, hidden_dim=None, drop=0.0):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class _LinformerBlock(nn.Module):
    """Pre-norm transformer block with Linformer attention."""

    def __init__(self, dim, num_heads, k=256, mlp_ratio=4.0,
                 drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _LinformerAttention(
            dim, num_heads=num_heads, k=k,
            attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _Mlp(dim, hidden_dim=int(dim * mlp_ratio), drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class _Stage(nn.Module):
    """Runs a stack of Linformer blocks on a 2D feature map."""

    def __init__(self, dim, depth, num_heads, k=256, mlp_ratio=4.0,
                 drop=0.0, attn_drop=0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            _LinformerBlock(dim, num_heads, k=k, mlp_ratio=mlp_ratio,
                            drop=drop, attn_drop=attn_drop)
            for _ in range(depth)
        ])

    def forward(self, x):
        B, C, H, W = x.shape
        x_seq = x.flatten(2).transpose(1, 2)        # (B, N, C), N=H*W
        for blk in self.blocks:
            x_seq = blk(x_seq)
        return x_seq.transpose(1, 2).reshape(B, C, H, W)


class _PatchEmbed(nn.Module):
    def __init__(self, in_channels, out_channels, stride=4, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.proj = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        x = self.proj(x)                            # (B, C, H', W')
        B, C, H, W = x.shape
        x_seq = x.flatten(2).transpose(1, 2)        # (B, N, C)
        x_seq = self.norm(x_seq)
        return x_seq.transpose(1, 2).reshape(B, C, H, W)


class _Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, stride=2, padding=1)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x_seq = x.flatten(2).transpose(1, 2)
        x_seq = self.norm(x_seq)
        return x_seq.transpose(1, 2).reshape(B, C, H, W)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

@ENCODER_REGISTRY.register("linformer")
class LinformerEncoder(nn.Module):
    """Hierarchical 4-stage Linformer encoder.

    Args:
        in_channels: input image channels (1 grayscale, 3 RGB, etc.). When
            != 3 a 1x1 conv stem maps the input to 3 channels before the
            standard stride-4 patch embed.
        img_size: nominal input edge length. Used only as a hint; the encoder
            is resolution-agnostic and derives all shapes at runtime.
        pretrained: accepted for API compatibility; no pretrained weights
            are published for this architecture.
        k: rank of the Linformer projection along the sequence axis.

    Returns from ``forward``: a list of 4 feature maps with channels
    ``[64, 128, 256, 512]`` at strides ``[4, 8, 16, 32]``. The deepest
    feature is LAST.
    """

    def __init__(self, in_channels: int = 3, img_size: int = 224,
                 pretrained: bool = False, k: int = 256, **kwargs):
        super().__init__()
        dims = [64, 128, 256, 512]
        depths = [2, 2, 4, 2]
        heads = [2, 4, 8, 16]
        self.k = k
        self.img_size = img_size
        self.out_channels: List[int] = list(dims)

        # Adapt non-3-channel inputs with a 1x1 conv stem
        if in_channels != 3:
            self.input_proj = nn.Conv2d(in_channels, 3, kernel_size=1)
            stem_in = 3
        else:
            self.input_proj = nn.Identity()
            stem_in = in_channels

        # Patch embed (stride 4) -> dims[0]
        self.patch_embed = _PatchEmbed(stem_in, dims[0],
                                       stride=4, kernel_size=7)

        # Encoder stages + 3 downsamples
        self.encoder_stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i in range(4):
            self.encoder_stages.append(
                _Stage(dims[i], depths[i], heads[i], k=k))
            if i < 3:
                self.downsamples.append(_Downsample(dims[i], dims[i + 1]))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.input_proj(x)
        feat = self.patch_embed(x)              # stride 4
        feats: List[torch.Tensor] = []
        for i in range(4):
            feat = self.encoder_stages[i](feat)
            feats.append(feat)
            if i < 3:
                feat = self.downsamples[i](feat)
        return feats
