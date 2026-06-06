"""PVT-Former: Pyramid Vision Transformer encoder + CNN decoder.

Uses a PVT-v2 style encoder (hierarchical transformer with spatial-reduction
attention) combined with a U-Net decoder for medical image segmentation.

Reference:
    CT Liver Segmentation via PVT-based Encoding and Refined Decoding.
    arXiv 2024.

Key components:
    - Pyramid Vision Transformer v2 encoder (4 stages)
    - Spatial-Reduction Attention (SRA) for efficiency
    - U-Net style decoder with skip connections
"""
# Source: https://github.com/whai362/PVT

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class _SRAttention(nn.Module):
    """Spatial-Reduction Attention: reduces KV spatial dim for efficiency."""

    def __init__(self, dim, num_heads=4, sr_ratio=2):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, sr_ratio, sr_ratio)
            self.sr_norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_2d = x.transpose(1, 2).view(B, C, H, W)
            x_sr = self.sr(x_2d).flatten(2).transpose(1, 2)
            x_sr = self.sr_norm(x_sr)
        else:
            x_sr = x

        kv = self.kv(x_sr).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class _PVTBlock(nn.Module):
    def __init__(self, dim, num_heads=4, sr_ratio=2, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _SRAttention(dim, num_heads, sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim),
        )

    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.mlp(self.norm2(x))
        return x


class _PVTStage(nn.Module):
    def __init__(self, in_c, out_c, depth, num_heads, sr_ratio, patch_size=3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_c, out_c, patch_size, 2, patch_size // 2, bias=False),
            nn.BatchNorm2d(out_c),
        )
        self.blocks = nn.ModuleList([
            _PVTBlock(out_c, num_heads, sr_ratio) for _ in range(depth)
        ])

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        for blk in self.blocks:
            tokens = blk(tokens, H, W)
        return tokens.transpose(1, 2).view(B, C, H, W)


class PVTUNet(nn.Module):
    """PVT-Former: Pyramid Vision Transformer U-Net.

    Args:
        in_channels: Input channels.
        num_classes: Segmentation classes.
        img_size: Input spatial size.
        embed_dims: Channel dims per encoder stage.
        depths: Blocks per stage.
        num_heads: Attention heads per stage.
        sr_ratios: Spatial reduction ratios per stage.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        img_size: int = 224,
        embed_dims: Optional[List[int]] = None,
        depths: Optional[List[int]] = None,
        num_heads: Optional[List[int]] = None,
        sr_ratios: Optional[List[int]] = None,
        embed_dim: int = None,
        **kwargs,
    ):
        super().__init__()
        if embed_dim is not None:
            embed_dims = [embed_dim, embed_dim * 2, embed_dim * 4, embed_dim * 8]
        embed_dims = embed_dims or [64, 128, 320, 512]
        depths = depths or [2, 2, 2, 2]
        num_heads = num_heads or [1, 2, 5, 8]
        sr_ratios = sr_ratios or [8, 4, 2, 1]

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, embed_dims[0] // 2, 7, 2, 3, bias=False),
            nn.BatchNorm2d(embed_dims[0] // 2),
            nn.GELU(),
            nn.Conv2d(embed_dims[0] // 2, embed_dims[0], 3, 1, 1, bias=False),
            nn.BatchNorm2d(embed_dims[0]),
            nn.GELU(),
        )

        # PVT encoder stages
        self.stages = nn.ModuleList()
        in_c = embed_dims[0]
        for i in range(len(embed_dims)):
            self.stages.append(_PVTStage(in_c, embed_dims[i], depths[i], num_heads[i], sr_ratios[i]))
            in_c = embed_dims[i]

        # Decoder
        self.up_convs = nn.ModuleList()
        self.dec_convs = nn.ModuleList()
        for i in range(len(embed_dims) - 1, 0, -1):
            up_in_c = embed_dims[i]
            up_out_c = embed_dims[i - 1]
            self.up_convs.append(nn.ConvTranspose2d(up_in_c, up_out_c, 2, 2))
            # After cat: up_out_c + skip_channels
            # Skip from stage i has embed_dims[i] channels
            skip_c = embed_dims[i]
            cat_c = up_out_c + skip_c
            self.dec_convs.append(nn.Sequential(
                nn.Conv2d(cat_c, embed_dims[i - 1], 3, 1, 1, bias=False),
                nn.BatchNorm2d(embed_dims[i - 1]),
                nn.ReLU(inplace=True),
            ))

        self.head = nn.Sequential(
            nn.Conv2d(embed_dims[0], embed_dims[0] // 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(embed_dims[0] // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dims[0] // 2, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H_in, W_in = x.shape[2:]
        x = self.stem(x)

        skips = []
        for stage in self.stages:
            x = stage(x)
            skips.append(x)

        for up, dec in zip(self.up_convs, self.dec_convs):
            x = up(x)
            skip = skips.pop()
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        return self.head(F.interpolate(x, size=(H_in, W_in), mode="bilinear", align_corners=False))
