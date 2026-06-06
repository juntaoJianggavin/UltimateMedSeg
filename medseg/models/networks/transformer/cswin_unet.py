"""CSWin-UNet: Transformer UNet with Cross-Shaped Windows.

Uses CSWin self-attention (cross-shaped window) in a U-Net architecture
for efficient medical image segmentation.

Reference:
    CSWin-UNet: Transformer UNet with Cross-Shaped Windows for Medical
    Image Segmentation. arXiv 2024.
    https://github.com/pgao-lab/CSWin-UNet

Key components:
    - Cross-shaped window attention (horizontal + vertical strips)
    - U-shaped encoder-decoder with skip connections
    - Merge blocks for multi-scale feature aggregation
"""
# Source: https://github.com/eatbeanss/CSWin-UNet

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class _CrossWindowAttention(nn.Module):
    """Cross-shaped window self-attention: horizontal + vertical strips."""

    def __init__(self, dim, num_heads=4, strip_size=7):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.strip_size = strip_size
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        qkv = self.qkv(tokens).reshape(B, -1, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, -1, C)
        out = self.proj(out).transpose(1, 2).view(B, C, H, W)
        return out


class _CSWinBlock(nn.Module):
    def __init__(self, dim, num_heads=4, strip_size=7, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _CrossWindowAttention(dim, num_heads, strip_size)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = tokens + self.attn(x).flatten(2).transpose(1, 2)
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens.transpose(1, 2).view(B, C, H, W)


class _CSWinStage(nn.Module):
    def __init__(self, dim, num_heads, depth, strip_size=7, downsample=False):
        super().__init__()
        self.blocks = nn.Sequential(*[
            _CSWinBlock(dim, num_heads, strip_size) for _ in range(depth)
        ])
        self.downsample = None
        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(dim, dim * 2, 3, 2, 1, bias=False),
                nn.BatchNorm2d(dim * 2),
            )

    def forward(self, x):
        x = self.blocks(x)
        out = x
        if self.downsample is not None:
            x = self.downsample(x)
            return out, x
        return out


class CSWinUNet(nn.Module):
    """CSWin-UNet: Cross-shaped Window Transformer UNet.

    Args:
        in_channels: Input channels.
        num_classes: Segmentation classes.
        img_size: Input spatial size.
        embed_dim: Base embedding dimension.
        depths: Blocks per stage.
        num_heads: Attention heads per stage.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        img_size: int = 224,
        embed_dim: int = 64,
        depths: Optional[List[int]] = None,
        num_heads: Optional[List[int]] = None,
        **kwargs,
    ):
        super().__init__()
        depths = depths or [2, 2, 6, 2]
        num_heads = num_heads or [2, 4, 8, 16]
        dims = [embed_dim * (2 ** i) for i in range(len(depths))]

        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], 4, 4, bias=False),
            nn.BatchNorm2d(dims[0]),
        )

        # Encoder
        self.enc_stages = nn.ModuleList()
        for i in range(len(depths)):
            ds = i < len(depths) - 1
            self.enc_stages.append(
                _CSWinStage(dims[i], num_heads[i], depths[i], downsample=ds)
            )

        # Decoder
        self.up_convs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in range(len(dims) - 1, 0, -1):
            self.up_convs.append(nn.ConvTranspose2d(dims[i], dims[i - 1], 2, 2))
            self.dec_blocks.append(nn.Sequential(
                nn.Conv2d(dims[i - 1] * 2, dims[i - 1], 3, 1, 1, bias=False),
                nn.BatchNorm2d(dims[i - 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(dims[i - 1], dims[i - 1], 3, 1, 1, bias=False),
                nn.BatchNorm2d(dims[i - 1]),
                nn.ReLU(inplace=True),
            ))

        self.head = nn.Sequential(
            nn.Conv2d(dims[0], dims[0] // 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(dims[0] // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(dims[0] // 2, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H_in, W_in = x.shape[2:]
        x = self.patch_embed(x)

        skips = []
        for i, stage in enumerate(self.enc_stages):
            if i < len(self.enc_stages) - 1:
                out, x = stage(x)
                skips.append(out)
            else:
                x = stage(x)

        for up, dec in zip(self.up_convs, self.dec_blocks):
            x = up(x)
            skip = skips.pop()
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        return self.head(F.interpolate(x, size=(H_in, W_in), mode="bilinear", align_corners=False))
