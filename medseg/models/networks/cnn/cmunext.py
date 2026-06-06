"""CMUNeXt: Efficient Medical Image Segmentation with Large Kernel + Skip Fusion.

Lightweight CNN-based architecture using large kernel convolutions and
a skip fusion module for efficient medical image segmentation.

Reference:
    CMUNeXt: An Efficient Medical Image Segmentation Network Based on
    Large Kernel and Skip Fusion. arXiv 2023.

Key components:
    - ConvNeXt-style blocks with large kernels (7x7)
    - Skip Fusion Module for multi-scale feature aggregation
    - Lightweight encoder-decoder design
"""
# Source: https://github.com/FengheTan9/CMUNeXt

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class _ConvNeXtBlock(nn.Module):
    """ConvNeXt-style block: DWConv(7x7) -> LN -> Linear -> GELU -> Linear."""

    def __init__(self, dim, kernel_size=7):
        super().__init__()
        pad = kernel_size // 2
        self.dw = nn.Conv2d(dim, dim, kernel_size, 1, pad, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pw1 = nn.Linear(dim, dim * 4)
        self.act = nn.GELU()
        self.pw2 = nn.Linear(dim * 4, dim)
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim))

    def forward(self, x):
        residual = x
        x = self.dw(x)
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.norm(x)
        x = self.pw2(self.act(self.pw1(x)))
        x = x * self.gamma
        x = x.permute(0, 3, 1, 2)
        return x + residual


class _SkipFusion(nn.Module):
    """Skip Fusion: channel attention to fuse encoder skip with decoder feature."""

    def __init__(self, dim):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, max(dim // 4, 4)),
            nn.ReLU(inplace=True),
            nn.Linear(max(dim // 4, 4), dim),
            nn.Sigmoid(),
        )

    def forward(self, skip, dec):
        B, C = skip.shape[:2]
        w = self.fc(self.gap(skip).view(B, C)).view(B, C, 1, 1)
        return skip * w + dec


class CMUNeXt(nn.Module):
    """CMUNeXt: Large kernel CNN + Skip Fusion.

    Args:
        in_channels: Input channels.
        num_classes: Segmentation classes.
        img_size: Input spatial size.
        embed_dims: Channel dims per stage.
        depths: Blocks per stage.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        img_size: int = 224,
        embed_dims: Optional[List[int]] = None,
        depths: Optional[List[int]] = None,
        **kwargs,
    ):
        super().__init__()
        embed_dims = embed_dims or [32, 64, 128, 256]
        depths = depths or [2, 2, 2, 2]

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, embed_dims[0], 4, 4, bias=False),
            nn.BatchNorm2d(embed_dims[0]),
        )

        # Encoder
        self.enc_stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i in range(len(embed_dims)):
            self.enc_stages.append(nn.Sequential(
                *[_ConvNeXtBlock(embed_dims[i]) for _ in range(depths[i])]
            ))
            if i < len(embed_dims) - 1:
                self.downsamples.append(nn.Sequential(
                    nn.Conv2d(embed_dims[i], embed_dims[i + 1], 2, 2),
                ))

        # Decoder
        self.upsamples = nn.ModuleList()
        self.fusions = nn.ModuleList()
        self.dec_stages = nn.ModuleList()
        for i in range(len(embed_dims) - 1, 0, -1):
            self.upsamples.append(nn.ConvTranspose2d(embed_dims[i], embed_dims[i - 1], 2, 2))
            self.fusions.append(_SkipFusion(embed_dims[i - 1]))
            self.dec_stages.append(nn.Sequential(
                *[_ConvNeXtBlock(embed_dims[i - 1]) for _ in range(depths[i - 1])]
            ))

        # Head: 4x upsample + conv
        self.head = nn.Sequential(
            nn.ConvTranspose2d(embed_dims[0], embed_dims[0], 4, 4),
            nn.Conv2d(embed_dims[0], num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H_in, W_in = x.shape[2:]
        x = self.stem(x)

        skips = []
        for i, stage in enumerate(self.enc_stages):
            x = stage(x)
            if i < len(self.downsamples):
                skips.append(x)
                x = self.downsamples[i](x)

        for up, fusion, dec in zip(self.upsamples, self.fusions, self.dec_stages):
            x = up(x)
            skip = skips.pop()
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = fusion(skip, x)
            x = dec(x)

        x = self.head(x)
        if x.shape[2:] != (H_in, W_in):
            x = F.interpolate(x, size=(H_in, W_in), mode="bilinear", align_corners=False)
        return x
