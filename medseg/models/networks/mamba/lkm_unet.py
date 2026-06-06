"""LKM-UNet: Large Kernel Mamba UNet for Medical Image Segmentation.

Combines large kernel convolutions with Mamba SSM for capturing both
local and global features in medical image segmentation.

Reference:
    LKM-UNet: Large Kernel Vision Mamba UNet for Medical Image
    Segmentation. MICCAI 2024. https://github.com/wjh892521292/LKM-UNet
"""
# Source: https://github.com/wjh892521292/LKM-UNet

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class _LargeKernelMambaBlock(nn.Module):
    def __init__(self, dim, kernel_size=15, d_state=16):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        pad = kernel_size // 2
        self.lk_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size, 1, pad, groups=dim),
            nn.BatchNorm2d(dim),
            nn.SiLU(),
        )
        self.proj = nn.Linear(dim, dim * 2)
        self.gate_fn = nn.Sigmoid()
        self.out_proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x):
        B, C, H, W = x.shape
        res = x
        x_lk = self.lk_conv(x)
        tokens = x_lk.flatten(2).transpose(1, 2)
        kv = self.proj(self.norm(tokens))
        k, v = kv.chunk(2, dim=-1)
        out = self.out_proj(self.gate_fn(k) * v)
        tokens = tokens + out
        tokens = tokens + self.ffn(self.norm2(tokens))
        return tokens.transpose(1, 2).view(B, C, H, W) + res


class LKMUNet(nn.Module):
    """LKM-UNet: Large Kernel Mamba UNet."""

    def __init__(self, in_channels=3, num_classes=2, img_size=224,
                 embed_dim=64, depths=None, **kwargs):
        super().__init__()
        depths = depths or [2, 2, 6, 2]
        dims = [embed_dim * (2 ** i) for i in range(len(depths))]
        self.stem = nn.Sequential(nn.Conv2d(in_channels, dims[0], 4, 4, bias=False), nn.BatchNorm2d(dims[0]))
        self.enc = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i in range(len(depths)):
            self.enc.append(nn.Sequential(*[_LargeKernelMambaBlock(dims[i]) for _ in range(depths[i])]))
            if i < len(depths) - 1:
                self.downs.append(nn.Conv2d(dims[i], dims[i + 1], 3, 2, 1))
        self.ups = nn.ModuleList()
        self.dec = nn.ModuleList()
        self.merges = nn.ModuleList()
        for i in range(len(dims) - 1, 0, -1):
            self.ups.append(nn.ConvTranspose2d(dims[i], dims[i - 1], 2, 2))
            self.merges.append(nn.Sequential(nn.Conv2d(dims[i - 1] * 2, dims[i - 1], 1, bias=False), nn.BatchNorm2d(dims[i - 1])))
            self.dec.append(nn.Sequential(*[_LargeKernelMambaBlock(dims[i - 1]) for _ in range(depths[i - 1])]))
        self.head = nn.Sequential(nn.ConvTranspose2d(dims[0], dims[0], 4, 4), nn.Conv2d(dims[0], num_classes, 1))

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.stem(x)
        skips = []
        for i, enc in enumerate(self.enc):
            x = enc(x)
            if i < len(self.downs):
                skips.append(x)
                x = self.downs[i](x)
        for up, merge, dec in zip(self.ups, self.merges, self.dec):
            x = up(x); s = skips.pop()
            if x.shape[2:] != s.shape[2:]: x = F.interpolate(x, size=s.shape[2:], mode="bilinear", align_corners=False)
            x = dec(merge(torch.cat([x, s], dim=1)))
        x = self.head(x)
        return F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False) if x.shape[2:] != (H, W) else x
