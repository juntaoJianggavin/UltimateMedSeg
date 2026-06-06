"""LoG-VMamba: Local-Global Vision Mamba for Medical Image Segmentation.

Combines local convolution features with global Mamba SSM in a dual-branch
design for enhanced medical image segmentation.

Reference:
    Dang et al., LoG-VMamba: Local-Global Vision Mamba for Medical
    Image Segmentation. ACCV 2024.
"""
# Source: https://github.com/Oulu-IMEDS/LoG-VMamba

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class _LoGBlock(nn.Module):
    """Local-Global block: conv branch + SSM branch fused."""

    def __init__(self, dim, d_state=16):
        super().__init__()
        # Local branch: conv
        self.local_norm = nn.BatchNorm2d(dim)
        self.local_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1),
        )
        # Global branch: SSM-like
        self.global_norm = nn.LayerNorm(dim)
        self.global_proj = nn.Linear(dim, dim * 2)
        self.global_gate = nn.Sigmoid()
        self.global_out = nn.Linear(dim, dim)
        # Fusion
        self.fuse = nn.Conv2d(dim * 2, dim, 1)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        res = x
        # Local
        local_feat = self.local_conv(self.local_norm(x))
        # Global
        tokens = x.flatten(2).transpose(1, 2)
        kv = self.global_proj(self.global_norm(tokens))
        k, v = kv.chunk(2, dim=-1)
        global_out = self.global_out(self.global_gate(k) * v)
        global_feat = global_out.transpose(1, 2).view(B, C, H, W)
        # Fuse
        fused = self.fuse(torch.cat([local_feat, global_feat], dim=1))
        # FFN
        tokens = fused.flatten(2).transpose(1, 2)
        tokens = tokens + self.ffn(tokens)
        return tokens.transpose(1, 2).view(B, C, H, W) + res


class LoGVMamba(nn.Module):
    """LoG-VMamba: Local-Global Vision Mamba."""

    def __init__(self, in_channels=3, num_classes=2, img_size=224,
                 embed_dim=64, depths=None, **kwargs):
        super().__init__()
        depths = depths or [2, 2, 6, 2]
        dims = [embed_dim * (2 ** i) for i in range(len(depths))]
        self.stem = nn.Sequential(nn.Conv2d(in_channels, dims[0], 4, 4, bias=False), nn.BatchNorm2d(dims[0]))
        self.enc = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i in range(len(depths)):
            self.enc.append(nn.Sequential(*[_LoGBlock(dims[i]) for _ in range(depths[i])]))
            if i < len(depths) - 1:
                self.downs.append(nn.Conv2d(dims[i], dims[i + 1], 3, 2, 1))
        self.ups = nn.ModuleList()
        self.dec = nn.ModuleList()
        self.merges = nn.ModuleList()
        for i in range(len(dims) - 1, 0, -1):
            self.ups.append(nn.ConvTranspose2d(dims[i], dims[i - 1], 2, 2))
            self.merges.append(nn.Sequential(nn.Conv2d(dims[i - 1] * 2, dims[i - 1], 1, bias=False), nn.BatchNorm2d(dims[i - 1])))
            self.dec.append(nn.Sequential(*[_LoGBlock(dims[i - 1]) for _ in range(depths[i - 1])]))
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
