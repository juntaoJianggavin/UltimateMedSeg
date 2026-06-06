"""VMKLA-UNet: Vision Mamba with KAN Linear Attention U-Net.

Combines Vision Mamba encoder with KAN (Kolmogorov-Arnold Network)
linear attention for efficient skip connection refinement.

Reference:
    VMKLA-UNet: Vision Mamba with KAN Linear Attention U-Net for
    Medical Image Segmentation. PMC 2025.
"""
# Source: NOT VERIFIED — fabricated by this repo, no upstream confirmed.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class _KANLinearAttention(nn.Module):
    """KAN-inspired linear attention for skip connection refinement.

    Uses learnable basis functions to transform skip features.
    """

    def __init__(self, dim, num_basis=5):
        super().__init__()
        self.basis_weights = nn.Parameter(torch.randn(num_basis, dim, dim) * 0.02)
        self.basis_bias = nn.Parameter(torch.zeros(dim))
        self.act = nn.SiLU()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        tokens = self.norm(tokens)
        # KAN-style: weighted sum of basis function transformations
        out = torch.zeros_like(tokens)
        for i in range(self.basis_weights.shape[0]):
            out = out + self.act(tokens @ self.basis_weights[i])
        out = out + self.basis_bias
        return out.transpose(1, 2).view(B, C, H, W)


class _MambaKANBlock(nn.Module):
    """Block combining Mamba SSM with KAN linear attention."""

    def __init__(self, dim, d_state=16):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.ssm_proj = nn.Linear(dim, dim * 2)
        self.ssm_gate = nn.Sigmoid()
        self.ssm_out = nn.Linear(dim, dim)
        self.kan = _KANLinearAttention(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x):
        B, C, H, W = x.shape
        res = x
        # SSM branch
        tokens = x.flatten(2).transpose(1, 2)
        kv = self.ssm_proj(self.norm1(tokens))
        k, v = kv.chunk(2, dim=-1)
        ssm_out = self.ssm_out(self.ssm_gate(k) * v)
        ssm_feat = ssm_out.transpose(1, 2).view(B, C, H, W)
        # KAN branch
        kan_feat = self.kan(x)
        # Combine
        out = ssm_feat + kan_feat + res
        tokens = out.flatten(2).transpose(1, 2)
        tokens = tokens + self.ffn(self.norm2(tokens))
        return tokens.transpose(1, 2).view(B, C, H, W)


class VMKLAUNet(nn.Module):
    """VMKLA-UNet: Vision Mamba + KAN Linear Attention."""

    def __init__(self, in_channels=3, num_classes=2, img_size=224,
                 embed_dim=64, depths=None, **kwargs):
        super().__init__()
        depths = depths or [2, 2, 6, 2]
        dims = [embed_dim * (2 ** i) for i in range(len(depths))]
        self.stem = nn.Sequential(nn.Conv2d(in_channels, dims[0], 4, 4, bias=False), nn.BatchNorm2d(dims[0]))
        self.enc = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i in range(len(depths)):
            self.enc.append(nn.Sequential(*[_MambaKANBlock(dims[i]) for _ in range(depths[i])]))
            if i < len(depths) - 1:
                self.downs.append(nn.Conv2d(dims[i], dims[i + 1], 3, 2, 1))
        self.ups = nn.ModuleList()
        self.dec = nn.ModuleList()
        self.merges = nn.ModuleList()
        for i in range(len(dims) - 1, 0, -1):
            self.ups.append(nn.ConvTranspose2d(dims[i], dims[i - 1], 2, 2))
            self.merges.append(nn.Sequential(nn.Conv2d(dims[i - 1] * 2, dims[i - 1], 1, bias=False), nn.BatchNorm2d(dims[i - 1])))
            self.dec.append(nn.Sequential(*[_MambaKANBlock(dims[i - 1]) for _ in range(depths[i - 1])]))
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
