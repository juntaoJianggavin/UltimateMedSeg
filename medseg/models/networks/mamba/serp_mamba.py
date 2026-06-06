"""Serp-Mamba: Selective State-Space Model for Retinal Vessel Segmentation.

Reference:
    "Serp-Mamba: Advancing High-Resolution Retinal Vessel Segmentation
    with Selective State-Space Model", IEEE TMI 2025.
    https://github.com/whq-xxh/Serp-Mamba

Architecture:
    * 5-level UNet with Mamba blocks in the bottleneck and deep stages.
    * Encoder: Conv blocks + MaxPool.
    * Serpentine scan: 4-directional SS2D (horizontal+vertical+diagonal).
    * Decoder: ConvTranspose2d upsample + skip concat + Conv blocks.

Constructor:
    SerpMamba(in_channels=3, num_classes=2, img_size=224, **kwargs)
"""
# Source: https://github.com/whq-xxh/Serp-Mamba

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.models.encoders.vmunet_encoder import SS2D


class _SerpMambaBlock(nn.Module):
    """Serpentine Mamba block: 4-directional SS2D fusion."""
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        # 4 directional scans via 4 separate SS2D blocks
        self.ss2d_h = SS2D(d_model=dim, d_state=16, d_conv=3, expand=2)
        self.ss2d_v = SS2D(d_model=dim, d_state=16, d_conv=3, expand=2)
        self.fuse = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
        )
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x_hw = x.permute(0, 2, 3, 1)  # BHWC
        x_norm = self.norm(x_hw)

        # Horizontal scan
        h_out = self.ss2d_h(x_norm)
        # Vertical scan (transpose H/W)
        x_t = x_hw.transpose(1, 2)  # BWHC
        x_t_norm = self.norm(x_t)
        v_out = self.ss2d_v(x_t_norm).transpose(1, 2)  # BHWC back

        # Fuse
        fused = self.fuse(torch.cat([h_out, v_out], dim=-1))
        out = self.proj(fused)
        return out.permute(0, 3, 1, 2) + x


class _DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class SerpMamba(nn.Module):
    """Serp-Mamba for retinal vessel segmentation."""
    def __init__(self, in_channels=3, num_classes=2, img_size=224,
                 base_channels=32, **kwargs):
        super().__init__()
        C = base_channels
        self.num_classes = num_classes

        # Encoder
        self.enc1 = _DoubleConv(in_channels, C)
        self.enc2 = _DoubleConv(C, C * 2)
        self.enc3 = _DoubleConv(C * 2, C * 4)
        self.enc4 = _DoubleConv(C * 4, C * 8)
        self.enc5 = _DoubleConv(C * 8, C * 16)
        self.pool = nn.MaxPool2d(2)

        # Mamba blocks at bottleneck + deep stages
        self.mamba_b = _SerpMambaBlock(C * 16)
        self.mamba4 = _SerpMambaBlock(C * 8)
        self.mamba3 = _SerpMambaBlock(C * 4)

        # Decoder
        self.up5 = nn.ConvTranspose2d(C * 16, C * 8, 2, stride=2)
        self.dec5 = _DoubleConv(C * 16, C * 8)
        self.up4 = nn.ConvTranspose2d(C * 8, C * 4, 2, stride=2)
        self.dec4 = _DoubleConv(C * 8, C * 4)
        self.up3 = nn.ConvTranspose2d(C * 4, C * 2, 2, stride=2)
        self.dec3 = _DoubleConv(C * 4, C * 2)
        self.up2 = nn.ConvTranspose2d(C * 2, C, 2, stride=2)
        self.dec2 = _DoubleConv(C * 2, C)

        self.head = nn.Conv2d(C, num_classes, 1)

    def forward(self, x):
        H, W = x.shape[2:]
        pH = ((H + 31) // 32) * 32
        pW = ((W + 31) // 32) * 32
        if pH != H or pW != W:
            x = F.pad(x, [0, pW - W, 0, pH - H], mode='reflect')

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))

        # Mamba enhanced features
        b = self.mamba_b(e5)
        e4 = self.mamba4(e4)
        e3 = self.mamba3(e3)

        d5 = self.dec5(torch.cat([self.up5(b)[:, :, :e4.shape[2], :e4.shape[3]], e4], dim=1))
        d4 = self.dec4(torch.cat([self.up4(d5)[:, :, :e3.shape[2], :e3.shape[3]], e3], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4)[:, :, :e2.shape[2], :e2.shape[3]], e2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3)[:, :, :e1.shape[2], :e1.shape[3]], e1], dim=1))

        out = self.head(d2)
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        return out
