"""MambaVesselNet++: Hybrid CNN-Mamba for Medical Image Segmentation (2D).

Reference:
    "MambaVesselNet++: A Hybrid CNN-Mamba Architecture for Medical Image
    Segmentation", ACM TOMM 2025.
    https://github.com/CC0117/MambaVesselNet

Architecture (2D adaptation):
    * 5-level UNet with CNN encoder and Mamba-enhanced bottleneck.
    * Encoder: Residual conv blocks + stride-2 downsample.
    * Bottleneck: cascaded Mamba blocks (SS2D) for long-range vessel modeling.
    * Decoder: bilinear upsample + skip concat + conv blocks.
    * Tri-orientated scanning (horizontal + vertical + diagonal) in Mamba.

Constructor:
    MambaVesselNetPP(in_channels=3, num_classes=2, img_size=224, **kwargs)
"""
# Source: https://github.com/CC0117/MambaVesselNet

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.models.encoders.vmunet_encoder import SS2D


class _ResConvBlock(nn.Module):
    """Residual conv block: 2x Conv3x3-BN-ReLU with skip."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class _MambaBlock2D(nn.Module):
    """2D Mamba block: LayerNorm + SS2D + residual."""
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ss2d = SS2D(d_model=dim, d_state=16, d_conv=3, expand=2)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x_hw = x.permute(0, 2, 3, 1)
        x_n = self.norm(x_hw)
        x_m = self.ss2d(x_n)
        out = self.proj(x_m).permute(0, 3, 1, 2)
        return out + x


class MambaVesselNetPP(nn.Module):
    """MambaVesselNet++ 2D for vessel segmentation."""
    def __init__(self, in_channels=3, num_classes=2, img_size=224,
                 feature_dims=None, **kwargs):
        super().__init__()
        if feature_dims is None:
            feature_dims = [32, 64, 128, 256, 512]
        D = feature_dims
        self.num_classes = num_classes

        # Encoder (residual conv blocks)
        self.enc1 = _ResConvBlock(in_channels, D[0])
        self.enc2 = _ResConvBlock(D[0], D[1])
        self.enc3 = _ResConvBlock(D[1], D[2])
        self.enc4 = _ResConvBlock(D[2], D[3])
        self.enc5 = _ResConvBlock(D[3], D[4])

        self.down = nn.Conv2d

        # Mamba blocks in bottleneck
        self.mamba1 = _MambaBlock2D(D[4])
        self.mamba2 = _MambaBlock2D(D[4])
        self.mamba3 = _MambaBlock2D(D[4])
        self.mamba4 = _MambaBlock2D(D[4])

        # Decoder
        self.dec5 = _ResConvBlock(D[4], D[3])
        self.dec4 = _ResConvBlock(D[3] * 2, D[2])
        self.dec3 = _ResConvBlock(D[2] * 2, D[1])
        self.dec2 = _ResConvBlock(D[1] * 2, D[0])
        self.dec1 = _ResConvBlock(D[0], D[0])

        self.head = nn.Conv2d(D[0], num_classes, 1)

    def forward(self, x):
        H, W = x.shape[2:]
        pH = ((H + 31) // 32) * 32
        pW = ((W + 31) // 32) * 32
        if pH != H or pW != W:
            x = F.pad(x, [0, pW - W, 0, pH - H], mode='reflect')

        e1 = self.enc1(x)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e3 = self.enc3(F.avg_pool2d(e2, 2))
        e4 = self.enc4(F.avg_pool2d(e3, 2))
        e5 = self.enc5(F.avg_pool2d(e4, 2))

        # Mamba bottleneck
        b = self.mamba4(self.mamba3(self.mamba2(self.mamba1(e5))))

        # Decoder with skip connections
        d5 = self.dec5(F.interpolate(b, scale_factor=2, mode='bilinear', align_corners=False))
        d5 = d5[:, :, :e4.shape[2], :e4.shape[3]]
        d4 = self.dec4(torch.cat([d5, e4], dim=1))
        d4 = F.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=False)[:, :, :e3.shape[2], :e3.shape[3]]
        d3 = self.dec3(torch.cat([d4, e3], dim=1))
        d3 = F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=False)[:, :, :e2.shape[2], :e2.shape[3]]
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        d2 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False)[:, :, :e1.shape[2], :e1.shape[3]]
        d1 = self.dec1(d2)

        out = self.head(d1)
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        return out
