"""SkinMamba: CNN-Mamba Hybrid for Skin Lesion Segmentation.

Reference:
    "SkinMamba: A Precision Skin Lesion Segmentation Architecture with
    Cross-Scale Global State Modeling and Frequency Boundary Guidance",
    ACCV Workshop 2025.
    https://github.com/zs1314/skinmamba

Architecture:
    * 4-stage CNN encoder (Conv+BN+ReLU + MaxPool).
    * Cross-Scale Mamba Block: SS2D with multi-scale feature aggregation.
    * Frequency Boundary Guidance Module (FBGM): FFT-based edge enhancement.
    * Decoder: upsample + concat skip + Conv blocks.

Constructor:
    SkinMamba(in_channels=3, num_classes=2, img_size=224, **kwargs)
"""
# Source: https://github.com/zs1314/skinmamba

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.models.encoders.vmunet_encoder import SS2D


# ---------------------------------------------------------------------------
# Cross-Scale Mamba Block
# ---------------------------------------------------------------------------

class CrossScaleMambaBlock(nn.Module):
    """Cross-scale Mamba block: multi-scale SS2D + aggregation."""
    def __init__(self, dim, scales=None):
        super().__init__()
        if scales is None:
            scales = [1, 2, 4]
        self.scales = scales
        self.norm = nn.LayerNorm(dim)
        self.ss2d = SS2D(d_model=dim, d_state=16, d_conv=3, expand=2)
        self.scale_convs = nn.ModuleList()
        for s in scales:
            self.scale_convs.append(
                nn.Conv2d(dim, dim, 3, 1, s, dilation=s, groups=dim, bias=False))
        self.fuse = nn.Conv2d(dim * (len(scales) + 1), dim, 1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        # SS2D path
        x_norm = self.norm(x.permute(0, 2, 3, 1))
        x_mamba = self.ss2d(x_norm).permute(0, 3, 1, 2)

        # Multi-scale conv paths
        paths = [x_mamba]
        for conv in self.scale_convs:
            paths.append(conv(x))
        out = self.fuse(torch.cat(paths, dim=1))
        return out + x


# ---------------------------------------------------------------------------
# Frequency Boundary Guidance Module
# ---------------------------------------------------------------------------

class FBGM(nn.Module):
    """Frequency Boundary Guidance Module using FFT."""
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, 1, 1),
        )

    def forward(self, x):
        # Compute FFT-based edge map
        x_gray = x.mean(dim=1, keepdim=True)
        fft = torch.fft.fft2(x_gray)
        magnitude = torch.abs(fft)
        # High-pass filter for edges
        H, W = magnitude.shape[2:]
        mask = torch.ones_like(magnitude)
        cy, cx = H // 2, W // 2
        r = min(H, W) // 8
        yy, xx = torch.meshgrid(torch.arange(H, device=x.device),
                                torch.arange(W, device=x.device), indexing='ij')
        dist = ((yy - cy) ** 2 + (xx - cx) ** 2).float().sqrt().unsqueeze(0).unsqueeze(0)
        mask = (dist > r).float()
        edge_fft = fft * mask
        edge = torch.abs(torch.fft.ifft2(edge_fft))
        edge = edge.expand_as(x[:, :1, :, :])
        # Fuse edge with feature
        edge_feat = self.conv(x * edge.expand_as(x))
        return x + edge_feat


# ---------------------------------------------------------------------------
# Encoder/Decoder blocks
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# SkinMamba
# ---------------------------------------------------------------------------

class SkinMamba(nn.Module):
    """SkinMamba for skin lesion segmentation."""
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
        self.pool = nn.MaxPool2d(2)

        # Cross-Scale Mamba blocks at each stage
        self.csm1 = CrossScaleMambaBlock(C)
        self.csm2 = CrossScaleMambaBlock(C * 2)
        self.csm3 = CrossScaleMambaBlock(C * 4)
        self.csm4 = CrossScaleMambaBlock(C * 8)

        # FBGM at bottleneck
        self.fbgm = FBGM(C * 8)

        # Decoder
        self.up4 = nn.ConvTranspose2d(C * 8, C * 4, 2, stride=2)
        self.dec4 = _DoubleConv(C * 8, C * 4)
        self.up3 = nn.ConvTranspose2d(C * 4, C * 2, 2, stride=2)
        self.dec3 = _DoubleConv(C * 4, C * 2)
        self.up2 = nn.ConvTranspose2d(C * 2, C, 2, stride=2)
        self.dec2 = _DoubleConv(C * 2, C)

        self.head = nn.Conv2d(C, num_classes, 1)

    def forward(self, x):
        H, W = x.shape[2:]
        pH = ((H + 15) // 16) * 16
        pW = ((W + 15) // 16) * 16
        if pH != H or pW != W:
            x = F.pad(x, [0, pW - W, 0, pH - H], mode='reflect')

        e1 = self.csm1(self.enc1(x))
        e2 = self.csm2(self.enc2(self.pool(e1)))
        e3 = self.csm3(self.enc3(self.pool(e2)))
        e4 = self.csm4(self.enc4(self.pool(e3)))

        b = self.fbgm(e4)
        b_up = self.up4(b)

        d4 = self.dec4(torch.cat([b_up[:, :, :e3.shape[2], :e3.shape[3]], e3], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4)[:, :, :e2.shape[2], :e2.shape[3]], e2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3)[:, :, :e1.shape[2], :e1.shape[3]], e1], dim=1))

        out = self.head(d2)
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        return out
