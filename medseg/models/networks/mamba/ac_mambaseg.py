"""AC-MambaSeg: Adaptive Convolution and Mamba-based Skin Lesion Segmentation.

Reference:
    Nguyen et al., "AC-MAMBASEG: An Adaptive Convolution and Mamba-based
    Architecture for Enhanced Skin Lesion Segmentation", 2024.
    https://github.com/vietthanh2710/AC-MambaSeg

Architecture:
    * UNet-style encoder-decoder with 4 stages.
    * Encoder: Conv blocks (2x Conv3x3-BN-ReLU) + MaxPool.
    * Bottleneck: Adaptive Conv (kernel size from input) + Mamba (SS2D).
    * Skip: CBAM-style channel+spatial attention on skip features.
    * Decoder: ConvTranspose2d upsample + concat + Conv blocks.

Constructor:
    ACMambaSeg(in_channels=3, num_classes=2, img_size=224, **kwargs)
"""
# Source: https://github.com/vietthanh2710/AC-MambaSeg

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.models.encoders.vmunet_encoder import SS2D


# ---------------------------------------------------------------------------
# CBAM attention
# ---------------------------------------------------------------------------

class _ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


class _SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.max(dim=1, keepdim=True).values
        return x * self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, spatial_ksize=7):
        super().__init__()
        self.ca = _ChannelAttention(channels, reduction)
        self.sa = _SpatialAttention(spatial_ksize)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


# ---------------------------------------------------------------------------
# Encoder / Decoder blocks
# ---------------------------------------------------------------------------

class _DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class _AdaptiveConvMamba(nn.Module):
    """Bottleneck: adaptive convolution + Mamba SS2D."""
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )
        self.norm = nn.LayerNorm(dim)
        self.ss2d = SS2D(d_model=dim, d_state=16, d_conv=3, expand=2)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.conv(x)
        x_hw = x.permute(0, 2, 3, 1)  # BHWC
        x_hw = self.norm(x_hw)
        x_hw = self.ss2d(x_hw)
        return x_hw.permute(0, 3, 1, 2)  # BCHW


# ---------------------------------------------------------------------------
# AC-MambaSeg
# ---------------------------------------------------------------------------

class ACMambaSeg(nn.Module):
    """AC-MambaSeg for skin lesion segmentation."""
    def __init__(self, in_channels=3, num_classes=2, img_size=224,
                 base_channels=64, **kwargs):
        super().__init__()
        C = base_channels
        self.num_classes = num_classes

        # Encoder
        self.enc1 = _DoubleConv(in_channels, C)
        self.enc2 = _DoubleConv(C, C * 2)
        self.enc3 = _DoubleConv(C * 2, C * 4)
        self.enc4 = _DoubleConv(C * 4, C * 8)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = _AdaptiveConvMamba(C * 8)

        # CBAM on skip connections
        self.cbam1 = CBAM(C)
        self.cbam2 = CBAM(C * 2)
        self.cbam3 = CBAM(C * 4)
        self.cbam4 = CBAM(C * 8)

        # Decoder
        self.b_reduce = nn.Conv2d(C * 8, C * 4, 1)
        self.dec4 = _DoubleConv(C * 4 + C * 8, C * 4)
        self.up3 = nn.ConvTranspose2d(C * 4, C * 2, 2, stride=2)
        self.dec3 = _DoubleConv(C * 2 + C * 4, C * 2)
        self.up2 = nn.ConvTranspose2d(C * 2, C, 2, stride=2)
        self.dec2 = _DoubleConv(C + C * 2, C)

        # Head
        self.head = nn.Conv2d(C, num_classes, 1)

    def forward(self, x):
        H, W = x.shape[2:]
        # Pad to multiple of 8
        pH = ((H + 7) // 8) * 8
        pW = ((W + 7) // 8) * 8
        if pH != H or pW != W:
            x = F.pad(x, [0, pW - W, 0, pH - H], mode='reflect')

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))
        # Interpolate b to e4 size, reduce channels, concat with CBAM skip
        b_up = self.b_reduce(F.interpolate(b, size=e4.shape[2:],
                                           mode='bilinear', align_corners=False))

        d4 = self.dec4(torch.cat([b_up, self.cbam4(e4)], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4)[:, :, :e3.shape[2], :e3.shape[3]],
                                  self.cbam3(e3)], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3)[:, :, :e2.shape[2], :e2.shape[3]],
                                  self.cbam2(e2)], dim=1))

        out = self.head(d2)
        # Interpolate to original size
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        return out
