"""SCSE-UNet – self-contained port from ai-med/squeeze_and_excitation.

Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional
Networks (Roy et al., MICCAI 2019).

Architecture: Standard UNet with scSE blocks applied after each encoder
and decoder convolution block. scSE combines channel SE (cSE) and
spatial SE (sSE) for joint recalibration.
"""
# Source: https://github.com/ai-med/squeeze_and_excitation

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class _DoubleConv(nn.Module):
    """Two consecutive Conv-BN-ReLU blocks."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class _cSEBlock(nn.Module):
    """Channel Squeeze & Excitation: GAP -> FC -> ReLU -> FC -> Sigmoid."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x)


class _sSEBlock(nn.Module):
    """Spatial Squeeze & Excitation: 1x1 conv -> Sigmoid."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.conv(x)


class _scSEBlock(nn.Module):
    """Concurrent Spatial and Channel SE: max(cSE, sSE)."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.cse = _cSEBlock(channels, reduction)
        self.sse = _sSEBlock(channels)

    def forward(self, x):
        return self.cse(x) + self.sse(x)


# ---------------------------------------------------------------------------
# SCSE-UNet
# ---------------------------------------------------------------------------
class SCSEUNet(nn.Module):
    """UNet with concurrent spatial and channel squeeze-excitation blocks.

    Args:
        in_channels: Number of input image channels (default 3).
        num_classes: Number of output segmentation classes (default 2).
        img_size: Expected input spatial resolution (default 224, unused).
    """

    def __init__(self, in_channels=3, num_classes=2, img_size=224, **kwargs):
        super().__init__()
        # Encoder
        self.enc1 = _DoubleConv(in_channels, 64)
        self.se1 = _scSEBlock(64)
        self.enc2 = _DoubleConv(64, 128)
        self.se2 = _scSEBlock(128)
        self.enc3 = _DoubleConv(128, 256)
        self.se3 = _scSEBlock(256)
        self.enc4 = _DoubleConv(256, 512)
        self.se4 = _scSEBlock(512)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = _DoubleConv(512, 1024)
        self.se_bn = _scSEBlock(1024)

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = _DoubleConv(1024, 512)
        self.se_d4 = _scSEBlock(512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = _DoubleConv(512, 256)
        self.se_d3 = _scSEBlock(256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = _DoubleConv(256, 128)
        self.se_d2 = _scSEBlock(128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = _DoubleConv(128, 64)
        self.se_d1 = _scSEBlock(64)

        # Output
        self.out_conv = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        # Encoder with scSE
        e1 = self.se1(self.enc1(x))
        e2 = self.se2(self.enc2(self.pool(e1)))
        e3 = self.se3(self.enc3(self.pool(e2)))
        e4 = self.se4(self.enc4(self.pool(e3)))

        # Bottleneck
        b = self.se_bn(self.bottleneck(self.pool(e4)))

        # Decoder with scSE
        d4 = self.se_d4(self.dec4(torch.cat([self.up4(b), e4], 1)))
        d3 = self.se_d3(self.dec3(torch.cat([self.up3(d4), e3], 1)))
        d2 = self.se_d2(self.dec2(torch.cat([self.up2(d3), e2], 1)))
        d1 = self.se_d1(self.dec1(torch.cat([self.up1(d2), e1], 1)))

        out = self.out_conv(d1)
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode='bilinear',
                                align_corners=False)
        return out
