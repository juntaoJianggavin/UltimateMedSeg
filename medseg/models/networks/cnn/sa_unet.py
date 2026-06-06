"""SA-UNet: Spatial Attention UNet for medical image segmentation.

Reference:
    Guo et al., "SA-UNet: Spatial Attention UNet for Retinal Vessel
    Segmentation", ICPR 2021.
    https://github.com/clguo/SA-UNet

Lightweight UNet with spatial-attention modules applied on skip connections.
The spatial attention module uses average + max pooling → concat → 7×7 conv
→ sigmoid, following CBAM-style spatial attention but lighter.
"""
# Source: https://github.com/clguo/SA-UNet

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBNReLU(nn.Module):
    """Conv → BN → ReLU."""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class _DoubleConv(nn.Module):
    """Two consecutive ConvBNReLU blocks."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            _ConvBNReLU(in_ch, out_ch),
            _ConvBNReLU(out_ch, out_ch),
        )

    def forward(self, x):
        return self.block(x)


class _SpatialAttention(nn.Module):
    """Spatial attention module (CBAM-style, lightweight).

    Applies channel-wise average-pool and max-pool, concatenates along the
    channel axis, then a 7×7 conv + sigmoid to produce a spatial attention map.
    """

    def __init__(self, kernel_size=7):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=pad, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        att = torch.cat([avg_out, max_out], dim=1)
        att = self.sigmoid(self.conv(att))
        return x * att


class _SAEncoder(nn.Module):
    """Encoder block with spatial attention at the output."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = _DoubleConv(in_ch, out_ch)
        self.sa = _SpatialAttention()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        feat = self.conv(x)
        feat = self.sa(feat)
        down = self.pool(feat)
        return feat, down


class SADecoder(nn.Module):
    """Decoder block: upsample → concat skip → double-conv → spatial attn."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = _DoubleConv(in_ch // 2 + skip_ch, out_ch)
        self.sa = _SpatialAttention()

    def forward(self, x, skip):
        x = self.up(x)
        # pad if needed
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)
        x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                       diff_h // 2, diff_h - diff_h // 2])
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.sa(x)
        return x


class SAUNet(nn.Module):
    """SA-UNet: Spatial Attention UNet.

    Args:
        in_channels: number of input channels.
        num_classes: number of output segmentation classes.
        img_size: spatial size (unused, kept for API consistency).
    """

    def __init__(self, in_channels=3, num_classes=2, img_size=224):
        super().__init__()
        # Encoder
        self.enc1 = _SAEncoder(in_channels, 32)
        self.enc2 = _SAEncoder(32, 64)
        self.enc3 = _SAEncoder(64, 128)
        self.enc4 = _SAEncoder(128, 256)

        # Bottleneck
        self.bottleneck = _DoubleConv(256, 512)

        # Decoder
        self.dec4 = SADecoder(512, 256, 256)
        self.dec3 = SADecoder(256, 128, 128)
        self.dec2 = SADecoder(128, 64, 64)
        self.dec1 = SADecoder(64, 32, 32)

        # Output
        self.out_conv = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        # Encoder
        s1, d1 = self.enc1(x)
        s2, d2 = self.enc2(d1)
        s3, d3 = self.enc3(d2)
        s4, d4 = self.enc4(d3)

        # Bottleneck
        b = self.bottleneck(d4)

        # Decoder
        d = self.dec4(b, s4)
        d = self.dec3(d, s3)
        d = self.dec2(d, s2)
        d = self.dec1(d, s1)

        return self.out_conv(d)
