"""PSPNet: Pyramid Scene Parsing Network for semantic segmentation.

Reference:
    Zhao et al., "Pyramid Scene Parsing Network", CVPR 2017.
    https://github.com/hszhao/PSPNet

Uses a ResNet-like encoder with a Pyramid Pooling Module (PPM) at the
bottleneck, followed by a simple decoder that upsamples to full resolution.
"""
# Source: https://github.com/hszhao/PSPNet

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class _ResBlock(nn.Module):
    """Residual block with optional downsampling."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.relu(out)


class _PyramidPoolModule(nn.Module):
    """Pyramid Pooling Module (PPM).

    Adaptive average pooling at 4 scales (1×1, 2×2, 3×3, 6×6) → 1×1 conv
    to reduce channels → bilinear upsample back to input resolution →
    concatenate with input.
    """

    def __init__(self, in_ch, pool_sizes=(1, 2, 3, 6)):
        super().__init__()
        self.pool_sizes = pool_sizes
        out_ch = in_ch // len(pool_sizes)
        self.stages = nn.ModuleList()
        for _ in pool_sizes:
            self.stages.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(1),  # placeholder
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ))
        # Bottleneck conv: in_ch + out_ch * len(pool_sizes) -> in_ch
        total = in_ch + out_ch * len(pool_sizes)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(total, in_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        features = [x]
        for stage, scale in zip(self.stages, self.pool_sizes):
            pool = F.adaptive_avg_pool2d(x, scale)
            feat = stage[1:](pool)  # skip the placeholder adaptive pool
            feat = F.interpolate(feat, size=(h, w), mode='bilinear',
                                 align_corners=True)
            features.append(feat)
        out = torch.cat(features, dim=1)
        return self.bottleneck(out)


class PSPNet(nn.Module):
    """Pyramid Scene Parsing Network.

    Architecture:
        - ResNet-like encoder with 4 stages
        - Pyramid Pooling Module at bottleneck
        - Dropout + bilinear upsample decoder

    Args:
        in_channels: number of input channels.
        num_classes: number of output segmentation classes.
        img_size: spatial size (unused, kept for API consistency).
    """

    def __init__(self, in_channels=3, num_classes=2, img_size=224):
        super().__init__()
        # Stem
        self.stem = nn.Sequential(
            _ConvBNReLU(in_channels, 64, 3, 1, 1),
            _ConvBNReLU(64, 64, 3, 1, 1),
        )

        # Encoder stages
        self.enc1 = nn.Sequential(
            _ResBlock(64, 64, stride=2),   # /2
            _ResBlock(64, 64),
        )
        self.enc2 = nn.Sequential(
            _ResBlock(64, 128, stride=2),  # /4
            _ResBlock(128, 128),
        )
        self.enc3 = nn.Sequential(
            _ResBlock(128, 256, stride=2), # /8
            _ResBlock(256, 256),
        )
        self.enc4 = nn.Sequential(
            _ResBlock(256, 512, stride=2), # /16
            _ResBlock(512, 512),
        )

        # Pyramid Pooling Module
        self.ppm = _PyramidPoolModule(512)

        # Dropout for regularization (as in original PSPNet)
        self.dropout = nn.Dropout2d(0.1)

        # Output conv
        self.out_conv = nn.Conv2d(512, num_classes, 1)

    def forward(self, x):
        inp_size = x.shape[2:]

        # Stem
        s = self.stem(x)

        # Encoder
        e1 = self.enc1(s)   # /2
        e2 = self.enc2(e1)  # /4
        e3 = self.enc3(e2)  # /8
        e4 = self.enc4(e3)  # /16

        # Pyramid Pooling Module
        p = self.ppm(e4)

        # Dropout + output conv
        p = self.dropout(p)
        out = self.out_conv(p)

        # Upsample to input resolution
        out = F.interpolate(out, size=inp_size, mode='bilinear', align_corners=True)

        return out
