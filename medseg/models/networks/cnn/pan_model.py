"""PAN: Pyramid Attention Network for semantic segmentation.

Reference:
    Li et al., "Pyramid Attention Network for Semantic Segmentation", 2018.
    https://github.com/JiaweiWang-AI/Pyramid-Attention-Networks-pytorch

Uses a ResNet-like encoder with pyramid attention modules that combine
multi-scale global context via spatial pyramid pooling with attention.
"""
# Source: https://github.com/JaveyWang/Pyramid-Attention-Networks-pytorch

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
    """Basic residual block with optional stride for downsampling."""

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


class _PyramidAttention(nn.Module):
    """Pyramid attention module.

    Multi-scale pooling (1×1, 2×2, 3×3, 6×6) → 1×1 conv reduce → upsample
    → concat → 1×1 conv → sigmoid attention map applied to input.
    """

    def __init__(self, in_ch, reduction=4):
        super().__init__()
        mid_ch = in_ch // reduction
        self.pool_scales = [1, 2, 3, 6]
        self.branches = nn.ModuleList()
        for _ in self.pool_scales:
            self.branches.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(1),  # placeholder, used per-scale in forward
                nn.Conv2d(in_ch, mid_ch, 1, bias=False),
                nn.BatchNorm2d(mid_ch),
                nn.ReLU(inplace=True),
            ))
        self.fuse = nn.Sequential(
            nn.Conv2d(mid_ch * len(self.pool_scales), in_ch, 1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.Sigmoid(),
        )

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        branches = []
        for branch, scale in zip(self.branches, self.pool_scales):
            pool = F.adaptive_avg_pool2d(x, scale)
            feat = branch[1:](pool)  # skip first adaptive pool in branch
            feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=True)
            branches.append(feat)
        att = torch.cat(branches, dim=1)
        att = self.fuse(att)
        return x * att


class _PANEncoder(nn.Module):
    """Encoder stage: stack of ResBlocks."""

    def __init__(self, in_ch, out_ch, num_blocks=2, stride=2):
        super().__init__()
        layers = [_ResBlock(in_ch, out_ch, stride)]
        for _ in range(1, num_blocks):
            layers.append(_ResBlock(out_ch, out_ch))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class _PANDecoder(nn.Module):
    """Decoder stage: upsample + skip concat + conv."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            _ConvBNReLU(in_ch + skip_ch, out_ch),
            _ConvBNReLU(out_ch, out_ch),
        )

    def forward(self, x, skip=None):
        if skip is not None:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, skip], dim=1)
        else:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return self.conv(x)


class PAN(nn.Module):
    """Pyramid Attention Network.

    Args:
        in_channels: number of input channels.
        num_classes: number of output segmentation classes.
        img_size: spatial size (unused, kept for API consistency).
    """

    def __init__(self, in_channels=3, num_classes=2, img_size=224):
        super().__init__()
        # Stem
        self.stem = nn.Sequential(
            _ConvBNReLU(in_channels, 32, 3, 1, 1),
            _ConvBNReLU(32, 64, 3, 1, 1),
        )

        # Encoder stages
        self.enc1 = _PANEncoder(64, 64, num_blocks=2, stride=2)    # /2
        self.enc2 = _PANEncoder(64, 128, num_blocks=2, stride=2)   # /4
        self.enc3 = _PANEncoder(128, 256, num_blocks=2, stride=2)  # /8
        self.enc4 = _PANEncoder(256, 512, num_blocks=2, stride=2)  # /16

        # Pyramid attention at bottleneck
        self.pa = _PyramidAttention(512)

        # Decoder
        self.dec3 = _PANDecoder(512, 256, 256)
        self.dec2 = _PANDecoder(256, 128, 128)
        self.dec1 = _PANDecoder(128, 64, 64)

        # Final upsampling decoder (no skip, goes back to stem resolution)
        self.dec0 = nn.Sequential(
            _ConvBNReLU(64, 64),
            _ConvBNReLU(64, 64),
        )

        # Output
        self.out_conv = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        inp = x
        # Stem
        s = self.stem(x)

        # Encoder
        e1 = self.enc1(s)    # /2
        e2 = self.enc2(e1)   # /4
        e3 = self.enc3(e2)   # /8
        e4 = self.enc4(e3)   # /16

        # Pyramid attention
        e4 = self.pa(e4)

        # Decoder
        d = self.dec3(e4, e3)  # /8
        d = self.dec2(d, e2)   # /4
        d = self.dec1(d, e1)   # /2

        # Upsample to original size
        d = F.interpolate(d, size=inp.shape[2:], mode='bilinear', align_corners=True)
        d = self.dec0(d)

        return self.out_conv(d)
