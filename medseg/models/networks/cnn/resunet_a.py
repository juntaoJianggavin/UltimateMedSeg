"""ResUNet-a – self-contained port from feevos/resuneta.

ResUNet-a: A deep learning framework for semantic segmentation of remotely
sensed data (Diakogiannis et al., ISPRS 2020).

Architecture: 4-level encoder with residual + dilated convolutions,
PSP pooling at the bottleneck, and multi-scale decoder with
conditional random field-inspired boundary refinement.
"""
# Source: https://github.com/feevos/resuneta

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class _ResBlock(nn.Module):
    """Residual block: 2x Conv-BN with identity shortcut."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            if in_ch != out_ch or stride != 1
            else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + self.shortcut(x))


class _DilatedBlock(nn.Module):
    """Dilated residual block with multi-dilation rates (1, 2, 4, 8)."""

    def __init__(self, channels, dilations=(1, 2, 4, 8)):
        super().__init__()
        self.blocks = nn.ModuleList()
        for d in dilations:
            self.blocks.append(nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=d, dilation=d,
                          bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            ))

    def forward(self, x):
        out = x
        for block in self.blocks:
            out = out + block(x)
        return out / (len(self.blocks) + 1)


class _PSPPool(nn.Module):
    """Pyramid Spatial Pooling: adaptive pooling at 4 scales + 1x1 conv."""

    def __init__(self, in_ch, out_ch, pool_sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(s),
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
            for s in pool_sizes
        ])
        self.final = nn.Sequential(
            nn.Conv2d(in_ch + out_ch * len(pool_sizes), out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        h, w = x.shape[2:]
        feats = [x]
        for stage in self.stages:
            f = stage(x)
            f = F.interpolate(f, size=(h, w), mode='bilinear',
                              align_corners=False)
            feats.append(f)
        return self.final(torch.cat(feats, dim=1))


# ---------------------------------------------------------------------------
# ResUNet-a
# ---------------------------------------------------------------------------
class ResUNetA(nn.Module):
    """ResUNet-a with 4 encoder levels, dilated bottleneck, and PSP pooling.

    Args:
        in_channels: Number of input image channels (default 3).
        num_classes: Number of output segmentation classes (default 2).
        img_size: Expected input spatial resolution (default 224, unused).
    """

    def __init__(self, in_channels=3, num_classes=2, img_size=224, **kwargs):
        super().__init__()
        filters = [32, 64, 128, 256, 512]

        # Encoder
        self.enc1 = _ResBlock(in_channels, filters[0])
        self.enc2 = _ResBlock(filters[0], filters[1], stride=2)
        self.enc3 = _ResBlock(filters[1], filters[2], stride=2)
        self.enc4 = _ResBlock(filters[2], filters[3], stride=2)

        # Bottleneck with dilated convolutions + PSP
        self.bottleneck = nn.Sequential(
            _ResBlock(filters[3], filters[4], stride=2),
            _DilatedBlock(filters[4]),
            _PSPPool(filters[4], filters[4]),
        )

        # Decoder
        self.up4 = nn.ConvTranspose2d(filters[4], filters[3], 2, stride=2)
        self.dec4 = _ResBlock(filters[3] * 2, filters[3])
        self.up3 = nn.ConvTranspose2d(filters[3], filters[2], 2, stride=2)
        self.dec3 = _ResBlock(filters[2] * 2, filters[2])
        self.up2 = nn.ConvTranspose2d(filters[2], filters[1], 2, stride=2)
        self.dec2 = _ResBlock(filters[1] * 2, filters[1])
        self.up1 = nn.ConvTranspose2d(filters[1], filters[0], 2, stride=2)
        self.dec1 = _ResBlock(filters[0] * 2, filters[0])

        # Output
        self.out_conv = nn.Conv2d(filters[0], num_classes, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Bottleneck
        b = self.bottleneck(e4)

        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up4(b), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))

        out = self.out_conv(d1)
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode='bilinear',
                                align_corners=False)
        return out
