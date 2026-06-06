"""UNet++ – self-contained port from MrGiovanni/UNetPlusPlus.

UNet++: A Nested U-Net Architecture for Medical Image Segmentation
(Zhou et al., IEEE TMI 2018).

Architecture: Nested dense skip pathways between encoder and decoder levels.
Each decoder node receives features from ALL higher-resolution encoder AND
decoder nodes, connected through dense convolution blocks.
"""
# Source: https://github.com/MrGiovanni/UNetPlusPlus

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class _ConvBlock(nn.Module):
    """Double Conv-BN-ReLU block."""

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


class _NestedBlock(nn.Module):
    """Nested decoder block: cat all inputs -> Conv-BN-ReLU."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = _ConvBlock(in_ch, out_ch)

    def forward(self, x):
        return self.conv(x)


# ---------------------------------------------------------------------------
# UNet++
# ---------------------------------------------------------------------------
class UNetPP(nn.Module):
    """UNet++ with 4 encoder levels, dense nested skip pathways, and
    deep supervision (optional).

    Args:
        in_channels: Number of input image channels (default 3).
        num_classes: Number of output segmentation classes (default 2).
        img_size: Expected input spatial resolution (default 224, unused).
        deep_supervision: Whether to enable deep supervision (default False).
    """

    def __init__(self, in_channels=3, num_classes=2, img_size=224,
                 deep_supervision=False, **kwargs):
        super().__init__()
        self.deep_supervision = deep_supervision
        filters = [32, 64, 128, 256, 512]

        # Encoder
        self.enc0 = _ConvBlock(in_channels, filters[0])
        self.enc1 = _ConvBlock(filters[0], filters[1])
        self.enc2 = _ConvBlock(filters[1], filters[2])
        self.enc3 = _ConvBlock(filters[2], filters[3])
        self.enc4 = _ConvBlock(filters[3], filters[4])
        self.pool = nn.MaxPool2d(2)

        # Decoder level 1 (X_01, X_11, X_21, X_31)
        self.up1_0 = nn.ConvTranspose2d(filters[1], filters[0], 2, stride=2)
        self.dec0_1 = _NestedBlock(filters[0] * 2, filters[0])
        self.up2_0 = nn.ConvTranspose2d(filters[2], filters[1], 2, stride=2)
        self.dec1_1 = _NestedBlock(filters[1] * 2, filters[1])
        self.up3_0 = nn.ConvTranspose2d(filters[3], filters[2], 2, stride=2)
        self.dec2_1 = _NestedBlock(filters[2] * 2, filters[2])
        self.up4_0 = nn.ConvTranspose2d(filters[4], filters[3], 2, stride=2)
        self.dec3_1 = _NestedBlock(filters[3] * 2, filters[3])

        # Decoder level 2 (X_02, X_12, X_22)
        self.up1_1 = nn.ConvTranspose2d(filters[1], filters[0], 2, stride=2)
        self.dec0_2 = _NestedBlock(filters[0] * 3, filters[0])
        self.up2_1 = nn.ConvTranspose2d(filters[2], filters[1], 2, stride=2)
        self.dec1_2 = _NestedBlock(filters[1] * 3, filters[1])
        self.up3_1 = nn.ConvTranspose2d(filters[3], filters[2], 2, stride=2)
        self.dec2_2 = _NestedBlock(filters[2] * 3, filters[2])

        # Decoder level 3 (X_03, X_13)
        self.up1_2 = nn.ConvTranspose2d(filters[1], filters[0], 2, stride=2)
        self.dec0_3 = _NestedBlock(filters[0] * 4, filters[0])
        self.up2_2 = nn.ConvTranspose2d(filters[2], filters[1], 2, stride=2)
        self.dec1_3 = _NestedBlock(filters[1] * 4, filters[1])

        # Decoder level 4 (X_04)
        self.up1_3 = nn.ConvTranspose2d(filters[1], filters[0], 2, stride=2)
        self.dec0_4 = _NestedBlock(filters[0] * 5, filters[0])

        # Output heads
        self.final = nn.Conv2d(filters[0], num_classes, 1)
        if deep_supervision:
            self.final1 = nn.Conv2d(filters[0], num_classes, 1)
            self.final2 = nn.Conv2d(filters[0], num_classes, 1)
            self.final3 = nn.Conv2d(filters[0], num_classes, 1)

    def forward(self, x):
        # Encoder
        x0_0 = self.enc0(x)
        x1_0 = self.enc1(self.pool(x0_0))
        x2_0 = self.enc2(self.pool(x1_0))
        x3_0 = self.enc3(self.pool(x2_0))
        x4_0 = self.enc4(self.pool(x3_0))

        # Decoder level 1
        x0_1 = self.dec0_1(torch.cat([x0_0, self.up1_0(x1_0)], 1))
        x1_1 = self.dec1_1(torch.cat([x1_0, self.up2_0(x2_0)], 1))
        x2_1 = self.dec2_1(torch.cat([x2_0, self.up3_0(x3_0)], 1))
        x3_1 = self.dec3_1(torch.cat([x3_0, self.up4_0(x4_0)], 1))

        # Decoder level 2
        x0_2 = self.dec0_2(torch.cat([x0_0, x0_1, self.up1_1(x1_1)], 1))
        x1_2 = self.dec1_2(torch.cat([x1_0, x1_1, self.up2_1(x2_1)], 1))
        x2_2 = self.dec2_2(torch.cat([x2_0, x2_1, self.up3_1(x3_1)], 1))

        # Decoder level 3
        x0_3 = self.dec0_3(torch.cat([x0_0, x0_1, x0_2, self.up1_2(x1_2)], 1))
        x1_3 = self.dec1_3(torch.cat([x1_0, x1_1, x1_2, self.up2_2(x2_2)], 1))

        # Decoder level 4
        x0_4 = self.dec0_4(
            torch.cat([x0_0, x0_1, x0_2, x0_3, self.up1_3(x1_3)], 1))

        if self.deep_supervision and self.training:
            o1 = self.final1(x0_1)
            o2 = self.final2(x0_2)
            o3 = self.final3(x0_3)
            o4 = self.final(x0_4)
            out = (o1 + o2 + o3 + o4) / 4
        else:
            out = self.final(x0_4)

        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode='bilinear',
                                align_corners=False)
        return out
