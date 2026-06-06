"""LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation.

Reference:
    Chaurasia & Culurciello, "LinkNet: Exploiting Encoder Representations
    for Efficient Semantic Segmentation", VCIP 2017.
    https://github.com/e-lab/pytorch-linknet

Efficient encoder-decoder architecture using residual encoder blocks
with summation-based skip connections (instead of concatenation).
"""
# Source: https://github.com/e-lab/pytorch-linknet

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


class _ConvBN(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        return self.block(x)


class _BasicBlock(nn.Module):
    """Basic residual block for LinkNet encoder."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = _ConvBNReLU(in_ch, out_ch, 3, stride, 1)
        self.conv2 = _ConvBN(out_ch, out_ch, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = _ConvBN(in_ch, out_ch, 1, stride, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + self.shortcut(x)
        return self.relu(out)


class _DecoderBlock(nn.Module):
    """LinkNet decoder block: 1×1 reduce → deconv → 1×1 restore."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        mid_ch = in_ch // 4
        self.block = nn.Sequential(
            # 1×1 reduce channels
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            # Transpose conv for upsampling
            nn.ConvTranspose2d(mid_ch, mid_ch, 3, stride=2, padding=1,
                               output_padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            # 1×1 restore channels
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class LinkNet(nn.Module):
    """LinkNet: efficient encoder-decoder with summation skip connections.

    Architecture:
        - Encoder: 4 stages of residual blocks (ResNet18-like)
        - Decoder: 4 stages of decoder blocks with summation skip connections
        - Final: full-resolution deconv + 1×1 output conv

    Args:
        in_channels: number of input channels.
        num_classes: number of output segmentation classes.
        img_size: spatial size (unused, kept for API consistency).
    """

    def __init__(self, in_channels=3, num_classes=2, img_size=224):
        super().__init__()
        filters = [64, 128, 256, 512]

        # Initial convolution (stride=2)
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, filters[0], 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        # Encoder stages (ResNet18-like)
        self.enc1 = self._make_layer(filters[0], filters[0], 3, stride=1)
        self.enc2 = self._make_layer(filters[0], filters[1], 2, stride=2)
        self.enc3 = self._make_layer(filters[1], filters[2], 2, stride=2)
        self.enc4 = self._make_layer(filters[2], filters[3], 2, stride=2)

        # Decoder stages (summation skip connections)
        self.dec4 = _DecoderBlock(filters[3], filters[2])
        self.dec3 = _DecoderBlock(filters[2], filters[1])
        self.dec2 = _DecoderBlock(filters[1], filters[0])
        self.dec1 = _DecoderBlock(filters[0], filters[0])

        # Final full-resolution deconv + output
        self.final_deconv = nn.Sequential(
            nn.ConvTranspose2d(filters[0], 32, 3, stride=2, padding=1,
                               output_padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.out_conv = nn.Conv2d(32, num_classes, 1)

    @staticmethod
    def _make_layer(in_ch, out_ch, num_blocks, stride):
        layers = [_BasicBlock(in_ch, out_ch, stride)]
        for _ in range(1, num_blocks):
            layers.append(_BasicBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        inp_size = x.shape[2:]

        # Initial: /4 resolution
        x0 = self.initial(x)

        # Encoder
        e1 = self.enc1(x0)  # /4
        e2 = self.enc2(e1)  # /8
        e3 = self.enc3(e2)  # /16
        e4 = self.enc4(e3)  # /32

        # Decoder with summation skip connections
        d4 = self.dec4(e4) + e3     # /16
        d3 = self.dec3(d4) + e2     # /8
        d2 = self.dec2(d3) + e1     # /4
        d1 = self.dec1(d2)          # /2

        # Final upsampling to input resolution
        out = self.final_deconv(d1)  # /1
        out = F.interpolate(out, size=inp_size, mode='bilinear', align_corners=True)

        return self.out_conv(out)
