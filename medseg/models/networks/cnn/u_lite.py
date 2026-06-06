"""U-Lite: Lightweight U-Net with Axial Depthwise Convolution.

Faithful reimplementation from:
  https://github.com/duong-db/U-Lite  (2023, ~878K params)

Key innovations:
  - AxialDW: Axial depthwise convolution (separate H and W 1D convs).
  - BottleNeckBlock: Multi-dilation axial DW conv bottleneck.
  - Ultra-lightweight encoder-decoder with only ~878K parameters.
"""
# Source: https://github.com/duong-db/U-Lite

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Axial Depthwise Conv
# ---------------------------------------------------------------------------

class AxialDW(nn.Module):
    """Axial depthwise convolution: separate H and W 1D convolutions."""
    def __init__(self, dim, mixer_kernel, dilation=1):
        super().__init__()
        h, w = mixer_kernel
        self.dw_h = nn.Conv2d(dim, dim, kernel_size=(h, 1), padding='same',
                               groups=dim, dilation=dilation)
        self.dw_w = nn.Conv2d(dim, dim, kernel_size=(1, w), padding='same',
                               groups=dim, dilation=dilation)

    def forward(self, x):
        return x + self.dw_h(x) + self.dw_w(x)


# ---------------------------------------------------------------------------
# Encoder / Decoder / Bottleneck blocks
# ---------------------------------------------------------------------------

class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.dw = AxialDW(in_c, mixer_kernel=(7, 7))
        self.bn = nn.BatchNorm2d(in_c)
        self.pw = nn.Conv2d(in_c, out_c, kernel_size=1)
        self.down = nn.MaxPool2d((2, 2))
        self.act = nn.GELU()

    def forward(self, x):
        skip = self.bn(self.dw(x))
        x = self.act(self.down(self.pw(skip)))
        return x, skip


class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.pw = nn.Conv2d(in_c + out_c, out_c, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_c)
        self.dw = AxialDW(out_c, mixer_kernel=(7, 7))
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(out_c, out_c, kernel_size=1)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = nn.functional.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.act(self.pw2(self.dw(self.bn(self.pw(x)))))
        return x


class BottleNeckBlock(nn.Module):
    """Axial dilated DW convolution bottleneck."""
    def __init__(self, dim):
        super().__init__()
        gc = dim // 4
        self.pw1 = nn.Conv2d(dim, gc, kernel_size=1)
        self.dw1 = AxialDW(gc, mixer_kernel=(3, 3), dilation=1)
        self.dw2 = AxialDW(gc, mixer_kernel=(3, 3), dilation=2)
        self.dw3 = AxialDW(gc, mixer_kernel=(3, 3), dilation=3)
        self.bn = nn.BatchNorm2d(4 * gc)
        self.pw2 = nn.Conv2d(4 * gc, dim, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.pw1(x)
        x = torch.cat([x, self.dw1(x), self.dw2(x), self.dw3(x)], 1)
        x = self.act(self.pw2(self.bn(x)))
        return x


# ---------------------------------------------------------------------------
# U-Lite
# ---------------------------------------------------------------------------

class ULite(nn.Module):
    """U-Lite: Lightweight UNet with Axial Depthwise Convolution."""

    def __init__(self, in_channels=3, num_classes=2, img_size=224,
                 channels=(16, 32, 64, 128, 256, 512), deep_supervision=False, **kwargs):
        super().__init__()
        c = list(channels)
        self.deep_supervision = deep_supervision

        # Encoder
        self.conv_in = nn.Conv2d(in_channels, c[0], kernel_size=7, padding=3)
        self.e1 = EncoderBlock(c[0], c[1])
        self.e2 = EncoderBlock(c[1], c[2])
        self.e3 = EncoderBlock(c[2], c[3])
        self.e4 = EncoderBlock(c[3], c[4])
        self.e5 = EncoderBlock(c[4], c[5])

        # Bottleneck
        self.b5 = BottleNeckBlock(c[5])

        # Decoder
        self.d5 = DecoderBlock(c[5], c[4])
        self.d4 = DecoderBlock(c[4], c[3])
        self.d3 = DecoderBlock(c[3], c[2])
        self.d2 = DecoderBlock(c[2], c[1])
        self.d1 = DecoderBlock(c[1], c[0])
        self.conv_out = nn.Conv2d(c[0], num_classes, kernel_size=1)

        # Deep supervision side output heads
        if deep_supervision:
            self.ds_heads = nn.ModuleList([
                nn.Conv2d(c[4], num_classes, 1),
                nn.Conv2d(c[3], num_classes, 1),
                nn.Conv2d(c[2], num_classes, 1),
                nn.Conv2d(c[1], num_classes, 1),
            ])

    def forward(self, x):
        x = self.conv_in(x)
        x, skip1 = self.e1(x)
        x, skip2 = self.e2(x)
        x, skip3 = self.e3(x)
        x, skip4 = self.e4(x)
        x, skip5 = self.e5(x)

        x = self.b5(x)

        d5 = self.d5(x, skip5)
        d4 = self.d4(d5, skip4)
        d3 = self.d3(d4, skip3)
        d2 = self.d2(d3, skip2)
        d1 = self.d1(d2, skip1)
        out = self.conv_out(d1)

        if self.training and self.deep_supervision:
            input_size = out.shape[2:]
            aux = []
            for feat, head in zip([d5, d4, d3, d2], self.ds_heads):
                a = head(feat)
                if a.shape[2:] != input_size:
                    a = F.interpolate(a, size=input_size, mode='bilinear', align_corners=False)
                aux.append(a)
            return [out] + aux

        return out
