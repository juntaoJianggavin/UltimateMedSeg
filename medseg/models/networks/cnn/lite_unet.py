"""Lite-UNet: Lightweight and Efficient Network for Cell Localization / Segmentation.

Faithful reimplementation from:
  https://github.com/Boom5426/MHFAN  (EAAI 2024)

Key innovations:
  - Conv2d_cd: Central difference convolution (gradient aggregation).
  - GhostModule: Ghost convolution for cheap feature generation.
  - GhostBottleneck_CBAM: Ghost bottleneck with CBAM attention.
  - Lightweight encoder-decoder with Ghost modules and CBAM.
"""
# Source: https://github.com/Boom5426/MHFAN

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Central Difference Convolution
# ---------------------------------------------------------------------------

class Conv2d_cd(nn.Module):
    """Central difference convolution: out = conv(x) - theta * sum_conv(x)."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, dilation=dilation,
                               groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)
        if abs(self.theta) < 1e-8:
            return out_normal
        kernel_diff = self.conv.weight.sum(2).sum(2)[:, :, None, None]
        out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias,
                            stride=self.conv.stride, padding=0, groups=self.conv.groups)
        return out_normal - self.theta * out_diff


# ---------------------------------------------------------------------------
# Ghost Module
# ---------------------------------------------------------------------------

class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super().__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)
        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential())
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2,
                      groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential())

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return torch.cat([x1, x2], dim=1)[:, :self.oup, :, :]


# ---------------------------------------------------------------------------
# Squeeze-Excite & CBAM
# ---------------------------------------------------------------------------

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def hard_sigmoid(x):
    return F.relu6(x + 3.) / 6.


class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, divisor=4):
        super().__init__()
        reduced_chs = _make_divisible(in_chs * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.act1(self.conv_reduce(x_se))
        x_se = self.conv_expand(x_se)
        return x * hard_sigmoid(x_se)


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super().__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        red = max(channel // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, red, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(red, channel, 1, bias=False))
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        ch_out = self.sigmoid(self.mlp(self.max_pool(x)) + self.mlp(self.avg_pool(x)))
        x = ch_out * x
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        sp_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        return sp_out * x


# ---------------------------------------------------------------------------
# Ghost Bottleneck with CBAM
# ---------------------------------------------------------------------------

class GhostBottleneckCBAM(nn.Module):
    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3, stride=1):
        super().__init__()
        self.stride = stride
        self.ghost1 = GhostModule(in_chs, mid_chs, relu=True)
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                                      padding=(dw_kernel_size - 1) // 2, groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)
        self.cbam = CBAM(mid_chs)
        self.ghost2 = GhostModule(mid_chs, out_chs, relu=False)
        if in_chs == out_chs and self.stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                          padding=(dw_kernel_size - 1) // 2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs))

    def forward(self, x):
        residual = x
        x = self.ghost1(x)
        if self.stride > 1:
            x = self.bn_dw(self.conv_dw(x))
        x = self.cbam(x)
        x = self.ghost2(x)
        return x + self.shortcut(residual)


# ---------------------------------------------------------------------------
# Conv Block & Up Conv
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, 3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out), nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, 3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


class UpConv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, 3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.up(x)


# ---------------------------------------------------------------------------
# GhostBottleneck (standard, for decoder)
# ---------------------------------------------------------------------------

class GhostBottleneck(nn.Module):
    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3, stride=1, se_ratio=1.):
        super().__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride
        self.ghost1 = GhostModule(in_chs, mid_chs, relu=True)
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                                      padding=(dw_kernel_size - 1) // 2, groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)
        self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio) if has_se else None
        self.ghost2 = GhostModule(mid_chs, out_chs, relu=False)
        if in_chs == out_chs and self.stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                          padding=(dw_kernel_size - 1) // 2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs))

    def forward(self, x):
        residual = x
        x = self.ghost1(x)
        if self.stride > 1:
            x = self.bn_dw(self.conv_dw(x))
        if self.se is not None:
            x = self.se(x)
        x = self.ghost2(x)
        return x + self.shortcut(residual)


# ---------------------------------------------------------------------------
# Lite-UNet
# ---------------------------------------------------------------------------

class LiteUNet(nn.Module):
    """Lite-UNet: Lightweight UNet with Ghost modules and CBAM."""

    def __init__(self, in_channels=3, num_classes=2, img_size=224,
                 channels=(64, 128, 256, 512, 1024), deep_supervision=False, **kwargs):
        super().__init__()
        c = list(channels)
        self.deep_supervision = deep_supervision

        # Encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = ConvBlock(in_channels, c[0])
        self.Conv2 = GhostBottleneckCBAM(c[0], c[0] * 2, c[1])
        self.Conv3 = GhostBottleneckCBAM(c[1], c[1] * 2, c[2])
        self.Conv4 = GhostBottleneckCBAM(c[2], c[2] * 2, c[3])
        self.Conv5 = GhostBottleneckCBAM(c[3], c[3] * 2, c[4])

        # Decoder
        self.Up5 = UpConv(c[4], c[3])
        self.Up_conv5 = GhostBottleneck(c[4], c[3] * 2, c[3])
        self.Up4 = UpConv(c[3], c[2])
        self.Up_conv4 = GhostBottleneck(c[3], c[2] * 2, c[2])
        self.Up3 = UpConv(c[2], c[1])
        self.Up_conv3 = GhostBottleneck(c[2], c[1] * 2, c[1])
        self.Up2 = UpConv(c[1], c[0])
        self.Up_conv2 = GhostBottleneck(c[1], c[0] * 2, c[0])

        self.Conv_1x1 = nn.Conv2d(c[0], num_classes, kernel_size=1, stride=1, padding=0)

        # Deep supervision side output heads
        if deep_supervision:
            self.ds_heads = nn.ModuleList([
                nn.Conv2d(c[3], num_classes, 1),
                nn.Conv2d(c[2], num_classes, 1),
                nn.Conv2d(c[1], num_classes, 1),
            ])

    def forward(self, x):
        x1 = self.Conv1(x)
        x2 = self.Conv2(self.Maxpool(x1))
        x3 = self.Conv3(self.Maxpool(x2))
        x4 = self.Conv4(self.Maxpool(x3))
        x5 = self.Conv5(self.Maxpool(x4))

        d5 = self.Up5(x5)
        if d5.shape[2:] != x4.shape[2:]:
            d5 = F.interpolate(d5, size=x4.shape[2:], mode='bilinear', align_corners=False)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        if d4.shape[2:] != x3.shape[2:]:
            d4 = F.interpolate(d4, size=x3.shape[2:], mode='bilinear', align_corners=False)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        if d3.shape[2:] != x2.shape[2:]:
            d3 = F.interpolate(d3, size=x2.shape[2:], mode='bilinear', align_corners=False)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        if d2.shape[2:] != x1.shape[2:]:
            d2 = F.interpolate(d2, size=x1.shape[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv_1x1(d2)

        if self.training and self.deep_supervision:
            input_size = out.shape[2:]
            aux = []
            for feat, head in zip([d5, d4, d3], self.ds_heads):
                a = head(feat)
                if a.shape[2:] != input_size:
                    a = F.interpolate(a, size=input_size, mode='bilinear', align_corners=False)
                aux.append(a)
            return [out] + aux

        return out
