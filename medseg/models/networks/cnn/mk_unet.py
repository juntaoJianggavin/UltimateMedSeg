"""MK-UNet: Multi-Kernel Lightweight CNN for Medical Image Segmentation.

Faithful reimplementation from:
  https://github.com/SLDGroup/MK-UNet  (ICCV 2025 CVAMD Oral)

Key innovations:
  - MultiKernelInvertedResidualBlock: MobileNetV2-style IRB with multi-kernel DW conv.
  - GroupedAttentionGate: Grouped conv attention gate for skip connections.
  - Channel + Spatial Attention (CBAM-style) at each decoder stage.
  - Channel shuffle for inter-group communication.
"""
# Source: https://github.com/SLDGroup/MK-UNet

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import trunc_normal_tf_
from timm.models.helpers import named_apply


# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d):
        fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
        fan_out //= module.groups
        nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


def _act_layer(act='relu6', inplace=False):
    act = act.lower()
    if act == 'relu':
        return nn.ReLU(inplace)
    elif act == 'relu6':
        return nn.ReLU6(inplace)
    elif act == 'gelu':
        return nn.GELU()
    elif act == 'hswish':
        return nn.Hardswish(inplace)
    else:
        return nn.ReLU(inplace)


def _channel_shuffle(x, groups):
    B, C, H, W = x.shape
    cpg = C // groups
    x = x.view(B, groups, cpg, H, W)
    x = torch.transpose(x, 1, 2).contiguous()
    return x.view(B, -1, H, W)


# ---------------------------------------------------------------------------
# Channel & Spatial Attention
# ---------------------------------------------------------------------------

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        ratio = min(ratio, in_planes)
        reduced = max(in_planes // ratio, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, reduced, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(reduced, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


# ---------------------------------------------------------------------------
# Grouped Attention Gate
# ---------------------------------------------------------------------------

class GroupedAttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int, kernel_size=1, groups=1):
        super().__init__()
        if kernel_size == 1:
            groups = 1
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size, stride=1, padding=kernel_size // 2,
                      groups=groups, bias=True),
            nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size, stride=1, padding=kernel_size // 2,
                      groups=groups, bias=True),
            nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        psi = self.relu(self.W_g(g) + self.W_x(x))
        psi = self.psi(psi)
        return x * psi


# ---------------------------------------------------------------------------
# Multi-Kernel Inverted Residual Block
# ---------------------------------------------------------------------------

class MultiKernelDWConv(nn.Module):
    def __init__(self, in_channels, kernel_sizes, stride, activation='relu6',
                 dw_parallel=True):
        super().__init__()
        self.dw_parallel = dw_parallel
        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, ks, stride, ks // 2,
                          groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                _act_layer(activation, inplace=True))
            for ks in kernel_sizes])

    def forward(self, x):
        outputs = []
        for dwconv in self.dwconvs:
            dw_out = dwconv(x)
            outputs.append(dw_out)
            if not self.dw_parallel:
                x = x + dw_out
        return outputs


class MKInvertedResidualBlock(nn.Module):
    """MobileNetV2-style Inverted Residual Block with multi-kernel DW conv."""
    def __init__(self, in_c, out_c, stride, expansion_factor=2,
                 dw_parallel=True, add=True, kernel_sizes=(1, 3, 5),
                 activation='relu6'):
        super().__init__()
        assert stride in [1, 2]
        self.stride = stride
        self.in_c = in_c
        self.out_c = out_c
        self.add = add
        self.n_scales = len(kernel_sizes)
        self.use_skip = (stride == 1)

        ex_c = int(in_c * expansion_factor)
        self.pconv1 = nn.Sequential(
            nn.Conv2d(in_c, ex_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(ex_c),
            _act_layer(activation, inplace=True))
        self.mk_dwconv = MultiKernelDWConv(ex_c, kernel_sizes, stride, activation,
                                            dw_parallel)
        combined = ex_c if add else ex_c * self.n_scales
        self.pconv2 = nn.Sequential(
            nn.Conv2d(combined, out_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_c))

        if self.use_skip and in_c != out_c:
            self.conv1x1 = nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False)
        else:
            self.conv1x1 = None

        self._combined = combined

    def forward(self, x):
        pout = self.pconv1(x)
        dw_outs = self.mk_dwconv(pout)
        if self.add:
            dout = sum(dw_outs)
        else:
            dout = torch.cat(dw_outs, dim=1)
        dout = _channel_shuffle(dout, _gcd(self._combined, self.out_c))
        out = self.pconv2(dout)
        if self.use_skip:
            if self.conv1x1 is not None:
                x = self.conv1x1(x)
            return x + out
        return out


def _mk_irb_bottleneck(in_c, out_c, n, s, expansion_factor=2, kernel_sizes=(1, 3, 5),
                        activation='relu6'):
    layers = [MKInvertedResidualBlock(in_c, out_c, s, expansion_factor,
                                       kernel_sizes=kernel_sizes, activation=activation)]
    for _ in range(1, n):
        layers.append(MKInvertedResidualBlock(out_c, out_c, 1, expansion_factor,
                                               kernel_sizes=kernel_sizes, activation=activation))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# MK-UNet
# ---------------------------------------------------------------------------

class MKUNet(nn.Module):
    """MK-UNet: Multi-Kernel Lightweight CNN UNet."""

    def __init__(self, in_channels=3, num_classes=2, img_size=224,
                 channels=(16, 32, 64, 96, 160), depths=(1, 1, 1, 1, 1),
                 kernel_sizes=(1, 3, 5), expansion_factor=2,
                 gag_kernel=3, deep_supervision=False, **kwargs):
        super().__init__()
        c = list(channels)
        depths = list(depths)
        kernel_sizes = list(kernel_sizes)
        self.deep_supervision = deep_supervision

        # Encoder
        self.encoder1 = _mk_irb_bottleneck(in_channels, c[0], depths[0], 1, expansion_factor, kernel_sizes)
        self.encoder2 = _mk_irb_bottleneck(c[0], c[1], depths[1], 1, expansion_factor, kernel_sizes)
        self.encoder3 = _mk_irb_bottleneck(c[1], c[2], depths[2], 1, expansion_factor, kernel_sizes)
        self.encoder4 = _mk_irb_bottleneck(c[2], c[3], depths[3], 1, expansion_factor, kernel_sizes)
        self.encoder5 = _mk_irb_bottleneck(c[3], c[4], depths[4], 1, expansion_factor, kernel_sizes)

        # Attention Gates
        self.AG1 = GroupedAttentionGate(c[3], c[3], c[3] // 2, gag_kernel, c[3] // 2)
        self.AG2 = GroupedAttentionGate(c[2], c[2], c[2] // 2, gag_kernel, c[2] // 2)
        self.AG3 = GroupedAttentionGate(c[1], c[1], c[1] // 2, gag_kernel, c[1] // 2)
        self.AG4 = GroupedAttentionGate(c[0], c[0], c[0] // 2, gag_kernel, c[0] // 2)

        # Decoder
        self.decoder1 = _mk_irb_bottleneck(c[4], c[3], 1, 1, expansion_factor, kernel_sizes)
        self.decoder2 = _mk_irb_bottleneck(c[3], c[2], 1, 1, expansion_factor, kernel_sizes)
        self.decoder3 = _mk_irb_bottleneck(c[2], c[1], 1, 1, expansion_factor, kernel_sizes)
        self.decoder4 = _mk_irb_bottleneck(c[1], c[0], 1, 1, expansion_factor, kernel_sizes)
        self.decoder5 = _mk_irb_bottleneck(c[0], c[0], 1, 1, expansion_factor, kernel_sizes)

        # Channel & Spatial Attention
        self.CA1 = ChannelAttention(c[4], ratio=16)
        self.CA2 = ChannelAttention(c[3], ratio=16)
        self.CA3 = ChannelAttention(c[2], ratio=16)
        self.CA4 = ChannelAttention(c[1], ratio=8)
        self.CA5 = ChannelAttention(c[0], ratio=4)
        self.SA = SpatialAttention()

        self.final = nn.Conv2d(c[0], num_classes, kernel_size=1)

        # Deep supervision side output heads
        if deep_supervision:
            self.ds_heads = nn.ModuleList([
                nn.Conv2d(c[3], num_classes, 1),
                nn.Conv2d(c[2], num_classes, 1),
                nn.Conv2d(c[1], num_classes, 1),
            ])

        named_apply(partial(_init_weights, scheme=''), self)

    def forward(self, x):
        # Encoder
        out = F.max_pool2d(self.encoder1(x), 2, 2)
        t1 = out
        out = F.max_pool2d(self.encoder2(out), 2, 2)
        t2 = out
        out = F.max_pool2d(self.encoder3(out), 2, 2)
        t3 = out
        out = F.max_pool2d(self.encoder4(out), 2, 2)
        t4 = out
        out = F.max_pool2d(self.encoder5(out), 2, 2)

        ds_collect = self.training and self.deep_supervision
        intermediates = []

        # Decoder
        out = self.CA1(out) * out
        out = self.SA(out) * out
        out = F.relu(F.interpolate(self.decoder1(out), scale_factor=2, mode='bilinear'))
        t4 = self.AG1(g=out, x=t4)
        out = out + t4
        if ds_collect:
            intermediates.append(out)

        out = self.CA2(out) * out
        out = self.SA(out) * out
        out = F.relu(F.interpolate(self.decoder2(out), scale_factor=2, mode='bilinear'))
        t3 = self.AG2(g=out, x=t3)
        out = out + t3
        if ds_collect:
            intermediates.append(out)

        out = self.CA3(out) * out
        out = self.SA(out) * out
        out = F.relu(F.interpolate(self.decoder3(out), scale_factor=2, mode='bilinear'))
        t2 = self.AG3(g=out, x=t2)
        out = out + t2
        if ds_collect:
            intermediates.append(out)

        out = self.CA4(out) * out
        out = self.SA(out) * out
        out = F.relu(F.interpolate(self.decoder4(out), scale_factor=2, mode='bilinear'))
        t1 = self.AG4(g=out, x=t1)
        out = out + t1

        out = self.CA5(out) * out
        out = self.SA(out) * out
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=2, mode='bilinear'))

        main_out = self.final(out)
        if ds_collect:
            return self._ds_forward(main_out, intermediates)
        return main_out

    def _ds_forward(self, main_out, intermediates):
        """Deep supervision helper."""
        input_size = main_out.shape[2:]
        aux = []
        for feat, head in zip(intermediates, self.ds_heads):
            a = head(feat)
            if a.shape[2:] != input_size:
                a = F.interpolate(a, size=input_size, mode='bilinear', align_corners=False)
            aux.append(a)
        return [main_out] + aux
