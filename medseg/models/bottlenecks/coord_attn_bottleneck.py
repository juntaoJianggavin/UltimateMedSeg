"""Coordinate Attention (CA) bottleneck — CVPR 2021.

Official source: https://github.com/houqb/CoordAttention

Reference:
    Hou et al., "Coordinate Attention for Efficient Mobile Network Design", CVPR 2021.
"""
# Source: INTERNAL — framework adaptation (this repo).

import torch
import torch.nn as nn
from medseg.registry import BOTTLENECK_REGISTRY


class _HSigmoid(nn.Module):
    """Hard-Sigmoid: ReLU6(x + 3) / 6 (matches official CoordAttention)."""

    def __init__(self, inplace=True):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class _HSwish(nn.Module):
    """Hard-Swish: x * h_sigmoid(x) (matches official CoordAttention)."""

    def __init__(self, inplace=True):
        super().__init__()
        self.sigmoid = _HSigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    """Coordinate attention — faithful port of official ``CoordAtt`` module.

    Decomposes channel attention into two 1-D feature encoding steps along
    the two spatial directions (height & width), then applies sigmoid gating.
    """

    def __init__(self, inp, oup, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = _HSwish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)                        # (N, C, H, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)   # (N, C, W, 1)
        y = torch.cat([x_h, x_w], dim=2)            # (N, C, H+W, 1)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        return out


@BOTTLENECK_REGISTRY.register("coord_attn")
class CoordAttnBottleneck(nn.Module):
    """Coordinate Attention bottleneck with residual connection.

    Args:
        in_channels: Number of input channels.
        reduction: Channel reduction ratio for the attention module.
    """

    def __init__(self, in_channels, reduction=32, **kwargs):
        super().__init__()
        self.attn = CoordAtt(in_channels, in_channels, reduction)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self._out_channels = in_channels

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, x):
        return self.conv(self.attn(x)) + x
