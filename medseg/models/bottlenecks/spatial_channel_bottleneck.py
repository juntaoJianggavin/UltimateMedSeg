"""Spatial-Channel Cross-Attention bottleneck.

Combines a spatial attention branch (inspired by Spatial Transformer Networks)
with a channel attention branch (SE-style), then fuses them through a 1×1
gating mechanism.

Reference concepts from:
    - Jaderberg et al., "Spatial Transformer Networks", NeurIPS 2015.
    - Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018.
"""
# Source: INTERNAL — framework adaptation (this repo).

import torch
import torch.nn as nn
from medseg.registry import BOTTLENECK_REGISTRY


class SpatialChannelAttention(nn.Module):
    """Dual-branch spatial + channel attention with learned gating."""

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        mid = max(in_channels // reduction, 4)

        # Channel attention branch
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_channels),
        )

        # Spatial attention branch
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
        )

        # Fusion gate
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Channel attention
        ca_w = torch.sigmoid(self.ca(x)).unsqueeze(-1).unsqueeze(-1)  # B,C,1,1
        ca_feat = x * ca_w

        # Spatial attention
        avg_out = x.mean(dim=1, keepdim=True)    # B,1,H,W
        max_out, _ = x.max(dim=1, keepdim=True)  # B,1,H,W
        sa_w = torch.sigmoid(self.sa(torch.cat([avg_out, max_out], dim=1)))
        sa_feat = x * sa_w

        # Fused gate
        g = self.gate(torch.cat([ca_feat, sa_feat], dim=1))
        return g * ca_feat + (1 - g) * sa_feat


@BOTTLENECK_REGISTRY.register("spatial_channel")
class SpatialChannelBottleneck(nn.Module):
    """Spatial-Channel cross-attention bottleneck with residual.

    Args:
        in_channels: Number of input/output channels.
        reduction: Channel-attention reduction ratio.
    """

    def __init__(self, in_channels, reduction=16, **kwargs):
        super().__init__()
        self.sca = SpatialChannelAttention(in_channels, reduction)
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
        return self.conv(self.sca(x)) + x
