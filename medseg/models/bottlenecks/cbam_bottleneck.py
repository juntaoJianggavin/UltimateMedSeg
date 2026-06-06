"""CBAM (Convolutional Block Attention Module) bottleneck.

Reference: Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018.

Sequentially applies channel attention and spatial attention to refine features.
"""
# Source: INTERNAL — framework adaptation (this repo).

import torch
import torch.nn as nn
from medseg.registry import BOTTLENECK_REGISTRY


class ChannelAttention(nn.Module):
    """Channel attention: squeeze spatial dims, learn channel weights."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.shared_mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # Max pool + Avg pool
        avg_out = self.shared_mlp(x.mean(dim=[2, 3]))       # (B, C)
        max_out = self.shared_mlp(x.amax(dim=[2, 3]))       # (B, C)
        w = torch.sigmoid(avg_out + max_out).view(B, C, 1, 1)
        return x * w


class SpatialAttention(nn.Module):
    """Spatial attention: channel-wise pool, learn spatial weights."""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg_out = x.mean(dim=1, keepdim=True)   # (B, 1, H, W)
        max_out = x.amax(dim=1, keepdim=True)   # (B, 1, H, W)
        w = torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * w


@BOTTLENECK_REGISTRY.register("cbam")
class CBAMBottleneck(nn.Module):
    """CBAM bottleneck: sequential channel + spatial attention with conv refinement.

    Args:
        in_channels: Number of input channels.
        reduction: Channel attention reduction ratio (default: 16).
        spatial_ks: Spatial attention kernel size (default: 7).
    """
    def __init__(self, in_channels, reduction=16, spatial_ks=7, **kwargs):
        super().__init__()
        self.ca = ChannelAttention(in_channels, reduction)
        self.sa = SpatialAttention(spatial_ks)
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
        out = self.ca(x)
        out = self.sa(out)
        return self.conv(out) + x  # residual
