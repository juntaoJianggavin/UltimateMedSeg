"""Basic convolutional bottleneck."""
# Source: INTERNAL — framework adaptation (this repo).

import torch.nn as nn
from medseg.registry import BOTTLENECK_REGISTRY


@BOTTLENECK_REGISTRY.register("basic")
class BasicBottleneck(nn.Module):
    """Basic conv bottleneck: two 3x3 convolutions."""
    def __init__(self, in_channels, mid_channels=None, **kwargs):
        super().__init__()
        mid = mid_channels or in_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self._out_channels = in_channels

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, x):
        return self.conv(x) + x  # residual
