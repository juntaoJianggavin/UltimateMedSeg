"""SE (Squeeze-and-Excitation) bottleneck."""
# Source: INTERNAL — framework adaptation (this repo).

import torch.nn as nn
from medseg.registry import BOTTLENECK_REGISTRY


@BOTTLENECK_REGISTRY.register("se")
class SEBottleneck(nn.Module):
    """SE bottleneck with channel attention."""
    def __init__(self, in_channels, reduction=16, **kwargs):
        super().__init__()
        mid = max(in_channels // reduction, 1)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_channels),
            nn.Sigmoid(),
        )
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
        w = self.se(x).unsqueeze(-1).unsqueeze(-1)
        return self.conv(x * w) + x
