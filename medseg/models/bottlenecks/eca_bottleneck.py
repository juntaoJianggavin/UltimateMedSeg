"""ECA (Efficient Channel Attention) bottleneck — CVPR 2020.

Official source: https://github.com/BangguWu/ECANet

Reference:
    Wang et al., "ECA-Net: Efficient Channel Attention for Deep Convolutional
    Neural Networks", CVPR 2020.
"""
# Source: INTERNAL — framework adaptation (this repo).

import math
import torch.nn as nn
from medseg.registry import BOTTLENECK_REGISTRY


class ECALayer(nn.Module):
    """Efficient Channel Attention — faithful port of official ``eca_layer``.

    Adaptive 1-D convolution instead of dimensionality reduction, achieving
    linear complexity w.r.t. channels.

    Args:
        channel: Number of channels of the input feature map.
        k_size: Adaptive kernel size (default 3, as in official code).
    """

    def __init__(self, channel, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size,
                              padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Feature descriptor on the global spatial information
        y = self.avg_pool(x)
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)
        return x * y.expand_as(x)


@BOTTLENECK_REGISTRY.register("eca")
class ECABottleneck(nn.Module):
    """ECA bottleneck with residual connection.

    Args:
        in_channels: Number of input channels.
        k_size: Kernel size for the 1-D conv (default 3, official).
    """

    def __init__(self, in_channels, k_size=3, **kwargs):
        super().__init__()
        self.eca = ECALayer(in_channels, k_size)
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
        return self.conv(self.eca(x)) + x
