"""Dense ASPP bottleneck."""
# Source: INTERNAL — framework adaptation (this repo).

import torch
import torch.nn as nn
from medseg.registry import BOTTLENECK_REGISTRY


class DenseASPPBlock(nn.Module):
    def __init__(self, in_ch, inter_ch, out_ch, dilation):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, inter_ch, 1, bias=False),
            nn.BatchNorm2d(inter_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_ch, out_ch, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


@BOTTLENECK_REGISTRY.register("dense_aspp")
class DenseASPPBottleneck(nn.Module):
    """Dense ASPP: each dilated conv takes all previous outputs as input."""
    def __init__(self, in_channels, growth_rate=64, dilations=(3, 6, 12, 18, 24), **kwargs):
        super().__init__()
        self.blocks = nn.ModuleList()
        cur_ch = in_channels
        for d in dilations:
            self.blocks.append(DenseASPPBlock(cur_ch, growth_rate * 2, growth_rate, d))
            cur_ch += growth_rate
        self.project = nn.Sequential(
            nn.Conv2d(cur_ch, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self._out_channels = in_channels

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, x):
        features = [x]
        for block in self.blocks:
            out = block(torch.cat(features, dim=1))
            features.append(out)
        return self.project(torch.cat(features, dim=1))
