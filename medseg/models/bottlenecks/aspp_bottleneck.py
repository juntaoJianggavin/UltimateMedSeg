"""ASPP (Atrous Spatial Pyramid Pooling) bottleneck."""
# Source: INTERNAL — framework adaptation (this repo).

import torch
import torch.nn as nn
import torch.nn.functional as F
from medseg.registry import BOTTLENECK_REGISTRY


@BOTTLENECK_REGISTRY.register("aspp")
class ASPPBottleneck(nn.Module):
    """ASPP bottleneck from DeepLabV3+."""
    def __init__(self, in_channels, out_channels=None, atrous_rates=(6, 12, 18), **kwargs):
        super().__init__()
        out_ch = out_channels or in_channels
        modules = [nn.Sequential(
            nn.Conv2d(in_channels, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )]
        for rate in atrous_rates:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_ch, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ))
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_ch, 1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * (len(atrous_rates) + 2), out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self._out_channels = out_ch

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, x):
        H, W = x.shape[2:]
        res = []
        for conv in self.convs:
            res.append(conv(x))
        gap_out = self.gap(x)
        gap_out = F.interpolate(gap_out, size=(H, W), mode='bilinear', align_corners=False)
        res.append(gap_out)
        return self.project(torch.cat(res, dim=1))
