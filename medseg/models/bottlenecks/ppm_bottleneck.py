"""PPM (Pyramid Pooling Module) bottleneck from PSPNet."""
# Source: INTERNAL — framework adaptation (this repo).

import torch
import torch.nn as nn
import torch.nn.functional as F
from medseg.registry import BOTTLENECK_REGISTRY


@BOTTLENECK_REGISTRY.register("ppm")
class PPMBottleneck(nn.Module):
    """Pyramid Pooling Module bottleneck."""
    def __init__(self, in_channels, pool_sizes=(1, 2, 3, 6), **kwargs):
        super().__init__()
        out_ch = in_channels // len(pool_sizes)
        self.stages = nn.ModuleList()
        for s in pool_sizes:
            self.stages.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(s),
                nn.Conv2d(in_channels, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ))
        self.project = nn.Sequential(
            nn.Conv2d(in_channels + out_ch * len(pool_sizes), in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self._out_channels = in_channels

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, x):
        H, W = x.shape[2:]
        priors = [x]
        for stage in self.stages:
            p = stage(x)
            p = F.interpolate(p, size=(H, W), mode='bilinear', align_corners=False)
            priors.append(p)
        return self.project(torch.cat(priors, dim=1))
