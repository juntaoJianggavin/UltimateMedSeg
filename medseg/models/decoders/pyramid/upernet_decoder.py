"""UPerNet Decoder - Unified Perceptual Parsing Network decoder.
Reference: Xiao et al. "Unified Perceptual Parsing for Scene Understanding"

Has its own internal FPN-style lateral connections + PPM.
External skip_connection parameter is IGNORED.
"""
# Source: https://github.com/CSAILVision/unifiedparsing

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from medseg.registry import DECODER_REGISTRY


class PPM(nn.Module):
    """Pyramid Pooling Module from PSPNet."""
    def __init__(self, in_ch, pool_sizes=(1, 2, 3, 6)):
        super().__init__()
        out_ch = in_ch // len(pool_sizes)
        self.stages = nn.ModuleList()
        for size in pool_sizes:
            self.stages.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(size),
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ))
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_ch + out_ch * len(pool_sizes), in_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        H, W = x.shape[2:]
        priors = [x]
        for stage in self.stages:
            p = stage(x)
            p = F.interpolate(p, size=(H, W), mode='bilinear', align_corners=False)
            priors.append(p)
        return self.bottleneck(torch.cat(priors, dim=1))


@DECODER_REGISTRY.register("upernet")
class UPerNetDecoder(nn.Module):
    """UPerNet decoder: PPM on deepest feature + FPN-style lateral connections.

    External skip_connection is IGNORED - UPerNet has its own FPN lateral mechanism.
    """
    has_internal_skip = True

    def __init__(self, encoder_channels: List[int], bottleneck_channels: int,
                 skip_connection=None, fpn_dim: int = 256, **kwargs):
        super().__init__()
        all_channels = list(encoder_channels) + [bottleneck_channels]

        self.ppm = PPM(all_channels[-1])

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for ch in all_channels:
            self.lateral_convs.append(nn.Sequential(
                nn.Conv2d(ch, fpn_dim, 1, bias=False),
                nn.BatchNorm2d(fpn_dim),
                nn.ReLU(inplace=True),
            ))
            self.fpn_convs.append(nn.Sequential(
                nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1, bias=False),
                nn.BatchNorm2d(fpn_dim),
                nn.ReLU(inplace=True),
            ))

        self.fpn_bottleneck = nn.Sequential(
            nn.Conv2d(fpn_dim * len(all_channels), fpn_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_dim),
            nn.ReLU(inplace=True),
        )
        self._out_channels = fpn_dim

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, bottleneck_feat: torch.Tensor, skip_features: List[torch.Tensor]) -> torch.Tensor:
        all_features = list(skip_features) + [bottleneck_feat]
        target_size = all_features[0].shape[2:]

        # Apply PPM to deepest feature
        all_features[-1] = self.ppm(all_features[-1])

        # Build FPN laterals (top-down)
        laterals = [lat(f) for f, lat in zip(all_features, self.lateral_convs)]
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[2:], mode='bilinear', align_corners=False)

        # FPN output
        fpn_outs = [fpn(lat) for lat, fpn in zip(laterals, self.fpn_convs)]
        # Upsample all to highest resolution
        fpn_outs = [F.interpolate(f, size=target_size, mode='bilinear', align_corners=False)
                     if f.shape[2:] != target_size else f for f in fpn_outs]

        return self.fpn_bottleneck(torch.cat(fpn_outs, dim=1))
