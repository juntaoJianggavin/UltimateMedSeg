"""H2Former decoder module.

Extracted from networks/transformer/h2former_model.py for modular reuse.
Faithful to the original H2Former decoder: bilinear 2x upsample + concat skip
+ two 3x3 Conv-BN-ReLU blocks per level.

Adapted to work with any encoder (not just h2former): the number of decoder
stages and channel dimensions are derived dynamically from encoder_channels.
"""
# Source: https://github.com/NKUhealong/H2Former

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from medseg.registry import DECODER_REGISTRY


class _DecoderStep(nn.Module):
    """Single decoder step: bilinear 2x up -> concat skip -> 2x Conv-BN-ReLU."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear',
                              align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + out_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear',
                              align_corners=True)
        return self.conv(torch.cat([x, skip], dim=1))


@DECODER_REGISTRY.register("h2former")
class H2FormerDecoder(nn.Module):
    """H2Former-style UNet decoder with bilinear upsampling + concat skip.

    Automatically adapts to any encoder: builds one decoder stage per skip
    feature. Each stage does bilinear 2x upsample + concat skip + two
    Conv-BN-ReLU blocks.

    Standard interface: ``forward(bottleneck_feat, skip_features)``
    where skip_features = [shallow, ..., deep] (shallow→deep, excluding bottleneck).
    """

    has_internal_skip = True

    def __init__(self, encoder_channels: List[int], bottleneck_channels: int,
                 skip_connection=None, **kwargs):
        super().__init__()
        # encoder_channels = [c0, c1, ..., cN] shallow→deep (skip channels)
        # Build decoder from deep to shallow:
        #   first step: bottleneck -> c_deep
        #   subsequent steps: c_i -> c_{i-1}
        skips_reversed = list(reversed(encoder_channels))  # [deep, ..., shallow]
        in_channels = [bottleneck_channels] + skips_reversed[:-1]
        out_channels = skips_reversed  # deep to shallow

        self.stages = nn.ModuleList()
        for in_ch, out_ch in zip(in_channels, out_channels):
            self.stages.append(_DecoderStep(in_ch, out_ch))

        self._out_channels = encoder_channels[0]  # shallowest skip

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, bottleneck_feat: torch.Tensor,
                skip_features: List[torch.Tensor]) -> torch.Tensor:
        # skip_features: [shallow, ..., deep]
        skips_reversed = list(reversed(skip_features))  # [deep, ..., shallow]

        x = bottleneck_feat
        for stage, skip in zip(self.stages, skips_reversed):
            x = stage(x, skip)
        return x
