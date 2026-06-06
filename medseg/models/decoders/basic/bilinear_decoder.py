"""Bilinear Upsampling Decoder."""
# Source: INTERNAL — framework adaptation (this repo).

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from medseg.registry import DECODER_REGISTRY


@DECODER_REGISTRY.register("bilinear")
class BilinearDecoder(nn.Module):
    """UNet decoder using bilinear upsampling instead of transposed convolutions."""

    def __init__(self, encoder_channels: List[int], bottleneck_channels: int,
                 skip_connection=None, **kwargs):
        super().__init__()
        self.skip_connection = skip_connection
        skip_channels = list(reversed(encoder_channels))

        self.up_convs = nn.ModuleList()
        in_ch = bottleneck_channels
        for skip_ch in skip_channels:
            if skip_connection is not None:
                merged_ch = skip_connection.get_out_channels(in_ch, skip_ch)
            else:
                merged_ch = in_ch + skip_ch
            out_ch = skip_ch
            self.up_convs.append(nn.Sequential(
                nn.Conv2d(merged_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ))
            in_ch = out_ch
        self._out_channels = skip_channels[-1] if skip_channels else bottleneck_channels

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, bottleneck_feat: torch.Tensor, skip_features: List[torch.Tensor]) -> torch.Tensor:
        skips = list(reversed(skip_features))
        x = bottleneck_feat
        for i, conv in enumerate(self.up_convs):
            skip = skips[i]
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            if self.skip_connection is not None:
                x = self.skip_connection(x, skip)
            else:
                x = torch.cat([x, skip], dim=1)
            x = conv(x)
        return x
