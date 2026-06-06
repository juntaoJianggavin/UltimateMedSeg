"""Transposed Convolution (Deconv) Decoder."""
# Source: INTERNAL — framework adaptation (this repo).

import torch
import torch.nn as nn
from typing import List, Optional
from medseg.registry import DECODER_REGISTRY


class DeconvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(self.up(x))


@DECODER_REGISTRY.register("deconv")
class DeconvDecoder(nn.Module):
    """Standard transposed-convolution UNet decoder."""

    def __init__(self, encoder_channels: List[int], bottleneck_channels: int,
                 skip_connection=None, **kwargs):
        super().__init__()
        self.skip_connection = skip_connection
        skip_channels = list(reversed(encoder_channels))  # high-res first after reverse
        in_channels = [bottleneck_channels] + [skip_channels[i] for i in range(len(skip_channels) - 1)]

        self.up_blocks = nn.ModuleList()
        self.channel_adapts = nn.ModuleList()
        for i, (in_ch, skip_ch) in enumerate(zip(in_channels, skip_channels)):
            # After skip merge, channel count depends on skip type
            if skip_connection is not None:
                merged_ch = skip_connection.get_out_channels(in_ch, skip_ch)
            else:
                merged_ch = in_ch + skip_ch  # default concat
            self.channel_adapts.append(nn.Identity())
            out_ch = skip_ch
            self.up_blocks.append(nn.Sequential(
                nn.ConvTranspose2d(merged_ch, out_ch, kernel_size=2, stride=2),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ))
        self._out_channels = skip_channels[-1] if skip_channels else bottleneck_channels

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, bottleneck_feat: torch.Tensor, skip_features: List[torch.Tensor]) -> torch.Tensor:
        skips = list(reversed(skip_features))  # now from deepest to shallowest
        x = bottleneck_feat
        for i, block in enumerate(self.up_blocks):
            skip = skips[i]
            if x.shape[2:] != skip.shape[2:]:
                x = nn.functional.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            if self.skip_connection is not None:
                x = self.skip_connection(x, skip)
            else:
                x = torch.cat([x, skip], dim=1)
            x = block(x)
        return x
