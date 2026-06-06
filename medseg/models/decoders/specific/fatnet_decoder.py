"""FATNet decoder module.

Extracted from networks/transformer/fatnet_model.py for modular reuse.
Faithful to the original DecoderBottleneckLayer: 1x1 reduce -> BN -> ReLU ->
transpose conv (or bilinear) up -> 1x1 project -> BN -> ReLU.
"""
# Source: https://github.com/SZUcsh/FAT-Net

import torch
import torch.nn as nn
from typing import List
from medseg.registry import DECODER_REGISTRY


class DecoderBottleneckLayer(nn.Module):
    """Single decoder step: reduce channels, upsample, project."""

    def __init__(self, in_channels, n_filters, use_transpose=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU(inplace=True)

        if use_transpose:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels // 4, in_channels // 4, 3,
                    stride=2, padding=1, output_padding=1
                ),
                nn.BatchNorm2d(in_channels // 4),
                nn.ReLU(inplace=True)
            )
        else:
            self.up = nn.Upsample(scale_factor=2, align_corners=True,
                                  mode="bilinear")

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.norm1(self.conv1(x)))
        x = self.up(x)
        x = self.relu3(self.norm3(self.conv3(x)))
        return x


@DECODER_REGISTRY.register("fatnet")
class FATNetDecoder(nn.Module):
    """FATNet cascade decoder with DecoderBottleneckLayers.

    Standard interface: ``forward(bottleneck_feat, skip_features)``

    Decoder path: 4 DecoderBottleneckLayers that progressively upsample
    and reduce channels from the bottleneck (768) to 32 channels,
    with optional concat of skip features at each level.
    """

    has_internal_skip = True

    def __init__(self, encoder_channels: List[int], bottleneck_channels: int,
                 skip_connection=None,
                 decoder_channels=None,
                 use_transpose=None,
                 **kwargs):
        super().__init__()
        if decoder_channels is None:
            decoder_channels = [256, 128, 64, 32]
        if use_transpose is None:
            use_transpose = [True, True, True, False]

        in_ch = bottleneck_channels
        self.decoder_layers = nn.ModuleList()
        for i, (out_ch, ut) in enumerate(
                zip(decoder_channels, use_transpose)):
            self.decoder_layers.append(
                DecoderBottleneckLayer(in_ch, out_ch, use_transpose=ut))
            in_ch = out_ch
        self._out_channels = decoder_channels[-1]

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, bottleneck_feat: torch.Tensor,
                skip_features: List[torch.Tensor]) -> torch.Tensor:
        x = bottleneck_feat
        for layer in self.decoder_layers:
            x = layer(x)
        return x
