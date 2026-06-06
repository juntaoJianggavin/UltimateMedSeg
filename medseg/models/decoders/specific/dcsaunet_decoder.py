"""DCSAU-Net Decoder: faithful port from https://github.com/xq141839/DCSAU-Net

Reference: Xu et al., "DCSAU-Net: A Deeper and More Compact Split-Attention U-Net
           for Medical Image Segmentation"
File: DCSAU_Net.py

Decoder uses bilinear upsampling + concat skip + CSA Bottleneck refinement layers.
Mirrors the encoder's CSA ResNet structure.

Has its own internal skip connection mechanism (Up concat + CSA layers).
External skip_connection parameter is IGNORED.
"""
# Source: https://github.com/xq141839/DCSAU-Net

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from medseg.registry import DECODER_REGISTRY
from medseg.models.encoders.dcsaunet_encoder import Bottleneck, SplAtConv2d


class Up(nn.Module):
    """Bilinear 2x upsample + pad + concat (from original DCSAU-Net)."""

    def __init__(self):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return x


def _make_csa_layer(block, inplanes, planes, blocks, radix=2, cardinality=1,
                    bottleneck_width=64, avd=True, avd_first=False,
                    norm_layer=nn.BatchNorm2d):
    """Build a CSA (Channel Split Attention) layer with correct input channels."""
    downsample = None
    if inplanes != planes * block.expansion:
        down_layers = [
            nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, bias=False),
            norm_layer(planes * block.expansion),
        ]
        downsample = nn.Sequential(*down_layers)

    layers = []
    layers.append(block(inplanes, planes, stride=1, downsample=downsample,
                        radix=radix, cardinality=cardinality,
                        bottleneck_width=bottleneck_width,
                        avd=avd, avd_first=avd_first,
                        norm_layer=norm_layer, is_first=False))
    cur_inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(cur_inplanes, planes,
                            radix=radix, cardinality=cardinality,
                            bottleneck_width=bottleneck_width,
                            avd=avd, avd_first=avd_first,
                            norm_layer=norm_layer))
    return nn.Sequential(*layers)


@DECODER_REGISTRY.register("dcsaunet")
class DCSAUNetDecoder(nn.Module):
    """DCSAU-Net decoder with Up (bilinear+concat) + CSA Bottleneck refinement.

    Faithful to the original DCSAU-Net Model decoder path.
    Architecture:
        Up(bottleneck, skip3) -> CSA_layer5 ->
        Up(layer5_out, skip2) -> CSA_layer6 ->
        Up(layer6_out, skip1) -> CSA_layer7 ->
        Up(layer7_out, skip0) -> CSA_layer8

    External skip_connection is IGNORED.
    """
    has_internal_skip = True

    def __init__(self, encoder_channels: List[int], bottleneck_channels: int,
                 skip_connection=None,
                 radix: int = 2,
                 blocks_per_layer: int = 2,
                 **kwargs):
        super().__init__()
        # encoder_channels for DCSAU-Net: [64, 64, 128, 256] (skip features)
        # bottleneck_channels: 512
        skip_channels = list(reversed(encoder_channels))  # [256, 128, 64, 64]

        # Decoder planes mirror the encoder decoder layers
        # Original: layer5(64), layer6(32), layer7(16), layer8(16) with expansion=4
        # Gives output channels: 256, 128, 64, 64
        n_skips = len(skip_channels)

        # Compute decoder planes to produce target output channels
        # After concat: deeper_ch + skip_ch; after CSA layer: planes * 4
        self.up_blocks = nn.ModuleList()
        self.csa_layers = nn.ModuleList()

        in_ch = bottleneck_channels
        decoder_planes = []
        for i, skip_ch in enumerate(skip_channels):
            # Target output for this level matches skip_ch
            target_out = skip_ch
            planes = target_out // Bottleneck.expansion  # planes for Bottleneck
            decoder_planes.append(planes)

            self.up_blocks.append(Up())
            concat_ch = in_ch + skip_ch  # After Up concat
            self.csa_layers.append(
                _make_csa_layer(Bottleneck, concat_ch, planes, blocks_per_layer,
                                radix=radix))
            in_ch = target_out  # output of this CSA layer

        self._out_channels = skip_channels[-1] if skip_channels else bottleneck_channels

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, bottleneck_feat: torch.Tensor, skip_features: List[torch.Tensor]) -> torch.Tensor:
        skips = list(reversed(skip_features))  # deep to shallow
        x = bottleneck_feat
        for i, (up_block, csa_layer) in enumerate(zip(self.up_blocks, self.csa_layers)):
            skip = skips[i]
            x = up_block(x, skip)  # bilinear upsample + concat
            x = csa_layer(x)       # CSA Bottleneck refinement
        return x
