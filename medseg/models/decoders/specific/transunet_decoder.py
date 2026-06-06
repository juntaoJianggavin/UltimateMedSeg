"""TransUNet CUP (Cascaded Upsampler) Decoder.
Faithfully ported from: https://github.com/Beckschen/TransUNet/blob/main/networks/vit_seg_modeling.py

Reference: Chen et al., "TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation"
All class/attribute names match the original DecoderCup for pretrained weight loading.

Has its own internal skip connection mechanism (concat).
External skip_connection parameter is IGNORED.
"""
# Source: https://github.com/Beckschen/TransUNet

import torch
import torch.nn as nn
from typing import List
from medseg.registry import DECODER_REGISTRY


class Conv2dReLU(nn.Sequential):
    """Conv2d + BatchNorm + ReLU, matching original TransUNet."""
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, use_batchnorm=True):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                         stride=stride, padding=padding, bias=not use_batchnorm)
        bn = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU(inplace=True)
        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    """Single decoder block: bilinear 2x upsample -> concat skip -> two Conv2dReLU."""
    def __init__(self, in_channels, out_channels, skip_channels=0, use_batchnorm=True):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels, out_channels,
            kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        self.conv2 = Conv2dReLU(
            out_channels, out_channels,
            kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = nn.functional.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


@DECODER_REGISTRY.register("transunet")
class TransUNetDecoder(nn.Module):
    """TransUNet CUP (Cascaded Upsampler) decoder.

    Faithful to the original DecoderCup class.
    Architecture:
        conv_more (hidden_size -> 512) ->
        4 DecoderBlocks with bilinear 2x upsample + concat skip + 2x Conv2dReLU
        decoder_channels = (256, 128, 64, 16), n_skip = 3

    External skip_connection is IGNORED - concat is built-in.
    """
    has_internal_skip = True

    def __init__(self, encoder_channels: List[int], bottleneck_channels: int,
                 skip_connection=None,
                 head_channels: int = 512,
                 decoder_channels: tuple = (256, 128, 64, 16),
                 **kwargs):
        super().__init__()
        # conv_more: reduce bottleneck channels to head_channels (original: 768 -> 512)
        self.conv_more = Conv2dReLU(bottleneck_channels, head_channels,
                                     kernel_size=3, padding=1, use_batchnorm=True)

        # Skip channels from encoder (reversed: deep to shallow)
        skip_channels = list(reversed(encoder_channels))
        # Pad with 0 for extra decoder blocks without skip
        while len(skip_channels) < len(decoder_channels):
            skip_channels.append(0)

        in_channels = [head_channels] + list(decoder_channels[:-1])
        self.blocks = nn.ModuleList([
            DecoderBlock(in_ch, out_ch, sk_ch)
            for in_ch, out_ch, sk_ch in zip(in_channels, decoder_channels, skip_channels)
        ])
        self._out_channels = decoder_channels[-1]

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, bottleneck_feat: torch.Tensor, skip_features: List[torch.Tensor]) -> torch.Tensor:
        skips = list(reversed(skip_features))  # deep to shallow
        x = self.conv_more(bottleneck_feat)
        for i, block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = block(x, skip=skip)
        return x
