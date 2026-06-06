"""Attention UNet encoder.

Extracts the 4-stage CNN encoder from Attention UNet (Oktay et al., 2018)
for use in bottleneck/decoder ablation studies.

Architecture:
    in → DoubleConv(in, 64) → pool → DoubleConv(64, 128) → pool →
    DoubleConv(128, 256) → pool → DoubleConv(256, 512)
Returns 4 multi-scale feature maps (deepest LAST).
out_channels = [64, 128, 256, 512].

NOTE: The (Conv-BN-ReLU)x2 helper mirrors ``_DoubleConv`` from
``medseg.models.networks.cnn.attention_unet`` and is inlined here to avoid a
circular import through ``medseg.models.networks/__init__.py`` (which eagerly
imports mamba modules that depend on ``medseg.models.encoders.vmunet_encoder``).
"""
# Source: https://github.com/ozan-oktay/Attention-Gated-Networks

from typing import List

import torch
import torch.nn as nn

from medseg.registry import ENCODER_REGISTRY


class ConvBlock(nn.Module):
    """Two consecutive Conv3x3-BN-ReLU blocks (DoubleConv)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


@ENCODER_REGISTRY.register("attention_unet")
class AttentionUNetEncoder(nn.Module):
    """Standard 4-stage Attention UNet encoder.

    Stem + 4 stages of (Conv-BN-ReLU)x2 with MaxPool2x2 between stages.

    Args:
        in_channels: Number of input image channels.
        img_size:    Expected input spatial resolution (unused, kept for
                     interface parity).
        channels:    Per-stage output channels (default [64, 128, 256, 512]).
    """

    def __init__(self, in_channels: int = 3, img_size: int = 224,
                 channels: List[int] = None, **kwargs):
        super().__init__()
        if channels is None:
            channels = [64, 128, 256, 512]

        self.out_channels = channels
        self.pool = nn.MaxPool2d(2, 2)

        self.stages = nn.ModuleList()
        ch_in = in_channels
        for ch_out in channels:
            self.stages.append(ConvBlock(ch_in, ch_out))
            ch_in = ch_out

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features: List[torch.Tensor] = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            features.append(x)
            if i < len(self.stages) - 1:
                x = self.pool(x)
        return features
