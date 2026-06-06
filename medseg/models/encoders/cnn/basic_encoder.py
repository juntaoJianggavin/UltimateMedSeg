"""Basic convolutional encoder (vanilla UNet-style).

A simple 4-stage convolutional encoder without pretrained weights.
Each stage: DoubleConv (Conv3x3-BN-ReLU × 2) → MaxPool2x2.
Output channels: [64, 128, 256, 512] by default.
"""
# Source: INTERNAL — framework adaptation (this repo).

import torch
import torch.nn as nn
from typing import List

from medseg.registry import ENCODER_REGISTRY


class DoubleConv(nn.Module):
    """Two consecutive Conv3x3-BN-ReLU blocks."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


@ENCODER_REGISTRY.register("basic")
class BasicEncoder(nn.Module):
    """Vanilla UNet encoder with 4 down-sampling stages.

    Architecture:
        in → DoubleConv(in, 64) → pool → DoubleConv(64, 128) → pool →
        DoubleConv(128, 256) → pool → DoubleConv(256, 512) → pool

    Returns 4 multi-scale feature maps (before each pooling).
    out_channels = [64, 128, 256, 512].
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
            self.stages.append(DoubleConv(ch_in, ch_out))
            ch_in = ch_out

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
            x = self.pool(x)
        return features
