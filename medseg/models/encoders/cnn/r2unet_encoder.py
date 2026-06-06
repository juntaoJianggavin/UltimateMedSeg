"""R2U-Net encoder.

Extracts the 5-stage RRCNN-based encoder from R2U-Net (Alom et al., 2018)
for use in bottleneck/decoder ablation studies.

Architecture:
    in → RRCNN(in, 64) → pool → RRCNN(64, 128) → pool →
    RRCNN(128, 256) → pool → RRCNN(256, 512) → pool → RRCNN(512, 1024)
Returns 5 multi-scale feature maps (deepest LAST).
out_channels = [64, 128, 256, 512, 1024].

NOTE: The RecurrentBlock / RRCNNBlock helpers mirror ``_RecurrentBlock`` and
``_RRCNNBlock`` from ``medseg.models.networks.cnn.r2unet`` and are inlined here to
avoid a circular import through ``medseg.models.networks/__init__.py``.
"""
# Source: https://github.com/LeeJunHyun/Image_Segmentation

from typing import List

import torch
import torch.nn as nn

from medseg.registry import ENCODER_REGISTRY


class RecurrentBlock(nn.Module):
    """Recurrent convolution: Conv3x3 + BN + ReLU iterated ``t`` times,
    adding the original input back to the accumulating activation at each
    iteration (RCNN cell from Alom et al., 2018).
    """

    def __init__(self, ch_out: int, t: int = 2):
        super().__init__()
        self.t = t
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1,
                      bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = None
        for i in range(self.t):
            if i == 0:
                x1 = self.conv(x)
            x1 = self.conv(x + x1)
        return x1


class RRCNNBlock(nn.Module):
    """Recurrent Residual CNN block.

    1x1 conv first projects the incoming features to ``ch_out`` channels,
    then two stacked recurrent conv blocks are applied. A residual skip
    adds the projected features back to the recurrent output.
    """

    def __init__(self, ch_in: int, ch_out: int, t: int = 2):
        super().__init__()
        self.RCNN = nn.Sequential(
            RecurrentBlock(ch_out, t=t),
            RecurrentBlock(ch_out, t=t),
        )
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1,
                                  padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x + x1


@ENCODER_REGISTRY.register("r2unet")
class R2UNetEncoder(nn.Module):
    """R2U-Net encoder with 5 RRCNN stages (4 MaxPool2x2 downsamples).

    Args:
        in_channels: Number of input image channels.
        img_size:    Expected input spatial resolution (unused, kept for
                     interface parity).
        channels:    Per-stage output channels
                     (default [64, 128, 256, 512, 1024]).
        t:           Number of recurrent iterations per RCNN cell (default 2).
    """

    def __init__(self, in_channels: int = 3, img_size: int = 224,
                 channels: List[int] = None, t: int = 2, **kwargs):
        super().__init__()
        if channels is None:
            channels = [64, 128, 256, 512, 1024]

        self.out_channels = channels
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.stages = nn.ModuleList()
        ch_in = in_channels
        for ch_out in channels:
            self.stages.append(RRCNNBlock(ch_in, ch_out, t=t))
            ch_in = ch_out

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features: List[torch.Tensor] = []
        for i, stage in enumerate(self.stages):
            if i > 0:
                x = self.pool(x)
            x = stage(x)
            features.append(x)
        return features
