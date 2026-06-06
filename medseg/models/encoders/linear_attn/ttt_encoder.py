"""TTT-UNet Encoder: ResBlock pyramid with a Test-Time Training (TTT) layer
at the bottleneck.

Adapted from ``medseg/networks/other/ttt_unet.py`` (Zhou et al., "TTT-UNet:
Enhancing U-Net with Test-Time Training Layers for Biomedical Image
Segmentation", NeurIPS 2024 Workshop, https://github.com/rongzhou7/TTT-Unet).

The source network combines a residual convolutional encoder (stage-0 at
stride 1, every subsequent stage strided 2) with a single ``TTTLayer``
applied at the bottleneck before the decoder. This module exposes that
encoder portion as a registered standalone encoder so it can be plugged
into any decoder via the project's registry.

Helper classes copied from the source are prefixed with ``_``; the
``TTTLayer`` is re-used directly from the source module.
"""
# Source: https://github.com/rongzhou7/TTT-Unet

from typing import List, Sequence

import torch
import torch.nn as nn

from medseg.registry import ENCODER_REGISTRY
from medseg.models.networks.other.ttt_unet import TTTLayer  # noqa: F401


# ---------------------------------------------------------------------------
# Residual conv block (copied from ttt_unet.ResBlock with ``_`` prefix)
# ---------------------------------------------------------------------------

class _ResBlock(nn.Module):
    """Residual conv block: (Conv-IN-LeakyReLU) x2 + identity/projection."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride,
                               padding=1, bias=False)
        self.norm1 = nn.InstanceNorm2d(out_ch, affine=True)
        self.act1 = nn.LeakyReLU(0.01, inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm2d(out_ch, affine=True)
        self.act2 = nn.LeakyReLU(0.01, inplace=True)
        if in_ch != out_ch or stride != 1:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=stride,
                                      bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act1(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return self.act2(y + self.shortcut(x))


# ---------------------------------------------------------------------------
# Standalone TTT encoder
# ---------------------------------------------------------------------------

@ENCODER_REGISTRY.register("ttt")
class TTTEncoder(nn.Module):
    """TTT-UNet encoder: ResBlock pyramid + ``TTTLayer`` at the bottleneck.

    Stage-0 runs at stride 1 (stem); every subsequent stage applies stride 2
    in its first ResBlock. Channel counts default to ``[32, 64, 128, 256,
    512]`` (5 stages, deepest spatial = ``img_size / 16``). After the
    deepest conv stage, a single ``TTTLayer`` (flatten -> TTTLinear ->
    reshape) refines the bottleneck feature, mirroring ``TTTUNet.forward``.

    Args:
        in_channels: Input image channels. If != 3 the existing first conv
            still handles arbitrary in-channels, but a 1x1 stem is prepended
            anyway to match the convention used by other encoders.
        img_size: Input spatial size (not architecturally baked in; the
            spatial state is derived from the runtime tensor shape).
        pretrained: Ignored (no pretrained weights available).
        features: Per-stage channel counts; doubled by default.
        n_blocks_per_stage: ResBlocks per encoder stage (int or sequence).
        ttt_d_state: Forwarded to ``TTTLayer`` for interface parity (the
            current ``TTTLayer`` does not use it internally).
    """

    def __init__(self,
                 in_channels: int = 3,
                 img_size: int = 224,
                 pretrained: bool = False,
                 features: Sequence[int] = None,
                 n_blocks_per_stage=None,
                 ttt_d_state: int = 16,
                 **kwargs):
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256, 512]
        features = list(features)
        num_stages = len(features)
        if num_stages < 2:
            raise ValueError(
                f"features must have >= 2 stages, got {num_stages}")

        if n_blocks_per_stage is None:
            n_blocks_per_stage = [2] * num_stages
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * num_stages
        n_blocks_per_stage = list(n_blocks_per_stage)
        if len(n_blocks_per_stage) != num_stages:
            raise ValueError(
                f"n_blocks_per_stage length {len(n_blocks_per_stage)} "
                f"does not match number of stages {num_stages}")

        self.img_size = img_size

        # Optional 1x1 stem so non-RGB inputs match the canonical interface.
        if in_channels != 3:
            self.stem = nn.Conv2d(in_channels, 3, 1, bias=False)
            stage_in = 3
        else:
            self.stem = nn.Identity()
            stage_in = in_channels

        # Conv pyramid: stage-0 stride 1, every subsequent stage stride 2.
        self.stages = nn.ModuleList()
        prev_ch = stage_in
        for i, f in enumerate(features):
            stride = 1 if i == 0 else 2
            blocks = [_ResBlock(prev_ch, f, stride=stride)]
            for _ in range(n_blocks_per_stage[i] - 1):
                blocks.append(_ResBlock(f, f))
            self.stages.append(nn.Sequential(*blocks))
            prev_ch = f

        # TTT bottleneck (flatten -> TTTLinear -> reshape).
        self.ttt_bottleneck = TTTLayer(features[-1], d_state=ttt_d_state)

        self.out_channels: List[int] = list(features)
        self.strides: List[int] = [1] + [2] * (num_stages - 1)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        feats: List[torch.Tensor] = []
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        # TTT on the deepest feature (faithful to ``TTTUNet.forward``).
        feats[-1] = self.ttt_bottleneck(feats[-1])
        return feats
