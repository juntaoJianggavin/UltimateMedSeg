"""xLSTM-UNet (Bot-style) standalone encoder.

Adapted from ``medseg/networks/other/xlstm_unet.py`` (Chen et al., "xLSTM-UNet
can be an Effective 2D & 3D Medical Image Segmentation Backbone with
Vision-LSTM (ViL)", arXiv:2407.01530, https://github.com/tianrun-chen/
xLSTM-UNet-PyTorch). The original ``XLSTMUNetBot`` model combines a residual
convolutional encoder (nnU-Net's residual ``UNetResEncoder``) with a single
``XLSTMLayer`` (patch-token mode) at the bottleneck before the decoder.

This module exposes that combination as a registered standalone encoder so it
can be plugged into any decoder via the project's registry. The "Bot" variant
is chosen (xLSTM only at the bottleneck) because it is simpler and works at
arbitrary resolutions: patch-token mode flattens spatial tokens at runtime, so
no spatial shape is baked into the parameters.

Building blocks ``BasicResBlock`` and ``BasicBlockD`` are reused from the
U-Mamba source, and ``XLSTMLayer`` is reused from the xLSTM-UNet source.
"""
# Source: https://github.com/tianrun-chen/xLSTM-UNet-PyTorch

from typing import List, Sequence

import torch
import torch.nn as nn

from medseg.registry import ENCODER_REGISTRY
from medseg.models.networks.mamba.umamba import BasicResBlock, BasicBlockD  # noqa: F401
from medseg.models.networks.other.xlstm_unet import XLSTMLayer  # noqa: F401


# ---------------------------------------------------------------------------
# Standalone xLSTM-UNet (Bot) encoder
# ---------------------------------------------------------------------------

@ENCODER_REGISTRY.register("xlstm")
class XLSTMEncoder(nn.Module):
    """xLSTM-UNet Bot encoder: residual conv pyramid + XLSTMLayer at bottleneck.

    The first stage is a stride-1 "stem" that lifts the input to
    ``features[0]`` channels; every subsequent stage applies stride 2 in its
    first block. After the deepest stage, a single ``XLSTMLayer`` in
    patch-token mode (spatial tokens as the sequence) refines the bottleneck
    feature.

    Args:
        in_channels: Input image channels. If ``in_channels != 3`` a 1x1
            conv stem is prepended to map it to 3 channels (consistent with
            the framework's convention for non-RGB inputs).
        img_size: Input spatial size. Kept for interface parity; the encoder
            derives all spatial state from the runtime tensor shape and works
            at arbitrary resolutions.
        pretrained: Unused (no pretrained weights are published for xLSTM-UNet).
        features: Per-stage channel counts. Defaults to ``[32, 64, 128, 256, 512]``
            (matches the source ``XLSTMUNetBot`` default).
        n_blocks_per_stage: Residual blocks per stage (int or sequence of
            length ``len(features)``). Default 2 per stage.
        kernel_size: Conv kernel size used in every stage.
        conv_bias: Whether to use bias in the conv blocks.
    """

    def __init__(self,
                 in_channels: int = 3,
                 img_size: int = 224,
                 pretrained: bool = False,
                 features: Sequence[int] = None,
                 n_blocks_per_stage=2,
                 kernel_size: int = 3,
                 conv_bias: bool = False,
                 **kwargs):
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256, 512]
        features = list(features)
        n_stages = len(features)
        if n_stages < 2:
            raise ValueError(f"features must have >= 2 stages, got {n_stages}")

        self.img_size = img_size

        # Per-stage block counts.
        if isinstance(n_blocks_per_stage, int):
            blocks_per_stage = [n_blocks_per_stage] * n_stages
        else:
            blocks_per_stage = list(n_blocks_per_stage)
            if len(blocks_per_stage) != n_stages:
                raise ValueError(
                    f"n_blocks_per_stage length {len(blocks_per_stage)} "
                    f"does not match number of stages {n_stages}")

        # Optional channel adapter: 1x1 conv when input is not 3-channel.
        if in_channels != 3:
            self.input_adapter = nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)
            stem_in = 3
        else:
            self.input_adapter = None
            stem_in = in_channels

        pad = kernel_size // 2

        # Stem: stride-1 residual block + extra residual blocks.
        stem_ch = features[0]
        self.stem = nn.Sequential(
            BasicResBlock(stem_in, stem_ch, kernel_size, pad,
                          stride=1, use_1x1conv=True),
            *[BasicBlockD(stem_ch, kernel_size, conv_bias)
              for _ in range(blocks_per_stage[0] - 1)],
        )

        # Strided stages 1..n_stages-1.
        self.stages = nn.ModuleList()
        prev_ch = stem_ch
        for s in range(1, n_stages):
            stage = nn.Sequential(
                BasicResBlock(prev_ch, features[s], kernel_size, pad,
                              stride=2, use_1x1conv=True),
                *[BasicBlockD(features[s], kernel_size, conv_bias)
                  for _ in range(blocks_per_stage[s] - 1)],
            )
            self.stages.append(stage)
            prev_ch = features[s]

        # xLSTM at the bottleneck, patch-token mode (resolution-friendly).
        self.xlstm_bottleneck = XLSTMLayer(dim=features[-1], channel_token=False)

        self.out_channels: List[int] = list(features)
        # Effective strides relative to input (stem at 1, then doubling).
        self.strides: List[int] = [1] + [2 ** s for s in range(1, n_stages)]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        if self.input_adapter is not None:
            x = self.input_adapter(x)
        features: List[torch.Tensor] = []
        x = self.stem(x)
        features.append(x)
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        # Apply xLSTM on the bottleneck (faithful to XLSTMUNetBot.forward,
        # which replaces ``skips[-1]`` with the xLSTM-refined feature).
        features[-1] = self.xlstm_bottleneck(features[-1])
        return features
