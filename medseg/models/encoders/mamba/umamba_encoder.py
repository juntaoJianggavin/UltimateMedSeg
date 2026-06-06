"""U-Mamba Encoder: plain-conv stages with a Mamba SSM bottleneck.

Adapted from ``medseg/networks/mamba/umamba.py`` (Ma et al., "U-Mamba: Enhancing
Long-range Dependency for Biomedical Image Segmentation", MICCAI 2024,
https://github.com/bowang-lab/U-Mamba). The original ``UMambaBot`` model
combines a strided convolutional encoder (nnU-Net's PlainConv stages) with a
single ``MambaLayer`` (patch-token mode) applied at the bottleneck before the
decoder. This module exposes that combination as a registered standalone
encoder so it can be plugged into any decoder via the project's registry.

Key components reused from the source implementation:
    - ``MambaLayer``         (Mamba SSM on flattened spatial tokens)
    - ``MambaSSM``           (mamba_ssm wrapper; hard dependency on mamba_ssm)

The conv stages here mirror nnU-Net's PlainConvEncoder (Conv-InstanceNorm-
LeakyReLU pairs, strided down-sampling) rather than the residual variant used
by ``UMambaBot``/``UMambaEnc``; this keeps the standalone encoder light while
preserving the original Mamba-at-bottleneck behaviour.
"""
# Source: https://github.com/bowang-lab/U-Mamba

from typing import List, Sequence

import torch
import torch.nn as nn

from medseg.registry import ENCODER_REGISTRY
from medseg.models.networks.mamba.umamba import MambaLayer, MambaSSM  # noqa: F401


# ---------------------------------------------------------------------------
# Plain convolutional building blocks (Conv-InstanceNorm-LeakyReLU)
# ---------------------------------------------------------------------------

class PlainConvBlock(nn.Module):
    """Single Conv-InstanceNorm-LeakyReLU block (nnU-Net plain-conv style)."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1,
                 conv_bias: bool = False):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=conv_bias)
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class PlainConvStage(nn.Module):
    """Stack of plain conv blocks: first block applies the stage stride."""

    def __init__(self, in_channels: int, out_channels: int,
                 n_blocks: int = 2, kernel_size: int = 3,
                 stride: int = 1, conv_bias: bool = False):
        super().__init__()
        layers = [PlainConvBlock(in_channels, out_channels,
                                 kernel_size=kernel_size,
                                 stride=stride, conv_bias=conv_bias)]
        for _ in range(n_blocks - 1):
            layers.append(PlainConvBlock(out_channels, out_channels,
                                         kernel_size=kernel_size,
                                         stride=1, conv_bias=conv_bias))
        self.blocks = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


# ---------------------------------------------------------------------------
# Standalone U-Mamba encoder
# ---------------------------------------------------------------------------

@ENCODER_REGISTRY.register("umamba")
class UMambaEncoder(nn.Module):
    """U-Mamba encoder: PlainConv pyramid + MambaLayer at the bottleneck.

    The first stage runs at stride 1 (nnU-Net "stem"); every subsequent stage
    applies stride 2 in its first conv. Per-stage channel counts double from
    ``base_features`` and are capped at ``max_features`` (320 by default, also
    nnU-Net's default cap). After the deepest stage, a single ``MambaLayer``
    in patch-token mode (spatial tokens as the sequence) refines the
    bottleneck feature.

    Args:
        in_channels: Input image channels.
        img_size: Input spatial size (not used architecturally but kept for
            interface parity with other encoders).
        base_features: Channels of the first stage; doubles per stage.
        num_stages: Number of pyramid stages including the stride-1 stem.
        max_features: Upper bound for per-stage channels (nnU-Net convention).
        n_blocks_per_stage: Plain-conv blocks per stage (int or sequence).
        kernel_size: Conv kernel size used in every stage.
        mamba_d_state: Mamba SSM state dimension at the bottleneck.
        mamba_d_conv: Mamba SSM conv width at the bottleneck.
        mamba_expand: Mamba SSM expansion factor at the bottleneck.
    """

    def __init__(self,
                 in_channels: int = 3,
                 img_size: int = 224,
                 base_features: int = 32,
                 num_stages: int = 6,
                 max_features: int = 320,
                 n_blocks_per_stage=2,
                 kernel_size: int = 3,
                 mamba_d_state: int = 16,
                 mamba_d_conv: int = 4,
                 mamba_expand: int = 2,
                 **kwargs):
        super().__init__()
        if num_stages < 2:
            raise ValueError(
                f"num_stages must be >= 2, got {num_stages}")

        self.img_size = img_size

        # Per-stage feature dims: doubled, capped at max_features.
        features = [min(base_features * (2 ** s), max_features)
                    for s in range(num_stages)]

        # Per-stage block counts.
        if isinstance(n_blocks_per_stage, int):
            blocks_per_stage = [n_blocks_per_stage] * num_stages
        else:
            blocks_per_stage = list(n_blocks_per_stage)
            if len(blocks_per_stage) != num_stages:
                raise ValueError(
                    f"n_blocks_per_stage length {len(blocks_per_stage)} "
                    f"does not match num_stages {num_stages}")

        # Strides: stem at 1, every subsequent stage at 2.
        strides = [1] + [2] * (num_stages - 1)

        # Build conv pyramid.
        self.stages = nn.ModuleList()
        prev_ch = in_channels
        for s in range(num_stages):
            self.stages.append(PlainConvStage(
                in_channels=prev_ch,
                out_channels=features[s],
                n_blocks=blocks_per_stage[s],
                kernel_size=kernel_size,
                stride=strides[s],
            ))
            prev_ch = features[s]

        # Mamba bottleneck: patch-token mode on the deepest feature.
        self.mamba_bottleneck = MambaLayer(
            dim=features[-1],
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
            channel_token=False,
        )

        self.out_channels: List[int] = list(features)
        self.strides: List[int] = strides

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features: List[torch.Tensor] = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        # Apply Mamba on the bottleneck (faithful to UMambaBot.forward, which
        # replaces ``skips[-1]`` with the Mamba-refined feature).
        features[-1] = self.mamba_bottleneck(features[-1])
        return features
