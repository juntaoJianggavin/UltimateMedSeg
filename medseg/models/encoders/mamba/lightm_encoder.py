"""LightM-UNet Encoder: lightweight Mamba-based hierarchical encoder.

Extracted from ``medseg/networks/mamba/lightm_unet.py`` (LightM-UNet,
https://github.com/MrBlankness/LightM-UNet, 2024). The original network
combines an encoder built from a DWConv stem followed by 4 stages of
``ResMambaBlock``s (with ``LightMambaLayer`` + MaxPool2x2 down-sampling
between stages) with a lightweight DWConv decoder. This file exposes the
encoder half as a standalone, registered module producing a 4-level
feature pyramid.

Pipeline per stage:
    Stage 0 (stride 1):  identity                       + ResMambaBlock x N0
    Stage i (stride 2):  LightMambaLayer + MaxPool2x2   + ResMambaBlock x Ni

Default channel widths follow the paper's ``init_filters=32`` recipe:
    out_channels = [32, 64, 128, 256].
Default block counts: ``blocks_down = [1, 2, 2, 4]``.

The Mamba SSM hard-depends on ``mamba_ssm`` (via ``MambaSSM`` from
``medseg.models.networks.mamba.umamba``), matching the official LightM-UNet
source which requires ``pip install mamba-ssm``.
"""
# Source: https://github.com/MrBlankness/LightM-UNet

from typing import List, Sequence

import torch
import torch.nn as nn

from medseg.registry import ENCODER_REGISTRY
from medseg.models.networks.mamba.umamba import MambaSSM


# ---------------------------------------------------------------------------
# Helper building blocks (prefixed with "_" since they are encoder-internal).
# ---------------------------------------------------------------------------


class _DWConv(nn.Module):
    """Depthwise separable convolution (depthwise 3x3 + pointwise 1x1)."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1, bias: bool = False):
        super().__init__()
        self.depth_conv = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=kernel_size, stride=stride,
            padding=kernel_size // 2,
            groups=in_channels, bias=bias,
        )
        self.point_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.point_conv(self.depth_conv(x))


class _LightMambaLayer(nn.Module):
    """Mamba layer with channel projection + learnable skip scale.

    Operates on (B, C, *spatial) tensors, flattening the spatial dims to a
    token sequence before applying the SSM and re-folding afterwards.
    Faithful to LightM-UNet's ``MambaLayer``.
    """

    def __init__(self, input_dim: int, output_dim: int,
                 d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.mamba = MambaSSM(
            input_dim, d_state=d_state, d_conv=d_conv, expand=expand,
        )
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.float16:
            x = x.float()
        B, C = x.shape[:2]
        assert C == self.input_dim
        img_dims = x.shape[2:]  # runtime-derived spatial shape
        n_tokens = img_dims.numel()

        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)  # (B, N, C)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm) + self.skip_scale * x_flat
        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        return x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)


class _ResMambaBlock(nn.Module):
    """Residual block with two ``_LightMambaLayer``s (faithful to source)."""

    def __init__(self, channels: int,
                 d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, channels), channels)
        self.norm2 = nn.GroupNorm(min(8, channels), channels)
        self.act = nn.ReLU(inplace=True)
        self.conv1 = _LightMambaLayer(
            channels, channels,
            d_state=d_state, d_conv=d_conv, expand=expand,
        )
        self.conv2 = _LightMambaLayer(
            channels, channels,
            d_state=d_state, d_conv=d_conv, expand=expand,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.act(self.norm1(x))
        x = self.conv1(x)
        x = self.act(self.norm2(x))
        x = self.conv2(x)
        return x + identity


# ---------------------------------------------------------------------------
# Standalone LightM-UNet encoder.
# ---------------------------------------------------------------------------


@ENCODER_REGISTRY.register("lightm")
class LightMUNetEncoder(nn.Module):
    """LightM-UNet encoder: DWConv stem + 4 Mamba stages with MaxPool down.

    Args:
        in_channels: Input image channels. If not equal to 3, a 1x1 conv
            stem maps to 3 channels before the DWConv init. (Kept simple
            because the DWConv stem already accepts arbitrary in_channels;
            the 1x1 here is just a convention for interface parity.)
        img_size: Nominal input spatial size. Not used architecturally;
            all spatial state is derived from runtime tensor shapes.
        pretrained: Unused (LightM-UNet has no canonical pretrained
            checkpoint); kept for interface parity.
        init_filters: Base channel count for the first stage. Subsequent
            stages double this.
        blocks_down: Number of ``_ResMambaBlock``s per encoder stage.
            Defaults to ``[1, 2, 2, 4]`` (paper recipe).
        d_state: Mamba SSM state dimension.
        d_conv: Mamba SSM conv width.
        expand: Mamba SSM expansion factor.
    """

    def __init__(self,
                 in_channels: int = 3,
                 img_size: int = 224,
                 pretrained: bool = False,
                 init_filters: int = 32,
                 blocks_down: Sequence[int] = None,
                 d_state: int = 16,
                 d_conv: int = 4,
                 expand: int = 2,
                 **kwargs):
        super().__init__()
        if blocks_down is None:
            blocks_down = [1, 2, 2, 4]
        blocks_down = list(blocks_down)

        self.img_size = img_size
        self.init_filters = init_filters
        self.blocks_down = blocks_down

        # Optional 1x1 stem when caller provides non-RGB input. The DWConv
        # below can already accept arbitrary in_channels, but we follow the
        # project convention of a 1x1 stem for parity with other encoders.
        if in_channels != 3:
            self.input_stem = nn.Conv2d(in_channels, 3, kernel_size=1, bias=True)
            stem_in = 3
        else:
            self.input_stem = nn.Identity()
            stem_in = in_channels

        # DWConv init: in -> init_filters, stride 1.
        self.conv_init = _DWConv(stem_in, init_filters)

        # Encoder stages.
        self.down_layers = nn.ModuleList()
        for i, n_blocks in enumerate(blocks_down):
            ch = init_filters * (2 ** i)
            if i > 0:
                downsample = nn.Sequential(
                    _LightMambaLayer(
                        ch // 2, ch,
                        d_state=d_state, d_conv=d_conv, expand=expand,
                    ),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                )
            else:
                downsample = nn.Identity()
            blocks = [
                _ResMambaBlock(
                    ch, d_state=d_state, d_conv=d_conv, expand=expand,
                )
                for _ in range(n_blocks)
            ]
            self.down_layers.append(nn.Sequential(downsample, *blocks))

        self.out_channels: List[int] = [
            init_filters * (2 ** i) for i in range(len(blocks_down))
        ]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Run the encoder and return per-stage feature maps.

        Args:
            x: (B, in_channels, H, W) input.
        Returns:
            List of feature maps, shallowest first, deepest LAST.
            Spatial sizes are roughly (H, H/2, H/4, H/8) for the default
            4-stage configuration.
        """
        x = self.input_stem(x)
        x = self.conv_init(x)

        features: List[torch.Tensor] = []
        for down in self.down_layers:
            x = down(x)
            features.append(x)
        return features
