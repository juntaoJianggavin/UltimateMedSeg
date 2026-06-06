"""LightM-UNet: Lightweight Mamba UNet for Medical Image Segmentation.

Faithful reimplementation from:
  https://github.com/MrBlankness/LightM-UNet  (2024)

Architecture:
  - Lightweight ~1M params Mamba-based encoder-decoder
  - Encoder: DWConv init → MambaLayer downsample + ResMambaBlock stages
  - Decoder: Conv1x1 + upsample + ResUpBlock (DWConv) stages
  - Uses Mamba SSM (hard dependency on mamba_ssm, matching official source)

Self-contained implementation without MONAI dependency.
Uses MambaSSM from umamba.py for Mamba SSM.
"""
# Source: https://github.com/MrBlankness/LightM-UNet

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from .umamba import MambaSSM


# ---------------------------------------------------------------------------
# DWConv Layer (depthwise separable convolution)
# ---------------------------------------------------------------------------

class DWConv(nn.Module):
    """Depthwise separable convolution."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, bias=False):
        super().__init__()
        self.depth_conv = nn.Conv2d(in_channels, in_channels,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=kernel_size // 2,
                                     groups=in_channels, bias=bias)
        self.point_conv = nn.Conv2d(in_channels, out_channels,
                                     kernel_size=1, bias=bias)

    def forward(self, x):
        return self.point_conv(self.depth_conv(x))


# ---------------------------------------------------------------------------
# MambaLayer for 2D features
# ---------------------------------------------------------------------------

class LightMambaLayer(nn.Module):
    """Mamba layer with optional channel projection + skip scaling.

    Faithful to LightM-UNet's MambaLayer implementation.
    """
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.mamba = MambaSSM(input_dim, d_state=d_state,
                               d_conv=d_conv, expand=expand)
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.float()
        B, C = x.shape[:2]
        assert C == self.input_dim
        img_dims = x.shape[2:]
        n_tokens = img_dims.numel()

        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)  # (B, N, C)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm) + self.skip_scale * x_flat
        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        return x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)


# ---------------------------------------------------------------------------
# ResMambaBlock: residual block with Mamba layers
# ---------------------------------------------------------------------------

class ResMambaBlock(nn.Module):
    """Residual block with two Mamba layers.

    Faithful to LightM-UNet's ResMambaBlock.
    """
    def __init__(self, channels, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, channels), channels)
        self.norm2 = nn.GroupNorm(min(8, channels), channels)
        self.act = nn.ReLU(inplace=True)
        self.conv1 = LightMambaLayer(channels, channels,
                                      d_state=d_state, d_conv=d_conv, expand=expand)
        self.conv2 = LightMambaLayer(channels, channels,
                                      d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x):
        identity = x
        x = self.act(self.norm1(x))
        x = self.conv1(x)
        x = self.act(self.norm2(x))
        x = self.conv2(x)
        return x + identity


# ---------------------------------------------------------------------------
# ResUpBlock: residual upsampling block with DWConv
# ---------------------------------------------------------------------------

class ResUpBlock(nn.Module):
    """Residual block with DWConv for decoder.

    Faithful to LightM-UNet's ResUpBlock.
    """
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, channels), channels)
        self.norm2 = nn.GroupNorm(min(8, channels), channels)
        self.act = nn.ReLU(inplace=True)
        self.conv = DWConv(channels, channels, kernel_size=kernel_size)
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        identity = x
        x = self.act(self.norm1(x))
        x = self.conv(x) + self.skip_scale * identity
        x = self.act(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# LightMUNet: main model
# ---------------------------------------------------------------------------

class LightMUNet(nn.Module):
    """LightM-UNet: Lightweight Mamba UNet (~1M params).

    Architecture:
      DWConv init → 4 encoder stages (MambaLayer downsample + ResMambaBlock)
      → 3 decoder stages (Conv1x1 + upsample + ResUpBlock)
      → GroupNorm + ReLU + DWConv head

    Args:
        in_channels: Input image channels.
        num_classes: Number of output classes.
        img_size: Input image size.
        init_filters: Base channel count.
        blocks_down: Number of ResMambaBlocks per encoder stage.
        blocks_up: Number of ResUpBlocks per decoder stage.
        d_state: Mamba SSM state dimension.
        d_conv: Mamba SSM conv width.
        expand: Mamba SSM expansion factor.
    """
    def __init__(self, in_channels=3, num_classes=2, img_size=224,
                 init_filters=32, blocks_down=None, blocks_up=None,
                 d_state=16, d_conv=4, expand=2,
                 dropout_prob=None, deep_supervision=False, **kwargs):
        super().__init__()
        if blocks_down is None:
            blocks_down = [1, 2, 2, 4]
        if blocks_up is None:
            blocks_up = [1, 1, 1]

        self.init_filters = init_filters
        self.blocks_down = blocks_down
        self.blocks_up = blocks_up

        # Initial DWConv
        self.conv_init = DWConv(in_channels, init_filters)

        # Encoder
        self.down_layers = nn.ModuleList()
        for i, n_blocks in enumerate(blocks_down):
            ch = init_filters * (2 ** i)
            if i > 0:
                downsample = nn.Sequential(
                    LightMambaLayer(ch // 2, ch, d_state=d_state,
                                    d_conv=d_conv, expand=expand),
                    nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                downsample = nn.Identity()
            blocks = [ResMambaBlock(ch, d_state=d_state, d_conv=d_conv,
                                    expand=expand) for _ in range(n_blocks)]
            self.down_layers.append(nn.Sequential(downsample, *blocks))

        # Decoder
        n_up = len(blocks_up)
        self.up_samples = nn.ModuleList()
        self.up_layers = nn.ModuleList()
        for i in range(n_up):
            sample_ch = init_filters * (2 ** (n_up - i))
            out_ch = sample_ch // 2
            self.up_samples.append(nn.Sequential(
                nn.Conv2d(sample_ch, out_ch, 1),
                nn.Upsample(scale_factor=2, mode='nearest'),
            ))
            self.up_layers.append(nn.Sequential(
                *[ResUpBlock(out_ch) for _ in range(blocks_up[i])]
            ))

        # Final head
        self.conv_final = nn.Sequential(
            nn.GroupNorm(min(8, init_filters), init_filters),
            nn.ReLU(inplace=True),
            DWConv(init_filters, num_classes, kernel_size=1, bias=True),
        )

        if dropout_prob is not None and dropout_prob > 0:
            self.dropout = nn.Dropout2d(dropout_prob)
        else:
            self.dropout = None

        self.deep_supervision = deep_supervision
        if deep_supervision:
            # DS heads for decoder intermediates (all except last stage)
            n_up = len(self.blocks_up)
            self.ds_heads = nn.ModuleList([
                nn.Conv2d(init_filters * (2 ** (n_up - 1 - i)), num_classes, 1)
                for i in range(n_up - 1)
            ])

    def forward(self, x):
        input_size = x.shape[2:]
        x = self.conv_init(x)
        if self.dropout is not None:
            x = self.dropout(x)

        # Encoder
        down_x = []
        for down in self.down_layers:
            x = down(x)
            down_x.append(x)

        # Reverse for decoder skip connections
        down_x.reverse()

        # Decoder
        ds_collect = self.training and self.deep_supervision
        intermediates = []
        for i, (up_sample, up_layer) in enumerate(zip(self.up_samples, self.up_layers)):
            x = up_sample(x)
            skip = down_x[i + 1]
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear',
                                  align_corners=False)
            x = x + skip
            x = up_layer(x)
            if ds_collect and i < len(self.up_samples) - 1:
                intermediates.append(x)

        # Final
        x = self.conv_final(x)
        if x.shape[2:] != input_size:
            x = F.interpolate(x, size=input_size, mode='bilinear',
                              align_corners=False)

        if ds_collect:
            aux = []
            for feat, head in zip(intermediates, self.ds_heads):
                a = head(feat)
                if a.shape[2:] != input_size:
                    a = F.interpolate(a, size=input_size, mode='bilinear',
                                      align_corners=False)
                aux.append(a)
            return [x] + aux
        return x
