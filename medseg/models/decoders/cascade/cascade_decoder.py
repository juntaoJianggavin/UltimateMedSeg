"""CASCADE Decoder – faithful port from SLDGroup/CASCADE.

Medical Image Segmentation via Cascaded Attention Decoding (WACV 2023).
https://github.com/SLDGroup/CASCADE/blob/main/lib/decoders.py

Key innovations:
  - Attention gates on skip connections
  - CBAM (Channel Attention + Spatial Attention) at every decoder stage
  - Multi-scale deep supervision (returns all decoder stages)

CASCADE has its own internal skip mechanism (attention gates + CBAM),
so external skip_connection module is IGNORED.
"""
# Source: https://github.com/SLDGroup/CASCADE

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from medseg.registry import DECODER_REGISTRY


class _ConvBlock(nn.Module):
    """Double conv block faithful to original CASCADE."""

    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, 3, 1, 1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, 3, 1, 1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class _UpConv(nn.Module):
    """Upsample + conv faithful to original CASCADE."""

    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, 3, 1, 1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.up(x)


class _AttentionBlock(nn.Module):
    """Attention gate faithful to original CASCADE."""

    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class _ChannelAttention(nn.Module):
    """Channel attention (CBAM) faithful to original CASCADE."""

    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class _SpatialAttention(nn.Module):
    """Spatial attention (CBAM) faithful to original CASCADE."""

    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


@DECODER_REGISTRY.register("cascade")
class CASCADEDecoder(nn.Module):
    """CASCADE decoder – faithful port from SLDGroup/CASCADE.

    Uses attention gates + CBAM (channel + spatial attention) at each
    decoder stage. External skip_connection parameter is IGNORED.

    Args:
        encoder_channels: List of encoder stage output channels (shallow→deep).
        bottleneck_channels: Deepest feature channels.
    """
    has_internal_skip = True

    def __init__(self, encoder_channels: List[int], bottleneck_channels: int,
                 skip_connection=None, **kwargs):
        super().__init__()
        # encoder_channels from model_builder is already the skip channels
        # (encoder.out_channels[:-1]).  Do NOT truncate again.
        # Reversed: deep to shallow, matching decoder upsample order.
        skip_chs = list(reversed(encoder_channels))
        channels = [bottleneck_channels] + skip_chs
        # e.g. enc=[64,128,256] (already skip-only), bottleneck=512 → skip_chs=[256,128,64], channels=[512,256,128,64]

        # 1×1 conv on bottleneck
        self.Conv_1x1 = nn.Conv2d(channels[0], channels[0], 1, 1, 0)

        # Stage 4 (deepest): CBAM + conv block on bottleneck
        self.ConvBlock4 = _ConvBlock(channels[0], channels[0])

        # Decoder stages (3 levels for 4 encoder stages)
        self.ups = nn.ModuleList()
        self.ags = nn.ModuleList()
        self.conv_blocks = nn.ModuleList()

        for i in range(len(skip_chs)):
            in_ch = channels[i]
            out_ch = channels[i + 1]
            self.ups.append(_UpConv(in_ch, out_ch))
            # Attention gate: F_int = min(out_ch, channels[i+2]) if available, else 32
            if i + 2 < len(channels):
                f_int = channels[i + 2]
            else:
                f_int = 32
            self.ags.append(_AttentionBlock(F_g=out_ch, F_l=out_ch, F_int=f_int))
            self.conv_blocks.append(_ConvBlock(2 * out_ch, out_ch))

        # Channel attention modules (one per decoder stage)
        self.cas = nn.ModuleList()
        self.cas.append(_ChannelAttention(channels[0]))  # for bottleneck
        for i in range(len(skip_chs)):
            self.cas.append(_ChannelAttention(2 * channels[i + 1]))

        # Shared spatial attention
        self.sa = _SpatialAttention()

        self._out_channels = skip_chs[-1] if skip_chs else bottleneck_channels

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, bottleneck_feat: torch.Tensor,
                skip_features: List[torch.Tensor]) -> torch.Tensor:
        skips = list(reversed(skip_features))  # deep to shallow

        # Stage 4: 1×1 conv → CAM → SAM → conv block on bottleneck
        d = self.Conv_1x1(bottleneck_feat)
        d = self.cas[0](d) * d
        d = self.sa(d) * d
        d = self.ConvBlock4(d)

        # Decoder stages with attention gates + CBAM
        for i in range(len(self.ups)):
            # Upsample
            d = self.ups[i](d)
            # Attention gate on skip feature
            x_skip = self.ags[i](g=d, x=skips[i])
            # Concatenate
            d = torch.cat([x_skip, d], dim=1)
            # CBAM: channel attention → spatial attention
            d = self.cas[i + 1](d) * d
            d = self.sa(d) * d
            # Conv block
            d = self.conv_blocks[i](d)

        return d
