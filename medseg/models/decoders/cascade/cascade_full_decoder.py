"""Cascade Full Decoder – representative CASCADE-style cascaded attention decoder.

A standalone decoder that mirrors the cascade-attention concat pattern used in
CASCADE (Medical Image Segmentation via Cascaded Attention Decoding, WACV 2023)
but kept self-contained and compatible with the framework's external skip
connection module (``has_internal_skip = False``).

Each decoder stage applies:
    interpolate-up  ->  skip-concat  ->  channel-attention  ->  spatial-attention
    ->  3x3 conv

The framework wires skips externally via the provided ``skip_connection``
module; when ``skip_connection`` is ``None`` we fall back to plain concat.
"""
# Source: https://github.com/SLDGroup/CASCADE

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from medseg.registry import DECODER_REGISTRY


class _ChannelAttention(nn.Module):
    """CBAM-style channel attention (avg + max pool, shared MLP)."""

    def __init__(self, in_planes: int, ratio: int = 16):
        super().__init__()
        hidden = max(in_planes // ratio, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, hidden, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class _SpatialAttention(nn.Module):
    """CBAM-style spatial attention (avg + max along channel)."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(out))


@DECODER_REGISTRY.register("cascade_full")
class CADecoder(nn.Module):
    """Cascade-attention decoder (representative concat pattern).

    Args:
        encoder_channels: Skip-connection channels (shallow -> deep). Typically
            this is ``encoder.out_channels[:-1]`` as wired by ``model_builder``.
        bottleneck_channels: Channels of the bottleneck feature handed to
            ``forward``.
        skip_connection: External skip fusion module (e.g. concat/add). When
            ``None`` we fall back to channel concatenation.
        img_size: Input spatial size (kept for API symmetry; unused internally).
    """

    has_internal_skip = False

    def __init__(self, encoder_channels: List[int], bottleneck_channels: int,
                 skip_connection: nn.Module = None, img_size: int = 224,
                 **kwargs):
        super().__init__()
        self.skip_connection = skip_connection
        self.img_size = img_size

        # Skips are processed from deep -> shallow during decoding.
        skip_channels = list(reversed(encoder_channels))

        # Channel attention modules operate on the post-fusion tensor.
        self.channel_attns = nn.ModuleList()
        self.spatial_attns = nn.ModuleList()
        self.convs = nn.ModuleList()

        in_ch = bottleneck_channels
        for skip_ch in skip_channels:
            if skip_connection is not None:
                merged_ch = skip_connection.get_out_channels(in_ch, skip_ch)
            else:
                merged_ch = in_ch + skip_ch  # plain concat fallback

            out_ch = skip_ch
            self.channel_attns.append(_ChannelAttention(merged_ch))
            self.spatial_attns.append(_SpatialAttention(kernel_size=7))
            self.convs.append(nn.Sequential(
                nn.Conv2d(merged_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ))
            in_ch = out_ch

        # Per spec: out_channels = shallowest encoder channel.
        self._out_channels = encoder_channels[0] if encoder_channels else bottleneck_channels

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def forward(self, bottleneck_feat: torch.Tensor,
                skip_features: List[torch.Tensor]) -> torch.Tensor:
        # Walk skips from deepest to shallowest in lockstep with the up stages.
        skips = list(reversed(skip_features))

        x = bottleneck_feat
        for i, (ca, sa, conv) in enumerate(
            zip(self.channel_attns, self.spatial_attns, self.convs)
        ):
            skip = skips[i]
            # 1) interpolate-up to the skip's spatial size
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:],
                                  mode='bilinear', align_corners=False)
            # 2) skip-concat (external skip module, else default concat)
            if self.skip_connection is not None:
                x = self.skip_connection(x, skip)
            else:
                x = torch.cat([x, skip], dim=1)
            # 3) channel attention
            x = ca(x) * x
            # 4) spatial attention
            x = sa(x) * x
            # 5) 3x3 conv (channel reduction to this stage's target)
            x = conv(x)
        return x
