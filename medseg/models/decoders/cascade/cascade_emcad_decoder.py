"""Cascade EMCAD Decoder – full cascade variant with channel + spatial attention.

Combines the CASCADE (Cascaded Attention Decoding, WACV 2023) staged refinement
idea with the EMCAD (Efficient Multi-scale Convolutional Attention Decoding,
CVPR 2024) attention block recipe.

Distinct from ``emcad_decoder.py``: that file is a faithful port of EMCAD with
LGAG gating + additive aggregation + MSDC.  This decoder is the cascade variant
where every stage applies the full attention recipe – depthwise conv -> channel
attention -> spatial attention -> 3x3 conv fusion – on a skip-concat tensor.

Three cascade stages walk the bottleneck back up at /16 -> /8 -> /4 relative to
the input, fusing one encoder skip per stage.  ``has_internal_skip = True``
because the decoder builds its own attention-based skip topology and does not
delegate to the framework's ``skip_connection`` module.
"""
# Source: https://github.com/SLDGroup/EMCAD

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from medseg.registry import DECODER_REGISTRY


class _DepthwiseConv(nn.Module):
    """3x3 depthwise conv + 1x1 pointwise (channel-preserving)."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1,
                      groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _ChannelAttention(nn.Module):
    """CBAM-style channel attention (avg + max pool, shared MLP)."""

    def __init__(self, in_planes: int, ratio: int = 16):
        super().__init__()
        if in_planes < ratio:
            ratio = max(in_planes, 1)
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


class _CascadeStage(nn.Module):
    """One cascade stage: upsample, skip-concat, depthwise conv, CA, SA, 3x3 fuse."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        merged_ch = in_ch + skip_ch
        self.dwconv = _DepthwiseConv(merged_ch)
        self.channel_attn = _ChannelAttention(merged_ch)
        self.spatial_attn = _SpatialAttention(kernel_size=7)
        self.fuse = nn.Sequential(
            nn.Conv2d(merged_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:],
                              mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.dwconv(x)
        x = self.channel_attn(x) * x
        x = self.spatial_attn(x) * x
        x = self.fuse(x)
        return x


@DECODER_REGISTRY.register("cascade_emcad")
class CascadeEMCADDecoder(nn.Module):
    """EMCAD-style cascade decoder with channel + spatial attention per stage.

    Three-stage cascade walking the bottleneck back up at /16 -> /8 -> /4.
    Each stage performs: bilinear upsample -> concat with skip -> depthwise
    conv -> channel attention -> spatial attention -> 3x3 conv fusion.

    Args:
        encoder_channels: Encoder stage output channels (shallow -> deep).
            The shallowest ``num_stages`` entries are used as per-stage skip
            channels.
        bottleneck_channels: Channels of the bottleneck feature.
        skip_connection: IGNORED – this decoder owns its skip topology.
        img_size: Kept for API symmetry; unused internally.

    Notes:
        ``has_internal_skip = True`` so the framework will not call
        ``skip_connection(decoder_feat, skip_feat)`` on our behalf.
    """

    has_internal_skip = True

    NUM_STAGES = 3

    def __init__(self, encoder_channels: List[int], bottleneck_channels: int,
                 skip_connection: nn.Module = None, img_size: int = 224,
                 **kwargs):
        super().__init__()
        self.img_size = img_size
        # External skip_connection intentionally ignored – kept on the instance
        # purely so introspection tools can see it was supplied.
        self.skip_connection = skip_connection

        if not encoder_channels:
            raise ValueError("encoder_channels must be non-empty for "
                             "CascadeEMCADDecoder")

        num_stages = min(self.NUM_STAGES, len(encoder_channels))
        # Use the shallowest ``num_stages`` encoder channels – the deepest
        # encoder stage typically coincides with the bottleneck and is not a
        # skip target here.
        skip_chs_shallow_to_deep = list(encoder_channels[:num_stages])
        # Cascade walks deep -> shallow.
        skip_chs = list(reversed(skip_chs_shallow_to_deep))

        self.stages = nn.ModuleList()
        in_ch = bottleneck_channels
        for skip_ch in skip_chs:
            out_ch = skip_ch
            self.stages.append(_CascadeStage(in_ch=in_ch, skip_ch=skip_ch,
                                             out_ch=out_ch))
            in_ch = out_ch

        # Final feature channel count BEFORE the segmentation head.
        self._out_channels = skip_chs_shallow_to_deep[0]
        self._num_stages = num_stages

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def forward(self, bottleneck_feat: torch.Tensor,
                skip_features: List[torch.Tensor]) -> torch.Tensor:
        if len(skip_features) < self._num_stages:
            raise ValueError(
                f"CascadeEMCADDecoder expects at least {self._num_stages} "
                f"skip features, got {len(skip_features)}"
            )

        # ``skip_features`` arrive shallow -> deep.  Take the matching slice
        # and walk deep -> shallow in lockstep with the stages.
        used_skips = skip_features[:self._num_stages]
        skips_deep_to_shallow = list(reversed(used_skips))

        x = bottleneck_feat
        for stage, skip in zip(self.stages, skips_deep_to_shallow):
            x = stage(x, skip)
        return x
