"""G-CASCADE Decoder – Efficient Cascaded Graph Convolutional Decoding.

Faithfully ported from: https://github.com/SLDGroup/G-CASCADE
Paper: G-CASCADE: Efficient Cascaded Graph Convolutional Decoding
       for 2D Medical Image Segmentation (WACV 2024, Rahman & Marclescu)

Uses graph convolution (GCB/Grapher) blocks with spatial attention (SPA) and
progressive skip aggregation. Two variants are supported:
  - ``gcascade``: additive skip aggregation
  - ``gcascade_cat``: concatenation skip aggregation

The decoder automatically adapts to any encoder: the number of decoder stages
is derived from encoder_channels. External skip_connection is IGNORED
(``has_internal_skip = True``).
"""
# Source: https://github.com/SLDGroup/G-CASCADE

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from medseg.registry import DECODER_REGISTRY
from ..gcn_lib import Grapher


def channel_shuffle(x, groups):
    """Channel shuffle (from ShuffleNet)."""
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x


def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a


# ── UCB (Up-Convolution Block) ──────────────────────────────────────────────

class UCB(nn.Module):
    """Up-convolution block: upsample + DW-separable conv + 1x1 pointwise."""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.in_channels = in_channels
        self.up_dwc = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, in_channels, kernel_size, 1,
                      kernel_size // 2, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.pwc = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=True)

    def forward(self, x):
        x = self.up_dwc(x)
        x = channel_shuffle(x, self.in_channels)
        x = self.pwc(x)
        return x


# ── SPA (Spatial Attention) ──────────────────────────────────────────────────

class SPA(nn.Module):
    """Spatial attention block (CBAM-style spatial branch)."""

    def __init__(self, kernel_size=7):
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(out))


# ── GCASCADE Decoder (additive skip) ────────────────────────────────────────

@DECODER_REGISTRY.register("gcascade")
class GCASCADEDecoder(nn.Module):
    """G-CASCADE decoder with additive skip aggregation.

    Automatically adapts to any encoder: builds N decoder stages where
    N = len(encoder_channels). Each stage: UCB (upsample) + GCB (graph conv)
    + SPA (spatial attention) + skip aggregation.

    Args:
        encoder_channels: List of encoder stage output channels (shallow to deep).
        bottleneck_channels: Channels of the bottleneck feature.
        img_size: Input image size (for Grapher relative pos).
        k: KNN graph kernel size.
        padding: Graph conv padding.
        conv: Graph conv type ('mr', 'edge', 'sage', 'gin').
        gcb_act: Grapher activation function.
        drop_path: Drop path rate for Grapher blocks.
        skip_connection: IGNORED (internal skip used).
    """

    has_internal_skip = True

    def __init__(self, encoder_channels: List[int], bottleneck_channels: int,
                 skip_connection=None, img_size: int = 224,
                 k: int = 11, padding: int = 5, conv: str = 'mr',
                 gcb_act: str = 'gelu', activation: str = 'relu',
                 drop_path: float = 0.0, **kwargs):
        super().__init__()
        self.img_size = img_size
        n_skips = len(encoder_channels)

        # Build channel list: [bottleneck, skip_deep, ..., skip_shallow]
        skip_chs = list(reversed(encoder_channels))
        channels = [bottleneck_channels] + skip_chs

        # GCB blocks (one per channel level)
        # Use max(16, ...) to ensure enough tokens for KNN at small img_size
        n_spatial = max(16, (img_size // 16) ** 2)
        self.gcb_blocks = nn.ModuleList([
            Grapher(ch, kernel_size=k, dilation=1, conv=conv, act=gcb_act,
                    n=n_spatial, drop_path=drop_path, padding=padding)
            for ch in channels
        ])

        # UCB blocks (upsample between stages)
        self.ucb_blocks = nn.ModuleList([
            UCB(channels[i], channels[i + 1])
            for i in range(n_skips)
        ])

        # SPA (shared)
        self.spa = SPA()

        # 1x1 convs for additive skip aggregation
        self.skip_convs = nn.ModuleList([
            nn.Conv2d(ch, ch, 1, bias=False) for ch in skip_chs
        ])

        # Channel reduction for multi-scale supervision
        shallowest_ch = channels[-1]
        self.h_1x1s = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, shallowest_ch, 1),
                nn.BatchNorm2d(shallowest_ch),
                nn.ReLU(inplace=True)
            )
            for ch in skip_chs[:-1]
        ])

        # Activation
        if activation.lower() == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif activation.lower() == 'relu6':
            self.act = nn.ReLU6(inplace=True)
        else:
            self.act = nn.ReLU(inplace=True)

        self._out_channels = shallowest_ch

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def forward(self, bottleneck_feat: torch.Tensor,
                skip_features: List[torch.Tensor]) -> torch.Tensor:
        skips = list(reversed(skip_features))  # deep to shallow

        # Stage 0: GCB + SPA on bottleneck
        x = self.gcb_blocks[0](bottleneck_feat)
        x = self.act(x)
        x = self.spa(x) * x

        # Remaining stages: UCB + interpolate + GCB + SPA + skip
        for i, (ucb, gcb, skip) in enumerate(zip(self.ucb_blocks, self.gcb_blocks[1:], skips)):
            x = ucb(x)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = gcb(x)
            x = self.act(x)
            x = self.spa(x) * x

        return x


# ── GCASCADE Decoder (concatenation skip) ───────────────────────────────────

@DECODER_REGISTRY.register("gcascade_cat")
class GCASCADECatDecoder(nn.Module):
    """G-CASCADE decoder with concatenation skip aggregation.

    Same architecture as GCASCADEDecoder but uses concatenation instead of
    additive skip. Extra 1x1 convolutions reduce concatenated channels.
    Automatically adapts to any encoder.

    Args: same as GCASCADEDecoder.
    """

    has_internal_skip = True

    def __init__(self, encoder_channels: List[int], bottleneck_channels: int,
                 skip_connection=None, img_size: int = 224,
                 k: int = 11, padding: int = 5, conv: str = 'mr',
                 gcb_act: str = 'gelu', activation: str = 'relu',
                 drop_path: float = 0.0, **kwargs):
        super().__init__()
        self.img_size = img_size
        n_skips = len(encoder_channels)

        skip_chs = list(reversed(encoder_channels))
        channels = [bottleneck_channels] + skip_chs

        n_spatial = (img_size // 16) ** 2

        # GCB blocks
        self.gcb_blocks = nn.ModuleList([
            Grapher(ch, kernel_size=k, dilation=1, conv=conv, act=gcb_act,
                    n=n_spatial, drop_path=drop_path, padding=padding)
            for ch in channels
        ])

        # UCB blocks
        self.ucb_blocks = nn.ModuleList([
            UCB(channels[i], channels[i + 1])
            for i in range(n_skips)
        ])

        # SPA
        self.spa = SPA()

        # Channel reduction after concat (2x channels -> 1x)
        self.cv_convs = nn.ModuleList([
            nn.Conv2d(ch * 2, ch, 1, bias=False) for ch in skip_chs
        ])

        # Multi-scale reduction
        shallowest_ch = channels[-1]
        self.h_1x1s = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch * 2, shallowest_ch, 1),
                nn.BatchNorm2d(shallowest_ch),
                nn.ReLU(inplace=True)
            )
            for ch in skip_chs[:-1]
        ])

        if activation.lower() == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif activation.lower() == 'relu6':
            self.act = nn.ReLU6(inplace=True)
        else:
            self.act = nn.ReLU(inplace=True)

        self._out_channels = shallowest_ch

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def forward(self, bottleneck_feat: torch.Tensor,
                skip_features: List[torch.Tensor]) -> torch.Tensor:
        skips = list(reversed(skip_features))

        # Stage 0: GCB + SPA on bottleneck
        x = self.gcb_blocks[0](bottleneck_feat)
        x = self.act(x)
        x = self.spa(x) * x

        # Remaining stages: UCB + concat skip + reduce + GCB + SPA
        for i, (ucb, gcb, skip, cv) in enumerate(
            zip(self.ucb_blocks, self.gcb_blocks[1:], skips, self.cv_convs)
        ):
            x = ucb(x)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat((x, skip), dim=1)
            x = cv(x)
            x = gcb(x)
            x = self.act(x)
            x = self.spa(x) * x

        return x
