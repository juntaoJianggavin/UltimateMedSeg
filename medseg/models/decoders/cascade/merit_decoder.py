"""MERIT Decoder – Multi-scale Hierarchical Vision Transformer with
Cascaded Attention Decoding.

Faithfully ported from: https://github.com/SLDGroup/MERIT
Paper: Multi-scale Hierarchical Vision Transformer with Cascaded Attention
       Decoding for Medical Image Segmentation (MIDL 2024, Rahman & Marclescu)

The decoder uses standard double-conv blocks with Attention Gates (AG) and
CBAM-style Channel/Spatial Attention Modules (CAM). Two variants:
  - ``merit_add``: additive skip aggregation
  - ``merit_cat``: concatenation skip aggregation

Pipeline per stage:
    bottleneck -> 1x1 -> CAM4+SA -> ConvBlock4
             -> Up3 -> AG3(skip) -> cat/add -> CAM3+SA -> ConvBlock3
             -> Up2 -> AG2(skip) -> cat/add -> CAM2+SA -> ConvBlock2
             -> Up1 -> AG1(skip) -> cat/add -> CAM1+SA -> ConvBlock1

External skip_connection parameter is IGNORED (``has_internal_skip = True``).
"""
# Source: https://github.com/SLDGroup/MERIT

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from medseg.registry import DECODER_REGISTRY


# ── Building Blocks ──────────────────────────────────────────────────────────

class _ConvBlock(nn.Module):
    """Standard double-conv block (conv-BN-ReLU x2)."""

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
    """Upsample 2x + conv + BN + ReLU."""

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


class _AttentionGate(nn.Module):
    """MERIT attention gate (1x1 conv based, non-grouped).

    AG(g, x) = sigmoid(BN(Conv1x1(relu(BN(Conv1x1(g)) + BN(Conv1x1(x)))))) * x
    """

    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, 1, 0, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, 1, 0, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, 1, 0, bias=True),
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
    """CBAM-style channel attention: avg+max pool -> shared MLP -> sigmoid."""

    def __init__(self, in_planes, ratio=16):
        super().__init__()
        hidden = max(in_planes // ratio, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, hidden, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(hidden, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class _SpatialAttention(nn.Module):
    """CBAM-style spatial attention: avg+max along channel -> 7x7 conv -> sigmoid."""

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


# ── MERIT Decoder (additive skip) ────────────────────────────────────────────

@DECODER_REGISTRY.register("merit_add")
class MERITAddDecoder(nn.Module):
    """MERIT decoder with additive skip aggregation.

    Args:
        encoder_channels: List of encoder stage output channels (shallow to deep).
        bottleneck_channels: Channels of the bottleneck feature.
        skip_connection: IGNORED (internal skip used).
    """

    has_internal_skip = True

    def __init__(self, encoder_channels: List[int],
                 bottleneck_channels: int,
                 skip_connection=None, **kwargs):
        super().__init__()

        skip_chs = list(reversed(encoder_channels))  # deep to shallow
        channels = [bottleneck_channels] + skip_chs  # e.g. [512, 320, 128, 64]

        # 1x1 conv on bottleneck
        self.conv_1x1 = nn.Conv2d(channels[0], channels[0], 1, 1, 0)

        # Stage 4 (bottleneck)
        self.conv_block4 = _ConvBlock(channels[0], channels[0])

        # Progressive decoding stages
        self.ups = nn.ModuleList()
        self.ags = nn.ModuleList()
        self.conv_blocks = nn.ModuleList()

        for i in range(len(skip_chs)):
            in_ch = channels[i]
            out_ch = channels[i + 1]
            self.ups.append(_UpConv(in_ch, out_ch))
            # F_int for AG: channels[i+2] if available, else out_ch//2
            if i + 2 < len(channels):
                f_int = channels[i + 2]
            else:
                f_int = out_ch // 2
            self.ags.append(_AttentionGate(F_g=out_ch, F_l=out_ch,
                                           F_int=f_int))
            self.conv_blocks.append(_ConvBlock(out_ch, out_ch))

        # CAM and SA (shared)
        self.cas = nn.ModuleList()
        self.cas.append(_ChannelAttention(channels[0]))
        for i in range(len(skip_chs)):
            self.cas.append(_ChannelAttention(channels[i + 1]))
        self.sa = _SpatialAttention()

        self._out_channels = skip_chs[-1] if skip_chs else bottleneck_channels

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def forward(self, bottleneck_feat: torch.Tensor,
                skip_features: List[torch.Tensor]) -> torch.Tensor:
        skips = list(reversed(skip_features))

        # Stage 4
        d4 = self.conv_1x1(bottleneck_feat)
        d4 = self.cas[0](d4) * d4
        d4 = self.sa(d4) * d4
        d4 = self.conv_block4(d4)

        # Progressive decoding
        d = d4
        for i in range(len(self.ups)):
            # UpConv
            d = self.ups[i](d)
            # Match spatial size
            skip_i = skips[i]
            if d.shape[2:] != skip_i.shape[2:]:
                d = F.interpolate(d, size=skip_i.shape[2:],
                                  mode='bilinear', align_corners=False)
            # AG (attention gate on skip)
            x_skip = self.ags[i](g=d, x=skip_i)
            # Additive aggregation
            d = d + x_skip
            # CAM + SA
            d = self.cas[i + 1](d) * d
            d = self.sa(d) * d
            # ConvBlock
            d = self.conv_blocks[i](d)

        return d


# ── MERIT Decoder (concatenation skip) ───────────────────────────────────────

@DECODER_REGISTRY.register("merit_cat")
class MERITCatDecoder(nn.Module):
    """MERIT decoder with concatenation skip aggregation.

    After concatenation, ChannelAttention operates on 2x channels.

    Args: same as MERITAddDecoder.
    """

    has_internal_skip = True

    def __init__(self, encoder_channels: List[int],
                 bottleneck_channels: int,
                 skip_connection=None, **kwargs):
        super().__init__()

        skip_chs = list(reversed(encoder_channels))
        channels = [bottleneck_channels] + skip_chs

        self.conv_1x1 = nn.Conv2d(channels[0], channels[0], 1, 1, 0)
        self.conv_block4 = _ConvBlock(channels[0], channels[0])

        self.ups = nn.ModuleList()
        self.ags = nn.ModuleList()
        self.conv_blocks = nn.ModuleList()

        for i in range(len(skip_chs)):
            in_ch = channels[i]
            out_ch = channels[i + 1]
            self.ups.append(_UpConv(in_ch, out_ch))
            if i + 2 < len(channels):
                f_int = channels[i + 2]
            else:
                f_int = out_ch // 2
            self.ags.append(_AttentionGate(F_g=out_ch, F_l=out_ch,
                                           F_int=f_int))
            # After cat: 2*out_ch channels
            self.conv_blocks.append(_ConvBlock(2 * out_ch, out_ch))

        # CAM: stage 4 uses channels[0], stages 3/2/1 use 2*channels[i+1]
        self.cas = nn.ModuleList()
        self.cas.append(_ChannelAttention(channels[0]))
        for i in range(len(skip_chs)):
            self.cas.append(_ChannelAttention(2 * channels[i + 1]))
        self.sa = _SpatialAttention()

        self._out_channels = skip_chs[-1] if skip_chs else bottleneck_channels

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def forward(self, bottleneck_feat: torch.Tensor,
                skip_features: List[torch.Tensor]) -> torch.Tensor:
        skips = list(reversed(skip_features))

        # Stage 4
        d4 = self.conv_1x1(bottleneck_feat)
        d4 = self.cas[0](d4) * d4
        d4 = self.sa(d4) * d4
        d4 = self.conv_block4(d4)

        # Progressive decoding
        d = d4
        for i in range(len(self.ups)):
            d = self.ups[i](d)
            skip_i = skips[i]
            if d.shape[2:] != skip_i.shape[2:]:
                d = F.interpolate(d, size=skip_i.shape[2:],
                                  mode='bilinear', align_corners=False)
            x_skip = self.ags[i](g=d, x=skip_i)
            # Concatenation
            d = torch.cat([x_skip, d], dim=1)
            # CAM + SA
            d = self.cas[i + 1](d) * d
            d = self.sa(d) * d
            # ConvBlock
            d = self.conv_blocks[i](d)

        return d
