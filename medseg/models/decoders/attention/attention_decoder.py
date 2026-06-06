"""Attention UNet Decoder with attention gates.
Reference: Oktay et al. "Attention U-Net: Learning Where to Look for the Pancreas"

Has its own internal skip mechanism (attention gates).
External skip_connection parameter is IGNORED.
"""
# Source: INTERNAL — framework adaptation (this repo).

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from medseg.registry import DECODER_REGISTRY


class AttentionGate(nn.Module):
    """Standard attention gate from Attention U-Net (Oktay et al.)."""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1, bias=True), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1, bias=True), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1, bias=True), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=False)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


@DECODER_REGISTRY.register("attention")
class AttentionDecoder(nn.Module):
    """Attention UNet decoder with built-in attention gates on skip connections.

    External skip_connection is IGNORED - attention gates ARE the skip mechanism.
    """
    has_internal_skip = True

    def __init__(self, encoder_channels: List[int], bottleneck_channels: int,
                 skip_connection=None, **kwargs):
        super().__init__()
        skip_channels = list(reversed(encoder_channels))

        self.attention_gates = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        in_ch = bottleneck_channels
        for skip_ch in skip_channels:
            self.attention_gates.append(
                AttentionGate(F_g=in_ch, F_l=skip_ch, F_int=max(skip_ch // 2, 1))
            )
            merged_ch = in_ch + skip_ch  # always concat after attention gating
            self.up_convs.append(nn.Sequential(
                nn.Conv2d(merged_ch, skip_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(skip_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(skip_ch, skip_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(skip_ch),
                nn.ReLU(inplace=True),
            ))
            in_ch = skip_ch
        self._out_channels = skip_channels[-1] if skip_channels else bottleneck_channels

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, bottleneck_feat: torch.Tensor, skip_features: List[torch.Tensor]) -> torch.Tensor:
        skips = list(reversed(skip_features))
        x = bottleneck_feat
        for i, (ag, conv) in enumerate(zip(self.attention_gates, self.up_convs)):
            skip = skips[i]
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            skip = ag(x, skip)  # attention gate modulates skip features
            x = torch.cat([x, skip], dim=1)  # always concat (built-in)
            x = conv(x)
        return x
