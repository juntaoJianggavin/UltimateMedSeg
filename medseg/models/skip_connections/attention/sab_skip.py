"""Spatial Attention Bridge (SAB) skip connection."""
# Source: INTERNAL — framework adaptation (this repo).

import torch
import torch.nn as nn
from medseg.registry import SKIP_REGISTRY


@SKIP_REGISTRY.register("sab")
class SABSkip(nn.Module):
    """Spatial Attention Bridge: applies spatial attention to skip features before fusion."""
    def __init__(self, kernel_size=7, **kwargs):
        super().__init__()
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid(),
        )

    def get_out_channels(self, decoder_ch, skip_ch):
        return decoder_ch + skip_ch

    def forward(self, decoder_feat, skip_feat):
        avg_out = torch.mean(skip_feat, dim=1, keepdim=True)
        max_out, _ = torch.max(skip_feat, dim=1, keepdim=True)
        spatial = torch.cat([avg_out, max_out], dim=1)
        attn = self.spatial_attn(spatial)
        skip_feat = skip_feat * attn
        return torch.cat([decoder_feat, skip_feat], dim=1)
