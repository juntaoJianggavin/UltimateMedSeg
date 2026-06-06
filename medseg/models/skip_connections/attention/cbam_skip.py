"""CBAM (Channel + Spatial Attention) skip connection."""
# Source: INTERNAL — framework adaptation (this repo).

import torch
import torch.nn as nn
import torch.nn.functional as F
from medseg.registry import SKIP_REGISTRY


@SKIP_REGISTRY.register("cbam")
class CBAMSkip(nn.Module):
    """CBAM Channel+Spatial Attention skip.

    Applies CBAM to the skip feature:
      1) Channel attention: shared MLP applied to GAP and GMP pooled features,
         summed and passed through sigmoid -> (B, C, 1, 1) channel weights.
      2) Spatial attention: channel-pool (avg + max along channel dim, concat),
         followed by a 7x7 conv + sigmoid -> (B, 1, H, W) spatial weights.
    The attended skip is then concatenated with the decoder feature.
    """

    def __init__(self, reduction=16, spatial_kernel=7, **kwargs):
        super().__init__()
        self.reduction = reduction
        self.spatial_kernel = spatial_kernel
        # Lazily-built submodules keyed by channel count
        self._channel_mlps = nn.ModuleDict()
        self._spatial_convs = nn.ModuleDict()

    def get_out_channels(self, decoder_ch, skip_ch):
        return decoder_ch + skip_ch

    def _build(self, channels, device):
        key = str(channels)
        if key not in self._channel_mlps:
            r = self.reduction
            hidden = max(channels // r, 1)
            mlp = nn.Sequential(
                nn.Linear(channels, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, channels),
            ).to(device)
            self._channel_mlps[key] = mlp

            pad = self.spatial_kernel // 2
            spatial_conv = nn.Conv2d(
                2, 1, kernel_size=self.spatial_kernel,
                padding=pad, bias=False,
            ).to(device)
            self._spatial_convs[key] = spatial_conv

    def forward(self, decoder_feat, skip_feat):
        # Spatial align skip to decoder if needed
        if skip_feat.shape[-2:] != decoder_feat.shape[-2:]:
            skip_feat = F.interpolate(
                skip_feat, size=decoder_feat.shape[-2:],
                mode="bilinear", align_corners=False,
            )

        B, C, H, W = skip_feat.shape
        self._build(C, skip_feat.device)
        key = str(C)
        mlp = self._channel_mlps[key]
        spatial_conv = self._spatial_convs[key]

        # --- Channel attention ---
        avg_pool = F.adaptive_avg_pool2d(skip_feat, 1).view(B, C)
        max_pool = F.adaptive_max_pool2d(skip_feat, 1).view(B, C)
        ch_attn = torch.sigmoid(mlp(avg_pool) + mlp(max_pool))
        ch_attn = ch_attn.view(B, C, 1, 1)
        x = skip_feat * ch_attn

        # --- Spatial attention ---
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        sp_in = torch.cat([avg_map, max_map], dim=1)
        sp_attn = torch.sigmoid(spatial_conv(sp_in))
        x = x * sp_attn

        return torch.cat([decoder_feat, x], dim=1)
