"""BiFusion (TransFuse-style) parallel-branch skip connection."""
# Source: INTERNAL — framework adaptation (this repo).

import torch
import torch.nn as nn
import torch.nn.functional as F
from medseg.registry import SKIP_REGISTRY


@SKIP_REGISTRY.register("bifusion")
class BiFusionSkip(nn.Module):
    """Parallel-branch BiFusion skip.

    Fuses decoder and skip features by combining:
      - channel-attention weights from concat(decoder, skip) via GAP + 2 FC
      - spatial-attention weights from concat via 1x1 conv + sigmoid
    out = w_c * (decoder + skip) + w_s * (decoder * skip), then 1x1 conv
    to a unified channel count (max of decoder_ch, skip_ch).
    """

    def __init__(self, reduction=16, **kwargs):
        super().__init__()
        self.reduction = reduction
        # Lazily-built submodules keyed by (decoder_ch, skip_ch)
        self._channel_attns = nn.ModuleDict()
        self._spatial_attns = nn.ModuleDict()
        self._dec_projs = nn.ModuleDict()
        self._skip_projs = nn.ModuleDict()
        self._out_convs = nn.ModuleDict()

    def get_out_channels(self, decoder_ch, skip_ch):
        return max(decoder_ch, skip_ch)

    def _key(self, dc, sc):
        return f"{dc}_{sc}"

    def _build(self, decoder_ch, skip_ch, device):
        key = self._key(decoder_ch, skip_ch)
        if key in self._out_convs:
            return
        unified = max(decoder_ch, skip_ch)
        r = self.reduction

        # Project both branches to unified channels so element-wise ops align
        dec_proj = (nn.Conv2d(decoder_ch, unified, kernel_size=1)
                    if decoder_ch != unified else nn.Identity())
        skip_proj = (nn.Conv2d(skip_ch, unified, kernel_size=1)
                     if skip_ch != unified else nn.Identity())

        # Channel attention on concat(decoder, skip): GAP + 2 FC -> sigmoid
        concat_ch = unified * 2
        channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(concat_ch, max(concat_ch // r, 1)),
            nn.ReLU(inplace=True),
            nn.Linear(max(concat_ch // r, 1), unified),
            nn.Sigmoid(),
        )

        # Spatial attention on concat: 1x1 conv -> sigmoid (per-pixel scalar)
        spatial_attn = nn.Sequential(
            nn.Conv2d(concat_ch, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # Final 1x1 conv to unified channel count
        out_conv = nn.Conv2d(unified, unified, kernel_size=1)

        self._dec_projs[key] = dec_proj.to(device)
        self._skip_projs[key] = skip_proj.to(device)
        self._channel_attns[key] = channel_attn.to(device)
        self._spatial_attns[key] = spatial_attn.to(device)
        self._out_convs[key] = out_conv.to(device)

    def forward(self, decoder_feat, skip_feat):
        # Spatial align skip to decoder if needed
        if skip_feat.shape[-2:] != decoder_feat.shape[-2:]:
            skip_feat = F.interpolate(
                skip_feat, size=decoder_feat.shape[-2:],
                mode="bilinear", align_corners=False,
            )

        decoder_ch = decoder_feat.shape[1]
        skip_ch = skip_feat.shape[1]
        self._build(decoder_ch, skip_ch, decoder_feat.device)
        key = self._key(decoder_ch, skip_ch)

        d = self._dec_projs[key](decoder_feat)
        s = self._skip_projs[key](skip_feat)

        concat = torch.cat([d, s], dim=1)

        # Channel attention weights -> (B, C, 1, 1)
        w_c = self._channel_attns[key](concat).unsqueeze(-1).unsqueeze(-1)
        # Spatial attention weights -> (B, 1, H, W)
        w_s = self._spatial_attns[key](concat)

        fused = w_c * (d + s) + w_s * (d * s)
        out = self._out_convs[key](fused)
        return out
