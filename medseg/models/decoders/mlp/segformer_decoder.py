"""SegFormer All-MLP Decoder.

Reference: Xie et al. "SegFormer: Simple and Efficient Design for Semantic
Segmentation with Transformers" (NeurIPS 2021).

This is the canonical SegFormer decoder head: each multi-scale feature is
projected to a common embedding dim with a 1x1 conv, bilinearly upsampled to
the shallowest (largest) feature's spatial resolution (~1/4 of input), the
four projections are concatenated channel-wise, fused with a 1x1 conv + BN +
GELU, and an embed_dim-sized "segmentation" projection is applied so that
``self.out_channels`` reports the feature width BEFORE the framework's
external seg head.

External ``skip_connection`` is IGNORED -- the SegFormer head builds its own
skip topology by ingesting all encoder features in parallel.
"""
# Source: https://github.com/NVlabs/SegFormer

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from medseg.registry import DECODER_REGISTRY


@DECODER_REGISTRY.register("segformer")
class SegFormerDecoder(nn.Module):
    """SegFormer all-MLP decoder (1x1 conv variant).

    Args:
        encoder_channels: Skip-connection channels, shallow -> deep.
        bottleneck_channels: Channels of the deepest (bottleneck) feature.
        skip_connection: Ignored (SegFormer has its own internal skip topology).
        img_size: Input spatial size (kept for API symmetry; unused internally).
        embed_dim: Common embedding dimension for all projected features.
        dropout: Dropout applied after the final fusion conv.
    """

    has_internal_skip = True

    def __init__(self, encoder_channels: List[int], bottleneck_channels: int,
                 skip_connection: nn.Module = None, img_size: int = 224,
                 embed_dim: int = 256, dropout: float = 0.1, **kwargs):
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim

        # All multi-scale features fed to the decoder: skips + bottleneck.
        # Some callers pass encoder_channels already including the bottleneck
        # stage; only append when it's clearly missing to avoid double-counting.
        all_channels = list(encoder_channels)
        if not all_channels or all_channels[-1] != bottleneck_channels:
            all_channels.append(bottleneck_channels)

        # 1x1 conv projection per scale -> common embed_dim.
        self.projections = nn.ModuleList([
            nn.Conv2d(ch, embed_dim, kernel_size=1, bias=False)
            for ch in all_channels
        ])

        # Fusion: concat all projected features -> 1x1 conv -> BN -> GELU.
        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim * len(all_channels), embed_dim,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

        # Embed-dim-sized segmentation projection (BEFORE the framework seg head).
        self.seg_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=False)

        self.dropout = nn.Dropout2d(dropout)
        self._out_channels = embed_dim

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def forward(self, bottleneck_feat: torch.Tensor,
                skip_features: List[torch.Tensor]) -> torch.Tensor:
        # All input features ordered shallow (largest) -> deep (smallest).
        all_features = list(skip_features) + [bottleneck_feat]

        # Target: shallowest feature's spatial size (~1/4 of input).
        target_size = all_features[0].shape[2:]

        projected = []
        for feat, proj in zip(all_features, self.projections):
            p = proj(feat)
            if p.shape[2:] != target_size:
                p = F.interpolate(p, size=target_size,
                                  mode='bilinear', align_corners=False)
            projected.append(p)

        x = self.fuse(torch.cat(projected, dim=1))
        x = self.seg_proj(x)
        x = self.dropout(x)
        return x
