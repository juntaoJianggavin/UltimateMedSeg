"""MLP Decoder - faithfully ported from SegFormer.
Reference: https://github.com/NVlabs/SegFormer/blob/master/mmseg/models/decode_heads/segformer_head.py

SegFormer MLP decoder takes ALL multi-scale features, projects each to embed_dim,
upsamples to highest resolution, concatenates, and fuses. No external skip connections.
"""
# Source: INTERNAL — framework adaptation (this repo).

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from medseg.registry import DECODER_REGISTRY


class MLP(nn.Module):
    """Linear Embedding - matches original SegFormer MLP class."""
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B,C,H,W -> B,N,C
        x = self.proj(x)
        return x


@DECODER_REGISTRY.register("mlp")
class MLPDecoder(nn.Module):
    """SegFormer-style all-MLP decoder - faithful port.

    Takes all encoder features (skip + bottleneck), projects each to embed_dim
    via linear layer, upsamples to highest resolution, concatenates, and fuses
    with a 1x1 convolution.

    External skip_connection parameter is IGNORED.
    """
    has_internal_skip = True

    def __init__(self, encoder_channels: List[int], bottleneck_channels: int,
                 skip_connection=None, embed_dim: int = 256, dropout: float = 0.1, **kwargs):
        super().__init__()
        all_channels = list(encoder_channels) + [bottleneck_channels]

        # One MLP (linear embedding) per feature scale - matches original linear_c1..c4
        self.linear_layers = nn.ModuleList([
            MLP(input_dim=ch, embed_dim=embed_dim)
            for ch in all_channels
        ])

        # linear_fuse: 1x1 conv to fuse concatenated features
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embed_dim * len(all_channels), embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

        self.dropout = nn.Dropout2d(dropout)
        self._out_channels = embed_dim

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, bottleneck_feat: torch.Tensor, skip_features: List[torch.Tensor]) -> torch.Tensor:
        all_features = list(skip_features) + [bottleneck_feat]
        # Target: highest resolution (first skip feature)
        n, _, h, w = all_features[0].shape

        projected = []
        for feat, linear in zip(all_features, self.linear_layers):
            # MLP: flatten -> linear -> reshape back
            _c = linear(feat)  # B, N, embed_dim
            _c = _c.permute(0, 2, 1).reshape(n, -1, feat.shape[2], feat.shape[3])
            # Upsample to highest resolution
            if _c.shape[2:] != (h, w):
                _c = F.interpolate(_c, size=(h, w), mode='bilinear', align_corners=False)
            projected.append(_c)

        # Concat all projected features and fuse
        _c = self.linear_fuse(torch.cat(projected, dim=1))
        x = self.dropout(_c)
        return x
