"""Basic skip connections: Add and Concat."""
# Source: INTERNAL — framework adaptation (this repo).

import torch
import torch.nn as nn
from medseg.registry import SKIP_REGISTRY


@SKIP_REGISTRY.register("add")
class AddSkip(nn.Module):
    """Element-wise addition skip connection."""
    def __init__(self, **kwargs):
        super().__init__()

    def get_out_channels(self, decoder_ch, skip_ch):
        return max(decoder_ch, skip_ch)

    def forward(self, decoder_feat, skip_feat):
        if decoder_feat.shape[1] != skip_feat.shape[1]:
            # Adapt channels via 1x1 conv (project skip to decoder channels)
            adapt = nn.Conv2d(skip_feat.shape[1], decoder_feat.shape[1], 1).to(skip_feat.device)
            skip_feat = adapt(skip_feat)
        return decoder_feat + skip_feat


@SKIP_REGISTRY.register("concat")
class ConcatSkip(nn.Module):
    """Channel concatenation skip connection."""
    def __init__(self, **kwargs):
        super().__init__()

    def get_out_channels(self, decoder_ch, skip_ch):
        return decoder_ch + skip_ch

    def forward(self, decoder_feat, skip_feat):
        return torch.cat([decoder_feat, skip_feat], dim=1)
