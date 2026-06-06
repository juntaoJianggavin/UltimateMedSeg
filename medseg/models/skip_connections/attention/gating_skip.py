"""Gating skip connection."""
# Source: INTERNAL — framework adaptation (this repo).

import torch
import torch.nn as nn
from medseg.registry import SKIP_REGISTRY


@SKIP_REGISTRY.register("gating")
class GatingSkip(nn.Module):
    """Gating skip: learns a gate from decoder features to modulate skip features."""
    def __init__(self, **kwargs):
        super().__init__()

    def get_out_channels(self, decoder_ch, skip_ch):
        return decoder_ch + skip_ch

    def forward(self, decoder_feat, skip_feat):
        # Simple learned gating: sigmoid(decoder_feat averaged) * skip
        gate = torch.sigmoid(decoder_feat.mean(dim=1, keepdim=True))
        skip_feat = skip_feat * gate
        return torch.cat([decoder_feat, skip_feat], dim=1)
