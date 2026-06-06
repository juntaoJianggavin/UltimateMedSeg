"""Dense skip connection."""
# Source: INTERNAL — framework adaptation (this repo).

import torch
import torch.nn as nn
from medseg.registry import SKIP_REGISTRY


@SKIP_REGISTRY.register("dense")
class DenseSkip(nn.Module):
    """Dense skip connection: concat with a learned projection to reduce channels."""
    def __init__(self, **kwargs):
        super().__init__()

    def get_out_channels(self, decoder_ch, skip_ch):
        return decoder_ch + skip_ch

    def forward(self, decoder_feat, skip_feat):
        return torch.cat([decoder_feat, skip_feat], dim=1)
