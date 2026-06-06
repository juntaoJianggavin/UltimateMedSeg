"""CCM (Cross Channel Module) skip connection."""
# Source: INTERNAL — framework adaptation (this repo).

import torch
import torch.nn as nn
from medseg.registry import SKIP_REGISTRY


@SKIP_REGISTRY.register("ccm")
class CCMSkip(nn.Module):
    """Cross Channel Module: cross-channel interaction between decoder and skip features."""
    def __init__(self, **kwargs):
        super().__init__()

    def get_out_channels(self, decoder_ch, skip_ch):
        return decoder_ch + skip_ch

    def forward(self, decoder_feat, skip_feat):
        # Channel-wise cross attention via global average pooling
        B, Cd, H, W = decoder_feat.shape
        Cs = skip_feat.shape[1]
        # Global descriptors
        d_pool = decoder_feat.mean(dim=[2, 3])  # B, Cd
        s_pool = skip_feat.mean(dim=[2, 3])  # B, Cs
        # Cross attention weights
        cross = torch.bmm(d_pool.unsqueeze(2), s_pool.unsqueeze(1))  # B, Cd, Cs
        d_weight = cross.mean(dim=2).sigmoid().unsqueeze(-1).unsqueeze(-1)  # B, Cd, 1, 1
        s_weight = cross.mean(dim=1).sigmoid().unsqueeze(-1).unsqueeze(-1)  # B, Cs, 1, 1
        return torch.cat([decoder_feat * d_weight, skip_feat * s_weight], dim=1)
