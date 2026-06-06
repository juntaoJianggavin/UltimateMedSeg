"""Multi-scale skip connection."""
# Source: INTERNAL — framework adaptation (this repo).

import torch
import torch.nn as nn
import torch.nn.functional as F
from medseg.registry import SKIP_REGISTRY


@SKIP_REGISTRY.register("multiscale")
class MultiscaleSkip(nn.Module):
    """Multi-scale skip: applies multi-scale convolutions to skip features before concat."""
    def __init__(self, **kwargs):
        super().__init__()

    def get_out_channels(self, decoder_ch, skip_ch):
        return decoder_ch + skip_ch

    def forward(self, decoder_feat, skip_feat):
        # Apply multi-scale pooling to skip features
        B, C, H, W = skip_feat.shape
        pool2 = F.adaptive_avg_pool2d(skip_feat, (H // 2, W // 2))
        pool2 = F.interpolate(pool2, size=(H, W), mode='bilinear', align_corners=False)
        pool4 = F.adaptive_avg_pool2d(skip_feat, (H // 4, W // 4))
        pool4 = F.interpolate(pool4, size=(H, W), mode='bilinear', align_corners=False)
        skip_feat = skip_feat + pool2 + pool4
        return torch.cat([decoder_feat, skip_feat], dim=1)
