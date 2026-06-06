"""Channel Attention Bridge (CAB) skip connection."""
# Source: INTERNAL — framework adaptation (this repo).

import torch
import torch.nn as nn
from medseg.registry import SKIP_REGISTRY


@SKIP_REGISTRY.register("cab")
class CABSkip(nn.Module):
    """Channel Attention Bridge: uses SE-style channel attention to weight skip features."""
    def __init__(self, reduction=16, **kwargs):
        super().__init__()
        self.reduction = reduction
        self._pools = {}

    def get_out_channels(self, decoder_ch, skip_ch):
        return decoder_ch + skip_ch

    def _get_se(self, channels, device):
        key = channels
        if key not in self._pools:
            r = self.reduction
            se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(channels, max(channels // r, 1)),
                nn.ReLU(inplace=True),
                nn.Linear(max(channels // r, 1), channels),
                nn.Sigmoid(),
            ).to(device)
            self._pools[key] = se
        return self._pools[key]

    def forward(self, decoder_feat, skip_feat):
        se = self._get_se(skip_feat.shape[1], skip_feat.device)
        w = se(skip_feat).unsqueeze(-1).unsqueeze(-1)
        skip_feat = skip_feat * w
        return torch.cat([decoder_feat, skip_feat], dim=1)
