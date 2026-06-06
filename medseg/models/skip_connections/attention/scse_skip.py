"""scSE (Concurrent Spatial and Channel SE) skip connection."""
# Source: INTERNAL — framework adaptation (this repo).

import torch
import torch.nn as nn
from medseg.registry import SKIP_REGISTRY


class ChannelSE(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, max(channels // reduction, 1)),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 1), channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(x).unsqueeze(-1).unsqueeze(-1)
        return x * w


class SpatialSE(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.conv(x))


@SKIP_REGISTRY.register("scse")
class ScSESkip(nn.Module):
    """scSE: concurrent spatial and channel squeeze-and-excitation on skip features."""
    def __init__(self, reduction=16, **kwargs):
        super().__init__()
        self.reduction = reduction
        self._modules_cache = {}

    def get_out_channels(self, decoder_ch, skip_ch):
        return decoder_ch + skip_ch

    def _get_scse(self, channels, device):
        if channels not in self._modules_cache:
            cse = ChannelSE(channels, self.reduction).to(device)
            sse = SpatialSE(channels).to(device)
            self._modules_cache[channels] = (cse, sse)
        return self._modules_cache[channels]

    def forward(self, decoder_feat, skip_feat):
        cse, sse = self._get_scse(skip_feat.shape[1], skip_feat.device)
        skip_feat = cse(skip_feat) + sse(skip_feat)
        return torch.cat([decoder_feat, skip_feat], dim=1)
