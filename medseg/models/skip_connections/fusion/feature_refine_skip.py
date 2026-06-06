"""Feature Refinement skip connection with CBAM."""
# Source: INTERNAL — framework adaptation (this repo).

import torch
import torch.nn as nn
from medseg.registry import SKIP_REGISTRY


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        mid = max(channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
        )
        # Spatial attention
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Channel attention
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        ca = torch.sigmoid(avg_out + max_out)
        x = x * ca
        # Spatial attention
        avg_s = torch.mean(x, dim=1, keepdim=True)
        max_s, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.spatial(torch.cat([avg_s, max_s], dim=1))
        return x * sa


@SKIP_REGISTRY.register("feature_refine")
class FeatureRefineSkip(nn.Module):
    """Feature refinement skip with CBAM attention on skip features."""
    def __init__(self, reduction=16, **kwargs):
        super().__init__()
        self.reduction = reduction
        self._cbam_cache = {}

    def get_out_channels(self, decoder_ch, skip_ch):
        return decoder_ch + skip_ch

    def _get_cbam(self, channels, device):
        if channels not in self._cbam_cache:
            self._cbam_cache[channels] = CBAM(channels, self.reduction).to(device)
        return self._cbam_cache[channels]

    def forward(self, decoder_feat, skip_feat):
        cbam = self._get_cbam(skip_feat.shape[1], skip_feat.device)
        skip_feat = cbam(skip_feat)
        return torch.cat([decoder_feat, skip_feat], dim=1)
