"""Transformer bottleneck."""
# Source: INTERNAL — framework adaptation (this repo).

import torch
import torch.nn as nn
from medseg.registry import BOTTLENECK_REGISTRY


class TransformerLayer(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        x2 = self.norm1(x)
        x = x + self.attn(x2, x2, x2)[0]
        x = x + self.mlp(self.norm2(x))
        return x


@BOTTLENECK_REGISTRY.register("transformer")
class TransformerBottleneck(nn.Module):
    """Transformer bottleneck: applies transformer layers to bottleneck features."""
    def __init__(self, in_channels, num_layers=2, num_heads=8, mlp_ratio=4.0, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(in_channels, num_heads, mlp_ratio) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(in_channels)
        self._out_channels = in_channels

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1)  # B, N, C
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x.permute(0, 2, 1).reshape(B, C, H, W)
