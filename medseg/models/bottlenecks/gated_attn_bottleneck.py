"""Gated Attention bottleneck.

Inspired by gated attention mechanisms from:
    - Ilse et al., "Attention-based Deep Multiple Instance Learning", ICML 2018.
    - Vaswani et al., "Attention Is All You Need", NeurIPS 2017.

Uses a learnable gate to modulate self-attention over spatial positions,
providing a lightweight alternative to full transformer bottlenecks.
"""
# Source: INTERNAL — framework adaptation (this repo).

import torch
import torch.nn as nn
from medseg.registry import BOTTLENECK_REGISTRY


class GatedAttentionLayer(nn.Module):
    """Gated spatial self-attention."""

    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        head_dim = max(channels // num_heads, 1)
        self.qkv = nn.Conv2d(channels, head_dim * num_heads * 3, 1, bias=False)
        self.proj = nn.Conv2d(head_dim * num_heads, channels, 1, bias=False)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, -1, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # each B,nh,d,HW

        attn = (q.transpose(-2, -1) @ k).softmax(dim=-1)  # B,nh,HW,HW
        out = (v @ attn.transpose(-2, -1))                # B,nh,d,HW
        out = out.reshape(B, -1, H, W)
        out = self.proj(out)
        g = self.gate(x)
        return x + g * out


@BOTTLENECK_REGISTRY.register("gated_attn")
class GatedAttnBottleneck(nn.Module):
    """Gated attention bottleneck with residual.

    Args:
        in_channels: Number of input/output channels.
        num_heads: Number of attention heads.
    """

    def __init__(self, in_channels, num_heads=4, **kwargs):
        super().__init__()
        # Adjust num_heads if channels don't divide evenly
        while num_heads > 1 and in_channels % num_heads != 0:
            num_heads //= 2
        self.ga = GatedAttentionLayer(in_channels, num_heads)
        self.norm = nn.BatchNorm2d(in_channels)
        self.act = nn.ReLU(inplace=True)
        self._out_channels = in_channels

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, x):
        return self.act(self.norm(self.ga(x)))
