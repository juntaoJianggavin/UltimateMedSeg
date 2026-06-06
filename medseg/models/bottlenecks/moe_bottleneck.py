"""Mixture-of-Experts (MoE) bottleneck.

Inspired by:
    - Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated
      Mixture-of-Experts Layer", ICLR 2017.
    - Fedus et al., "Switch Transformers: Scaling to Trillion Parameter Models
      with Simple and Efficient Sparsity", JMLR 2022.

Each expert is a small 2-layer conv block; a lightweight router selects the
top-k experts per spatial position and combines them.
"""
# Source: INTERNAL — framework adaptation (this repo).

import torch
import torch.nn as nn
import torch.nn.functional as F
from medseg.registry import BOTTLENECK_REGISTRY


class _Expert(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return self.block(x)


class MoELayer(nn.Module):
    """Spatially-gated Mixture-of-Experts."""

    def __init__(self, channels, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([_Expert(channels) for _ in range(num_experts)])
        self.router = nn.Conv2d(channels, num_experts, 1, bias=True)

    def forward(self, x):
        # Router logits: B, E, H, W
        logits = self.router(x)
        # Top-k gating
        topk_logits, topk_idx = logits.topk(self.top_k, dim=1)
        gates = F.softmax(topk_logits, dim=1)  # B, k, H, W

        # Evaluate all experts (efficient for small E)
        expert_outs = torch.stack([e(x) for e in self.experts], dim=1)  # B,E,C,H,W

        # Gather selected experts
        B, E, C, H, W = expert_outs.shape
        idx_expanded = topk_idx.unsqueeze(2).expand(-1, -1, C, -1, -1)  # B,k,C,H,W
        selected = expert_outs.gather(1, idx_expanded)                   # B,k,C,H,W

        # Weighted sum
        out = (selected * gates.unsqueeze(2)).sum(dim=1)  # B,C,H,W
        return x + out


@BOTTLENECK_REGISTRY.register("moe")
class MoEBottleneck(nn.Module):
    """Mixture-of-Experts bottleneck with residual.

    Args:
        in_channels: Number of input/output channels.
        num_experts: Number of expert sub-networks.
        top_k: Number of experts activated per position.
    """

    def __init__(self, in_channels, num_experts=4, top_k=2, **kwargs):
        super().__init__()
        top_k = min(top_k, num_experts)
        self.moe = MoELayer(in_channels, num_experts, top_k)
        self.act = nn.ReLU(inplace=True)
        self._out_channels = in_channels

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, x):
        return self.act(self.moe(x))
