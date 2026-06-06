"""TA-MoSC (Task-Adaptive Mixture of Skip Connections) — adapted from
UTANet (AAAI 2025, Luo et al.).

Adapted from: https://github.com/AshleyLuo001/UTANet
Paper: Rethinking U-Net: Task-Adaptive Mixture of Skip Connections for
       Enhanced Medical Image Segmentation (AAAI 2025)

The original TA-MoSC module fuses ALL encoder features into a single
representation, routes through a Mixture of Experts (MoE), and distributes
task-adaptive skip features to each decoder level:

    fused = fuse(align_and_cat([e1, e2, e3, e4]))  -> (B, C, H, W)
    o1, o2, o3, o4 = MoE(fused)                    -> 4 expert outputs
    skip_i = docker_i(oi)                           -> per-level skip

Adapted to the framework's per-pair skip interface:
    1. Project ``decoder_feat`` and ``skip_feat`` to a unified channel dim
    2. Concatenate -> MoE input
    3. MoE routes through ``num_experts`` MLP experts with top-k selection
    4. Weighted sum of top-k expert outputs -> refined skip feature

MoE design (cleaned from original, which had debug breakpoints):
    - **Expert**: 1×1 conv -> BN -> ReLU -> 1×1 conv (residual MLP)
    - **Gate**: GAP -> Linear -> Softmax -> top-k selection
    - **Load balancing**: CV² loss encouraging uniform expert usage
    - The auxiliary loss is stored on ``self.aux_loss`` and can be
      accessed externally (e.g. added to the training loss).

Output channel count: ``max(decoder_ch, skip_ch)`` (unified dimension).
"""
# Source: https://github.com/AshleyLuo001/UTANet

import torch
import torch.nn as nn
import torch.nn.functional as F
from medseg.registry import SKIP_REGISTRY


class _Expert(nn.Module):
    """MLP expert: 1×1 conv -> hidden -> BN -> ReLU -> 1×1 conv."""

    def __init__(self, channels: int, hidden_rate: int = 2):
        super().__init__()
        hidden = channels * hidden_rate
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.Conv2d(hidden, hidden, 1, bias=True),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
        )

    def forward(self, x):
        return self.net(x)


class _MoEGate(nn.Module):
    """MoE gating: GAP -> Linear -> Softmax -> top-k selection.

    Returns (top_weights, top_indices, load_balance_loss).
    """

    def __init__(self, channels: int, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gate_weight = nn.Parameter(torch.zeros(channels, num_experts))
        nn.init.xavier_uniform_(self.gate_weight)

    @staticmethod
    def _cv_squared(x):
        """Squared coefficient of variation for load balancing."""
        eps = 1e-10
        if x.numel() <= 1:
            return torch.tensor(0.0, device=x.device)
        x_f = x.float()
        return x_f.var() / (x_f.mean() ** 2 + eps)

    def forward(self, x):
        B = x.shape[0]
        pooled = self.gap(x).view(B, -1)  # (B, C)
        logits = pooled @ self.gate_weight  # (B, num_experts)
        probs = F.softmax(logits, dim=1)

        # Load balancing loss
        expert_usage = probs.sum(dim=0)
        lb_loss = self._cv_squared(expert_usage)

        # Top-k selection
        top_w, top_idx = torch.topk(probs, self.top_k, dim=1)
        top_w = F.softmax(top_w, dim=1)  # re-normalize

        return top_w, top_idx, lb_loss


@SKIP_REGISTRY.register("ta_mosc")
class TAMoSCSkip(nn.Module):
    """TA-MoSC skip connection with Mixture of Experts routing.

    Args:
        num_experts: Number of expert networks.
        top_k: Number of top experts to select per forward.
        hidden_rate: Hidden channel multiplier for each expert MLP.
    """

    def __init__(self, num_experts: int = 4, top_k: int = 2,
                 hidden_rate: int = 2, **kwargs):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.hidden_rate = hidden_rate
        # Lazily-built submodules keyed by (decoder_ch, skip_ch)
        self._cache: dict = {}
        # Auxiliary load balancing loss (accessible externally)
        self.aux_loss = torch.tensor(0.0)

    def get_out_channels(self, decoder_ch: int, skip_ch: int) -> int:
        return max(decoder_ch, skip_ch)

    def _build(self, decoder_ch: int, skip_ch: int, device):
        """Lazily build layers for a (decoder_ch, skip_ch) pair."""
        key = (decoder_ch, skip_ch, str(device))
        if key in self._cache:
            return self._cache[key]

        unified = max(decoder_ch, skip_ch)

        # Project both to unified channels
        dec_proj = (nn.Conv2d(decoder_ch, unified, 1, bias=False)
                    if decoder_ch != unified else nn.Identity()).to(device)
        skip_proj = (nn.Conv2d(skip_ch, unified, 1, bias=False)
                     if skip_ch != unified else nn.Identity()).to(device)

        # Concat projection: 2*unified -> unified
        concat_proj = nn.Sequential(
            nn.Conv2d(unified * 2, unified, 1, bias=False),
            nn.BatchNorm2d(unified),
            nn.ReLU(inplace=True),
        ).to(device)

        # Expert networks
        experts = nn.ModuleList([
            _Expert(unified, hidden_rate=self.hidden_rate)
            for _ in range(self.num_experts)
        ]).to(device)

        # Gating mechanism
        gate = _MoEGate(unified, self.num_experts, self.top_k).to(device)

        # Output projection
        out_conv = nn.Sequential(
            nn.Conv2d(unified, unified, 3, 1, 1, bias=False),
            nn.BatchNorm2d(unified),
            nn.ReLU(inplace=True),
        ).to(device)

        mod = nn.ModuleDict({
            "dec_proj": dec_proj,
            "skip_proj": skip_proj,
            "concat_proj": concat_proj,
            "experts": experts,
            "gate": gate,
            "out_conv": out_conv,
        })
        safe_name = (f"_tamosc_{decoder_ch}_{skip_ch}_"
                     f"{str(device).replace(':', '_')}")
        setattr(self, safe_name, mod)
        self._cache[key] = mod
        return mod

    def forward(self, decoder_feat: torch.Tensor,
                skip_feat: torch.Tensor) -> torch.Tensor:
        # Spatial align skip to decoder if needed
        if skip_feat.shape[2:] != decoder_feat.shape[2:]:
            skip_feat = F.interpolate(
                skip_feat, size=decoder_feat.shape[2:],
                mode='bilinear', align_corners=False
            )

        dec_ch = decoder_feat.shape[1]
        skip_ch = skip_feat.shape[1]
        mod = self._build(dec_ch, skip_ch, decoder_feat.device)

        # Project both to unified channels
        d = mod["dec_proj"](decoder_feat)
        s = mod["skip_proj"](skip_feat)

        # Concatenate and project to unified dim
        fused = mod["concat_proj"](torch.cat([d, s], dim=1))

        # MoE routing
        top_w, top_idx, lb_loss = mod["gate"](fused)
        self.aux_loss = lb_loss

        B, C, H, W = fused.shape
        experts = mod["experts"]

        # Process top-k experts
        result = torch.zeros_like(fused)
        for k in range(self.top_k):
            w_k = top_w[:, k].view(B, 1, 1, 1)  # (B, 1, 1, 1)
            idx_k = top_idx[:, k]  # (B,)

            # Group batch items by selected expert for efficiency
            for expert_i in range(self.num_experts):
                mask = (idx_k == expert_i)
                if mask.any():
                    x_expert = fused[mask]
                    y_expert = experts[expert_i](x_expert)
                    result[mask] = result[mask] + w_k[mask] * y_expert

        # Output refinement
        return mod["out_conv"](result)
