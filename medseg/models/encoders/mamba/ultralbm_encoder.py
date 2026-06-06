"""UltraLBM-UNet encoder — extracted from medseg/networks/cnn/ultralbm_unet.py.

UltraLBM-UNet (Ultra-Lightweight Bidirectional Mamba UNet, 2024) uses a 6-stage
encoder with conv stems on the first 3 levels, then LMBP (Local Mamba Block
Parallel) and GLMBP (Global-Local Mamba Block Parallel) on deeper levels, and
finally a GLMBP bottleneck. We expose all 6 stage outputs (t1..t5, plus the
GLMBP bottleneck feature) so downstream decoders can use any subset as
skips.

Default channels follow the paper: c_list=(8, 16, 24, 32, 48, 64) at strides
(/2, /4, /8, /16, /32, /32) — last stage's `encoder6` does NOT downsample.

Multi-resolution: works at 224/256/512 since stages are 5 MaxPool2d(2)
downsamples (each level must be even); 224 → 112/56/28/14/7, 256 →
128/64/32/16/8, 512 → 256/128/64/32/16 — all OK.
"""
# Source: https://github.com/wurenkai/UltraLight-VM-UNet

from __future__ import annotations
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.registry import ENCODER_REGISTRY
# Re-use the bespoke LMBP / GLMBP blocks directly from the source network file
# (the network is now physically located in medseg/networks/mamba/).
from medseg.models.networks.mamba.ultralbm_unet import LMBP, GLMBP


@ENCODER_REGISTRY.register("ultralbm")
class UltraLBMEncoder(nn.Module):
    """UltraLBM-UNet encoder, exposed as a standalone multi-scale backbone.

    Args:
        in_channels: input image channels.
        img_size: kept for interface parity; not baked into any layer.
        pretrained: no-op (no public UltraLBM-UNet vision weights bundled).
        c_list: per-stage channels (6 entries: stages 1-5 downsample, stage 6 is bottleneck).
        channel_multiplier: scales each entry of c_list (rounded to multiple of 4, min 4).

    Outputs of forward(x) — 6 feature maps:
        [t1 (C0, H/2), t2 (C1, H/4), t3 (C2, H/8), t4 (C3, H/16),
         t5 (C4, H/32), bot (C5, H/32)]
    `self.out_channels = list(c_list)`.
    """

    def __init__(
        self,
        in_channels: int = 3,
        img_size: int = 224,
        pretrained: bool = False,
        c_list=(8, 16, 24, 32, 48, 64),
        channel_multiplier: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        if channel_multiplier != 1.0:
            c_list = [max(4, (int(c * channel_multiplier) // 4) * 4) for c in c_list]
        c_list = list(c_list)
        assert len(c_list) == 6, f"c_list must have 6 entries, got {len(c_list)}"
        self.out_channels: List[int] = c_list

        # Stages 1-3: plain Conv3x3 stems
        self.encoder1 = nn.Conv2d(in_channels, c_list[0], 3, stride=1, padding=1)
        self.encoder2 = nn.Conv2d(c_list[0], c_list[1], 3, stride=1, padding=1)
        self.encoder3 = nn.Conv2d(c_list[1], c_list[2], 3, stride=1, padding=1)
        # Stages 4-5: parallel local / global-local Mamba blocks
        self.encoder4 = LMBP(c_list[2], c_list[3], sep_conv_kernel=3)
        self.encoder5 = GLMBP(c_list[3], c_list[4], sep_conv_kernel=5)
        # Stage 6: GLMBP bottleneck (no downsample)
        self.encoder6 = GLMBP(c_list[4], c_list[5], sep_conv_kernel=7)

        self.ebn1 = nn.GroupNorm(min(4, c_list[0]), c_list[0])
        self.ebn2 = nn.GroupNorm(min(4, c_list[1]), c_list[1])
        self.ebn3 = nn.GroupNorm(min(4, c_list[2]), c_list[2])
        self.ebn4 = nn.GroupNorm(min(4, c_list[3]), c_list[3])
        self.ebn5 = nn.GroupNorm(min(4, c_list[4]), c_list[4])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out
        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out
        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out
        out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(out)), 2, 2))
        t4 = out
        out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out)), 2, 2))
        t5 = out
        bot = F.gelu(self.encoder6(out))
        return [t1, t2, t3, t4, t5, bot]
