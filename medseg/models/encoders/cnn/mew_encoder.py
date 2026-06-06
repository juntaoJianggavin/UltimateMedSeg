"""MEW-UNet Encoder.

Standalone encoder extracted from ``medseg.models.networks.cnn.mew_unet.MEWUNet``.

6-stage encoder with channel widths ``c_list=(8, 16, 24, 32, 48, 64)``. Each
stage is Conv3x3 + GN + GELU + MaxPool2. Stages 4 and 5 additionally apply the
MEW (Multi-axis External Weight) block to the post-pool feature. Stage 6 is
the bottleneck (no pool).

Returns 6 multi-scale features (shallow -> deep, deepest LAST)::

    [c0 @ H/2, c1 @ H/4, c2 @ H/8, c3 @ H/16, c4 @ H/32, c5 @ H/32]

For default ``c_list``, the channel list is ``[8, 16, 24, 32, 48, 64]``.

The MEW blocks bake an external weight at construction time sized for the
provided ``img_size``; for other runtime resolutions the complex spectrum is
bilinearly interpolated inside ``_MEW.forward``.
"""
# Source: https://github.com/JCruan519/MEW-UNet

from __future__ import annotations

import math
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.registry import ENCODER_REGISTRY


def _get_MEW():
    """延迟导入 _MEW 避免 encoders ↔ networks 循环依赖。
    Lazy import _MEW to avoid encoders ↔ networks circular dependency."""
    from medseg.models.networks.cnn.mew_unet import _MEW
    return _MEW


# 模块级占位，在类 __init__ 里用 _get_MEW() 实际加载
_MEW = None


@ENCODER_REGISTRY.register("mew")
class MEWEncoder(nn.Module):
    """Standalone MEW-UNet encoder.

    Args:
        in_channels: Number of input channels (e.g. 1 for grayscale, 3 for RGB).
        img_size: Reference spatial resolution used to size the construction-
            time MEW spectra (deeper stages run at H/16 and H/32). For other
            runtime sizes the MEW spectrum is bilinearly interpolated.
        pretrained: Unused for MEW (no canonical public weights); kept for
            standard encoder interface parity.
        c_list: Six channel widths for the six encoder stages.
    """

    def __init__(
        self,
        in_channels: int = 3,
        img_size: int = 224,
        pretrained: bool = False,
        c_list: Sequence[int] = (8, 16, 24, 32, 48, 64),
        **kwargs,
    ) -> None:
        super().__init__()
        c_list = list(c_list)
        assert len(c_list) == 6, "MEW encoder expects exactly 6 channel widths."

        # Construction-time spectrum sizes for the two MEW-bearing stages.
        s4 = max(img_size // 16, 4)
        s5 = max(img_size // 32, 2)

        # -- Conv stages --
        self.encoder1 = nn.Conv2d(in_channels, c_list[0], 3, 1, 1)
        self.encoder2 = nn.Conv2d(c_list[0], c_list[1], 3, 1, 1)
        self.encoder3 = nn.Conv2d(c_list[1], c_list[2], 3, 1, 1)
        self.encoder4 = nn.Conv2d(c_list[2], c_list[3], 3, 1, 1)
        self.encoder5 = nn.Conv2d(c_list[3], c_list[4], 3, 1, 1)
        self.encoder6 = nn.Conv2d(c_list[4], c_list[5], 3, 1, 1)

        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])
        self.ebn6 = nn.GroupNorm(4, c_list[5])

        # MEW blocks at the deeper encoder stages (post-pool features).
        MEW = _get_MEW()
        self.mew_e4 = MEW(c_list[3], s4, s4)
        self.mew_e5 = MEW(c_list[4], s5, s5)

        # Channel list for each returned feature (deepest LAST).
        self.out_channels: List[int] = list(c_list)

        self._pretrained_requested = bool(pretrained)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0.0, math.sqrt(2.0 / n))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= max(m.groups, 1)
            m.weight.data.normal_(0.0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        e1 = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))   # H/2,  c0
        e2 = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(e1)), 2, 2))  # H/4,  c1
        e3 = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(e2)), 2, 2))  # H/8,  c2
        e4 = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(e3)), 2, 2))  # H/16, c3
        e4 = self.mew_e4(e4)
        e5 = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(e4)), 2, 2))  # H/32, c4
        e5 = self.mew_e5(e5)
        bn = F.gelu(self.ebn6(self.encoder6(e5)))                      # H/32, c5

        return [e1, e2, e3, e4, e5, bn]
