"""UltraLight VM-UNet Encoder.

Extracted from ``medseg/networks/mamba/ultralight_vmunet.py`` (Wu et al.,
"UltraLight-VM-UNet", 2024, https://github.com/wurenkai/UltraLight-VM-UNet).

The standalone encoder mirrors the original 6-stage backbone:
    - 3 plain Conv3x3 + GroupNorm + MaxPool stages (the conv stem).
    - 3 ``PVMLayer`` (Parallel Vision Mamba: channels split into 4 groups,
      each fed through a shared Mamba SSM) stages, the first two followed by
      MaxPool, the last (encoder6) left at the bottleneck resolution.

Returns six multi-scale feature maps with the deepest LAST, matching the
project's encoder convention. Spatial state is derived from the runtime
tensor shape, so any input resolution divisible by 32 works.
"""
# Source: https://github.com/wurenkai/UltraLight-VM-UNet

import math
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.registry import ENCODER_REGISTRY
from medseg.models.networks.mamba.umamba import MambaSSM


# ---------------------------------------------------------------------------
# PVMLayer (Parallel Vision Mamba) - copied from the source network.
# ---------------------------------------------------------------------------

class _PVMLayer(nn.Module):
    """Parallel Vision Mamba: split channels into 4 groups, share one Mamba."""

    def __init__(self, input_dim: int, output_dim: int,
                 d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        if input_dim % 4 != 0:
            raise ValueError(
                f"PVMLayer input_dim must be divisible by 4, got {input_dim}")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.mamba = MambaSSM(
            d_model=input_dim // 4,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.float16:
            x = x.float()
        B, C = x.shape[:2]
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)

        x1, x2, x3, x4 = torch.chunk(x_norm, 4, dim=2)
        x_mamba1 = self.mamba(x1) + self.skip_scale * x1
        x_mamba2 = self.mamba(x2) + self.skip_scale * x2
        x_mamba3 = self.mamba(x3) + self.skip_scale * x3
        x_mamba4 = self.mamba(x4) + self.skip_scale * x4
        x_mamba = torch.cat([x_mamba1, x_mamba2, x_mamba3, x_mamba4], dim=2)

        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        return x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)


# ---------------------------------------------------------------------------
# Pretrained-download SSL fallback (kept for parity even though this encoder
# does not currently fetch any weights from the network).
# ---------------------------------------------------------------------------

def _load_with_ssl_fallback(load_fn, *args, **kwargs):
    import ssl
    import warnings
    try:
        return load_fn(*args, **kwargs)
    except Exception:
        prev = ssl._create_default_https_context
        try:
            ssl._create_default_https_context = ssl._create_unverified_context
            return load_fn(*args, **kwargs)
        except Exception as e2:
            warnings.warn(
                f'Pretrained download failed ({e2}); using random init.')
            return load_fn(*args, **{**kwargs, 'pretrained': False})
        finally:
            ssl._create_default_https_context = prev


# ---------------------------------------------------------------------------
# UltraLight VM-UNet encoder
# ---------------------------------------------------------------------------

@ENCODER_REGISTRY.register("ultralight_vm")
class UltraLightVMUNetEncoder(nn.Module):
    """Standalone encoder portion of UltraLight VM-UNet.

    Pipeline (matches ``UltraLightVMUNet.forward`` up to ``encoder6``):

        x -> Conv(c0) -> GN -> MaxPool 2x  -> t1 (c0 @ H/2)
          -> Conv(c1) -> GN -> MaxPool 2x  -> t2 (c1 @ H/4)
          -> Conv(c2) -> GN -> MaxPool 2x  -> t3 (c2 @ H/8)
          -> PVMLayer(c3) -> GN -> MaxPool -> t4 (c3 @ H/16)
          -> PVMLayer(c4) -> GN -> MaxPool -> t5 (c4 @ H/32)
          -> PVMLayer(c5) -> GELU          -> t6 (c5 @ H/32, deepest)

    Args:
        in_channels: Input image channels (1x1 conv stem prepended if != 3).
        img_size: Nominal input spatial size; kept for interface parity.
        pretrained: Unused (no public pretrained weights).
        c_list: Per-stage channel widths (default matches the paper).
        d_state / d_conv / expand: PVMLayer Mamba hyperparameters.
    """

    def __init__(self,
                 in_channels: int = 3,
                 img_size: int = 224,
                 pretrained: bool = False,
                 c_list: Sequence[int] = (8, 16, 24, 32, 48, 64),
                 d_state: int = 16,
                 d_conv: int = 4,
                 expand: int = 2,
                 **kwargs):
        super().__init__()
        c_list = list(c_list)
        if len(c_list) != 6:
            raise ValueError(
                f"c_list must have exactly 6 entries, got {len(c_list)}")
        for i in (3, 4, 5):
            if c_list[i] % 4 != 0:
                raise ValueError(
                    f"c_list[{i}]={c_list[i]} must be divisible by 4 "
                    "(PVMLayer 4-way channel split).")

        self.img_size = img_size
        self.c_list = c_list

        # Optional 1x1 stem to handle non-RGB inputs without touching the
        # downstream conv weights' channel count.
        if in_channels != 3:
            self.stem = nn.Conv2d(in_channels, 3, kernel_size=1)
            stage1_in = 3
        else:
            self.stem = nn.Identity()
            stage1_in = in_channels

        # Stages 1-3: pure conv.
        self.encoder1 = nn.Conv2d(stage1_in, c_list[0], 3, stride=1, padding=1)
        self.encoder2 = nn.Conv2d(c_list[0], c_list[1], 3, stride=1, padding=1)
        self.encoder3 = nn.Conv2d(c_list[1], c_list[2], 3, stride=1, padding=1)
        # Stages 4-6: PVMLayer.
        self.encoder4 = _PVMLayer(c_list[2], c_list[3],
                                  d_state=d_state, d_conv=d_conv, expand=expand)
        self.encoder5 = _PVMLayer(c_list[3], c_list[4],
                                  d_state=d_state, d_conv=d_conv, expand=expand)
        self.encoder6 = _PVMLayer(c_list[4], c_list[5],
                                  d_state=d_state, d_conv=d_conv, expand=expand)

        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])

        self.out_channels: List[int] = list(c_list)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= max(m.groups, 1)
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)

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
        out = F.gelu(self.encoder6(out))
        t6 = out

        # Deepest LAST.
        return [t1, t2, t3, t4, t5, t6]
