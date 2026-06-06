"""MEW-UNet: Multi-axis External Weight UNet for Medical Image Segmentation.

Faithful PyTorch port of:
  J. Ruan et al., "MEW-UNet: Multi-axis representation learning in frequency
  domain for medical image segmentation", MICCAI 2023.
  Upstream: https://github.com/JCruan519/MEW-UNet

Core idea — the MEW block. For an input feature x in (B, C, H, W) we apply
three Fourier-domain branches, one per "axis":

  * H-axis branch: permute (B, C, H, W) -> (B, H, C, W), 2D rFFT over (C, W),
    multiply by a learnable complex external-weight tensor, then iFFT back.
  * W-axis branch: analogous with W as the leading axis.
  * C-axis branch: standard spatial 2D rFFT over (H, W); the channel axis is
    free, and the weight is per-channel.

The three branches are summed to give the MEW output.

The external-weight tensors are *baked at construction time* (default 7x7 for
the deepest stage at img_size=224). For runtime sizes 256/512, the complex
spectrum is bilinearly interpolated on the (real, imag) channels separately
inside ``_MEW.forward`` — this is the key for multi-resolution support.

Architecture: 6-stage encoder with c_list=(8, 16, 24, 32, 48, 64). Each stage
is Conv3x3 + GN + GELU + MaxPool2; stages 4 and 5 additionally apply MEW to the
post-pool feature. Stage 6 is the bottleneck (no pool). An optional
SC_Att_Bridge (re-used from MALUNet) fuses the five encoder skips. The decoder
mirrors the encoder with GN+GELU+bilinear-up x2 + skip-add, with MEW mirrored
into the two deepest decoder stages. Final 1x1 head + a final 2x interpolate
recovers the input resolution.
"""
# Source: https://github.com/JCruan519/MEW-UNet

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F

from medseg.models.networks.cnn.malunet import SCABridge


# ---------------------------------------------------------------------------
# MEW block
# ---------------------------------------------------------------------------


class _MEW(nn.Module):
    """Multi-axis External Weight block.

    The three branches each transform the input into the frequency domain along
    a different "axis" interpretation, multiply by a learnable complex
    external-weight tensor, and inverse-transform back. Outputs are summed.
    """

    def __init__(self, dim: int, h: int, w: int) -> None:
        super().__init__()
        self.dim = dim
        self.h = h
        self.w = w

        a_h, a_w = dim, w // 2 + 1  # H-axis branch spectrum (C, W//2+1)
        b_h, b_w = dim, h // 2 + 1  # W-axis branch spectrum (C, H//2+1)
        c_h, c_w = h, w // 2 + 1    # C-axis branch spectrum (H, W//2+1)

        # Storage layout: (2, 1, H, W) with dim 0 = real/imag so F.interpolate
        # can treat dim 0 as batch and dim 1 as channel.
        self.a_weight = nn.Parameter(torch.ones(2, 1, a_h, a_w))
        self.b_weight = nn.Parameter(torch.ones(2, 1, b_h, b_w))
        self.c_weight = nn.Parameter(torch.ones(2, 1, c_h, c_w))

        # Group-norm the residual output for stability.
        self.norm = nn.GroupNorm(4, dim)

    @staticmethod
    def _to_complex(weight_4d: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        """Interpolate the stored (2, 1, H, W) weight to target spectrum shape
        and return a complex tensor of shape ``target_hw``.
        """
        if weight_4d.shape[-2:] != tuple(target_hw):
            weight_4d = F.interpolate(
                weight_4d, size=tuple(target_hw), mode="bilinear", align_corners=True
            )
        real = weight_4d[0, 0]
        imag = weight_4d[1, 0]
        return torch.complex(real, imag)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        B, C, H, W = x.shape

        # ----- a-branch: H is the leading axis -----
        xa = x.permute(0, 2, 1, 3).contiguous()            # (B, H, C, W)
        xa_spec = torch.fft.rfft2(xa, dim=(2, 3), norm="ortho")
        a_w = self._to_complex(self.a_weight, (C, W // 2 + 1))
        xa_spec = xa_spec * a_w
        xa = torch.fft.irfft2(xa_spec, s=(C, W), dim=(2, 3), norm="ortho")
        xa = xa.permute(0, 2, 1, 3).contiguous()           # (B, C, H, W)

        # ----- b-branch: W is the leading axis -----
        xb = x.permute(0, 3, 1, 2).contiguous()            # (B, W, C, H)
        xb_spec = torch.fft.rfft2(xb, dim=(2, 3), norm="ortho")
        b_w = self._to_complex(self.b_weight, (C, H // 2 + 1))
        xb_spec = xb_spec * b_w
        xb = torch.fft.irfft2(xb_spec, s=(C, H), dim=(2, 3), norm="ortho")
        xb = xb.permute(0, 2, 3, 1).contiguous()           # (B, C, H, W)

        # ----- c-branch: classic spatial 2D FFT, channel is free axis -----
        xc_spec = torch.fft.rfft2(x, dim=(2, 3), norm="ortho")
        c_w = self._to_complex(self.c_weight, (H, W // 2 + 1))
        xc_spec = xc_spec * c_w
        xc = torch.fft.irfft2(xc_spec, s=(H, W), dim=(2, 3), norm="ortho")

        out = xa + xb + xc
        return self.norm(out) + x


# ---------------------------------------------------------------------------
# MEW-UNet
# ---------------------------------------------------------------------------


class MEWUNet(nn.Module):
    """MEW-UNet, MALUNet-style 6-stage backbone with MEW blocks at deep stages."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        img_size: int = 224,
        c_list: Sequence[int] = (8, 16, 24, 32, 48, 64),
        bridge: bool = True,
        deep_supervision: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        c_list = list(c_list)
        assert len(c_list) == 6, "MEW-UNet expects exactly 6 channel widths."
        self.bridge = bridge
        self.deep_supervision = deep_supervision

        # Construction-time spectrum sizes for the two MEW-bearing stages.
        s4 = max(img_size // 16, 4)   # 14 for img_size=224
        s5 = max(img_size // 32, 2)   #  7 for img_size=224

        # ---------------- Encoder ----------------
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
        self.mew_e4 = _MEW(c_list[3], s4, s4)
        self.mew_e5 = _MEW(c_list[4], s5, s5)

        # Optional spatial-channel attention bridge over 5 encoder skips.
        if bridge:
            self.scab = SCABridge(c_list, split_att="fc")

        # ---------------- Decoder ----------------
        self.decoder1 = nn.Conv2d(c_list[5], c_list[4], 3, 1, 1)
        self.decoder2 = nn.Conv2d(c_list[4], c_list[3], 3, 1, 1)
        self.decoder3 = nn.Conv2d(c_list[3], c_list[2], 3, 1, 1)
        self.decoder4 = nn.Conv2d(c_list[2], c_list[1], 3, 1, 1)
        self.decoder5 = nn.Conv2d(c_list[1], c_list[0], 3, 1, 1)

        self.dbn1 = nn.GroupNorm(4, c_list[4])
        self.dbn2 = nn.GroupNorm(4, c_list[3])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])
        self.dbn5 = nn.GroupNorm(4, c_list[0])

        # MEW blocks mirrored into the two deepest decoder stages.
        self.mew_d1 = _MEW(c_list[4], s5, s5)
        self.mew_d2 = _MEW(c_list[3], s4, s4)

        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)

        if deep_supervision:
            self.ds_heads = nn.ModuleList([
                nn.Conv2d(c_list[3], num_classes, 1),
                nn.Conv2d(c_list[2], num_classes, 1),
                nn.Conv2d(c_list[1], num_classes, 1),
                nn.Conv2d(c_list[0], num_classes, 1),
            ])

        self.apply(self._init_weights)

    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        H_in, W_in = x.shape[-2:]

        # ---- Encoder ----
        e1 = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))   # H/2,  c0
        t1 = e1
        e2 = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(e1)), 2, 2))  # H/4,  c1
        t2 = e2
        e3 = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(e2)), 2, 2))  # H/8,  c2
        t3 = e3
        e4 = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(e3)), 2, 2))  # H/16, c3
        e4 = self.mew_e4(e4)
        t4 = e4
        e5 = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(e4)), 2, 2))  # H/32, c4
        e5 = self.mew_e5(e5)
        t5 = e5

        if self.bridge:
            t1, t2, t3, t4, t5 = self.scab([t1, t2, t3, t4, t5])

        # ---- Bottleneck (stage 6, no pool, GELU only on the activation) ----
        bn = F.gelu(self.ebn6(self.encoder6(e5)))                      # H/32, c5

        # ---- Decoder ----
        d1 = F.gelu(self.dbn1(self.decoder1(bn)))                      # H/32, c4
        d1 = self.mew_d1(d1)
        d1 = d1 + t5

        d2 = F.gelu(F.interpolate(self.dbn2(self.decoder2(d1)),
                                  scale_factor=2, mode="bilinear", align_corners=True))
        d2 = self.mew_d2(d2)
        d2 = d2 + t4                                                   # H/16, c3

        d3 = F.gelu(F.interpolate(self.dbn3(self.decoder3(d2)),
                                  scale_factor=2, mode="bilinear", align_corners=True))
        d3 = d3 + t3                                                   # H/8,  c2

        d4 = F.gelu(F.interpolate(self.dbn4(self.decoder4(d3)),
                                  scale_factor=2, mode="bilinear", align_corners=True))
        d4 = d4 + t2                                                   # H/4,  c1

        d5 = F.gelu(F.interpolate(self.dbn5(self.decoder5(d4)),
                                  scale_factor=2, mode="bilinear", align_corners=True))
        d5 = d5 + t1                                                   # H/2,  c0

        out = self.final(d5)
        if out.shape[-2:] != (H_in, W_in):
            out = F.interpolate(out, size=(H_in, W_in),
                                mode="bilinear", align_corners=True)

        if self.training and self.deep_supervision:
            return [out] + self._ds_outputs([d2, d3, d4, d5], (H_in, W_in))
        return out

    # ------------------------------------------------------------------
    def _ds_outputs(self, feats: List[torch.Tensor],
                    out_size: Tuple[int, int]) -> List[torch.Tensor]:
        aux: List[torch.Tensor] = []
        for f, head in zip(feats, self.ds_heads):
            a = head(f)
            if a.shape[-2:] != out_size:
                a = F.interpolate(a, size=out_size,
                                  mode="bilinear", align_corners=True)
            aux.append(a)
        return aux
