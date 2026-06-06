"""KiU-Net (2D) — self-contained port of the upstream reference.

Reference: Valanarasu et al., "KiU-Net: Overcomplete Convolutional Architectures
for Biomedical Image and Volumetric Segmentation" (MICCAI 2020).
Upstream: https://github.com/jeya-maria-jose/KiU-Net-pytorch

Implementation notes
--------------------
The upstream code uses hard-coded ``scale_factor`` values inside
``F.interpolate`` (e.g. ``0.0625``, ``16``, ``64``) which silently assume a
fixed 128x128 input. We replace those with explicit ``size=`` targets
computed from the running spatial dims so the network is fully size-agnostic.

The Ki-Net branch is *overcomplete*: it doubles spatial size at every encoder
stage. For an H=512 input this would produce a level-3 feature of
8H x 8W = 4096 x 4096 (>4 GB of activation at 64 channels in float32) which
is impractical. We therefore introduce ``max_kite_size`` (default 1024) which
caps the Ki-Net branch's spatial dimension at each level. Because we use
*explicit-size* interpolation in both the Ki-Net encoder and decoder, the
mirror symmetry is preserved regardless of the cap — the final Ki-Net stage
always returns to the original (H, W) before fusion with the U-Net stream.

Architecture summary
--------------------
* U-Net branch (undercomplete): channel ladder ``in -> b -> 2b -> 4b`` with
  Conv3x3+BN+ReLU and MaxPool2d stride-2 at every encoder level; mirror
  decoder with bilinear upsampling.
* Ki-Net branch (overcomplete): same channel ladder; encoder upsamples by 2x
  (capped), decoder mirrors by interpolating back down.
* Cross-Residual Fusion Block (CRFB) at every encoder/decoder scale: a
  Conv3x3 + BN + ReLU applied on each branch's feature, then bilinear
  interpolated to the partner branch's spatial size and added residually.
* Within-branch skip connections (``u1<->dec2``, ``u2<->dec1``, etc.).
* Final 1x1 conv producing ``num_classes`` channels at the original
  input resolution.
"""
# Source: https://github.com/jeya-maria-jose/KiU-Net-pytorch

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _interp(x: torch.Tensor, size) -> torch.Tensor:
    """Bilinear interpolate to an explicit (H, W) target."""
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class KiUNet(nn.Module):
    """KiU-Net (2D) — overcomplete + undercomplete dual-branch segmentation net."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        img_size: int = 224,
        base: int = 16,
        max_kite_size: int = 1024,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.img_size = int(img_size)
        self.max_kite_size = int(max_kite_size)

        b = int(base)
        c1, c2, c3 = b, b * 2, b * 4  # 16, 32, 64 by default
        # Decoder output channel for the final pre-fusion stage. Upstream uses
        # a fixed 8 regardless of base; keep that proportionality (b // 2).
        c_fuse = max(b // 2, 1)

        # ---------------- U-Net branch (undercomplete) ----------------
        self.encoder1 = nn.Conv2d(self.in_channels, c1, 3, stride=1, padding=1)
        self.en1_bn = nn.BatchNorm2d(c1)
        self.encoder2 = nn.Conv2d(c1, c2, 3, stride=1, padding=1)
        self.en2_bn = nn.BatchNorm2d(c2)
        self.encoder3 = nn.Conv2d(c2, c3, 3, stride=1, padding=1)
        self.en3_bn = nn.BatchNorm2d(c3)

        self.decoder1 = nn.Conv2d(c3, c2, 3, stride=1, padding=1)
        self.de1_bn = nn.BatchNorm2d(c2)
        self.decoder2 = nn.Conv2d(c2, c1, 3, stride=1, padding=1)
        self.de2_bn = nn.BatchNorm2d(c1)
        self.decoder3 = nn.Conv2d(c1, c_fuse, 3, stride=1, padding=1)
        self.de3_bn = nn.BatchNorm2d(c_fuse)

        # ---------------- Ki-Net branch (overcomplete) ----------------
        self.encoderf1 = nn.Conv2d(self.in_channels, c1, 3, stride=1, padding=1)
        self.enf1_bn = nn.BatchNorm2d(c1)
        self.encoderf2 = nn.Conv2d(c1, c2, 3, stride=1, padding=1)
        self.enf2_bn = nn.BatchNorm2d(c2)
        self.encoderf3 = nn.Conv2d(c2, c3, 3, stride=1, padding=1)
        self.enf3_bn = nn.BatchNorm2d(c3)

        self.decoderf1 = nn.Conv2d(c3, c2, 3, stride=1, padding=1)
        self.def1_bn = nn.BatchNorm2d(c2)
        self.decoderf2 = nn.Conv2d(c2, c1, 3, stride=1, padding=1)
        self.def2_bn = nn.BatchNorm2d(c1)
        self.decoderf3 = nn.Conv2d(c1, c_fuse, 3, stride=1, padding=1)
        self.def3_bn = nn.BatchNorm2d(c_fuse)

        # ------------- Cross-Residual Fusion Blocks (encoder) -------------
        # _1 = Ki-Net -> U-Net (downscaled), _2 = U-Net -> Ki-Net (upscaled).
        self.intere1_1 = nn.Conv2d(c1, c1, 3, stride=1, padding=1)
        self.inte1_1bn = nn.BatchNorm2d(c1)
        self.intere2_1 = nn.Conv2d(c2, c2, 3, stride=1, padding=1)
        self.inte2_1bn = nn.BatchNorm2d(c2)
        self.intere3_1 = nn.Conv2d(c3, c3, 3, stride=1, padding=1)
        self.inte3_1bn = nn.BatchNorm2d(c3)

        self.intere1_2 = nn.Conv2d(c1, c1, 3, stride=1, padding=1)
        self.inte1_2bn = nn.BatchNorm2d(c1)
        self.intere2_2 = nn.Conv2d(c2, c2, 3, stride=1, padding=1)
        self.inte2_2bn = nn.BatchNorm2d(c2)
        self.intere3_2 = nn.Conv2d(c3, c3, 3, stride=1, padding=1)
        self.inte3_2bn = nn.BatchNorm2d(c3)

        # ------------- Cross-Residual Fusion Blocks (decoder) -------------
        self.interd1_1 = nn.Conv2d(c2, c2, 3, stride=1, padding=1)
        self.intd1_1bn = nn.BatchNorm2d(c2)
        self.interd2_1 = nn.Conv2d(c1, c1, 3, stride=1, padding=1)
        self.intd2_1bn = nn.BatchNorm2d(c1)

        self.interd1_2 = nn.Conv2d(c2, c2, 3, stride=1, padding=1)
        self.intd1_2bn = nn.BatchNorm2d(c2)
        self.interd2_2 = nn.Conv2d(c1, c1, 3, stride=1, padding=1)
        self.intd2_2bn = nn.BatchNorm2d(c1)

        # Final classifier: 1x1 conv at the original input resolution.
        self.final = nn.Conv2d(c_fuse, self.num_classes, 1, stride=1, padding=0)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _hw(t: torch.Tensor):
        return t.shape[-2], t.shape[-1]

    def _kite_size(self, h: int, w: int):
        """Double (h, w) but cap each dim at ``max_kite_size``."""
        cap = self.max_kite_size
        return min(h * 2, cap), min(w * 2, cap)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0, w0 = x.shape[-2], x.shape[-1]

        # Pre-compute Ki-Net branch sizes (with capping).
        k1 = self._kite_size(h0, w0)
        k2 = self._kite_size(*k1)
        k3 = self._kite_size(*k2)

        # ===== Encoder, level 1 =====
        # U-Net branch: spatial /2 (via MaxPool2d stride 2).
        out = F.relu(self.en1_bn(F.max_pool2d(self.encoder1(x), 2, 2)))
        # Ki-Net branch: spatial x2 (capped via explicit-size interpolation).
        out1 = F.relu(self.enf1_bn(_interp(self.encoderf1(x), k1)))

        tmp = out
        # CRFB: bring Ki-Net feature down to U-Net resolution, and vice-versa.
        out = out + _interp(
            F.relu(self.inte1_1bn(self.intere1_1(out1))), self._hw(out)
        )
        out1 = out1 + _interp(
            F.relu(self.inte1_2bn(self.intere1_2(tmp))), self._hw(out1)
        )

        u1 = out  # U-Net skip
        o1 = out1  # Ki-Net skip

        # ===== Encoder, level 2 =====
        out = F.relu(self.en2_bn(F.max_pool2d(self.encoder2(out), 2, 2)))
        out1 = F.relu(self.enf2_bn(_interp(self.encoderf2(out1), k2)))

        tmp = out
        out = out + _interp(
            F.relu(self.inte2_1bn(self.intere2_1(out1))), self._hw(out)
        )
        out1 = out1 + _interp(
            F.relu(self.inte2_2bn(self.intere2_2(tmp))), self._hw(out1)
        )

        u2 = out
        o2 = out1

        # ===== Encoder, level 3 =====
        out = F.relu(self.en3_bn(F.max_pool2d(self.encoder3(out), 2, 2)))
        out1 = F.relu(self.enf3_bn(_interp(self.encoderf3(out1), k3)))

        tmp = out
        out = out + _interp(
            F.relu(self.inte3_1bn(self.intere3_1(out1))), self._hw(out)
        )
        out1 = out1 + _interp(
            F.relu(self.inte3_2bn(self.intere3_2(tmp))), self._hw(out1)
        )

        # ===== Decoder, level 1 =====
        # U-Net upsamples to the partner-skip resolution; Ki-Net downsamples
        # to mirror back toward the original input size.
        out = F.relu(self.de1_bn(_interp(self.decoder1(out), self._hw(u2))))
        out1 = F.relu(self.def1_bn(_interp(self.decoderf1(out1), k2)))

        tmp = out
        out = out + _interp(
            F.relu(self.intd1_1bn(self.interd1_1(out1))), self._hw(out)
        )
        out1 = out1 + _interp(
            F.relu(self.intd1_2bn(self.interd1_2(tmp))), self._hw(out1)
        )

        out = out + u2  # within-branch skip
        out1 = out1 + o2

        # ===== Decoder, level 2 =====
        out = F.relu(self.de2_bn(_interp(self.decoder2(out), self._hw(u1))))
        out1 = F.relu(self.def2_bn(_interp(self.decoderf2(out1), k1)))

        tmp = out
        out = out + _interp(
            F.relu(self.intd2_1bn(self.interd2_1(out1))), self._hw(out)
        )
        out1 = out1 + _interp(
            F.relu(self.intd2_2bn(self.interd2_2(tmp))), self._hw(out1)
        )

        out = out + u1
        out1 = out1 + o1

        # ===== Decoder, level 3 (back to input resolution) =====
        out = F.relu(self.de3_bn(_interp(self.decoder3(out), (h0, w0))))
        out1 = F.relu(self.def3_bn(_interp(self.decoderf3(out1), (h0, w0))))

        # Branch fusion + final 1x1 classifier.
        out = out + out1
        out = self.final(out)

        # Safety: guarantee exact (H, W) match.
        if out.shape[-2] != h0 or out.shape[-1] != w0:
            out = _interp(out, (h0, w0))
        return out


__all__ = ["KiUNet"]
