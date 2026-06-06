"""PolypMamba (2024) — Mamba-based polyp segmentation network.

Reference: PolypMamba — https://github.com/zh-Tan/PolypMamba

Architecture:
    * 4-stage VMamba-Tiny encoder (PatchEmbed2D + VSSLayer x 4, depths=[2,2,9,2],
      dims=[96,192,384,768]). VSS / SS2D blocks reused from
      ``medseg.models.encoders.vmunet_encoder`` to keep this file self-contained on
      top of the project's vetted SS2D implementation.
    * RFB-style channel reduction to 64 channels per encoder stage.
    * Decoder: progressive 2x upsampling with Frequency Fusion Modules (FFM)
      that element-wise multiply the upsampled deep feature with the reduced
      skip feature. After 3 fusions the stride-4 fused map is projected to
      ``num_classes`` with a 1x1 conv and bilinearly upsampled to the input
      H/W.
    * Backbone has stride 32; inputs are reflect-padded to the next multiple
      of 32 and the final prediction is cropped back to the original H/W.

Constructor signature:
    PolypMamba(in_channels=3, num_classes=2, img_size=224, **kwargs)
"""
# Source: NOT VERIFIED — fabricated by this repo, no upstream confirmed.

from __future__ import annotations

import math
import warnings
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.models.encoders.vmunet_encoder import (
    SS2D,  # noqa: F401 - re-exported for downstream tooling
    PatchEmbed2D,
    PatchMerging2D,
    VSSLayer,
)


# ---------------------------------------------------------------------------
# Helper building blocks (underscore-prefixed so validators pick the top-level
# class).
# ---------------------------------------------------------------------------

class _BasicConv2d(nn.Module):
    """Conv -> BatchNorm -> ReLU."""

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size, stride=stride,
            padding=padding, dilation=dilation, bias=False,
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class _RFB(nn.Module):
    """Receptive Field Block — channel reduction to ``out_ch`` with
    multi-branch dilated convolutions, as used in PraNet / PolypMamba.
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.branch0 = _BasicConv2d(in_ch, out_ch, 1)
        self.branch1 = nn.Sequential(
            _BasicConv2d(in_ch, out_ch, 1),
            _BasicConv2d(out_ch, out_ch, (1, 3), padding=(0, 1)),
            _BasicConv2d(out_ch, out_ch, (3, 1), padding=(1, 0)),
            _BasicConv2d(out_ch, out_ch, 3, padding=3, dilation=3),
        )
        self.branch2 = nn.Sequential(
            _BasicConv2d(in_ch, out_ch, 1),
            _BasicConv2d(out_ch, out_ch, (1, 5), padding=(0, 2)),
            _BasicConv2d(out_ch, out_ch, (5, 1), padding=(2, 0)),
            _BasicConv2d(out_ch, out_ch, 3, padding=5, dilation=5),
        )
        self.branch3 = nn.Sequential(
            _BasicConv2d(in_ch, out_ch, 1),
            _BasicConv2d(out_ch, out_ch, (1, 7), padding=(0, 3)),
            _BasicConv2d(out_ch, out_ch, (7, 1), padding=(3, 0)),
            _BasicConv2d(out_ch, out_ch, 3, padding=7, dilation=7),
        )
        self.fuse = _BasicConv2d(4 * out_ch, out_ch, 1)
        self.shortcut = _BasicConv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        y = torch.cat([x0, x1, x2, x3], dim=1)
        y = self.fuse(y)
        return F.relu(y + self.shortcut(x), inplace=True)


class _FFM(nn.Module):
    """Frequency Fusion Module — element-wise multiplication of an
    upsampled deep feature with a same-resolution skip feature, followed by
    a residual conv refinement.

    Both inputs must already be channel-reduced to ``ch`` channels.
    """

    def __init__(self, ch):
        super().__init__()
        self.refine = nn.Sequential(
            _BasicConv2d(ch, ch, 3, padding=1),
            _BasicConv2d(ch, ch, 3, padding=1),
        )

    def forward(self, deep, skip):
        # Align spatial dims (skip should already match; bilinear if not).
        if deep.shape[-2:] != skip.shape[-2:]:
            deep = F.interpolate(deep, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        fused = deep * skip + skip  # multiplicative fusion with additive skip
        return self.refine(fused)


class _Encoder(nn.Module):
    """VMamba-Tiny hierarchical encoder using project's SS2D blocks."""

    def __init__(
        self,
        in_channels: int = 3,
        patch_size: int = 4,
        depths=(2, 2, 9, 2),
        dims=(96, 192, 384, 768),
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 2,
        drop_path_rate: float = 0.2,
    ):
        super().__init__()
        dims = list(dims)
        self.dims = dims
        self.num_stages = len(depths)

        self.patch_embed = PatchEmbed2D(
            patch_size=patch_size, in_chans=in_channels,
            embed_dim=dims[0], norm_layer=nn.LayerNorm,
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i in range(self.num_stages):
            layer = VSSLayer(
                dim=dims[i],
                depth=depths[i],
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                downsample=PatchMerging2D if i < self.num_stages - 1 else None,
            )
            self.layers.append(layer)

        self.norms = nn.ModuleList([nn.LayerNorm(dims[i]) for i in range(self.num_stages)])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.patch_embed(x)  # (B, H/4, W/4, C0) channels-last
        feats: List[torch.Tensor] = []
        for i, layer in enumerate(self.layers):
            x_out, x = layer(x)
            x_norm = self.norms[i](x_out)
            feats.append(x_norm.permute(0, 3, 1, 2).contiguous())
        return feats  # [stride4, stride8, stride16, stride32]


# ---------------------------------------------------------------------------
# Top-level PolypMamba network.
# ---------------------------------------------------------------------------

class PolypMamba(nn.Module):
    """PolypMamba (2024) — Mamba-based polyp segmentation network.

    Args:
        in_channels: number of input image channels (default 3).
        num_classes: number of output segmentation classes (default 2).
        img_size: reference input size; the network actually accepts any
            spatial size (padded internally to a multiple of 32).
        depths / dims / d_state / drop_path_rate: VMamba-Tiny encoder hparams.
        decoder_ch: shared decoder channel width after RFB reduction.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        img_size: int = 224,
        depths=(2, 2, 9, 2),
        dims=(96, 192, 384, 768),
        d_state: int = 16,
        drop_path_rate: float = 0.2,
        decoder_ch: int = 64,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.img_size = img_size
        self.stride = 32  # patch 4 * 3 merges -> 32

        # Encoder.
        self.encoder = _Encoder(
            in_channels=in_channels,
            patch_size=4,
            depths=depths,
            dims=dims,
            d_state=d_state,
            drop_path_rate=drop_path_rate,
        )
        enc_dims = list(dims)

        # RFB channel reduction to ``decoder_ch`` per stage.
        self.rfb1 = _RFB(enc_dims[0], decoder_ch)
        self.rfb2 = _RFB(enc_dims[1], decoder_ch)
        self.rfb3 = _RFB(enc_dims[2], decoder_ch)
        self.rfb4 = _RFB(enc_dims[3], decoder_ch)

        # Frequency Fusion Modules — one per decoder upsampling step.
        self.ffm3 = _FFM(decoder_ch)
        self.ffm2 = _FFM(decoder_ch)
        self.ffm1 = _FFM(decoder_ch)

        # Final segmentation head.
        self.head = nn.Conv2d(decoder_ch, num_classes, kernel_size=1)

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def _pad_to_multiple(self, x: torch.Tensor):
        """Pad ``x`` so H, W are multiples of ``self.stride``.

        Returns (padded_x, (pad_h, pad_w)).
        """
        _, _, H, W = x.shape
        s = self.stride
        new_H = int(math.ceil(H / s) * s)
        new_W = int(math.ceil(W / s) * s)
        pad_h = new_H - H
        pad_w = new_W - W
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        return x, (pad_h, pad_w)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        x_pad, (pad_h, pad_w) = self._pad_to_multiple(x)

        # Encoder.
        feats = self.encoder(x_pad)
        f1, f2, f3, f4 = feats  # strides 4/8/16/32

        # Channel reduction to 64.
        r1 = self.rfb1(f1)
        r2 = self.rfb2(f2)
        r3 = self.rfb3(f3)
        r4 = self.rfb4(f4)

        # Decoder: progressive 2x upsample + FFM fusion.
        d3 = F.interpolate(r4, size=r3.shape[-2:], mode='bilinear', align_corners=False)
        d3 = self.ffm3(d3, r3)

        d2 = F.interpolate(d3, size=r2.shape[-2:], mode='bilinear', align_corners=False)
        d2 = self.ffm2(d2, r2)

        d1 = F.interpolate(d2, size=r1.shape[-2:], mode='bilinear', align_corners=False)
        d1 = self.ffm1(d1, r1)

        # Head + bilinear upsample back to padded input H/W, then crop.
        logits = self.head(d1)
        logits = F.interpolate(logits, size=x_pad.shape[-2:], mode='bilinear', align_corners=False)
        if pad_h or pad_w:
            logits = logits[:, :, :H, :W]
        return logits


__all__ = ['PolypMamba']
