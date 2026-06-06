"""Wav-KAN Encoder.

Standalone encoder extracted from
``medseg.models.networks.kan_mlp.wav_kan_unet.WavKANUNet``.

Pipeline (default ``dims=(32, 64, 128, 256, 512)``):
    in -> [1x1 conv if in_channels != 3] ->
    encoder1 (DoubleConv,  in     -> 32)  -> MaxPool/2 -> t1 (C=32,  H/2)
    encoder2 (DoubleConv,  32     -> 64)  -> MaxPool/2 -> t2 (C=64,  H/4)
    encoder3 (DoubleConv,  64     -> 128) -> MaxPool/2 -> t3 (C=128, H/8)
    patch_embed4 (stride 2) + WavKANBlock + LN ->          t4 (C=256, H/16)
    patch_embed5 (stride 2) + WavKANBlock + LN ->          t5 (C=512, H/32)

Returns 5 multi-scale features ordered shallow -> deep, deepest LAST.
Inputs should be divisible by 32 (2^3 strided pool * 2 PatchEmbeds).
"""
# Source: https://github.com/zavareh1/Wav-KAN

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.registry import ENCODER_REGISTRY
from medseg.models.networks.kan_mlp.wav_kan_unet import (
    _ConvLayer,
    _PatchEmbed,
    _WavKANBlock,
)


@ENCODER_REGISTRY.register("wav_kan")
class WavKANEncoder(nn.Module):
    """Wav-KAN encoder.

    Returns 5 multi-scale features (shallow -> deep, deepest LAST):
        [t1 @ H/2, t2 @ H/4, t3 @ H/8, t4 @ H/16, t5 @ H/32]
    with channel counts equal to ``dims`` (default ``[32, 64, 128, 256, 512]``).

    Args:
        in_channels: Number of input channels. If != 3, a 1x1 stem maps to 3.
        img_size: Reference spatial resolution (kept only for interface
            parity; forward reads true H/W from the input tensor).
        pretrained: Unused for Wav-KAN (no public weights); kept for the
            standard encoder interface.
        dims: 5 channel dimensions, one per UNet level. Default
            ``(32, 64, 128, 256, 512)``.
        wavelet: Wavelet kernel name (``mexican_hat``, ``morlet``, ``dog``,
            ``shannon``).
        drop_rate: Dropout in Wav-KAN blocks.
    """

    def __init__(
        self,
        in_channels: int = 3,
        img_size: int = 224,
        pretrained: bool = False,
        dims=(32, 64, 128, 256, 512),
        wavelet: str = "mexican_hat",
        drop_rate: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__()
        dims = tuple(dims)
        assert len(dims) == 5, "dims must have 5 entries (5-level encoder)"
        self.dims = dims
        self.img_size = img_size

        # Optional 1x1 stem for non-RGB inputs.
        if in_channels != 3:
            self.in_proj = nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)
            stem_in = 3
        else:
            self.in_proj = nn.Identity()
            stem_in = in_channels

        # ---- Encoder: 3 conv stages (downsample via max-pool in forward) ----
        self.encoder1 = _ConvLayer(stem_in, dims[0])
        self.encoder2 = _ConvLayer(dims[0], dims[1])
        self.encoder3 = _ConvLayer(dims[1], dims[2])

        # ---- Encoder: 2 Wav-KAN stages (stride-2 patch embed each) ----
        self.patch_embed4 = _PatchEmbed(
            dims[2], dims[3], patch_size=3, stride=2)
        self.patch_embed5 = _PatchEmbed(
            dims[3], dims[4], patch_size=3, stride=2)
        self.block4 = _WavKANBlock(dims[3], wavelet=wavelet, drop=drop_rate)
        self.block5 = _WavKANBlock(dims[4], wavelet=wavelet, drop=drop_rate)
        self.norm4 = nn.LayerNorm(dims[3])
        self.norm5 = nn.LayerNorm(dims[4])

        # Channel list for each returned feature (deepest LAST).
        self.out_channels: List[int] = list(dims)

        self._pretrained_requested = bool(pretrained)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.in_proj(x)
        B = x.shape[0]

        # ---- Conv encoder stages ----
        out = F.relu(F.max_pool2d(self.encoder1(x), 2, 2))
        t1 = out                                  # dims[0], H/2
        out = F.relu(F.max_pool2d(self.encoder2(out), 2, 2))
        t2 = out                                  # dims[1], H/4
        out = F.relu(F.max_pool2d(self.encoder3(out), 2, 2))
        t3 = out                                  # dims[2], H/8

        # ---- Wav-KAN stage 4 ----
        out, H, W = self.patch_embed4(out)        # dims[3], H/16
        out = self.block4(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out

        # ---- Wav-KAN bottleneck (stage 5) ----
        out, H, W = self.patch_embed5(out)        # dims[4], H/32
        out = self.block5(out, H, W)
        out = self.norm5(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t5 = out

        return [t1, t2, t3, t4, t5]


__all__ = ["WavKANEncoder"]
