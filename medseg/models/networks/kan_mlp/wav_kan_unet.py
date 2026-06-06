"""Wav-KAN UNet -- UNet backbone with Wav-KAN (wavelet-basis KAN) blocks.

Self-contained port combining:
  - U-KAN architecture layout (CUHK-AIM-Group/U-KAN, AAAI 2025) extended to
    5 levels.
  - Wav-KAN linear layer from zavareh1/Wav-KAN: each KAN linear replaces the
    MLP with a learnable mixture of wavelet basis functions.

The Wav-KAN linear formula:
    output_j = sum_i [ W_ij * psi((x_i - b_ij) / s_ij) ]
where psi is a fixed wavelet kernel (default ``mexican_hat`` ->
``(1 - x^2) * exp(-x^2 / 2)``) and (W, b, s) are learnable per
(input, output) pair.

The UNet is 5 levels with channel dims [32, 64, 128, 256, 512].  The
first three encoder stages are plain Conv-BN-ReLU blocks; the deepest two
stages use Wav-KAN.  The decoder mirrors the encoder with Wav-KAN blocks
at the two deepest decoder positions, ConvTranspose2d upsampling, and
skip-connections from the encoder.
"""
# Source: https://github.com/zavareh1/Wav-KAN

from __future__ import annotations

import math
from typing import Callable, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Wavelet kernels (psi)
# ---------------------------------------------------------------------------

def _mexican_hat(x: torch.Tensor) -> torch.Tensor:
    return (1.0 - x * x) * torch.exp(-0.5 * x * x)


def _morlet(x: torch.Tensor, omega0: float = 5.0) -> torch.Tensor:
    return torch.cos(omega0 * x) * torch.exp(-0.5 * x * x)


def _dog(x: torch.Tensor) -> torch.Tensor:
    # Derivative of Gaussian: -x * exp(-x^2 / 2)
    return -x * torch.exp(-0.5 * x * x)


def _shannon(x: torch.Tensor) -> torch.Tensor:
    return torch.sinc(x) * torch.cos(2.0 * math.pi * x)


_WAVELETS: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "mexican_hat": _mexican_hat,
    "morlet": _morlet,
    "dog": _dog,
    "shannon": _shannon,
}


# ---------------------------------------------------------------------------
# Wav-KAN linear layer
# ---------------------------------------------------------------------------

class _WavKANLinear(nn.Module):
    """Wavelet-basis KAN linear layer.

    Maps an input tensor of shape ``(..., in_features)`` to one of shape
    ``(..., out_features)`` using the formula

        out_j = sum_i W_ij * psi((x_i - b_ij) / s_ij) + base_j(x)

    where ``W``, ``b`` (translation) and ``s`` (scale) are learnable
    per-(input, output) pairs and ``base_j(x)`` is an optional standard
    linear branch with SiLU activation (matches Wav-KAN reference).
    A LayerNorm is applied on the output channels for training stability,
    again following the reference implementation.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        wavelet: str = "mexican_hat",
        with_base: bool = True,
        out_chunk: int = 32,
    ) -> None:
        super().__init__()
        if wavelet not in _WAVELETS:
            raise ValueError(
                f"unknown wavelet '{wavelet}', expected one of "
                f"{list(_WAVELETS)}"
            )
        self.in_features = in_features
        self.out_features = out_features
        self.wavelet_name = wavelet
        self._psi = _WAVELETS[wavelet]
        self.out_chunk = max(1, int(out_chunk))

        # Learnable per (out, in) parameters of the wavelet basis.
        self.scale = nn.Parameter(torch.ones(out_features, in_features))
        self.translation = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        self.with_base = with_base
        if with_base:
            self.base_weight = nn.Parameter(
                torch.empty(out_features, in_features))
            nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
            self.base_activation = nn.SiLU()

        self.norm = nn.LayerNorm(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        assert orig_shape[-1] == self.in_features, (
            f"_WavKANLinear: expected last dim {self.in_features}, "
            f"got {orig_shape[-1]}"
        )
        x_flat = x.reshape(-1, self.in_features)  # (N, I)

        out_chunks = []
        for j_start in range(0, self.out_features, self.out_chunk):
            j_end = min(j_start + self.out_chunk, self.out_features)
            b_c = self.translation[j_start:j_end]   # (c, I)
            s_c = self.scale[j_start:j_end]          # (c, I)
            w_c = self.weight[j_start:j_end]         # (c, I)
            # (N, 1, I) - (c, I) -> (N, c, I)
            scaled = (x_flat.unsqueeze(1) - b_c) / (s_c + 1e-4)
            psi = self._psi(scaled)
            out_chunks.append((w_c * psi).sum(dim=-1))  # (N, c)
        out = torch.cat(out_chunks, dim=1) if len(out_chunks) > 1 \
            else out_chunks[0]

        if self.with_base:
            out = out + F.linear(self.base_activation(x_flat), self.base_weight)

        out = self.norm(out)
        return out.reshape(*orig_shape[:-1], self.out_features)


# ---------------------------------------------------------------------------
# Helper building blocks
# ---------------------------------------------------------------------------

class _DWConv(nn.Module):
    """Depth-wise 3x3 conv operating on tokenized (B, N, C) features."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=True)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)


class _WavKANBlock(nn.Module):
    """LayerNorm -> Wav-KAN linear -> depth-wise conv (residual)."""

    def __init__(self, dim: int, wavelet: str = "mexican_hat",
                 drop: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.kan = _WavKANLinear(dim, dim, wavelet=wavelet)
        self.dwconv = _DWConv(dim)
        self.post_norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        shortcut = x
        x = self.norm(x)
        x = self.kan(x)
        x = self.dwconv(x, H, W)
        x = self.post_norm(x)
        x = self.drop(x)
        return shortcut + x


class _PatchEmbed(nn.Module):
    """Overlapping patch embedding: Conv stride-2 downsample then norm."""

    def __init__(self, in_chans: int, embed_dim: int,
                 patch_size: int = 3, stride: int = 2) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=stride,
            padding=patch_size // 2,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class _ConvLayer(nn.Module):
    """Standard double conv block: Conv3x3-BN-ReLU x 2."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _UpBlock(nn.Module):
    """ConvTranspose2d 2x upsample then refining conv."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        return self.conv(x)


class _ConvRefine(nn.Module):
    """Refining block used by the bottom-most decoder stage (no upsample)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ---------------------------------------------------------------------------
# Wav-KAN UNet
# ---------------------------------------------------------------------------

class WavKANUNet(nn.Module):
    """Wav-KAN UNet for medical image segmentation.

    Args:
        in_channels:  Input image channels (default 3).
        num_classes:  Output segmentation classes (default 2).
        img_size:     Nominal input spatial size (used only as a hint;
                      ``forward`` accepts arbitrary H, W).
        dims:         5 channel dimensions, one per UNet level.
                      Default ``(32, 64, 128, 256, 512)``.
        wavelet:      Wavelet kernel name (``mexican_hat``, ``morlet``,
                      ``dog``, ``shannon``).
        drop_rate:    Dropout in Wav-KAN blocks.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        img_size: int = 224,
        dims=(32, 64, 128, 256, 512),
        wavelet: str = "mexican_hat",
        drop_rate: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__()
        dims = tuple(dims)
        assert len(dims) == 5, "dims must have 5 entries (5-level UNet)"
        self.dims = dims
        self.img_size = img_size
        self.num_classes = num_classes

        # ---- Encoder: 3 conv stages (downsample via max-pool in forward) ----
        self.encoder1 = _ConvLayer(in_channels, dims[0])  # /1 then /2
        self.encoder2 = _ConvLayer(dims[0], dims[1])      # /2 then /4
        self.encoder3 = _ConvLayer(dims[1], dims[2])      # /4 then /8

        # ---- Encoder: 2 Wav-KAN stages (stride-2 patch embed each) ----
        self.patch_embed4 = _PatchEmbed(dims[2], dims[3], patch_size=3, stride=2)
        self.patch_embed5 = _PatchEmbed(dims[3], dims[4], patch_size=3, stride=2)
        self.block4 = _WavKANBlock(dims[3], wavelet=wavelet, drop=drop_rate)
        self.block5 = _WavKANBlock(dims[4], wavelet=wavelet, drop=drop_rate)
        self.norm4 = nn.LayerNorm(dims[3])
        self.norm5 = nn.LayerNorm(dims[4])

        # ---- Decoder: ConvTranspose2d 2x upsample at each level ----
        # The two deepest decoder positions get a Wav-KAN block.
        self.decoder1 = _UpBlock(dims[4], dims[3])  # /32 -> /16, 512 -> 256
        self.decoder2 = _UpBlock(dims[3], dims[2])  # /16 -> /8,  256 -> 128
        self.decoder3 = _UpBlock(dims[2], dims[1])  # /8  -> /4,  128 -> 64
        self.decoder4 = _UpBlock(dims[1], dims[0])  # /4  -> /2,  64  -> 32
        self.decoder5 = _UpBlock(dims[0], dims[0])  # /2  -> /1,  32  -> 32

        self.dblock1 = _WavKANBlock(dims[3], wavelet=wavelet, drop=drop_rate)
        self.dblock2 = _WavKANBlock(dims[2], wavelet=wavelet, drop=drop_rate)
        self.dnorm1 = nn.LayerNorm(dims[3])
        self.dnorm2 = nn.LayerNorm(dims[2])

        self.final = nn.Conv2d(dims[0], num_classes, kernel_size=1)

        self.apply(self._init_weights)

    # ------------------------------------------------------------------
    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= max(1, m.groups)
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_in",
                                    nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    @staticmethod
    def _match_spatial(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != ref.shape[-2:]:
            x = F.interpolate(x, size=ref.shape[-2:], mode="bilinear",
                              align_corners=False)
        return x

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H_in, W_in = x.shape

        # ---- Encoder: convolutional stages ----
        out = F.relu(F.max_pool2d(self.encoder1(x), 2, 2))
        t1 = out                                # dim[0], /2
        out = F.relu(F.max_pool2d(self.encoder2(out), 2, 2))
        t2 = out                                # dim[1], /4
        out = F.relu(F.max_pool2d(self.encoder3(out), 2, 2))
        t3 = out                                # dim[2], /8

        # ---- Encoder: Wav-KAN stage 4 ----
        out, H, W = self.patch_embed4(out)      # dim[3], /16
        out = self.block4(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out                                # dim[3], /16

        # ---- Encoder: Wav-KAN bottleneck (stage 5) ----
        out, H, W = self.patch_embed5(out)      # dim[4], /32
        out = self.block5(out, H, W)
        out = self.norm5(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # ---- Decoder: Wav-KAN level 4 ----
        out = self.decoder1(out)                # /32 -> /16, dim[3]
        out = out + self._match_spatial(t4, out)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        out = self.dblock1(out, H, W)
        out = self.dnorm1(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # ---- Decoder: Wav-KAN level 3 ----
        out = self.decoder2(out)                # /16 -> /8, dim[2]
        out = out + self._match_spatial(t3, out)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        out = self.dblock2(out, H, W)
        out = self.dnorm2(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # ---- Decoder: convolutional levels ----
        out = self.decoder3(out)                # /8 -> /4, dim[1]
        out = out + self._match_spatial(t2, out)
        out = self.decoder4(out)                # /4 -> /2, dim[0]
        out = out + self._match_spatial(t1, out)
        out = self.decoder5(out)                # /2 -> /1, dim[0]

        out = self.final(out)
        if out.shape[-2:] != (H_in, W_in):
            out = F.interpolate(out, size=(H_in, W_in), mode="bilinear",
                                align_corners=False)
        return out


__all__ = ["WavKANUNet"]
