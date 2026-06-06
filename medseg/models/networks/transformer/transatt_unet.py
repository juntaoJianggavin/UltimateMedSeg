"""TransAttUNet — self-contained port.

Reference:
    Chen et al., "TransAttUnet: Multi-level Attention-guided U-Net with
    Transformer for Medical Image Segmentation." Expert Systems with
    Applications, 2023.
    Repo: https://github.com/McGregorWwww/TransAttUnet

Architecture:
    * Standard U-Net backbone with (Conv-BN-ReLU)x2 ConvBlocks
    * 4 down stages (maxpool + ConvBlock)
    * Transformer Self-Attention (TSA) module at the bottleneck:
        standard transformer encoder layer (MHSA + MLP, with LN) operating
        on flattened bottleneck tokens — no positional embedding so the
        module is resolution-agnostic.
    * Global Spatial Attention (GSA) module that fuses every decoder
        feature map: each scale is projected with a 1x1 conv, upsampled to
        the largest decoder resolution, concatenated, and then weighted
        per-pixel with softmax attention.
    * 4 up stages (transposed-conv + concat skip + ConvBlock)
    * Final fusion of last decoder feature with GSA output, then 1x1 head.

Standard interface:
    model = TransAttUNet(in_channels=3, num_classes=2, img_size=224)
    y = model(torch.randn(1, 3, 224, 224))   # -> (1, 2, 224, 224)
"""
# Source: https://github.com/McGregorWwww/TransAttUnet

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class _ConvBlock(nn.Module):
    """(Conv-BN-ReLU) x 2."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _DownBlock(nn.Module):
    """MaxPool -> ConvBlock."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = _ConvBlock(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class _UpBlock(nn.Module):
    """Transposed conv -> concat skip -> ConvBlock."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = _ConvBlock(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:],
                              mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


# ---------------------------------------------------------------------------
# Transformer Self-Attention (TSA) module — bottleneck
# ---------------------------------------------------------------------------

class _TSA(nn.Module):
    """Standard transformer encoder layer over flattened spatial tokens.

    No positional embedding is used so the module is resolution-agnostic.
    """

    def __init__(self, dim: int, num_heads: int = 8,
                 mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        # ensure num_heads divides dim
        while dim % num_heads != 0 and num_heads > 1:
            num_heads //= 2
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        z = x.flatten(2).transpose(1, 2)               # (B, HW, C)
        z_n = self.norm1(z)
        attn_out, _ = self.attn(z_n, z_n, z_n, need_weights=False)
        z = z + attn_out
        z = z + self.mlp(self.norm2(z))
        return z.transpose(1, 2).reshape(B, C, H, W)


# ---------------------------------------------------------------------------
# Global Spatial Attention (GSA) — multi-scale decoder fusion
# ---------------------------------------------------------------------------

class _GSA(nn.Module):
    """Fuse decoder features at multiple scales with softmax spatial attention.

    Each input is projected to ``out_channels`` via a 1x1 conv, upsampled
    bilinearly to the largest decoder resolution, concatenated, and the
    concatenation is mapped to per-scale attention logits which are
    softmax-normalised over the scale axis. The output is the
    attention-weighted sum of the projected features.
    """

    def __init__(self, in_channels_list, out_channels: int):
        super().__init__()
        self.n = len(in_channels_list)
        self.projs = nn.ModuleList([
            nn.Conv2d(c, out_channels, kernel_size=1, bias=True)
            for c in in_channels_list
        ])
        self.attn = nn.Sequential(
            nn.Conv2d(out_channels * self.n, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, self.n, kernel_size=1, bias=True),
        )

    def forward(self, features):
        # ``features`` ordered coarse -> fine, fuse at finest resolution.
        target_size = features[-1].shape[-2:]
        projected = []
        for proj, feat in zip(self.projs, features):
            f = proj(feat)
            if f.shape[-2:] != target_size:
                f = F.interpolate(f, size=target_size,
                                  mode="bilinear", align_corners=False)
            projected.append(f)
        concat = torch.cat(projected, dim=1)
        weights = F.softmax(self.attn(concat), dim=1)   # (B, n, H, W)
        out = 0
        for i, f in enumerate(projected):
            out = out + weights[:, i:i + 1] * f
        return out


# ---------------------------------------------------------------------------
# Full TransAttUNet
# ---------------------------------------------------------------------------

class TransAttUNet(nn.Module):
    """TransAttUNet — UNet + Transformer Self-Attention + Global Spatial Attention."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        img_size: int = 224,
        base_features: int = 64,
        tsa_heads: int = 8,
        tsa_mlp_ratio: float = 4.0,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.img_size = img_size

        f = base_features
        # Encoder
        self.inc = _ConvBlock(in_channels, f)
        self.down1 = _DownBlock(f, f * 2)
        self.down2 = _DownBlock(f * 2, f * 4)
        self.down3 = _DownBlock(f * 4, f * 8)
        self.down4 = _DownBlock(f * 8, f * 16)

        # Bottleneck transformer
        self.tsa = _TSA(f * 16, num_heads=tsa_heads, mlp_ratio=tsa_mlp_ratio)

        # Decoder
        self.up1 = _UpBlock(f * 16, f * 8, f * 8)
        self.up2 = _UpBlock(f * 8, f * 4, f * 4)
        self.up3 = _UpBlock(f * 4, f * 2, f * 2)
        self.up4 = _UpBlock(f * 2, f, f)

        # Global Spatial Attention over multi-scale decoder features
        self.gsa = _GSA([f * 8, f * 4, f * 2, f], f)

        # Output head: fuse decoder-finest with GSA, then 1x1 classifier
        self.fuse = nn.Sequential(
            nn.Conv2d(f * 2, f, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(f),
            nn.ReLU(inplace=True),
        )
        self.out_conv = nn.Conv2d(f, num_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]

        # Encoder
        x1 = self.inc(x)        # (B,  f,   H,   W)
        x2 = self.down1(x1)     # (B, 2f,  H/2, W/2)
        x3 = self.down2(x2)     # (B, 4f,  H/4, W/4)
        x4 = self.down3(x3)     # (B, 8f,  H/8, W/8)
        x5 = self.down4(x4)     # (B, 16f, H/16, W/16)

        # Bottleneck transformer self-attention
        x5 = self.tsa(x5)

        # Decoder
        d1 = self.up1(x5, x4)   # (B, 8f,  H/8,  W/8)
        d2 = self.up2(d1, x3)   # (B, 4f,  H/4,  W/4)
        d3 = self.up3(d2, x2)   # (B, 2f,  H/2,  W/2)
        d4 = self.up4(d3, x1)   # (B,  f,   H,    W)

        # Global Spatial Attention across all decoder scales
        gsa = self.gsa([d1, d2, d3, d4])    # (B, f, H, W)

        # Fuse + classify
        merged = torch.cat([d4, gsa], dim=1)
        merged = self.fuse(merged)
        out = self.out_conv(merged)

        if out.shape[-2:] != (H, W):
            out = F.interpolate(out, size=(H, W),
                                mode="bilinear", align_corners=False)
        return out


__all__ = ["TransAttUNet"]
