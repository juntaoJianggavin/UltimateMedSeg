"""U-RWKV: Medical Image Segmentation with RWKV Attention Mechanism.

Reimplementation based on:
  https://github.com/hbyecoding/U-RWKV  (MICCAI 2025)

Architecture: Conv encoder with RWKV spatial/channel mixing at each stage,
multi-direction scan (q_shift), SE-enhanced fusion decoder.

WKV is computed by the unified dispatcher in :mod:`medseg.kernels.wkv` so this
architecture automatically uses the official Vision-RWKV CUDA op when running
on a GPU and falls back to a vectorised PyTorch implementation otherwise. Both
paths are autograd-differentiable.
"""
# Source: https://github.com/hbyecoding/U-RWKV

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from medseg.kernels.wkv import run_wkv as _run_wkv


# ---------------------------------------------------------------------------
# WKV computation (CUDA-accelerated when available, pure PyTorch otherwise)
# ---------------------------------------------------------------------------

def wkv_pytorch(B, T, C, w, u, k, v):
    """Backward-compatible WKV entry-point used inside this file.

    Forwards to :func:`medseg.kernels.wkv.run_wkv` which dispatches to the
    JIT-compiled CUDA op on GPU and to a vectorised PyTorch fallback on CPU.
    Both code paths are autograd-differentiable; the previous in-file loop
    that ran on Python (and could not benefit from CUDA acceleration) has
    been replaced.
    """
    return _run_wkv(B, T, C, w, u, k, v)


# ---------------------------------------------------------------------------
# Q-Shift: spatial token shifting for multi-direction information flow
# ---------------------------------------------------------------------------

def q_shift(x, shift_pixel=1, gamma=0.25):
    """Shift tokens in 4 directions for spatial mixing.

    x: (B, N, C) where N = H*W. Shifts C/4 channels in each direction.
    """
    B, N, C = x.shape
    H = W = int(math.sqrt(N))
    assert H * W == N, f"q_shift requires square feature maps, got N={N}"

    x = x.transpose(1, 2).reshape(B, C, H, W)
    out = torch.zeros_like(x)
    g = int(C * gamma)

    # Shift right
    out[:, 0:g, :, shift_pixel:W] = x[:, 0:g, :, 0:W - shift_pixel]
    # Shift left
    out[:, g:2*g, :, 0:W - shift_pixel] = x[:, g:2*g, :, shift_pixel:W]
    # Shift down
    out[:, 2*g:3*g, shift_pixel:H, :] = x[:, 2*g:3*g, 0:H - shift_pixel, :]
    # Shift up
    out[:, 3*g:4*g, 0:H - shift_pixel, :] = x[:, 3*g:4*g, shift_pixel:H, :]
    # No shift
    out[:, 4*g:, ...] = x[:, 4*g:, ...]

    return out.flatten(2).transpose(1, 2)


# ---------------------------------------------------------------------------
# RWKV Spatial Mix and Channel Mix
# ---------------------------------------------------------------------------

class SpatialMix(nn.Module):
    """RWKV Spatial Mixing (time mixing adapted for 2D).

    Uses WKV attention with learnable decay and first-token bias.
    """
    def __init__(self, n_embd, n_layer, layer_id, shift_pixel=1, key_norm=True):
        super().__init__()
        self.n_embd = n_embd
        self.layer_id = layer_id

        # Learnable parameters
        ratio_0_to_1 = layer_id / max(n_layer - 1, 1)
        ratio_1_to_0 = 1.0 - layer_id / max(n_layer, 1)

        decay_speed = torch.ones(n_embd)
        for h in range(n_embd):
            decay_speed[h] = -5 + 8 * (h / max(n_embd - 1, 1)) ** (
                0.7 + 1.3 * ratio_0_to_1)
        self.spatial_decay = nn.Parameter(decay_speed)

        zigzag = torch.tensor(
            [(i + 1) % 3 - 1 for i in range(n_embd)], dtype=torch.float32) * 0.5
        self.spatial_first = nn.Parameter(
            torch.ones(n_embd) * math.log(0.3) + zigzag)

        # Mixing coefficients
        x = torch.ones(1, 1, n_embd)
        for i in range(n_embd):
            x[0, 0, i] = i / n_embd
        self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_0))
        self.spatial_mix_v = nn.Parameter(
            torch.pow(x, ratio_1_to_0) + 0.3 * ratio_0_to_1)
        self.spatial_mix_r = nn.Parameter(
            torch.pow(x, 0.5 * ratio_1_to_0))

        self.shift_pixel = shift_pixel

        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.output = nn.Linear(n_embd, n_embd, bias=False)

        if key_norm:
            self.key_norm = nn.LayerNorm(n_embd)
        else:
            self.key_norm = None

    def forward(self, x):
        B, T, C = x.shape

        if self.shift_pixel > 0:
            x_shifted = q_shift(x, self.shift_pixel)
        else:
            x_shifted = x

        xk = x * self.spatial_mix_k + x_shifted * (1 - self.spatial_mix_k)
        xv = x * self.spatial_mix_v + x_shifted * (1 - self.spatial_mix_v)
        xr = x * self.spatial_mix_r + x_shifted * (1 - self.spatial_mix_r)

        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)

        if self.key_norm is not None:
            k = self.key_norm(k)

        sr = torch.sigmoid(r)

        rwkv = wkv_pytorch(
            B, T, C,
            self.spatial_decay.float(),
            self.spatial_first.float(),
            k.float(), v.float()
        ).to(x.dtype)

        return self.output(sr * rwkv)


class ChannelMix(nn.Module):
    """RWKV Channel Mixing (feed-forward with gating).

    Uses squared ReLU activation (ReLU^2) for channel mixing.
    """
    def __init__(self, n_embd, n_layer, layer_id, hidden_ratio=4,
                 shift_pixel=1):
        super().__init__()
        self.n_embd = n_embd
        hidden = int(n_embd * hidden_ratio)

        ratio_1_to_0 = 1.0 - layer_id / max(n_layer, 1)
        x = torch.ones(1, 1, n_embd)
        for i in range(n_embd):
            x[0, 0, i] = i / n_embd
        self.channel_mix_k = nn.Parameter(
            torch.pow(x, ratio_1_to_0))
        self.channel_mix_r = nn.Parameter(
            torch.pow(x, ratio_1_to_0))

        self.shift_pixel = shift_pixel

        self.key = nn.Linear(n_embd, hidden, bias=False)
        self.value = nn.Linear(hidden, n_embd, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x):
        if self.shift_pixel > 0:
            x_shifted = q_shift(x, self.shift_pixel)
        else:
            x_shifted = x

        xk = x * self.channel_mix_k + x_shifted * (1 - self.channel_mix_k)
        xr = x * self.channel_mix_r + x_shifted * (1 - self.channel_mix_r)

        k = self.key(xk)
        k = torch.relu(k) ** 2  # squared ReLU
        kv = self.value(k)

        return torch.sigmoid(self.receptance(xr)) * kv


# ---------------------------------------------------------------------------
# RWKV Block (Spatial Mix + Channel Mix)
# ---------------------------------------------------------------------------

class RWKVBlock(nn.Module):
    """RWKV block: LayerNorm → SpatialMix → LayerNorm → ChannelMix."""
    def __init__(self, n_embd, n_layer, layer_id, shift_pixel=1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.spatial = SpatialMix(n_embd, n_layer, layer_id,
                                   shift_pixel=shift_pixel)
        self.channel = ChannelMix(n_embd, n_layer, layer_id,
                                   shift_pixel=shift_pixel)

    def forward(self, x):
        x = x + self.spatial(self.ln1(x))
        x = x + self.channel(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# SE Module
# ---------------------------------------------------------------------------

class SE(nn.Module):
    """Squeeze-and-Excitation block."""
    def __init__(self, channels, rd_ratio=0.25):
        super().__init__()
        rd = max(int(channels * rd_ratio), 1)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, rd, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(rd, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x).unsqueeze(-1).unsqueeze(-1)


# ---------------------------------------------------------------------------
# Conv Encoder Stage with RWKV
# ---------------------------------------------------------------------------

class ConvRWKVStage(nn.Module):
    """Conv block followed by RWKV blocks for one encoder stage."""
    def __init__(self, in_ch, out_ch, n_rwkv_layers, total_layers,
                 layer_offset, stride=2, shift_pixel=1):
        super().__init__()
        # Downsampling conv
        if stride > 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        # RWKV blocks
        self.rwkv_blocks = nn.ModuleList()
        for i in range(n_rwkv_layers):
            self.rwkv_blocks.append(
                RWKVBlock(out_ch, total_layers, layer_offset + i,
                          shift_pixel=shift_pixel))

    def forward(self, x):
        x = self.downsample(x)
        B, C, H, W = x.shape
        x_seq = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        for blk in self.rwkv_blocks:
            x_seq = blk(x_seq)
        return x_seq.transpose(1, 2).reshape(B, C, H, W)


# ---------------------------------------------------------------------------
# Fusion Decoder with SE
# ---------------------------------------------------------------------------

class FusionUpBlock(nn.Module):
    """Decoder block: upsample + concat skip + Conv + SE."""
    def __init__(self, in_ch, skip_ch, out_ch, se_ratio=0.25):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear',
                               align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.se = SE(out_ch, rd_ratio=se_ratio) if se_ratio > 0 else nn.Identity()

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear',
                              align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.se(x)
        return x


# ---------------------------------------------------------------------------
# U-RWKV
# ---------------------------------------------------------------------------

class URWKV(nn.Module):
    """U-RWKV: UNet with RWKV attention at each encoder stage.

    Architecture:
      - Stem: Conv → BN → ReLU
      - 4 encoder stages with ConvRWKV (downsample + RWKV blocks)
      - 4 decoder stages with FusionUpBlock (upsample + concat + SE)
      - 1x1 conv head

    Args:
        in_channels: Input channels.
        num_classes: Output classes.
        img_size: Input image size (must be divisible by 16).
        embed_dims: Channel dimensions for each encoder stage.
        depths: Number of RWKV layers per encoder stage.
        shift_pixel: Pixel shift for q_shift spatial mixing.
        se_ratio: Squeeze-and-excitation ratio in decoder.
    """
    def __init__(self, in_channels=3, num_classes=2, img_size=224,
                 embed_dims=None, depths=None, shift_pixel=1,
                 se_ratio=0.25, deep_supervision=False, **kwargs):
        super().__init__()
        if embed_dims is None:
            embed_dims = [64, 128, 256, 512]
        if depths is None:
            depths = [2, 2, 2, 2]
        self.deep_supervision = deep_supervision
        self._embed_dims = embed_dims

        total_layers = sum(depths)

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, embed_dims[0], 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(embed_dims[0]),
            nn.ReLU(inplace=True),
        )

        # Encoder stages
        self.enc_stages = nn.ModuleList()
        layer_offset = 0
        for i in range(len(embed_dims)):
            in_ch = embed_dims[0] if i == 0 else embed_dims[i - 1]
            stride = 1 if i == 0 else 2
            self.enc_stages.append(ConvRWKVStage(
                in_ch, embed_dims[i], depths[i], total_layers,
                layer_offset, stride=stride, shift_pixel=shift_pixel))
            layer_offset += depths[i]

        # Decoder stages
        self.dec_stages = nn.ModuleList()
        for i in range(len(embed_dims) - 1):
            dec_in = embed_dims[-(i + 1)]
            skip_ch = embed_dims[-(i + 2)]
            dec_out = skip_ch
            self.dec_stages.append(
                FusionUpBlock(dec_in, skip_ch, dec_out, se_ratio=se_ratio))

        # Final upsample (stem did 2x downsample)
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(embed_dims[0], embed_dims[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dims[0]),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(embed_dims[0], num_classes, 1)

        if deep_supervision:
            self.ds_heads = nn.ModuleList([
                nn.Conv2d(embed_dims[i], num_classes, 1)
                for i in range(len(embed_dims) - 2, -1, -1)
            ][:len(embed_dims) - 2])  # exclude last (shallowest = main output level)

    def forward(self, x):
        input_size = x.shape[2:]

        # Stem
        x = self.stem(x)

        # Encoder
        enc_feats = []
        for stage in self.enc_stages:
            x = stage(x)
            enc_feats.append(x)

        # Decoder
        x = enc_feats[-1]
        ds_collect = self.training and self.deep_supervision
        intermediates = []
        for i, dec in enumerate(self.dec_stages):
            skip = enc_feats[-(i + 2)]
            x = dec(x, skip)
            if ds_collect and i < len(self.dec_stages) - 1:
                intermediates.append(x)

        # Final upsample to input resolution
        x = self.final_up(x)
        x = self.head(x)

        if x.shape[2:] != input_size:
            x = F.interpolate(x, size=input_size, mode='bilinear',
                              align_corners=False)

        if ds_collect:
            aux = []
            for feat, head in zip(intermediates, self.ds_heads):
                a = head(feat)
                if a.shape[2:] != input_size:
                    a = F.interpolate(a, size=input_size, mode='bilinear',
                                      align_corners=False)
                aux.append(a)
            return [x] + aux
        return x
