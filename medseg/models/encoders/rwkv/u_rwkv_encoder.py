"""U-RWKV encoder: stride-2 stem + 4 ConvRWKV stages (Stage0 stride 1, Stages 1-3 stride 2).

Extracted from ``medseg.models.networks.rwkv.u_rwkv`` (U-RWKV, MICCAI 2025
reimplementation). Wraps the Conv+RWKV trunk and exposes the 4 multi-scale
feature maps (deepest last) consumed by a decoder.

Each stage uses:
  - Downsample conv (stride 2 except Stage 0 which is stride 1)
  - ``n`` RWKV blocks: LayerNorm -> SpatialMix (q_shift + WKV) -> LayerNorm -> ChannelMix

WKV is dispatched through :func:`medseg.kernels.wkv.run_wkv` (CUDA op on GPU,
pure-PyTorch fallback on CPU).
"""
# Source: https://github.com/hbyecoding/U-RWKV

import math
from typing import List

import torch
import torch.nn as nn

from medseg.registry import ENCODER_REGISTRY
from medseg.kernels.wkv import run_wkv as _run_wkv


# ---------------------------------------------------------------------------
# WKV entry point
# ---------------------------------------------------------------------------

def _wkv(B, T, C, w, u, k, v):
    return _run_wkv(B, T, C, w, u, k, v)


# ---------------------------------------------------------------------------
# Q-Shift: spatial token shifting (4 directions + identity), resolution-friendly
# ---------------------------------------------------------------------------

def _q_shift(x, H, W, shift_pixel=1, gamma=0.25):
    """Shift tokens in 4 directions for spatial mixing.

    x: (B, N, C) where N = H*W. Uses runtime (H, W) -- no baked size.
    """
    B, N, C = x.shape
    assert H * W == N, f"_q_shift: H*W ({H}*{W}) != N ({N})"

    x = x.transpose(1, 2).reshape(B, C, H, W)
    out = torch.zeros_like(x)
    g = int(C * gamma)

    # Shift right
    out[:, 0:g, :, shift_pixel:W] = x[:, 0:g, :, 0:W - shift_pixel]
    # Shift left
    out[:, g:2 * g, :, 0:W - shift_pixel] = x[:, g:2 * g, :, shift_pixel:W]
    # Shift down
    out[:, 2 * g:3 * g, shift_pixel:H, :] = x[:, 2 * g:3 * g, 0:H - shift_pixel, :]
    # Shift up
    out[:, 3 * g:4 * g, 0:H - shift_pixel, :] = x[:, 3 * g:4 * g, shift_pixel:H, :]
    # Identity tail
    out[:, 4 * g:, ...] = x[:, 4 * g:, ...]

    return out.flatten(2).transpose(1, 2)


# ---------------------------------------------------------------------------
# RWKV Spatial Mix (time mixing adapted for 2D)
# ---------------------------------------------------------------------------

class _SpatialMix(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, shift_pixel=1, key_norm=True):
        super().__init__()
        self.n_embd = n_embd
        self.layer_id = layer_id

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

        self.key_norm = nn.LayerNorm(n_embd) if key_norm else None

    def forward(self, x, H, W):
        B, T, C = x.shape

        if self.shift_pixel > 0:
            x_shifted = _q_shift(x, H, W, self.shift_pixel)
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

        rwkv = _wkv(
            B, T, C,
            self.spatial_decay.float(),
            self.spatial_first.float(),
            k.float(), v.float()
        ).to(x.dtype)

        return self.output(sr * rwkv)


# ---------------------------------------------------------------------------
# RWKV Channel Mix (feed-forward with gating)
# ---------------------------------------------------------------------------

class _ChannelMix(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, hidden_ratio=4, shift_pixel=1):
        super().__init__()
        self.n_embd = n_embd
        hidden = int(n_embd * hidden_ratio)

        ratio_1_to_0 = 1.0 - layer_id / max(n_layer, 1)
        x = torch.ones(1, 1, n_embd)
        for i in range(n_embd):
            x[0, 0, i] = i / n_embd
        self.channel_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_0))
        self.channel_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_0))

        self.shift_pixel = shift_pixel

        self.key = nn.Linear(n_embd, hidden, bias=False)
        self.value = nn.Linear(hidden, n_embd, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x, H, W):
        if self.shift_pixel > 0:
            x_shifted = _q_shift(x, H, W, self.shift_pixel)
        else:
            x_shifted = x

        xk = x * self.channel_mix_k + x_shifted * (1 - self.channel_mix_k)
        xr = x * self.channel_mix_r + x_shifted * (1 - self.channel_mix_r)

        k = self.key(xk)
        k = torch.relu(k) ** 2  # squared ReLU
        kv = self.value(k)

        return torch.sigmoid(self.receptance(xr)) * kv


# ---------------------------------------------------------------------------
# RWKV Block (LayerNorm -> SpatialMix -> LayerNorm -> ChannelMix)
# ---------------------------------------------------------------------------

class _RWKVBlock(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, shift_pixel=1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.spatial = _SpatialMix(n_embd, n_layer, layer_id,
                                   shift_pixel=shift_pixel)
        self.channel = _ChannelMix(n_embd, n_layer, layer_id,
                                   shift_pixel=shift_pixel)

    def forward(self, x, H, W):
        x = x + self.spatial(self.ln1(x), H, W)
        x = x + self.channel(self.ln2(x), H, W)
        return x


# ---------------------------------------------------------------------------
# Conv encoder stage with RWKV blocks
# ---------------------------------------------------------------------------

class _ConvRWKVStage(nn.Module):
    def __init__(self, in_ch, out_ch, n_rwkv_layers, total_layers,
                 layer_offset, stride=2, shift_pixel=1):
        super().__init__()
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

        self.rwkv_blocks = nn.ModuleList([
            _RWKVBlock(out_ch, total_layers, layer_offset + i,
                       shift_pixel=shift_pixel)
            for i in range(n_rwkv_layers)
        ])

    def forward(self, x):
        x = self.downsample(x)
        B, C, H, W = x.shape
        x_seq = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        for blk in self.rwkv_blocks:
            x_seq = blk(x_seq, H, W)
        return x_seq.transpose(1, 2).reshape(B, C, H, W)


# ---------------------------------------------------------------------------
# U-RWKV Encoder
# ---------------------------------------------------------------------------

@ENCODER_REGISTRY.register("u_rwkv")
class URWKVEncoder(nn.Module):
    """U-RWKV encoder: 7x7 stride-2 stem + 4 ConvRWKV stages.

    Stage strides: [1, 2, 2, 2] -> output strides relative to input are
    [2, 4, 8, 16] (stem contributes one 2x downsample).

    Default ``embed_dims = [64, 128, 256, 512]`` and ``depths = [2, 2, 2, 2]``.
    The 4 returned feature maps are ordered shallowest-first / deepest-last,
    matching the framework convention.

    Args:
        in_channels: Input channels. If != 3, a 1x1 conv prepends to remap to
            3 channels so the rest of the network can stay unchanged.
        img_size: Nominal input image size (unused at runtime -- spatial state
            is always taken from the live tensor shape).
        pretrained: Accepted for interface uniformity; this encoder has no
            published pretrained weights.
        embed_dims / depths / shift_pixel: Architectural knobs forwarded to the
            internal Conv+RWKV stages.
    """

    def __init__(self, in_channels: int = 3, img_size: int = 224,
                 pretrained: bool = False,
                 embed_dims: List[int] = None, depths: List[int] = None,
                 shift_pixel: int = 1, **kwargs):
        super().__init__()
        if embed_dims is None:
            embed_dims = [64, 128, 256, 512]
        if depths is None:
            depths = [2, 2, 2, 2]
        assert len(embed_dims) == len(depths), \
            "embed_dims and depths must have equal length"

        self.in_channels = in_channels
        self.img_size = img_size
        self._embed_dims = list(embed_dims)
        self._depths = list(depths)

        # Optional 1x1 channel-remap when the dataset is not RGB.
        if in_channels != 3:
            self.input_proj = nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)
            stem_in = 3
        else:
            self.input_proj = nn.Identity()
            stem_in = in_channels

        total_layers = sum(depths)

        # Stem: 7x7 stride-2 conv
        self.stem = nn.Sequential(
            nn.Conv2d(stem_in, embed_dims[0], 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(embed_dims[0]),
            nn.ReLU(inplace=True),
        )

        # 4 encoder stages: Stage 0 stride 1, Stages 1..3 stride 2
        self.enc_stages = nn.ModuleList()
        layer_offset = 0
        for i in range(len(embed_dims)):
            in_ch = embed_dims[0] if i == 0 else embed_dims[i - 1]
            stride = 1 if i == 0 else 2
            self.enc_stages.append(_ConvRWKVStage(
                in_ch, embed_dims[i], depths[i], total_layers,
                layer_offset, stride=stride, shift_pixel=shift_pixel))
            layer_offset += depths[i]

        # Multi-scale channel dims (deepest LAST, framework convention).
        self.out_channels = list(embed_dims)

        if pretrained:
            import warnings
            warnings.warn(
                "URWKVEncoder has no published pretrained weights; "
                "using random init.")

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.input_proj(x)
        x = self.stem(x)
        feats: List[torch.Tensor] = []
        for stage in self.enc_stages:
            x = stage(x)
            feats.append(x)
        return feats
