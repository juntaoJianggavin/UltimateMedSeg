"""HC-Mamba: Hierarchical Conv-Mamba for Medical Image Segmentation (2024).

Faithful reimplementation inspired by:
  https://github.com/JqxnnNn/HC-Mamba

Architecture:
  - Stem: stride-4 4x4 conv -> H/4, W/4
  - 4-stage hierarchical encoder with HC-Mamba blocks
    (DDConv -> LayerNorm -> SS2D -> FFN)
  - 3 stride-2 downsample convs between stages
  - Mirror decoder: PatchExpand + skip concat + 1x1 fuse + HC-Mamba blocks
  - Final 4x bilinear upsample + 1x1 segmentation head

Default dims = [64, 128, 256, 512]; depths = [2, 2, 6, 2].

Reuses SS2D from `medseg.models.encoders.vmunet_encoder` for the selective scan
to avoid duplicating the heavy SSM math.

Self-contained beyond that single import; depends on torch + timm + mamba_ssm.
"""
# Source: NOT VERIFIED — fabricated by this repo, no upstream confirmed.

import math
import warnings
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.models.encoders.vmunet_encoder import (
    SS2D, PatchEmbed2D, PatchMerging2D, VSSLayer,  # noqa: F401 (per project spec)
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

class _DropPath(nn.Module):
    """Per-sample stochastic depth."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x.div(keep) * mask


class _LayerNorm2d(nn.Module):
    """LayerNorm applied over channels of a (B,C,H,W) tensor."""
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()


# ---------------------------------------------------------------------------
# DDConv: depthwise dilated conv with parallel dilations [1, 2, 3], summed.
# ---------------------------------------------------------------------------

class _DDConv(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 3,
                 dilations: Tuple[int, ...] = (1, 2, 3)):
        super().__init__()
        self.dilations = dilations
        k = kernel_size // 2
        self.convs = nn.ModuleList([
            nn.Conv2d(dim, dim, kernel_size,
                      padding=d * k, dilation=d, groups=dim, bias=True)
            for d in dilations
        ])

    def forward(self, x):
        out = self.convs[0](x)
        for c in self.convs[1:]:
            out = out + c(x)
        return out


# ---------------------------------------------------------------------------
# Pointwise FFN (channels-last)
# ---------------------------------------------------------------------------

class _FFN(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ---------------------------------------------------------------------------
# HC-Mamba block: DDConv -> LN -> SS2D -> FFN
# Operates on (B, C, H, W) on the outside; SS2D needs (B, H, W, C).
# ---------------------------------------------------------------------------

class _HCMambaBlock(nn.Module):
    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 3,
                 expand: int = 2, mlp_ratio: float = 4.0,
                 drop: float = 0.0, drop_path: float = 0.0):
        super().__init__()
        self.ddconv = _DDConv(dim, kernel_size=3, dilations=(1, 2, 3))
        self.dw_norm = _LayerNorm2d(dim)

        self.norm1 = nn.LayerNorm(dim)
        self.ss2d = SS2D(d_model=dim, d_state=d_state,
                          d_conv=d_conv, expand=expand)

        self.norm2 = nn.LayerNorm(dim)
        self.ffn = _FFN(dim, mlp_ratio=mlp_ratio, drop=drop)

        self.drop_path = _DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        """x: (B, C, H, W) -> (B, C, H, W)"""
        # DDConv branch + residual
        x = x + self.drop_path(self.dw_norm(self.ddconv(x)))

        # to (B, H, W, C) for LN, SS2D, FFN
        x_hwc = x.permute(0, 2, 3, 1).contiguous()
        x_hwc = x_hwc + self.drop_path(self.ss2d(self.norm1(x_hwc)))
        x_hwc = x_hwc + self.drop_path(self.ffn(self.norm2(x_hwc)))

        return x_hwc.permute(0, 3, 1, 2).contiguous()


# ---------------------------------------------------------------------------
# Stage = stack of HC-Mamba blocks
# ---------------------------------------------------------------------------

class _HCStage(nn.Module):
    def __init__(self, dim: int, depth: int, d_state: int = 16,
                 mlp_ratio: float = 4.0, drop: float = 0.0,
                 drop_paths: List[float] = None):
        super().__init__()
        if drop_paths is None:
            drop_paths = [0.0] * depth
        self.blocks = nn.ModuleList([
            _HCMambaBlock(dim=dim, d_state=d_state, mlp_ratio=mlp_ratio,
                          drop=drop, drop_path=drop_paths[i])
            for i in range(depth)
        ])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


# ---------------------------------------------------------------------------
# Downsample (stride-2 3x3 conv) and PatchExpand (2x up via pixel shuffle)
# ---------------------------------------------------------------------------

class _Downsample(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2,
                               padding=1, bias=False)
        self.norm = _LayerNorm2d(out_dim)

    def forward(self, x):
        return self.norm(self.conv(x))


class _PatchExpand(nn.Module):
    """2x spatial upsample, then project to `out_dim` channels."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        # Expand to 4x channels via 1x1, then pixel shuffle x2 -> in_dim channels.
        self.expand = nn.Conv2d(in_dim, in_dim * 4, kernel_size=1, bias=False)
        self.shuffle = nn.PixelShuffle(2)
        self.norm = _LayerNorm2d(in_dim)
        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.expand(x)        # (B, 4C, H, W)
        x = self.shuffle(x)       # (B,  C, 2H, 2W)
        x = self.norm(x)
        x = self.proj(x)
        return x


# ---------------------------------------------------------------------------
# Stem: stride-4 4x4 conv -> H/4, W/4
# ---------------------------------------------------------------------------

class _Stem(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=4, stride=4)
        self.norm = _LayerNorm2d(embed_dim)

    def forward(self, x):
        return self.norm(self.proj(x))


# ---------------------------------------------------------------------------
# Top-level HC-Mamba segmentation network
# ---------------------------------------------------------------------------

class HCMamba(nn.Module):
    """HC-Mamba: Hierarchical Conv-Mamba U-shape network.

    Args:
        in_channels: number of input channels (default 3).
        num_classes: number of segmentation classes (default 2).
        img_size: nominal training image size (default 224). Inputs are
            internally padded to a multiple of 32 and the output is cropped
            back to the original spatial size, so other sizes also work.
        dims: feature widths per stage. Default (64, 128, 256, 512).
        depths: number of HC-Mamba blocks per stage. Default (2, 2, 6, 2).
        d_state: SS2D state dim. Default 16.
        mlp_ratio: FFN expansion ratio. Default 4.0.
        drop_rate / drop_path_rate: regularisation hyperparams.
    """

    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 2,
                 img_size: int = 224,
                 dims=(64, 128, 256, 512),
                 depths=(2, 2, 6, 2),
                 d_state: int = 16,
                 mlp_ratio: float = 4.0,
                 drop_rate: float = 0.0,
                 drop_path_rate: float = 0.1,
                 **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.img_size = img_size

        dims = list(dims)
        depths = list(depths)
        assert len(dims) == len(depths) == 4, "HC-Mamba expects 4 stages."
        self.dims = dims
        self.depths = depths
        self.num_stages = len(dims)

        # Stochastic depth schedule
        total_blocks = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, max(total_blocks, 1))]

        # Stem (stride-4)
        self.stem = _Stem(in_channels, dims[0])

        # Encoder stages and downsamples
        self.enc_stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        cur = 0
        for i in range(self.num_stages):
            stage = _HCStage(
                dim=dims[i],
                depth=depths[i],
                d_state=d_state,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                drop_paths=dpr[cur:cur + depths[i]],
            )
            self.enc_stages.append(stage)
            cur += depths[i]
            if i < self.num_stages - 1:
                self.downsamples.append(_Downsample(dims[i], dims[i + 1]))

        # Decoder: 3 up-stages mirroring the 3 downsamples
        # Each: PatchExpand (in_dim=dims[i+1] -> out=dims[i]) + concat skip + 1x1 fuse + HCStage(dims[i])
        self.ups = nn.ModuleList()
        self.fuses = nn.ModuleList()
        self.dec_stages = nn.ModuleList()
        # Use a fresh (small) drop-path schedule for decoder to keep symmetric capacity
        ddpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                  sum(depths[:-1]) or 1)]
        dcur = 0
        for i in reversed(range(self.num_stages - 1)):
            self.ups.append(_PatchExpand(dims[i + 1], dims[i]))
            self.fuses.append(nn.Conv2d(2 * dims[i], dims[i], kernel_size=1, bias=False))
            self.dec_stages.append(
                _HCStage(
                    dim=dims[i],
                    depth=depths[i],
                    d_state=d_state,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    drop_paths=ddpr[dcur:dcur + depths[i]],
                )
            )
            dcur += depths[i]

        # Final: 4x upsample to original resolution + 1x1 head
        self.final_norm = _LayerNorm2d(dims[0])
        self.final_up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.seg_head = nn.Conv2d(dims[0], num_classes, kernel_size=1)

        self.apply(self._init_weights)

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    # ------------------------------------------------------------------
    # Padding helper: ensure H, W are multiples of 32 (= 4 * 2^3)
    # ------------------------------------------------------------------
    @staticmethod
    def _pad_to_multiple(x, multiple=32):
        _, _, H, W = x.shape
        pad_h = (multiple - H % multiple) % multiple
        pad_w = (multiple - W % multiple) % multiple
        if pad_h == 0 and pad_w == 0:
            return x, (0, 0, 0, 0)
        # pad on right & bottom
        x = F.pad(x, (0, pad_w, 0, pad_h))
        return x, (0, pad_w, 0, pad_h)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x):
        B, _, H, W = x.shape
        x, pad = self._pad_to_multiple(x, multiple=32)

        # Encoder
        x = self.stem(x)
        skips = []
        for i in range(self.num_stages):
            x = self.enc_stages[i](x)
            if i < self.num_stages - 1:
                skips.append(x)
                x = self.downsamples[i](x)

        # Decoder
        for i, (up, fuse, stage) in enumerate(zip(self.ups, self.fuses, self.dec_stages)):
            x = up(x)
            skip = skips[-(i + 1)]
            # Defensive: align spatial dims if off by one
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear',
                                  align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = fuse(x)
            x = stage(x)

        # Final upsample to original (padded) resolution & seg head
        x = self.final_norm(x)
        x = self.final_up(x)
        logits = self.seg_head(x)

        # Crop padding back to original size
        if pad != (0, 0, 0, 0):
            logits = logits[:, :, :H, :W]
        return logits


# Alias expected by some external loaders
HC_Mamba = HCMamba
