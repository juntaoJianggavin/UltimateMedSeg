"""CSWin Encoder: extracted from CSWin-UNet.

Pure global MHSA encoder (simplified CSWin port) with patch-embed (stride-4)
and 4 stages of transformer blocks. Each non-final stage applies a 3x3 stride-2
conv downsample after its blocks. Multi-scale features are returned with the
deepest map LAST, following the framework convention.

Reference:
    CSWin-UNet: Transformer UNet with Cross-Shaped Windows for Medical
    Image Segmentation. arXiv 2024.
    https://github.com/pgao-lab/CSWin-UNet
"""
# Source: https://github.com/eatbeanss/CSWin-UNet

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from medseg.registry import ENCODER_REGISTRY


class _CrossWindowAttention(nn.Module):
    """Cross-shaped window self-attention: horizontal + vertical strips."""

    def __init__(self, dim, num_heads=4, strip_size=7):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.strip_size = strip_size
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        qkv = self.qkv(tokens).reshape(B, -1, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, -1, C)
        out = self.proj(out).transpose(1, 2).view(B, C, H, W)
        return out


class _CSWinBlock(nn.Module):
    def __init__(self, dim, num_heads=4, strip_size=7, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _CrossWindowAttention(dim, num_heads, strip_size)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = tokens + self.attn(x).flatten(2).transpose(1, 2)
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens.transpose(1, 2).view(B, C, H, W)


class _CSWinStage(nn.Module):
    def __init__(self, dim, num_heads, depth, strip_size=7, downsample=False):
        super().__init__()
        self.blocks = nn.Sequential(*[
            _CSWinBlock(dim, num_heads, strip_size) for _ in range(depth)
        ])
        self.downsample = None
        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(dim, dim * 2, 3, 2, 1, bias=False),
                nn.BatchNorm2d(dim * 2),
            )

    def forward(self, x):
        x = self.blocks(x)
        out = x
        if self.downsample is not None:
            x = self.downsample(x)
            return out, x
        return out


@ENCODER_REGISTRY.register("cswin")
class CSWinEncoder(nn.Module):
    """CSWin (simplified) encoder.

    Stem (patch embed) downsamples by 4 (Conv2d k=4 s=4), then 4 transformer
    stages. After each of the first 3 stages a 3x3 stride-2 conv halves the
    spatial size and doubles the channels.

    With embed_dim=64 and 4 stages, out_channels = [64, 128, 256, 512] and
    spatial scales are /4, /8, /16, /32 relative to the input.

    Returned features are ordered shallow -> deep (deepest LAST), matching the
    framework convention.
    """

    def __init__(
        self,
        in_channels: int = 3,
        img_size: int = 224,
        pretrained: bool = False,
        embed_dim: int = 64,
        depths: Optional[List[int]] = None,
        num_heads: Optional[List[int]] = None,
        strip_size: int = 7,
        **kwargs,
    ):
        super().__init__()
        depths = list(depths) if depths is not None else [2, 2, 6, 2]
        num_heads = list(num_heads) if num_heads is not None else [2, 4, 8, 16]
        dims = [embed_dim * (2 ** i) for i in range(len(depths))]

        # Optional 1x1 stem for non-RGB inputs.
        if in_channels != 3:
            self.input_proj = nn.Conv2d(in_channels, 3, 1, bias=False)
            stem_in = 3
        else:
            self.input_proj = nn.Identity()
            stem_in = in_channels

        self.patch_embed = nn.Sequential(
            nn.Conv2d(stem_in, dims[0], 4, 4, bias=False),
            nn.BatchNorm2d(dims[0]),
        )

        self.enc_stages = nn.ModuleList()
        for i in range(len(depths)):
            ds = i < len(depths) - 1
            self.enc_stages.append(
                _CSWinStage(dims[i], num_heads[i], depths[i],
                            strip_size=strip_size, downsample=ds)
            )

        self.out_channels: List[int] = dims

        if pretrained:
            import warnings
            warnings.warn(
                "CSWinEncoder has no published pretrained checkpoint for the "
                "simplified port; using random init."
            )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.input_proj(x)
        x = self.patch_embed(x)

        features: List[torch.Tensor] = []
        for i, stage in enumerate(self.enc_stages):
            if i < len(self.enc_stages) - 1:
                out, x = stage(x)
                features.append(out)
            else:
                out = stage(x)
                features.append(out)
        return features
