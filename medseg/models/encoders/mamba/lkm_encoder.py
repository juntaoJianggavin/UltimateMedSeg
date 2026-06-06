"""LKM-UNet encoder (LKMUNetEncoder).

Standalone encoder extracted from ``medseg.models.networks.mamba.lkm_unet``.

Architecture (from the source network):
    - stride-4 patch-embed stem (Conv2d k=4 s=4)
    - 4 encoder stages, each a stack of large-kernel + gated-SSM-proxy blocks
    - 3 strided 3x3 convolutions for stride 2x downsampling between stages

Default config: depths=[2, 2, 6, 2], dims=[64, 128, 256, 512]
Feature pyramid strides: 4 / 8 / 16 / 32. Deepest map LAST.
"""
# Source: https://github.com/wjh892521292/LKM-UNet

import ssl
import warnings
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.registry import ENCODER_REGISTRY


def _load_with_ssl_fallback(load_fn, *args, **kwargs):
    """Try a download/load, falling back to unverified SSL, then random init."""
    try:
        return load_fn(*args, **kwargs)
    except Exception as e1:
        prev = ssl._create_default_https_context
        try:
            ssl._create_default_https_context = ssl._create_unverified_context
            return load_fn(*args, **kwargs)
        except Exception as e2:
            warnings.warn(
                f"Pretrained download failed ({e2}); using random init."
            )
            kwargs2 = {**kwargs, 'pretrained': False}
            return load_fn(*args, **kwargs2)
        finally:
            ssl._create_default_https_context = prev


class _LargeKernelMambaBlock(nn.Module):
    """Large-kernel depthwise conv + gated SSM-proxy + FFN, with two residuals.

    Copied (with underscore prefix) from the source LKM-UNet network so that
    this encoder is fully self-contained.
    """

    def __init__(self, dim: int, kernel_size: int = 15, d_state: int = 16):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        pad = kernel_size // 2
        self.lk_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size, 1, pad, groups=dim),
            nn.BatchNorm2d(dim),
            nn.SiLU(),
        )
        self.proj = nn.Linear(dim, dim * 2)
        self.gate_fn = nn.Sigmoid()
        self.out_proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        res = x
        x_lk = self.lk_conv(x)
        tokens = x_lk.flatten(2).transpose(1, 2)  # (B, HW, C)
        kv = self.proj(self.norm(tokens))
        k, v = kv.chunk(2, dim=-1)
        out = self.out_proj(self.gate_fn(k) * v)
        tokens = tokens + out
        tokens = tokens + self.ffn(self.norm2(tokens))
        return tokens.transpose(1, 2).view(B, C, H, W) + res


@ENCODER_REGISTRY.register("lkm")
class LKMUNetEncoder(nn.Module):
    """LKM-UNet encoder: stride-4 stem + 3 strided downsamples + 4 LKM stages.

    Returns a 4-level feature pyramid in (B, C, H, W) format at strides
    4 / 8 / 16 / 32. The deepest (stride-32) feature is LAST.

    Args:
        in_channels: number of input image channels. If != 3, a 1x1 conv stem
            is prepended to map to 3 channels before the patch-embed.
        img_size: nominal input resolution (informational only, not baked).
        pretrained: kept for interface parity; the original LKM-UNet does not
            ship official pretrained weights, so this is a best-effort no-op.
        embed_dim: base channel width (default 64).
        depths: blocks per stage (default [2, 2, 6, 2]).
        kernel_size: large-kernel size used inside every block.
        d_state: SSM state dim (proxy, kept for parity with the source block).
        pretrained_path: optional local checkpoint path.
    """

    def __init__(
        self,
        in_channels: int = 3,
        img_size: int = 224,
        pretrained: bool = False,
        embed_dim: int = 64,
        depths: Optional[List[int]] = None,
        kernel_size: int = 15,
        d_state: int = 16,
        pretrained_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        if depths is None:
            depths = [2, 2, 6, 2]
        assert len(depths) == 4, "LKMUNetEncoder expects 4 stages."

        self.in_channels = in_channels
        self.img_size = img_size
        self.depths = list(depths)
        dims = [embed_dim * (2 ** i) for i in range(len(depths))]
        self.dims = dims
        self.out_channels: List[int] = list(dims)

        # Optional 1x1 stem if caller provides non-RGB input.
        if in_channels != 3:
            self.input_stem = nn.Conv2d(in_channels, 3, kernel_size=1, bias=True)
            stem_in = 3
        else:
            self.input_stem = nn.Identity()
            stem_in = in_channels

        # Stride-4 patch-embed stem.
        self.stem = nn.Sequential(
            nn.Conv2d(stem_in, dims[0], kernel_size=4, stride=4, bias=False),
            nn.BatchNorm2d(dims[0]),
        )

        # Encoder stages and inter-stage 2x strided downsamples.
        self.enc = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i in range(len(depths)):
            self.enc.append(nn.Sequential(*[
                _LargeKernelMambaBlock(dims[i], kernel_size=kernel_size, d_state=d_state)
                for _ in range(depths[i])
            ]))
            if i < len(depths) - 1:
                self.downs.append(nn.Conv2d(dims[i], dims[i + 1], 3, 2, 1))

        if pretrained:
            _load_with_ssl_fallback(self._maybe_load_pretrained, pretrained_path)

    # ---- Pretrained loading ---------------------------------------------------

    def _maybe_load_pretrained(self, pretrained_path: Optional[str] = None, **_):
        """Best-effort local checkpoint load. No-op without a path."""
        if not pretrained_path:
            warnings.warn(
                "LKMUNetEncoder: no pretrained_path provided; using random init."
            )
            return
        state = torch.load(pretrained_path, map_location='cpu')
        if isinstance(state, dict):
            if 'model' in state:
                state = state['model']
            if 'state_dict' in state:
                state = state['state_dict']
        cleaned = {}
        for k, v in state.items():
            nk = k
            for pfx in ('encoder.', 'backbone.', 'module.'):
                if nk.startswith(pfx):
                    nk = nk[len(pfx):]
            cleaned[nk] = v
        msg = self.load_state_dict(cleaned, strict=False)
        print(f"LKMUNetEncoder loaded pretrained from {pretrained_path}: {msg}")

    # ---- Forward --------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Args:
            x: (B, in_channels, H, W).
        Returns:
            List of 4 feature maps (B, C, H, W) at strides 4 / 8 / 16 / 32.
            Deepest map LAST.
        """
        x = self.input_stem(x)
        x = self.stem(x)  # stride-4

        features: List[torch.Tensor] = []
        for i, enc in enumerate(self.enc):
            x = enc(x)
            features.append(x)
            if i < len(self.downs):
                x = self.downs[i](x)
        return features
