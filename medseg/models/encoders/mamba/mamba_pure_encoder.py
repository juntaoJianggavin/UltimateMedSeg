"""Pure 2D Mamba-vision Encoder (MambaPureEncoder).

A representative pure-Mamba vision backbone built from the SS2D/VSSBlock
primitives shared with VM-UNet. The architecture mirrors VMamba-T:
    depths = [2, 2, 9, 2]
    dims   = [96, 192, 384, 768]

Produces a 4-level feature pyramid at strides 4 / 8 / 16 / 32.
"""
# Source: UNCHECKED — please verify

import ssl
import warnings
from typing import List, Optional

import torch
import torch.nn as nn

from medseg.registry import ENCODER_REGISTRY
from .vmunet_encoder import (
    SS2D,
    VSSBlock,
    PatchEmbed2D,
    PatchMerging2D,
)


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
            warnings.warn(f"Pretrained download failed ({e2}); using random init.")
            kwargs2 = {**kwargs, 'pretrained': False}
            return load_fn(*args, **kwargs2)
        finally:
            ssl._create_default_https_context = prev


class _MambaStage(nn.Module):
    """One stage: a stack of VSSBlocks + optional 2x patch-merging downsample.

    Operates on (B, H, W, C) tensors (channels-last) as used by VSSBlock/SS2D.
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        drop_path_rates: List[float],
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 2,
        downsample: bool = True,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                drop_path=drop_path_rates[i],
            )
            for i in range(depth)
        ])
        self.downsample = PatchMerging2D(dim) if downsample else None

    def forward(self, x: torch.Tensor):
        """x: (B, H, W, C). Returns (stage_feat_before_down, x_for_next_stage)."""
        for blk in self.blocks:
            x = blk(x)
        feat = x
        if self.downsample is not None:
            x = self.downsample(x)
        return feat, x


@ENCODER_REGISTRY.register("mambavision")
class MambaPureEncoder(nn.Module):
    """Pure 2D Mamba-vision encoder (VMamba-T config).

    4-stage hierarchical encoder of SS2D-based VSSBlocks. Returns a list of
    feature maps in (B, C, H, W) format with the deepest map LAST.

    Args:
        in_channels: input image channels. If != 3, a 1x1 conv stem maps to 3.
        img_size: nominal input resolution (informational only).
        pretrained: attempt to load a pretrained checkpoint if available.
        patch_size: patch-embed stride (default 4 -> stride-4 first feature).
        depths: blocks per stage (VMamba-T: [2, 2, 9, 2]).
        dims:   channels per stage (VMamba-T: [96, 192, 384, 768]).
        d_state / d_conv / expand: SS2D hyperparams.
        drop_path_rate: stochastic depth, linearly scaled across blocks.
        pretrained_path: optional local path to a checkpoint.
    """

    def __init__(
        self,
        in_channels: int = 3,
        img_size: int = 224,
        pretrained: bool = True,
        patch_size: int = 4,
        depths: tuple = (2, 2, 9, 2),
        dims: tuple = (96, 192, 384, 768),
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 2,
        drop_path_rate: float = 0.2,
        pretrained_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        assert len(depths) == 4 and len(dims) == 4, (
            "MambaPureEncoder expects 4 stages: depths and dims must have length 4."
        )

        self.in_channels = in_channels
        self.img_size = img_size
        self.depths = tuple(depths)
        self.dims = tuple(dims)
        self.out_channels: List[int] = list(dims)

        # 1x1 stem if the backbone needs RGB but caller provides other channels.
        if in_channels != 3:
            self.input_stem = nn.Conv2d(in_channels, 3, kernel_size=1, bias=True)
            stem_out = 3
        else:
            self.input_stem = nn.Identity()
            stem_out = in_channels

        # Patch embedding: stride=patch_size, channels-last output.
        self.patch_embed = PatchEmbed2D(
            patch_size=patch_size,
            in_chans=stem_out,
            embed_dim=dims[0],
        )

        # Linearly scaled stochastic depth across all blocks.
        total_blocks = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        self.stages = nn.ModuleList()
        cursor = 0
        for i in range(4):
            stage = _MambaStage(
                dim=dims[i],
                depth=depths[i],
                drop_path_rates=dpr[cursor:cursor + depths[i]],
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                downsample=(i < 3),
            )
            self.stages.append(stage)
            cursor += depths[i]

        # Per-stage LayerNorm applied on (B, H, W, C) outputs before reshape.
        self.norms = nn.ModuleList([nn.LayerNorm(dims[i]) for i in range(4)])

        if pretrained:
            _load_with_ssl_fallback(self._maybe_load_pretrained, pretrained_path)

    # ---- Pretrained loading ---------------------------------------------------

    def _maybe_load_pretrained(self, pretrained_path: Optional[str] = None, **_):
        """Best-effort local checkpoint load. No-op if no path is provided.

        A canonical pretrained VMamba-T checkpoint is not hosted in a stable
        place we can rely on offline, so we only honor an explicit local path.
        """
        if not pretrained_path:
            warnings.warn(
                "MambaPureEncoder: no pretrained_path provided; using random init."
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
            if nk.startswith('encoder.'):
                nk = nk[len('encoder.'):]
            if nk.startswith('backbone.'):
                nk = nk[len('backbone.'):]
            if nk.startswith('module.'):
                nk = nk[len('module.'):]
            cleaned[nk] = v
        msg = self.load_state_dict(cleaned, strict=False)
        print(f"MambaPureEncoder loaded pretrained from {pretrained_path}: {msg}")

    # ---- Forward --------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Args:
            x: (B, in_channels, H, W).
        Returns:
            List of 4 feature maps in (B, C, H, W) at strides 4/8/16/32.
            Deepest (stride-32) is LAST.
        """
        x = self.input_stem(x)
        x = self.patch_embed(x)  # (B, H/4, W/4, C0)

        features: List[torch.Tensor] = []
        for i, stage in enumerate(self.stages):
            feat, x = stage(x)  # feat: (B, H, W, C_i)
            feat = self.norms[i](feat)
            feat = feat.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
            features.append(feat)
        return features
