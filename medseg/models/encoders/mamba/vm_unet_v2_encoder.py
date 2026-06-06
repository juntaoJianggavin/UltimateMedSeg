"""VM-UNet-V2 Encoder.

Faithful port of the VM-UNet-V2 backbone (VSSM) from
https://github.com/nobodyplayer1/VM-UNetV2.

Uses the proper SS2D (4-direction selective scan) and VSSBlock from
the original implementation, matching the original VSSM encoder structure:

    PatchEmbed2D (stride-4) -> VSSLayer_0 (VSSBlock x d0 + PatchMerging2D)
                             -> VSSLayer_1 (VSSBlock x d1 + PatchMerging2D)
                             -> VSSLayer_2 (VSSBlock x d2 + PatchMerging2D)
                             -> VSSLayer_3 (VSSBlock x d3)

Returns multi-scale features in framework convention (deepest LAST),
converted to (B, C, H, W) format.
"""
# Source: https://github.com/nobodyplayer1/VM-UNetV2

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from functools import partial

from medseg.registry import ENCODER_REGISTRY

# Reuse SS2D, VSSBlock, and helpers from vmunet_encoder (same directory).
# vmunet_encoder is imported first in __init__.py, so this is always available.
from .vmunet_encoder import (
    SS2D,
    VSSBlock,
    DropPath,
    PatchEmbed2D,
    PatchMerging2D,
)


class _VSSLayer(nn.Module):
    """One stage of VSSBlocks + optional PatchMerging2D downsampling.

    Faithful to the original VM-UNetV2 VSSLayer.
    Data format: (B, H, W, C) throughout.
    """

    def __init__(self, dim, depth, d_state=16, drop_path=0.0,
                 downsample=None):
        super().__init__()
        if isinstance(drop_path, (list, tuple)):
            dp_rates = drop_path
        else:
            dp_rates = [drop_path] * depth

        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                d_state=d_state,
                drop_path=dp_rates[i],
            )
            for i in range(depth)
        ])
        self.downsample = downsample(dim) if downsample else None

    def forward(self, x):
        """x: (B, H, W, C) -> (pre_down, post_down)"""
        for blk in self.blocks:
            x = blk(x)
        pre_down = x
        if self.downsample is not None:
            x = self.downsample(x)
        return pre_down, x


@ENCODER_REGISTRY.register("vm_unet_v2")
class VMUNetV2Encoder(nn.Module):
    """VM-UNet-V2 Encoder (VSSM backbone).

    Faithful to https://github.com/nobodyplayer1/VM-UNetV2
    4-stage hierarchical encoder with SS2D selective scan and
    PatchMerging2D downsampling.

    Returns features at strides 4, 8, 16, 32 with out_channels
    [embed_dim, embed_dim*2, embed_dim*4, embed_dim*8] (deepest LAST).
    Note: stride 32 applies when img_size is divisible by 32 (e.g. 224).
    """

    def __init__(
        self,
        in_channels: int = 3,
        img_size: int = 224,
        pretrained: bool = False,
        embed_dim: int = 64,
        depths: List[int] = None,
        d_state: int = 16,
        drop_path_rate: float = 0.2,
        pretrained_path: str = None,
        **kwargs,
    ):
        super().__init__()
        depths = depths or [2, 2, 6, 2]
        num_stages = len(depths)
        dims = [embed_dim * (2 ** i) for i in range(num_stages)]
        self.dims = dims

        # Patch embedding: stride-4 Conv2d -> (B, H/4, W/4, C) channels-last
        self.patch_embed = PatchEmbed2D(
            patch_size=4, in_chans=in_channels, embed_dim=embed_dim)

        # Stochastic depth (linearly increasing)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # VSS layers with PatchMerging2D downsampling between stages
        self.layers = nn.ModuleList()
        for i in range(num_stages):
            layer = _VSSLayer(
                dim=dims[i],
                depth=depths[i],
                d_state=d_state,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                downsample=PatchMerging2D if i < num_stages - 1 else None,
            )
            self.layers.append(layer)

        # LayerNorm for each stage output
        self.norms = nn.ModuleList([nn.LayerNorm(dims[i]) for i in range(num_stages)])

        self.out_channels = dims

        if pretrained and pretrained_path:
            self._load_pretrained(pretrained_path)
        elif pretrained:
            import warnings
            warnings.warn(
                "VMUNetV2Encoder: no default pretrained weights available; "
                "using random initialisation."
            )

    def _load_pretrained(self, path):
        state = torch.load(path, map_location='cpu')
        if 'model' in state:
            state = state['model']
        if 'state_dict' in state:
            state = state['state_dict']
        encoder_state = {}
        for k, v in state.items():
            if k.startswith('encoder.') or k.startswith('vmunet.'):
                key = k.split('.', 1)[-1] if '.' in k else k
                encoder_state[key] = v
            elif not any(k.startswith(p) for p in ('decoder', 'head', 'final', 'seg_head')):
                encoder_state[k] = v
        msg = self.load_state_dict(encoder_state, strict=False)
        print(f"VM-UNetV2 encoder loaded: {msg}")

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # Patch embed: (B, C, H, W) -> (B, H/4, W/4, C)
        x = self.patch_embed(x)

        features: List[torch.Tensor] = []
        for i, layer in enumerate(self.layers):
            x_out, x = layer(x)
            x_out = self.norms[i](x_out)
            # Convert from channels-last (B, H, W, C) to channels-first (B, C, H, W)
            feat = x_out.permute(0, 3, 1, 2).contiguous()
            features.append(feat)

        return features
