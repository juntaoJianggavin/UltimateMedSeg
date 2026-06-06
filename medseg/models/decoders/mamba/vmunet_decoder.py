"""VM-UNet Decoder: faithful port from https://github.com/JCruan519/VM-UNet

Reference: "VM-UNet: Vision Mamba UNet for Medical Image Segmentation"
Decoder mirrors the encoder with PatchExpand2D upsampling + VSSBlock processing.
Skip connections via element-wise addition.

Has its own internal skip connection mechanism (add).
External skip_connection parameter is IGNORED.
"""
# Source: https://github.com/JCruan519/VM-UNet

import math
import torch
import torch.nn as nn
from typing import List
from einops import rearrange

from medseg.registry import DECODER_REGISTRY
from medseg.models.encoders.vmunet_encoder import VSSBlock


# ---------- PatchExpand2D ----------

class PatchExpand2D(nn.Module):
    """Patch Expand for 2x upsampling (from original VM-UNet)."""
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim * 2  # input dim is 2x the output dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale * self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        """x: (B, H, W, C) -> (B, 2H, 2W, C/2)"""
        B, H, W, C = x.shape
        x = self.expand(x)
        x = rearrange(x, 'b h w (p1 p2 c) -> b (h p1) (w p2) c',
                       p1=self.dim_scale, p2=self.dim_scale, c=C // self.dim_scale)
        x = self.norm(x)
        return x


class FinalPatchExpand2D(nn.Module):
    """Final Patch Expand for 4x upsampling to full resolution (from original VM-UNet)."""
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, dim_scale * dim, bias=False)
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """x: (B, H, W, C) -> (B, 4H, 4W, C/4)"""
        B, H, W, C = x.shape
        x = self.expand(x)
        x = rearrange(x, 'b h w (p1 p2 c) -> b (h p1) (w p2) c',
                       p1=self.dim_scale, p2=self.dim_scale, c=C // self.dim_scale)
        x = self.norm(x)
        return x


# ---------- VSSLayer_up ----------

class VSSLayer_up(nn.Module):
    """Decoder stage: optional PatchExpand2D + VSSBlocks."""
    def __init__(self, dim, depth, d_state=16, d_conv=3, expand=2,
                 drop_path=0.0, upsample=None):
        super().__init__()
        if isinstance(drop_path, (list, tuple)):
            dp_rates = drop_path
        else:
            dp_rates = [drop_path] * depth

        self.blocks = nn.ModuleList([
            VSSBlock(dim, d_state=d_state, d_conv=d_conv, expand=expand, drop_path=dp_rates[i])
            for i in range(depth)
        ])
        self.upsample = upsample(dim) if upsample is not None else None

    def forward(self, x, skip=None):
        """x: (B, H, W, C). If skip provided, upsample first then add skip."""
        if self.upsample is not None:
            x = self.upsample(x)  # 2x spatial upsample + channel reduction
        if skip is not None:
            x = x + skip
        for blk in self.blocks:
            x = blk(x)
        return x


# ---------- VM-UNet Decoder ----------

@DECODER_REGISTRY.register("vmunet")
class VMUNetDecoder(nn.Module):
    """VM-UNet decoder with PatchExpand2D upsampling and VSSBlock processing.

    Faithful to the original VM-UNet decoder architecture.
    Architecture:
        layers_up[0]: VSSBlocks only (no upsample, process bottleneck)
        layers_up[1..N-1]: PatchExpand2D + VSSBlocks (with skip add)
        Final: FinalPatchExpand2D (4x upsample to full resolution)

    Skip connections via element-wise addition (not concat).
    External skip_connection is IGNORED.
    """
    has_internal_skip = True

    def __init__(self, encoder_channels: List[int], bottleneck_channels: int,
                 skip_connection=None,
                 depths_decoder: tuple = (2, 2, 2, 2),
                 d_state: int = 16,
                 d_conv: int = 3,
                 expand: int = 2,
                 drop_path_rate: float = 0.1,
                 patch_size: int = 4,
                 **kwargs):
        super().__init__()
        # dims_decoder: from deepest to shallowest
        # encoder_channels = [c0, c1, c2] (skip channels, shallow to deep)
        # bottleneck_channels = c3 (deepest)
        all_channels = list(encoder_channels) + [bottleneck_channels]
        dims_decoder = list(reversed(all_channels))  # [c3, c2, c1, c0]

        num_layers = len(dims_decoder)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_decoder[:num_layers]))]

        self.layers_up = nn.ModuleList()
        for i in range(num_layers):
            layer = VSSLayer_up(
                dim=dims_decoder[i],
                depth=depths_decoder[i] if i < len(depths_decoder) else 2,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                drop_path=dpr[sum(depths_decoder[:i]):sum(depths_decoder[:i+1])] if i < len(depths_decoder) else 0.0,
                upsample=PatchExpand2D if i != 0 else None,  # First layer: no upsample
            )
            self.layers_up.append(layer)

        # Final 4x upsample
        self.final_up = FinalPatchExpand2D(
            dim=dims_decoder[-1], dim_scale=patch_size, norm_layer=nn.LayerNorm
        )
        self._out_channels = dims_decoder[-1] // patch_size

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, bottleneck_feat: torch.Tensor, skip_features: List[torch.Tensor]) -> torch.Tensor:
        # Convert bottleneck (B, C, H, W) -> (B, H, W, C)
        x = bottleneck_feat.permute(0, 2, 3, 1).contiguous()

        # Reverse skip features: deep to shallow
        skips = list(reversed(skip_features))  # [c2, c1, c0]

        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                # First layer: just process bottleneck, no skip
                x = layer_up(x)
            else:
                # PatchExpand upsample first, then add skip, then VSSBlocks
                skip = skips[inx - 1].permute(0, 2, 3, 1).contiguous()
                x = layer_up(x, skip=skip)

        # Final 4x upsample
        x = self.final_up(x)  # (B, 4H, 4W, C/4)

        # Convert to (B, C, H, W)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x
