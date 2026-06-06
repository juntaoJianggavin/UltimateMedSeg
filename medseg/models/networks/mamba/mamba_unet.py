"""Mamba-UNet: Pure Vision Mamba encoder-decoder for Medical Image Segmentation.

Faithful reimplementation from:
  https://github.com/ziyangwang007/Mamba-UNet  (MedIA 2024, 798+ stars)

Architecture: PatchEmbed2D → VSSLayer encoder (PatchMerging downsample)
  → VSSLayer_up decoder (PatchExpand upsample) → FinalPatchExpand_X4
Skip connections via concatenation + linear projection (different from VM-UNet's addition).

Reuses SS2D/VSSBlock from the project's vmunet_encoder to avoid code duplication.
"""
# Source: https://github.com/ziyangwang007/Mamba-UNet

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from functools import partial
from einops import rearrange

from medseg.models.encoders.vmunet_encoder import (
    SS2D, VSSBlock, PatchEmbed2D, PatchMerging2D
)


# ---------------------------------------------------------------------------
# PatchExpand & FinalPatchExpand_X4 (from Mamba-UNet original)
# ---------------------------------------------------------------------------

class PatchExpand(nn.Module):
    """Patch expand for 2x spatial upsampling (Mamba-UNet style)."""
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """x: (B, H, W, C) -> (B, 2H, 2W, C/2)"""
        x = self.expand(x)
        B, H, W, C = x.shape
        x = rearrange(x, 'b h w (p1 p2 c) -> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = self.norm(x)
        return x


class FinalPatchExpand_X4(nn.Module):
    """Final 4x spatial upsampling to original resolution."""
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.norm = norm_layer(dim)

    def forward(self, x):
        """x: (B, H, W, C) -> (B, 4H, 4W, C)"""
        x = self.expand(x)
        B, H, W, C = x.shape
        x = rearrange(x, 'b h w (p1 p2 c) -> b (h p1) (w p2) c',
                       p1=self.dim_scale, p2=self.dim_scale,
                       c=C // (self.dim_scale ** 2))
        x = self.norm(x)
        return x


# ---------------------------------------------------------------------------
# VSSLayer (encoder stage)
# ---------------------------------------------------------------------------

class VSSLayer(nn.Module):
    """Encoder stage: VSSBlocks + optional PatchMerging2D downsample."""
    def __init__(self, dim, depth, d_state=16, d_conv=3, expand=2,
                 drop_path=0.0, downsample=None):
        super().__init__()
        if isinstance(drop_path, (list, tuple)):
            dp_rates = drop_path
        else:
            dp_rates = [drop_path] * depth

        self.blocks = nn.ModuleList([
            VSSBlock(dim, d_state=d_state, d_conv=d_conv, expand=expand,
                     drop_path=dp_rates[i])
            for i in range(depth)
        ])
        self.downsample = downsample(dim) if downsample else None

    def forward(self, x):
        """x: (B, H, W, C) -> x_out before downsample, x after downsample"""
        for blk in self.blocks:
            x = blk(x)
        x_out = x
        if self.downsample is not None:
            x = self.downsample(x)
        return x_out, x


# ---------------------------------------------------------------------------
# VSSLayer_up (decoder stage)
# ---------------------------------------------------------------------------

class VSSLayer_up(nn.Module):
    """Decoder stage: VSSBlocks + optional PatchExpand upsample."""
    def __init__(self, dim, depth, d_state=16, d_conv=3, expand=2,
                 drop_path=0.0, upsample=None):
        super().__init__()
        if isinstance(drop_path, (list, tuple)):
            dp_rates = drop_path
        else:
            dp_rates = [drop_path] * depth

        self.blocks = nn.ModuleList([
            VSSBlock(dim, d_state=d_state, d_conv=d_conv, expand=expand,
                     drop_path=dp_rates[i])
            for i in range(depth)
        ])
        self.upsample = PatchExpand(dim, dim_scale=2) if upsample is not None else None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


# ---------------------------------------------------------------------------
# VSSM: full VMamba encoder-decoder (Mamba-UNet core)
# ---------------------------------------------------------------------------

class VSSM(nn.Module):
    """VMamba Symmetric encoder-decoder with concat skip connections.

    This is the core architecture from Mamba-UNet (Wang et al., 2024).
    Key difference from VM-UNet: uses concatenation + linear for skip connections
    instead of element-wise addition.
    """
    def __init__(self, patch_size=4, in_chans=3, num_classes=2,
                 depths=(2, 2, 9, 2), dims=96,
                 d_state=16, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm,
                 **kwargs):
        super().__init__()
        num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i) for i in range(num_layers)]
        self.num_layers = num_layers
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims

        # Patch embedding
        self.patch_embed = PatchEmbed2D(patch_size, in_chans, self.embed_dim,
                                        norm_layer=norm_layer)

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Encoder layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = VSSLayer(
                dim=dims[i],
                depth=depths[i],
                d_state=d_state,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                downsample=PatchMerging2D if i < num_layers - 1 else None,
            )
            self.layers.append(layer)

        # Decoder layers
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i in range(num_layers):
            target_dim = dims[num_layers - 1 - i]
            concat_linear = (
                nn.Linear(2 * target_dim, target_dim)
                if i > 0 else nn.Identity()
            )
            if i == 0:
                layer_up = PatchExpand(
                    dim=dims[num_layers - 1 - i], dim_scale=2)
            else:
                layer_up = VSSLayer_up(
                    dim=dims[num_layers - 1 - i],
                    depth=depths[num_layers - 1 - i],
                    d_state=d_state,
                    drop_path=dpr[sum(depths[:num_layers - 1 - i]):sum(depths[:num_layers - i])],
                    upsample=PatchExpand if i < num_layers - 1 else None,
                )
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        self.norm = norm_layer(self.num_features)
        self.norm_up = norm_layer(self.embed_dim)

        # Final 4x upsample + head
        self.up = FinalPatchExpand_X4(dim=self.embed_dim, dim_scale=4)
        self.output = nn.Conv2d(self.embed_dim, num_classes, kernel_size=1, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.patch_embed(x)  # (B, H/4, W/4, C)
        x_downsample = []
        for layer in self.layers:
            x_downsample.append(x)
            x_out, x = layer(x)
        x = self.norm(x_out)  # use x_out (last stage output before downsample)
        return x, x_downsample

    def forward_up_features(self, x, x_downsample, return_intermediates=False):
        intermediates = []
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = torch.cat([x, x_downsample[self.num_layers - 1 - inx]], -1)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)
            if return_intermediates and inx < len(self.layers_up) - 1:
                intermediates.append(x)
        x = self.norm_up(x)
        if return_intermediates:
            return x, intermediates
        return x

    def forward(self, x, return_intermediates=False):
        input_size = x.shape[2:]
        x, x_downsample = self.forward_features(x)
        if return_intermediates:
            x, intermediates = self.forward_up_features(x, x_downsample, return_intermediates=True)
        else:
            x = self.forward_up_features(x, x_downsample)
        B, H, W, C = x.shape
        x = self.up(x)
        x = x.view(B, 4 * H, 4 * W, -1)
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        x = self.output(x)
        if x.shape[2:] != input_size:
            x = F.interpolate(x, size=input_size, mode='bilinear',
                              align_corners=False)
        if return_intermediates:
            return x, intermediates
        return x


# ---------------------------------------------------------------------------
# MambaUNet: wrapper with standard special_arch interface
# ---------------------------------------------------------------------------

class MambaUNet(nn.Module):
    """Mamba-UNet: Pure VMamba encoder-decoder for medical image segmentation.

    Architecture:
      PatchEmbed2D(4x) → 4 VSSLayer encoder stages → 4 VSSLayer_up decoder stages
      → FinalPatchExpand_X4 → 1x1 conv head
    Skip connections: concatenation + linear projection

    Args:
        in_channels: Input image channels.
        num_classes: Number of output classes.
        img_size: Input image size.
        embed_dim: Base embedding dimension.
        depths: Depth of each stage.
        d_state: SSM state dimension.
        drop_path_rate: Stochastic depth rate.
    """
    def __init__(self, in_channels=3, num_classes=2, img_size=224,
                 embed_dim=96, depths=None, d_state=16,
                 drop_path_rate=0.1, deep_supervision=False, **kwargs):
        super().__init__()
        if depths is None:
            depths = [2, 2, 9, 2]
        self.deep_supervision = deep_supervision
        num_layers = len(depths)
        dims = [int(embed_dim * 2 ** i) for i in range(num_layers)]
        self.model = VSSM(
            patch_size=4,
            in_chans=in_channels,
            num_classes=num_classes,
            depths=depths,
            dims=embed_dim,
            d_state=d_state,
            drop_path_rate=drop_path_rate,
        )

        if deep_supervision:
            # Intermediates from decoder: after step 0 → dims[-2], step 1 → dims[-3]
            self.ds_heads = nn.ModuleList([
                nn.Conv2d(dims[num_layers - 2 - i], num_classes, 1)
                for i in range(num_layers - 2)
            ])

    def forward(self, x):
        if self.training and self.deep_supervision:
            main_out, intermediates = self.model(x, return_intermediates=True)
            input_size = main_out.shape[2:]
            aux = []
            for feat_bhwc, head in zip(intermediates, self.ds_heads):
                # Convert (B, H, W, C) to (B, C, H, W)
                feat = feat_bhwc.permute(0, 3, 1, 2).contiguous()
                a = head(feat)
                if a.shape[2:] != input_size:
                    a = F.interpolate(a, size=input_size, mode='bilinear',
                                      align_corners=False)
                aux.append(a)
            return [main_out] + aux
        return self.model(x)
