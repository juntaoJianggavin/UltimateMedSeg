"""Swin-UNet Decoder: faithful port from https://github.com/HuCaoFighting/Swin-Unet

Reference: Cao et al., "Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation"
File: networks/swin_transformer_unet_skip_expand_decoder_sys.py

Decoder components: PatchExpand, FinalPatchExpand_X4, BasicLayer_up
All class/attribute names match the original for pretrained weight loading.

Has its own internal skip connection mechanism (concat + linear projection).
External skip_connection parameter is IGNORED.
"""
# Source: https://github.com/HuCaoFighting/Swin-Unet

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from typing import List
from einops import rearrange

from medseg.registry import DECODER_REGISTRY
from medseg.models.encoders.swinunet_encoder import SwinTransformerBlock


class PatchExpand(nn.Module):
    """Patch expanding layer for 2x upsampling (from original Swin-UNet)."""

    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)
        return x


class FinalPatchExpand_X4(nn.Module):
    """Final 4x patch expanding for full resolution recovery."""

    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c',
                       p1=self.dim_scale, p2=self.dim_scale, c=C // (self.dim_scale ** 2))
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x)
        return x


class BasicLayer_up(nn.Module):
    """A basic Swin Transformer layer for decoder (with optional upsample)."""

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        if upsample is not None:
            self.upsample = PatchExpand(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


@DECODER_REGISTRY.register("swinunet")
class SwinUNetDecoder(nn.Module):
    """Swin-UNet decoder with PatchExpand upsampling and Swin Transformer blocks.

    Faithful to the original SwinTransformerSys decoder part.
    Architecture:
        PatchExpand (bottleneck 2x up) ->
        [concat skip + Linear + BasicLayer_up with SwinBlocks + PatchExpand] x (N-1) ->
        [concat skip + Linear + BasicLayer_up with SwinBlocks] x 1 ->
        norm_up -> FinalPatchExpand_X4

    External skip_connection is IGNORED.
    """
    has_internal_skip = True
    requires_encoder = "swin"  # Requires Swin Transformer encoder with PatchEmbed

    def __init__(self, encoder_channels: List[int], bottleneck_channels: int,
                 skip_connection=None,
                 embed_dim: int = 96,
                 depths_decoder: tuple = (1, 2, 2, 2),
                 num_heads: tuple = (3, 6, 12, 24),
                 window_size: int = 7,
                 img_size: int = 224,
                 patch_size: int = 4,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = True,
                 qk_scale=None,
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.1,
                 use_checkpoint: bool = False,
                 **kwargs):
        super().__init__()
        norm_layer = nn.LayerNorm
        num_layers = len(encoder_channels)  # should be 3 for skip features (excluding bottleneck)
        # Actually num_layers is total encoder stages including bottleneck
        self.num_layers = num_layers + 1  # total stages
        patches_resolution = [img_size // patch_size, img_size // patch_size]
        self.patches_resolution = patches_resolution

        # Stochastic depth for decoder
        depths_dec = list(depths_decoder)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_dec))]

        # Build decoder layers and concat_back_dim
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()

        for i_layer in range(self.num_layers):
            dim_layer = int(embed_dim * 2 ** (self.num_layers - 1 - i_layer))
            res = (patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                   patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer)))

            concat_linear = nn.Linear(2 * dim_layer, dim_layer) if i_layer > 0 else nn.Identity()

            if i_layer == 0:
                # First decoder layer: just PatchExpand (no Swin blocks)
                layer_up = PatchExpand(
                    input_resolution=res,
                    dim=dim_layer, dim_scale=2, norm_layer=norm_layer)
            else:
                layer_up = BasicLayer_up(
                    dim=dim_layer,
                    input_resolution=res,
                    depth=depths_dec[self.num_layers - 1 - i_layer],
                    num_heads=num_heads[self.num_layers - 1 - i_layer],
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths_dec[:(self.num_layers - 1 - i_layer)]):
                                  sum(depths_dec[:(self.num_layers - 1 - i_layer) + 1])],
                    norm_layer=norm_layer,
                    upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                    use_checkpoint=use_checkpoint)

            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        self.norm_up = norm_layer(embed_dim)

        # Final 4x expand
        self.up = FinalPatchExpand_X4(
            input_resolution=(patches_resolution[0], patches_resolution[1]),
            dim_scale=4, dim=embed_dim)

        self._out_channels = embed_dim
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, bottleneck_feat: torch.Tensor, skip_features: List[torch.Tensor]) -> torch.Tensor:
        # Convert bottleneck from (B, C, H, W) to (B, L, C) sequence format
        B, C, H, W = bottleneck_feat.shape
        x = bottleneck_feat.flatten(2).transpose(1, 2)  # (B, H*W, C)

        # Convert skip features from (B, C, H, W) to (B, L, C) - keep original order (shallow to deep)
        skip_seqs = []
        for feat in skip_features:
            skip_seqs.append(feat.flatten(2).transpose(1, 2))

        # Decoder: process layers
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                # Concat with skip feature (from deep to shallow)
                skip_idx = len(skip_seqs) - inx  # maps to correct skip level
                if 0 <= skip_idx < len(skip_seqs):
                    x = torch.cat([x, skip_seqs[skip_idx]], -1)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)

        x = self.norm_up(x)

        # Final 4x expand: (B, L, C) -> (B, 4H*4W, C)
        x = self.up(x)

        # Convert back to (B, C, H, W)
        H_out = self.patches_resolution[0] * 4  # full resolution
        W_out = self.patches_resolution[1] * 4
        x = x.view(B, H_out, W_out, -1).permute(0, 3, 1, 2).contiguous()
        return x
