"""RWKV-UNet Decoder: adapted from the official implementation.

Reference: "RWKV-UNet: Improving UNet with Linear Complexity for Medical Image
Segmentation" — https://github.com/juntaoJianggavin/RWKV-UNet

The decoder closely mirrors the official RWKV_UNet head:

    1) CCMix : cross-channel mixer that fuses skip features via
       VRWKV_ChannelMix on a common feature plane.
    2) decoder blocks : UpBlocks (1x1 conv -> DWConv k=9 -> SE? ->
       1x1 proj -> bilinear 2x upsample).
    3) Final 1x1 conv is owned by the framework's SegmentationHead.

Adapted to work with any encoder: the number of decoder stages and
CCMix features are derived dynamically from encoder_channels.

This decoder advertises ``has_internal_skip = True`` so the model builder
skips its external skip-connection injection logic.
"""
# Source: https://github.com/juntaoJianggavin/RWKV-UNet

import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.registry import DECODER_REGISTRY
from medseg.models.encoders.rwkv_encoder import (
    ConvNormAct,
    DropPath,
    SE,
    VRWKV_ChannelMix,
    get_norm,
)


# ---------- UpBlock (1:1 with the official UpBlock in rwkv_unet.py) ----------


class UpBlock(nn.Module):
    """RWKV-UNet decoder block.

    Pipeline: optional norm -> 1x1 conv -> (DWConv k=dw_ks, optionally with SE,
    optional residual via has_skip) -> 1x1 proj -> dropout -> bilinear 2x up.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        norm_in: bool = False,
        has_skip: bool = False,
        exp_ratio: float = 1.0,
        norm_layer: str = "bn_2d",
        act_layer: str = "relu",
        dw_ks: int = 9,
        stride: int = 1,
        dilation: int = 1,
        se_ratio: float = 0.0,
        drop_path: float = 0.0,
        drop: float = 0.0,
    ):
        super().__init__()
        self.has_skip = has_skip
        self.norm = get_norm(norm_layer)(dim_in) if norm_in else nn.Identity()
        dim_mid = int(dim_in * exp_ratio)
        self.ln1 = nn.LayerNorm(dim_mid)
        self.conv = ConvNormAct(dim_in, dim_mid, kernel_size=1)
        if se_ratio > 0.0:
            self.se = SE(dim_mid, rd_ratio=se_ratio)
        else:
            self.se = nn.Identity()
        self.proj_drop = nn.Dropout(drop) if drop > 0 else nn.Identity()
        self.proj = ConvNormAct(dim_mid, dim_out, kernel_size=1, norm_layer="bn_2d", act_layer="relu")
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.conv_local = ConvNormAct(
            dim_mid, dim_mid,
            kernel_size=dw_ks, stride=stride, dilation=dilation, groups=dim_mid,
            norm_layer="bn_2d", act_layer="silu",
        )
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x):
        x = self.norm(x)
        x = self.conv(x)
        if self.has_skip:
            x = x + self.se(self.conv_local(x))
        else:
            x = self.se(self.conv_local(x))
        x = self.proj(x)
        x = self.proj_drop(x)
        x = self.upsample(x)
        return x


# ---------- CCMix (generalized for N skip features) ----------


class CCMix(nn.Module):
    """Cross-Channel Mixer: fuses N skip features through VRWKV_ChannelMix.

    Args:
        in_dims: per-skip channel counts in the order ``[deep, ..., shallow]``.
        target_dim: common channel count after the per-skip 1x1 projection.
        target_size: spatial size on which the mixing is performed.
    """

    def __init__(self, in_dims, target_dim: int, target_size: int):
        super().__init__()
        self.n_skips = len(in_dims)
        self.in_dims = list(in_dims)
        self.target_dim = target_dim
        self.target_size = target_size

        self.projections = nn.ModuleList([
            nn.Conv2d(c, target_dim, kernel_size=1) for c in self.in_dims
        ])
        total_ch = target_dim * self.n_skips
        self.ln1 = nn.LayerNorm(total_ch)
        # Official code unconditionally uses DropPath(0.05)
        self.drop_path = DropPath(0.05)
        self.channel = VRWKV_ChannelMix(
            n_embd=total_ch, channel_gamma=1 / 4, shift_pixel=1, hidden_rate=2,
        )
        self.final_projections = nn.ModuleList([
            nn.Conv2d(target_dim, c, kernel_size=1) for c in self.in_dims
        ])
        # Original spatial sizes: each skip is at a progressively larger scale
        # deepest skip at target_size//2^(n-1), shallowest at target_size
        self.original_sizes = []
        for i in range(self.n_skips):
            scale = 2 ** (self.n_skips - 1 - i)
            self.original_sizes.append(target_size // scale)

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        # Step 1: bring every skip up to (target_dim, target_size).
        upsampled = []
        for i, feat in enumerate(features):
            feat = F.interpolate(feat, size=self.target_size, mode="bilinear", align_corners=False)
            feat = self.projections[i](feat)
            upsampled.append(feat)

        # Step 2: concat along channels, reshape to sequence and apply VRWKV_ChannelMix.
        cat = torch.cat(upsampled, dim=1)  # (B, target_dim*N, T, T)
        B, C, H, W = cat.shape
        cat_seq = cat.view(B, C, -1).permute(0, 2, 1).contiguous()  # (B, N_seq, C)
        attn = cat_seq + self.drop_path(self.ln1(self.channel(cat_seq, (self.target_size, self.target_size))))

        # Step 3: reshape back to (B, C, H, W).
        B2, N_seq, hidden = attn.shape
        h = w = int(math.sqrt(N_seq))
        attn = attn.permute(0, 2, 1).contiguous().view(B2, hidden, h, w)

        # Step 4: split per-skip, project back to original channels and resize.
        chunks = torch.split(attn, self.target_dim, dim=1)
        outputs = []
        for i, chunk in enumerate(chunks):
            chunk = self.final_projections[i](chunk)
            chunk = F.interpolate(chunk, size=self.original_sizes[i], mode="bilinear", align_corners=False)
            outputs.append(chunk)
        return outputs


# ---------- Full RWKV-UNet decoder (adaptive) ----------


@DECODER_REGISTRY.register("rwkv_unet")
class RWKVUNetDecoder(nn.Module):
    """Adaptive RWKV-UNet decoder head.

    Automatically adapts to any encoder: the number of decoder stages and
    CCMix features are derived from encoder_channels.

    Architecture (for N skip features)::

        # skip_features = [enc_shallow, ..., enc_deep]  (shallow -> deep)
        # bottleneck_feat = deepest encoder output
        fused = CCMix([deep, ..., shallow])
        x = bottleneck_feat
        for i in range(N+1):
            x = decoder_block_i(x or cat(x, fused_skip))
    """

    has_internal_skip = True

    def __init__(
        self,
        encoder_channels: List[int],
        bottleneck_channels: int,
        skip_connection=None,
        img_size: int = 224,
        dw_ks: int = 9,
        se_ratio: float = 0.0,
        final_channels: int = 24,
        **kwargs,
    ):
        super().__init__()
        n_skips = len(encoder_channels)
        # encoder_channels = [c0, c1, ..., cN] shallow -> deep
        c_shallow = encoder_channels[0]
        c_deep = encoder_channels[-1]
        self._final_channels = final_channels

        # CCMix: fuses [deep, ..., shallow] skip features
        skips_reversed = list(reversed(encoder_channels))  # [deep, ..., shallow]
        self.ccm = CCMix(
            in_dims=skips_reversed,
            target_dim=c_shallow,
            target_size=img_size // 2,
        )

        # Build decoder UpBlock chain:
        # Block 0: bottleneck -> c_deep (no skip concat)
        # Block 1..N-1: concat with fused skip -> next level
        # Block N: concat with shallowest fused skip -> final_channels
        self.decoder_blocks = nn.ModuleList()

        # First block: bottleneck to deepest skip channels
        self.decoder_blocks.append(
            UpBlock(bottleneck_channels, c_deep, dw_ks=dw_ks, se_ratio=se_ratio)
        )

        # Intermediate blocks: concat fused skip (doubling channels) -> next level
        for i in range(n_skips - 1):
            in_ch = skips_reversed[i] * 2  # concat with fused skip
            out_ch = skips_reversed[i + 1]
            self.decoder_blocks.append(
                UpBlock(in_ch, out_ch, dw_ks=dw_ks, se_ratio=se_ratio)
            )

        # Final block: concat shallowest fused skip -> final_channels
        in_ch = skips_reversed[-1] * 2  # concat with shallowest fused skip
        self.decoder_blocks.append(
            UpBlock(in_ch, final_channels, dw_ks=dw_ks, se_ratio=se_ratio)
        )

    @property
    def out_channels(self) -> int:
        return self._final_channels

    def forward(
        self,
        bottleneck_feat: torch.Tensor,
        skip_features: List[torch.Tensor],
    ) -> torch.Tensor:
        # skip_features: [shallow, ..., deep]
        n_skips = len(skip_features)
        skips_reversed = list(reversed(skip_features))  # [deep, ..., shallow]

        # CCMix fuses [deep, ..., shallow] and returns in the same order
        fused = self.ccm(skips_reversed)  # [fused_deep, ..., fused_shallow]

        # Block 0: bottleneck -> upsample
        x = self.decoder_blocks[0](bottleneck_feat)

        # Blocks 1..N: concat with fused skip -> upsample
        for i in range(n_skips):
            if x.shape[2:] != fused[i].shape[2:]:
                x = F.interpolate(x, size=fused[i].shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, fused[i]], dim=1)
            x = self.decoder_blocks[i + 1](x)

        return x
