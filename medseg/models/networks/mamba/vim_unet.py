"""ViM-UNet: Vision Mamba for Biomedical Instance Segmentation.

Reference:
    Archit et al., "ViM-UNet: Vision Mamba for Biomedical Segmentation",
    MIDL 2024.
    https://github.com/constantinpape/torch-em

Architecture:
    * Vision Mamba (ViM) encoder with patch embedding + Mamba blocks.
    * UNet-style decoder with skip connections from encoder stages.
    * 4-stage encoder: PatchEmbed + Mamba blocks + PatchMerge.
    * Decoder: linear upsample + skip concat + MLP.

Constructor:
    ViMUNet(in_channels=3, num_classes=2, img_size=224, **kwargs)
"""
# Source: https://github.com/constantinpape/torch-em

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from medseg.models.encoders.vmunet_encoder import SS2D, PatchEmbed2D, PatchMerging2D


class _ViMBlock(nn.Module):
    """Vision Mamba block: bidirectional SS2D + MLP."""
    def __init__(self, dim, d_state=16, d_conv=3, expand=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.ss2d = SS2D(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(),
            nn.Linear(dim * 4, dim))

    def forward(self, x):
        # x: (B, H, W, C)
        h = self.ss2d(self.norm1(x))
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class ViMUNet(nn.Module):
    """ViM-UNet for biomedical segmentation."""
    def __init__(self, in_channels=3, num_classes=2, img_size=224,
                 embed_dim=64, depths=None, **kwargs):
        super().__init__()
        if depths is None:
            depths = [2, 2, 2, 2]
        self.num_classes = num_classes
        dims = [embed_dim * (2 ** i) for i in range(4)]

        # Encoder stages
        self.patch_embed = PatchEmbed2D(patch_size=4,
                                        in_chans=in_channels, embed_dim=dims[0])
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i in range(4):
            blocks = nn.ModuleList([_ViMBlock(dims[i]) for _ in range(depths[i])])
            self.stages.append(blocks)
            if i < 3:
                self.downsamples.append(PatchMerging2D(dims[i]))

        # Decoder
        self.decoder_up = nn.ModuleList()
        self.decoder_conv = nn.ModuleList()
        for i in range(3, 0, -1):
            self.decoder_up.append(nn.Linear(dims[i], dims[i - 1]))
            self.decoder_conv.append(nn.Sequential(
                nn.LayerNorm(dims[i - 1] * 2),
                nn.Linear(dims[i - 1] * 2, dims[i - 1]),
                nn.GELU()))

        # Head
        self.head = nn.Sequential(
            nn.LayerNorm(dims[0]),
            nn.Linear(dims[0], num_classes))
        self.patch_size = 4

    def forward(self, x):
        B, _, H, W = x.shape
        pH = ((H + self.patch_size - 1) // self.patch_size) * self.patch_size
        pW = ((W + self.patch_size - 1) // self.patch_size) * self.patch_size
        pad_needed = (pH != H or pW != W)
        if pad_needed:
            x = F.pad(x, [0, pW - W, 0, pH - H], mode='reflect')

        # Encoder
        feats = []
        z = self.patch_embed(x)  # (B, H', W', C)
        for i in range(4):
            for blk in self.stages[i]:
                z = blk(z)
            feats.append(z)
            if i < 3:
                z = self.downsamples[i](z)

        # Decoder
        z = feats[3]
        for j in range(3):
            z = self.decoder_up[j](z)
            # Upsample spatially to match skip
            skip = feats[2 - j]
            zH, zW = z.shape[1], z.shape[2]
            sH, sW = skip.shape[1], skip.shape[2]
            if zH != sH or zW != sW:
                z = rearrange(z, 'b h w c -> b c h w')
                z = F.interpolate(z, size=(sH, sW), mode='bilinear', align_corners=False)
                z = rearrange(z, 'b c h w -> b h w c')
            z = torch.cat([z, skip], dim=-1)
            z = self.decoder_conv[j](z)

        # Head
        out = self.head(z)  # (B, H', W', num_classes)
        out = rearrange(out, 'b h w c -> b c h w')
        # Upsample to original resolution
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        return out
