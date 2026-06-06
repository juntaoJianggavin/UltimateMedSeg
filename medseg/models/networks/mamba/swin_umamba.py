"""Swin-UMamba: VMamba Encoder + UNETR-style Conv Decoder for Medical Image Segmentation.

Faithful reimplementation from:
  https://github.com/JiarunLiu/Swin-UMamba  (2024, 384+ stars)

Architecture:
  - Encoder: VMamba (PatchEmbed2D → VSSLayer stages with PatchMerging2D)
  - Decoder: UNETR-style conv blocks with upsampling (self-contained, no MONAI)
  - Skip connections via concatenation + conv fusion

Reuses SS2D/VSSBlock/PatchEmbed2D/PatchMerging2D from vmunet_encoder.
"""
# Source: https://github.com/JiarunLiu/Swin-UMamba

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from medseg.models.encoders.vmunet_encoder import (
    SS2D, VSSBlock, VSSLayer, PatchEmbed2D, PatchMerging2D
)


# ---------------------------------------------------------------------------
# UNETR-style decoder blocks (self-contained, no MONAI dependency)
# ---------------------------------------------------------------------------

class UnetrBasicBlock(nn.Module):
    """Basic residual conv block for UNETR decoder."""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size,
                               padding=kernel_size // 2, bias=False)
        self.norm1 = nn.InstanceNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size,
                               padding=kernel_size // 2, bias=False)
        self.norm2 = nn.InstanceNorm2d(out_channels)
        self.act = nn.LeakyReLU(inplace=True)
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + residual)


class UnetrUpBlock(nn.Module):
    """UNETR-style upsample + concat skip + conv block."""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels,
                                            kernel_size=2, stride=2)
        self.conv_block = UnetrBasicBlock(out_channels * 2, out_channels,
                                           kernel_size=kernel_size)

    def forward(self, x, skip):
        x = self.upsample(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear',
                              align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv_block(x)


# ---------------------------------------------------------------------------
# VMamba Encoder (from Swin-UMamba)
# ---------------------------------------------------------------------------

class VSSMEncoder(nn.Module):
    """VMamba encoder with 4 hierarchical stages."""
    def __init__(self, in_channels=3, embed_dim=96, depths=(2, 2, 9, 2),
                 d_state=16, d_conv=3, expand=2, drop_path_rate=0.2):
        super().__init__()
        num_stages = len(depths)
        dims = [embed_dim * (2 ** i) for i in range(num_stages)]
        self.dims = dims
        self.patch_embed = PatchEmbed2D(4, in_channels, embed_dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i in range(num_stages):
            layer = VSSLayer(
                dim=dims[i], depth=depths[i], d_state=d_state,
                d_conv=d_conv, expand=expand,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                downsample=PatchMerging2D if i < num_stages - 1 else None,
            )
            self.layers.append(layer)

        self.norms = nn.ModuleList([nn.LayerNorm(dims[i]) for i in range(num_stages)])

    def forward(self, x):
        x = self.patch_embed(x)  # (B, H/4, W/4, C)
        features = []
        for i, layer in enumerate(self.layers):
            x_out, x = layer(x)
            x_out = self.norms[i](x_out)
            feat = x_out.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
            features.append(feat)
        return features


# ---------------------------------------------------------------------------
# SwinUMamba: main model
# ---------------------------------------------------------------------------

class SwinUMamba(nn.Module):
    """Swin-UMamba: VMamba encoder + UNETR-style decoder.

    Architecture:
      - VMamba encoder: 4 stages producing multi-scale features
      - Encoder0 block: additional conv block on stem features
      - UNETR decoder: 4 upsampling stages with skip connections
      - Final conv head

    Args:
        in_channels: Input image channels.
        num_classes: Number of output classes.
        img_size: Input image size.
        embed_dim: VMamba base embedding dimension.
        depths: Number of VSSBlocks per encoder stage.
        d_state: SS2D state dimension.
        drop_path_rate: Stochastic depth rate.
    """
    def __init__(self, in_channels=3, num_classes=2, img_size=224,
                 embed_dim=96, depths=None, d_state=16,
                 drop_path_rate=0.2, deep_supervision=False, **kwargs):
        super().__init__()
        if depths is None:
            depths = [2, 2, 9, 2]
        self.deep_supervision = deep_supervision

        dims = [embed_dim * (2 ** i) for i in range(len(depths))]
        self.dims = dims

        # VMamba encoder
        self.encoder = VSSMEncoder(
            in_channels=in_channels, embed_dim=embed_dim,
            depths=depths, d_state=d_state,
            drop_path_rate=drop_path_rate)

        # Encoder0: process input to get stem features at stride 4
        self.encoder0 = UnetrBasicBlock(in_channels, embed_dim, kernel_size=3)
        self.down0 = nn.Conv2d(embed_dim, embed_dim, kernel_size=4, stride=4)

        # Decoder
        # Stage 4: bottleneck → up + skip3
        self.decoder4 = UnetrUpBlock(dims[3], dims[2])
        # Stage 3: → up + skip2
        self.decoder3 = UnetrUpBlock(dims[2], dims[1])
        # Stage 2: → up + skip1
        self.decoder2 = UnetrUpBlock(dims[1], dims[0])
        # Stage 1: → up + encoder0 features
        self.decoder1 = UnetrUpBlock(dims[0], embed_dim)

        # Final head
        self.out = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=4, stride=4),
            UnetrBasicBlock(embed_dim // 2, embed_dim // 2),
            nn.Conv2d(embed_dim // 2, num_classes, 1),
        )

        self.apply(self._init_weights)

        if deep_supervision:
            self.ds_heads = nn.ModuleList([
                nn.Conv2d(dims[2], num_classes, 1),
                nn.Conv2d(dims[1], num_classes, 1),
                nn.Conv2d(dims[0], num_classes, 1),
            ])

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        input_size = x.shape[2:]

        # Encoder0: stem features
        enc0 = self.encoder0(x)  # (B, embed_dim, H, W)
        enc0_down = self.down0(enc0)  # (B, embed_dim, H/4, W/4)

        # VMamba encoder
        features = self.encoder(x)  # [f0, f1, f2, f3]

        # Decoder
        d4 = self.decoder4(features[3], features[2])
        d3 = self.decoder3(d4, features[1])
        d2 = self.decoder2(d3, features[0])
        d1 = self.decoder1(d2, enc0_down)

        # Final
        out = self.out(d1)
        if out.shape[2:] != input_size:
            out = F.interpolate(out, size=input_size, mode='bilinear',
                                align_corners=False)

        if self.training and self.deep_supervision:
            aux = []
            for feat, head in zip([d4, d3, d2], self.ds_heads):
                a = head(feat)
                if a.shape[2:] != input_size:
                    a = F.interpolate(a, size=input_size, mode='bilinear',
                                      align_corners=False)
                aux.append(a)
            return [out] + aux
        return out
