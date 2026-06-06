"""DA-TransUNet: Dual Attention + Transformer U-Net for Medical Image Segmentation.

Combines TransUNet with spatial and channel dual attention blocks (DA-Block)
to enhance feature representation in skip connections.

Reference:
    DA-TransUNet: integrating spatial and channel dual attention with
    Transformer U-net for medical image segmentation.
    Frontiers in Bioengineering and Biotechnology, 2024.
    https://github.com/SUN-1024/DA-TransUnet

Key components:
    - ResNet-50 encoder with multi-scale feature extraction
    - Transformer bottleneck for global context modelling
    - Dual Attention Block (spatial + channel) in skip connections
    - Progressive decoder with bilinear upsampling
"""
# Source: https://github.com/SUN-1024/DA-TransUnet

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class _ConvBnReLU(nn.Module):
    def __init__(self, in_c, out_c, ks=3, stride=1, pad=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, ks, stride, pad, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class _ResBlock(nn.Module):
    """Basic residual block with optional stride-2 downsampling."""

    def __init__(self, in_c, out_c, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return F.relu(out + identity, inplace=True)


class _ResStage(nn.Module):
    """Stack of ResBlocks with optional initial downsampling."""

    def __init__(self, in_c, out_c, num_blocks, stride_first=1):
        super().__init__()
        layers = []
        ds = None
        if stride_first != 1 or in_c != out_c:
            ds = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride_first, bias=False),
                nn.BatchNorm2d(out_c),
            )
        layers.append(_ResBlock(in_c, out_c, stride_first, ds))
        for _ in range(1, num_blocks):
            layers.append(_ResBlock(out_c, out_c))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# ---------------------------------------------------------------------------
# Dual Attention Block
# ---------------------------------------------------------------------------

class _ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        B, C = x.shape[:2]
        w = self.pool(x).view(B, C)
        w = self.fc(w).view(B, C, 1, 1)
        return x * w


class _SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=pad, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.max(dim=1, keepdim=True)[0]
        w = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * w


class DualAttentionBlock(nn.Module):
    """Channel attention followed by spatial attention."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.ca = _ChannelAttention(channels, reduction)
        self.sa = _SpatialAttention()

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


# ---------------------------------------------------------------------------
# Transformer Bottleneck
# ---------------------------------------------------------------------------

class _TransformerBottleneck(nn.Module):
    """Lightweight transformer encoder for global context."""

    def __init__(self, embed_dim, num_heads=8, num_layers=4, mlp_ratio=4.0):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)
        return tokens.transpose(1, 2).view(B, C, H, W)


# ---------------------------------------------------------------------------
# DA-TransUNet
# ---------------------------------------------------------------------------

class DATransUNet(nn.Module):
    """DA-TransUNet: TransUNet with Dual Attention.

    Args:
        in_channels: Input channels (default 3).
        num_classes: Number of output segmentation classes.
        img_size: Input spatial size (default 224).
        embed_dims: Channel dims for each encoder stage.
        depths: Number of ResBlocks per stage.
        transformer_heads: Multi-head attention heads in transformer.
        transformer_layers: Number of transformer encoder layers.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        img_size: int = 224,
        embed_dims: Optional[List[int]] = None,
        depths: Optional[List[int]] = None,
        transformer_heads: int = 8,
        transformer_layers: int = 4,
        embed_dim: int = None,
        **kwargs,
    ):
        super().__init__()
        if embed_dim is not None:
            embed_dims = [embed_dim, embed_dim * 2, embed_dim * 4, embed_dim * 8]
        embed_dims = embed_dims or [64, 128, 256, 512]
        depths = depths or [3, 4, 6, 3]

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, embed_dims[0] // 2, 7, 2, 3, bias=False),
            nn.BatchNorm2d(embed_dims[0] // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dims[0] // 2, embed_dims[0], 3, 1, 1, bias=False),
            nn.BatchNorm2d(embed_dims[0]),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(3, 2, 1)

        # Encoder stages
        stages = []
        in_c = embed_dims[0]
        for i, (out_c, num_blocks) in enumerate(zip(embed_dims, depths)):
            stride = 2 if i > 0 else 1
            stages.append(_ResStage(in_c, out_c, num_blocks, stride))
            in_c = out_c
        self.stages = nn.ModuleList(stages)

        # Transformer bottleneck
        self.transformer = _TransformerBottleneck(
            embed_dims[-1], transformer_heads, transformer_layers,
        )

        # Dual attention blocks for skip connections
        self.da_blocks = nn.ModuleList([
            DualAttentionBlock(c) for c in embed_dims
        ])

        # Decoder: progressive upsampling + concat
        # At each step: input channels = embed_dims[-1-i]*2 (upsampled + skip)
        self.dec_convs = nn.ModuleList()
        for i in range(len(embed_dims) - 1):
            c_in = embed_dims[-1 - i] * 2
            c_out = embed_dims[-2 - i]
            self.dec_convs.append(nn.Sequential(
                _ConvBnReLU(c_in, c_out),
                _ConvBnReLU(c_out, c_out),
            ))

        # Final head
        self.head = nn.Sequential(
            _ConvBnReLU(embed_dims[0], embed_dims[0] // 2),
            nn.Conv2d(embed_dims[0] // 2, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H_in, W_in = x.shape[2:]

        # Stem + pool
        s0 = self.stem(x)          # (B, d0, H/2, W/2)
        s0_pool = self.pool(s0)    # (B, d0, H/4, W/4)

        # Encoder features (only stage outputs, not stem)
        feats = []
        cur = s0_pool
        for i, stage in enumerate(self.stages):
            cur = stage(cur) if i > 0 else stage(s0)
            feats.append(cur)
        # feats: [s1(d0), s2(d1), s3(d2), s4(d3)] - 4 features

        # Transformer bottleneck
        bot = self.transformer(feats[-1])

        # Decoder with DA skip connections
        x = bot
        for i, dec_conv in enumerate(self.dec_convs):
            skip_idx = len(feats) - 1 - i
            skip = self.da_blocks[skip_idx](feats[skip_idx])
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = dec_conv(x)

        # Upsample to input size and predict
        x = F.interpolate(x, size=(H_in, W_in), mode="bilinear", align_corners=False)
        return self.head(x)
