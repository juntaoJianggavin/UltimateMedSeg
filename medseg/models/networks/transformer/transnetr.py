"""TransNetR: Transformer-based Residual Network for Segmentation.

Encoder-decoder with transformer blocks at the bottleneck and residual
connections throughout for efficient biomedical image segmentation.

Reference:
    Jha et al., TransNetR: Transformer-based Residual Network for Polyp
    Segmentation with Multi-Center Out-of-Distribution Testing.
    PMLR 2024. https://github.com/DebeshJha/TransNetR

Key components:
    - CNN encoder with residual blocks
    - Transformer bottleneck for global context
    - Residual decoder with skip connections
"""
# Source: https://github.com/DebeshJha/TransNetR

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class _ResBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.shortcut = nn.Identity()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride, bias=False),
                nn.BatchNorm2d(out_c),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x), inplace=True)


class _TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        normed = self.norm1(tokens)
        attn_out, _ = self.attn(normed, normed, normed)
        tokens = tokens + attn_out
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens.transpose(1, 2).view(B, C, H, W)


class TransNetR(nn.Module):
    """TransNetR: Transformer + Residual network.

    Args:
        in_channels: Input channels.
        num_classes: Segmentation classes.
        img_size: Input spatial size.
        base_channels: Base channel dimension.
        transformer_depth: Number of transformer blocks at bottleneck.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        img_size: int = 224,
        base_channels: int = 32,
        transformer_depth: int = 4,
        embed_dim: int = None,
        **kwargs,
    ):
        super().__init__()
        if embed_dim is not None:
            base_channels = embed_dim
        c = base_channels

        # Encoder
        self.enc1 = nn.Sequential(_ResBlock(in_channels, c), _ResBlock(c, c))
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(_ResBlock(c, c * 2), _ResBlock(c * 2, c * 2))
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = nn.Sequential(_ResBlock(c * 2, c * 4), _ResBlock(c * 4, c * 4))
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = nn.Sequential(_ResBlock(c * 4, c * 8), _ResBlock(c * 8, c * 8))
        self.pool4 = nn.MaxPool2d(2)

        # Transformer bottleneck
        self.transformer = nn.Sequential(
            *[_TransformerBlock(c * 8) for _ in range(transformer_depth)]
        )

        # Decoder with residual connections
        self.up4 = nn.ConvTranspose2d(c * 8, c * 4, 2, 2)
        self.dec4 = nn.Sequential(_ResBlock(c * 8 + c * 4, c * 4), _ResBlock(c * 4, c * 4))
        self.up3 = nn.ConvTranspose2d(c * 4, c * 2, 2, 2)
        self.dec3 = nn.Sequential(_ResBlock(c * 4 + c * 2, c * 2), _ResBlock(c * 2, c * 2))
        self.up2 = nn.ConvTranspose2d(c * 2, c, 2, 2)
        self.dec2 = nn.Sequential(_ResBlock(c * 2 + c, c), _ResBlock(c, c))

        self.head = nn.Conv2d(c, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H_in, W_in = x.shape[2:]

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        bn = self.transformer(self.pool4(e4))

        d4 = self.up4(bn)
        if d4.shape[2:] != e4.shape[2:]:
            d4 = F.interpolate(d4, size=e4.shape[2:], mode="bilinear", align_corners=False)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        if d3.shape[2:] != e3.shape[2:]:
            d3 = F.interpolate(d3, size=e3.shape[2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        if d2.shape[2:] != e2.shape[2:]:
            d2 = F.interpolate(d2, size=e2.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        return self.head(F.interpolate(d2, size=(H_in, W_in), mode="bilinear", align_corners=False))
