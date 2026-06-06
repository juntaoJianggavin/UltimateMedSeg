"""Mobile-U-ViT: Large Kernel + U-shaped Vision Transformer.

Lightweight hybrid network combining large-kernel depthwise convolutions
with U-shaped ViT for efficient medical image segmentation.

Reference:
    Mobile U-ViT: Revisiting large kernel and U-shaped ViT for efficient
    medical image segmentation. ACM MM 2025.
    https://github.com/FengheTan9/Mobile-U-ViT

Key components:
    - Large kernel depthwise conv stem
    - U-shaped ViT encoder-decoder with skip connections
    - Lightweight design suitable for mobile deployment
"""
# Source: https://github.com/FengheTan9/Mobile-U-ViT

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class _LargeKernelDWConv(nn.Module):
    """Large kernel depthwise separable convolution."""

    def __init__(self, dim, kernel_size=7):
        super().__init__()
        pad = kernel_size // 2
        self.dw = nn.Conv2d(dim, dim, kernel_size, 1, pad, groups=dim, bias=False)
        self.pw = nn.Conv2d(dim, dim, 1, bias=False)
        self.norm = nn.BatchNorm2d(dim)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.pw(self.dw(x))))


class _MobileBlock(nn.Module):
    """Mobile building block: large-kernel DWConv + FFN."""

    def __init__(self, dim, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.lk = _LargeKernelDWConv(dim, kernel_size=7)
        self.norm2 = nn.BatchNorm2d(dim)
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, dim, 1),
        )

    def forward(self, x):
        x = x + self.lk(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class _ViTBlock(nn.Module):
    """Lightweight ViT block with reduced heads."""

    def __init__(self, dim, num_heads=4, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = tokens + self.attn(self.norm1(tokens), self.norm1(tokens), self.norm1(tokens))[0]
        tokens = tokens + self.ffn(self.norm2(tokens))
        return tokens.transpose(1, 2).view(B, C, H, W)


class MobileUViT(nn.Module):
    """Mobile-U-ViT: Large kernel conv + U-shaped ViT.

    Args:
        in_channels: Input channels.
        num_classes: Segmentation classes.
        img_size: Input spatial size.
        embed_dims: Channel dims per stage.
        depths: Blocks per stage.
        use_vit: Use ViT blocks at bottleneck (True) or conv (False).
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        img_size: int = 224,
        embed_dims: Optional[List[int]] = None,
        depths: Optional[List[int]] = None,
        use_vit: bool = True,
        **kwargs,
    ):
        super().__init__()
        embed_dims = embed_dims or [32, 64, 128, 256]
        depths = depths or [2, 2, 2, 2]

        # Stem: large kernel
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, embed_dims[0], 7, 2, 3, bias=False),
            nn.BatchNorm2d(embed_dims[0]),
            nn.GELU(),
        )

        # Encoder
        self.enc_stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i in range(len(embed_dims)):
            blocks = nn.Sequential(*[_MobileBlock(embed_dims[i]) for _ in range(depths[i])])
            self.enc_stages.append(blocks)
            if i < len(embed_dims) - 1:
                self.downsamples.append(nn.Sequential(
                    nn.Conv2d(embed_dims[i], embed_dims[i + 1], 3, 2, 1, bias=False),
                    nn.BatchNorm2d(embed_dims[i + 1]),
                ))

        # Bottleneck: ViT block
        if use_vit:
            self.bottleneck = nn.Sequential(
                _ViTBlock(embed_dims[-1], num_heads=4),
                _ViTBlock(embed_dims[-1], num_heads=4),
            )
        else:
            self.bottleneck = nn.Sequential(
                _MobileBlock(embed_dims[-1]),
                _MobileBlock(embed_dims[-1]),
            )

        # Decoder
        self.upsamples = nn.ModuleList()
        self.dec_stages = nn.ModuleList()
        for i in range(len(embed_dims) - 1, 0, -1):
            self.upsamples.append(nn.Sequential(
                nn.ConvTranspose2d(embed_dims[i], embed_dims[i - 1], 2, 2),
            ))
            self.dec_stages.append(nn.Sequential(
                *[_MobileBlock(embed_dims[i - 1]) for _ in range(depths[i - 1])]
            ))

        # Head
        self.head = nn.Sequential(
            nn.Conv2d(embed_dims[0], embed_dims[0], 3, 1, 1, bias=False),
            nn.BatchNorm2d(embed_dims[0]),
            nn.GELU(),
            nn.Conv2d(embed_dims[0], num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H_in, W_in = x.shape[2:]
        x = self.stem(x)

        # Encoder
        skips = []
        for i, stage in enumerate(self.enc_stages):
            x = stage(x)
            skips.append(x)
            if i < len(self.downsamples):
                x = self.downsamples[i](x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for i, (up, dec) in enumerate(zip(self.upsamples, self.dec_stages)):
            x = up(x)
            skip = skips[len(skips) - 2 - i]
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = x + skip
            x = dec(x)

        x = F.interpolate(x, size=(H_in, W_in), mode="bilinear", align_corners=False)
        return self.head(x)
