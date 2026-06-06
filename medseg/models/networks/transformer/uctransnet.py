"""UCTransNet: Rethinking the Skip Connections in U-Net via Channel-wise Transformer.

Full end-to-end network using CNN encoder + CTrans (Channel-wise Cross-
fusion Transformer) module replacing conventional skip connections.

Reference:
    Wang et al., UCTransNet: Rethinking the Skip Connections in U-Net
    from a Channel-wise Perspective with Transformer. AAAI 2022.
    https://github.com/McGregorWwww/UCTransNet

Key components:
    - CNN encoder with multi-scale feature extraction
    - CTrans module: channel-wise cross-fusion transformer for skip
    - U-Net style decoder with concatenation
"""
# Source: https://github.com/McGregorWwww/UCTransNet

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


# ---------------------------------------------------------------------------
# CNN Encoder
# ---------------------------------------------------------------------------

class _DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class _Encoder(nn.Module):
    def __init__(self, in_channels, base_channels=64):
        super().__init__()
        self.enc1 = _DoubleConv(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = _DoubleConv(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = _DoubleConv(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = _DoubleConv(base_channels * 4, base_channels * 8)
        self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = _DoubleConv(base_channels * 8, base_channels * 16)
        self.out_channels = [base_channels * (2 ** i) for i in range(5)]

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        bn = self.bottleneck(self.pool4(e4))
        return [e1, e2, e3, e4, bn]


# ---------------------------------------------------------------------------
# CTrans: Channel-wise Cross-fusion Transformer
# ---------------------------------------------------------------------------

class _CTransModule(nn.Module):
    """Channel-wise Cross-fusion Transformer for multi-scale feature fusion.

    Performs self-attention on skip features to produce channel-refined skip
    features. For very high-resolution scales, spatial pooling is applied
    before attention and results are upsampled back to preserve tractability.
    """
    # Max spatial tokens before we apply pooling for attention
    _MAX_TOKENS = 4096

    def __init__(self, channels: List[int], num_heads: int = 4):
        super().__init__()
        self.num_scales = len(channels) - 1  # exclude bottleneck
        self.projections = nn.ModuleList()
        self.cross_attns = nn.ModuleList()
        self.fusions = nn.ModuleList()

        for i in range(self.num_scales):
            c = channels[i]
            self.projections.append(nn.Conv2d(c, c, 1))
            self.cross_attns.append(
                nn.MultiheadAttention(c, num_heads, batch_first=True)
            )
            self.fusions.append(nn.Sequential(
                nn.Conv2d(c * 2, c, 1, bias=False),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
            ))

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            features: [e1, e2, e3, e4, bottleneck]
        Returns:
            List of refined skip features [r1, r2, r3, r4]
        """
        skips = features[:-1]  # [e1, e2, e3, e4]
        refined = []

        for i in range(self.num_scales):
            skip = skips[i]
            B, C, H, W = skip.shape
            skip_proj = self.projections[i](skip)

            # Pool if spatial size is too large for efficient attention
            n_tokens = H * W
            pool_factor = 1
            if n_tokens > self._MAX_TOKENS:
                pool_factor = max(1, int((n_tokens / self._MAX_TOKENS) ** 0.5) + 1)
                skip_proj_pool = F.avg_pool2d(skip_proj, pool_factor)
                _, _, Hp, Wp = skip_proj_pool.shape
                tokens = skip_proj_pool.flatten(2).transpose(1, 2)
            else:
                Hp, Wp = H, W
                tokens = skip_proj.flatten(2).transpose(1, 2)

            attn_out, _ = self.cross_attns[i](tokens, tokens, tokens)

            if pool_factor > 1:
                attn_feat = attn_out.transpose(1, 2).view(B, C, Hp, Wp)
                attn_feat = F.interpolate(attn_feat, size=(H, W),
                                          mode='bilinear', align_corners=False)
            else:
                attn_feat = attn_out.transpose(1, 2).view(B, C, H, W)

            # Fuse attention with original skip
            fused = torch.cat([skip, attn_feat], dim=1)
            refined.append(self.fusions[i](fused))

        return refined


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class _Decoder(nn.Module):
    def __init__(self, channels: List[int]):
        super().__init__()
        # channels: [64, 128, 256, 512, 1024]
        self.up_convs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        for i in range(len(channels) - 1, 0, -1):
            self.up_convs.append(
                nn.ConvTranspose2d(channels[i], channels[i - 1], 2, 2)
            )
            self.dec_blocks.append(
                _DoubleConv(channels[i - 1] * 2, channels[i - 1])
            )

    def forward(self, bottleneck: torch.Tensor, skips: List[torch.Tensor]):
        x = bottleneck
        for i, (up, dec) in enumerate(zip(self.up_convs, self.dec_blocks)):
            x = up(x)
            skip = skips[len(skips) - 1 - i]
            # Handle size mismatch
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)
        return x


# ---------------------------------------------------------------------------
# UCTransNet
# ---------------------------------------------------------------------------

class UCTransNet(nn.Module):
    """UCTransNet: U-Net with Channel-wise Transformer skip connections.

    Args:
        in_channels: Input channels (default 3).
        num_classes: Number of segmentation classes.
        img_size: Input spatial size (default 224).
        base_channels: Base channel dimension (also accepts embed_dim).
        ctrans_heads: Number of attention heads in CTrans (also accepts num_heads).
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        img_size: int = 224,
        base_channels: int = 64,
        ctrans_heads: int = 4,
        embed_dim: Optional[int] = None,
        num_heads: Optional[int] = None,
        depths: Optional[List[int]] = None,
        drop_path_rate: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        # Support alternative parameter names from configs
        if embed_dim is not None:
            base_channels = embed_dim
        if num_heads is not None:
            ctrans_heads = num_heads

        channels = [base_channels * (2 ** i) for i in range(5)]

        self.encoder = _Encoder(in_channels, base_channels)
        self.ctrans = _CTransModule(channels, ctrans_heads)
        self.decoder = _Decoder(channels)
        self.head = nn.Conv2d(base_channels, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H_in, W_in = x.shape[2:]
        features = self.encoder(x)  # [e1, e2, e3, e4, bottleneck]

        # Refine skip connections via CTrans
        refined_skips = self.ctrans(features)

        # Decode
        decoded = self.decoder(features[-1], refined_skips)

        # Upsample and predict
        out = F.interpolate(decoded, size=(H_in, W_in), mode="bilinear", align_corners=False)
        return self.head(out)
