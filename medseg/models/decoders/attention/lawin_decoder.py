"""Lawin Decoder - Large Window Attention decoder.
Reference: Yan et al. "Lawin Transformer: Improving Semantic Segmentation Transformer
           with Multi-Scale Representations via Large Window Attention"
Ported from: https://github.com/yan-hao-tian/lawin

Architecture: MLP projection on multi-scale features (SegFormer-style) ->
              Lawin attention spatial pyramid pooling -> low-level feature fusion.

Has its own internal multi-scale fusion mechanism.
External skip_connection parameter is IGNORED.
"""
# Source: https://arxiv.org/abs/2201.01615

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from medseg.registry import DECODER_REGISTRY


class MLP(nn.Module):
    """Linear Embedding - matches SegFormer/Lawin MLP class."""
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B,C,H,W -> B,N,C
        x = self.proj(x)
        return x


class PatchEmbed(nn.Module):
    """Patch embedding via adaptive pooling (for context downsampling)."""
    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(patch_size),
            nn.Conv2d(in_channels, embed_dim, 1, bias=False),
            nn.LayerNorm([embed_dim, patch_size, patch_size]),
        )

    def forward(self, x):
        return self.proj(x)


class LawinAttn(nn.Module):
    """Large Window Attention module - faithful to original.

    Uses query patches and multi-scale context for cross-attention.
    """
    def __init__(self, in_channels, reduction=2, use_scale=True, head=4, patch_size=8):
        super().__init__()
        self.head = head
        self.patch_size = patch_size
        self.use_scale = use_scale
        inter_channels = in_channels // reduction

        self.g = nn.Conv2d(in_channels, inter_channels, 1)
        self.theta = nn.Conv2d(in_channels, inter_channels, 1)
        self.phi = nn.Conv2d(in_channels, inter_channels, 1)
        self.conv_out = nn.Sequential(
            nn.Conv2d(inter_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, query, context):
        """
        query: (B*nH*nW, C, pH, pW) - query patches
        context: (B*nH*nW, C, cH, cW) - context patches (from PatchEmbed)
        """
        B, C, pH, pW = query.shape
        _, _, cH, cW = context.shape

        # Project
        g_x = self.g(context).reshape(B, self.head, -1, cH * cW)  # B, head, C', cN
        theta_x = self.theta(query).reshape(B, self.head, -1, pH * pW)  # B, head, C', qN
        phi_x = self.phi(context).reshape(B, self.head, -1, cH * cW)  # B, head, C', cN

        # Attention: theta^T @ phi -> softmax -> g
        pairwise = torch.matmul(theta_x.permute(0, 1, 3, 2), phi_x)  # B, head, qN, cN
        if self.use_scale:
            pairwise = pairwise / (theta_x.shape[2] ** 0.5)
        pairwise = F.softmax(pairwise, dim=-1)

        y = torch.matmul(pairwise, g_x.permute(0, 1, 3, 2))  # B, head, qN, C'
        y = y.permute(0, 1, 3, 2).reshape(B, -1, pH, pW)  # B, inter_ch, pH, pW
        y = self.conv_out(y)
        return query + y


@DECODER_REGISTRY.register("lawin")
class LawinDecoder(nn.Module):
    """Lawin decoder - faithful port of LawinHead.

    Architecture:
    1. MLP projection on c2, c3, c4 -> embed_dim, upsample to c2 resolution, fuse
    2. Lawin attention spatial pyramid at 3 scales (r=8,4,2) + short path + image pool
    3. Low-level fusion with c1 projection (48-dim)

    External skip_connection parameter is IGNORED.
    """
    has_internal_skip = True

    def __init__(self, encoder_channels: List[int], bottleneck_channels: int,
                 skip_connection=None, embed_dim: int = 768, reduction: int = 2,
                 patch_size: int = 8, **kwargs):
        super().__init__()
        all_channels = list(encoder_channels) + [bottleneck_channels]

        # Need at least 4 levels for faithful Lawin (c1, c2, c3, c4)
        # If fewer, pad with the last channel
        while len(all_channels) < 4:
            all_channels = [all_channels[0]] + all_channels

        self.n_levels = len(all_channels)
        self.patch_size = patch_size
        c1_ch = all_channels[0]
        mid_channels = all_channels[1:-1]  # c2..c_{n-1}
        c_deep = all_channels[-1]  # deepest

        # MLP projections for c2..c4 to embed_dim
        self.linear_layers = nn.ModuleList()
        for ch in mid_channels + [c_deep]:
            self.linear_layers.append(MLP(input_dim=ch, embed_dim=embed_dim))

        # Linear fuse: concat projected c2..c4 -> fused_channels
        fused_channels = 512
        n_mid = len(mid_channels) + 1  # number of projected features
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embed_dim * n_mid, fused_channels, 1, bias=False),
            nn.BatchNorm2d(fused_channels),
            nn.ReLU(inplace=True),
        )

        # Lawin attention at 3 scales
        self.lawin_8 = LawinAttn(fused_channels, reduction=reduction, head=64, patch_size=patch_size)
        self.lawin_4 = LawinAttn(fused_channels, reduction=reduction, head=16, patch_size=patch_size)
        self.lawin_2 = LawinAttn(fused_channels, reduction=reduction, head=4, patch_size=patch_size)

        # Context downsampling
        self.ds_8 = PatchEmbed(fused_channels, fused_channels, patch_size)
        self.ds_4 = PatchEmbed(fused_channels, fused_channels, patch_size)
        self.ds_2 = PatchEmbed(fused_channels, fused_channels, patch_size)

        # Short path + image pool
        self.short_path = nn.Sequential(
            nn.Conv2d(fused_channels, fused_channels, 1, bias=False),
            nn.BatchNorm2d(fused_channels),
            nn.ReLU(inplace=True),
        )
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(fused_channels, fused_channels, 1, bias=False),
            nn.BatchNorm2d(fused_channels),
            nn.ReLU(inplace=True),
        )

        # Cat: short_path + image_pool + 3 lawin outputs
        self.cat_conv = nn.Sequential(
            nn.Conv2d(fused_channels * 5, fused_channels, 1, bias=False),
            nn.BatchNorm2d(fused_channels),
            nn.ReLU(inplace=True),
        )

        # Low-level fusion with c1
        c1_embed = 48  # matches original
        self.linear_c1 = MLP(input_dim=c1_ch, embed_dim=c1_embed)
        self.low_level_fuse = nn.Sequential(
            nn.Conv2d(fused_channels + c1_embed, fused_channels, 1, bias=False),
            nn.BatchNorm2d(fused_channels),
            nn.ReLU(inplace=True),
        )

        self._out_channels = fused_channels

    @property
    def out_channels(self):
        return self._out_channels

    def _get_patches(self, x, patch_size):
        """Unfold feature map into non-overlapping patches."""
        B, C, H, W = x.shape
        nH, nW = H // patch_size, W // patch_size
        # Reshape to patches: (B*nH*nW, C, pH, pW)
        x = x.reshape(B, C, nH, patch_size, nW, patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(B * nH * nW, C, patch_size, patch_size)
        return x, nH, nW

    def _merge_patches(self, x, nH, nW, patch_size):
        """Fold patches back to feature map."""
        BnHnW, C, pH, pW = x.shape
        B = BnHnW // (nH * nW)
        x = x.reshape(B, nH, nW, C, pH, pW)
        x = x.permute(0, 3, 1, 4, 2, 5).reshape(B, C, nH * pH, nW * pW)
        return x

    def _get_context(self, x, patch_size, r, ds_module):
        """Get context for Lawin attention at scale r."""
        B, C, H, W = x.shape
        nH, nW = H // patch_size, W // patch_size
        # Unfold with larger window (r * patch_size) and stride patch_size
        pad = int((r - 1) / 2 * patch_size)
        x_padded = F.pad(x, [pad, pad, pad, pad], mode='reflect')
        # Use unfold to extract overlapping patches
        ctx = F.unfold(x_padded, kernel_size=patch_size * r, stride=patch_size)
        # ctx: (B, C*pH*pW, nH*nW)
        ctx = ctx.reshape(B, C, patch_size * r, patch_size * r, nH * nW)
        ctx = ctx.permute(0, 4, 1, 2, 3).reshape(B * nH * nW, C, patch_size * r, patch_size * r)
        # Downsample context to patch_size
        ctx = ds_module(ctx)
        return ctx

    def forward(self, bottleneck_feat: torch.Tensor, skip_features: List[torch.Tensor]) -> torch.Tensor:
        all_features = list(skip_features) + [bottleneck_feat]

        # Pad to at least 4 levels
        while len(all_features) < 4:
            all_features = [all_features[0]] + all_features

        c1 = all_features[0]
        mid_and_deep = all_features[1:]  # c2..c4

        # Target resolution: c2 (second highest)
        n = mid_and_deep[0].shape[0]
        target_h, target_w = mid_and_deep[0].shape[2:]

        # MLP projection on c2..c4
        projected = []
        for feat, linear in zip(mid_and_deep, self.linear_layers):
            _c = linear(feat)  # B, N, embed_dim
            _c = _c.permute(0, 2, 1).reshape(n, -1, feat.shape[2], feat.shape[3])
            if _c.shape[2:] != (target_h, target_w):
                _c = F.interpolate(_c, size=(target_h, target_w), mode='bilinear', align_corners=False)
            projected.append(_c)

        # Fuse c2..c4 projections
        _c = self.linear_fuse(torch.cat(projected, dim=1))  # (B, 512, H, W)

        # Ensure spatial dims are divisible by patch_size
        _, _, h, w = _c.shape
        ps = self.patch_size
        pad_h = (ps - h % ps) % ps
        pad_w = (ps - w % ps) % ps
        if pad_h > 0 or pad_w > 0:
            _c = F.pad(_c, [0, pad_w, 0, pad_h], mode='reflect')
        _, _, h_pad, w_pad = _c.shape

        # Lawin attention spatial pyramid
        output = [self.short_path(_c)]
        output.append(F.interpolate(self.image_pool(_c), size=(h_pad, w_pad),
                                     mode='bilinear', align_corners=False))

        # Get query patches
        query, nH, nW = self._get_patches(_c, ps)

        # Multi-scale Lawin attention
        for r, lawin, ds in [(8, self.lawin_8, self.ds_8),
                              (4, self.lawin_4, self.ds_4),
                              (2, self.lawin_2, self.ds_2)]:
            ctx = self._get_context(_c, ps, r, ds)
            _out = lawin(query, ctx)
            _out = self._merge_patches(_out, nH, nW, ps)
            output.append(_out)

        # Cat all outputs
        x = self.cat_conv(torch.cat(output, dim=1))

        # Remove padding if added
        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :h, :w]

        # Low-level fusion with c1
        _c1 = self.linear_c1(c1)
        _c1 = _c1.permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        x = F.interpolate(x, size=c1.shape[2:], mode='bilinear', align_corners=False)
        x = self.low_level_fuse(torch.cat([x, _c1], dim=1))

        return x
