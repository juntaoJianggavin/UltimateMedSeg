"""FCBFormer: FCN-Transformer Feature Fusion for Polyp Segmentation.

Faithful port from github.com/ESandML/FCBFormer (MIUA 2022).

Official architecture:
    - TB (Transformer Branch): PVTv2-B3 backbone with pyramid features
    - FCB (Fully Convolutional Branch): 6-level CNN with GroupNorm+SiLU+RB
    - LE (Local Enhancement): upsample each pyramid level to fixed size
    - SFA (Scale Feature Aggregation): top-down fusion of enhanced features
    - PH (Prediction Head): RB + RB + 1x1 conv

Reference:
    Sanderson & Matuszewski, FCN-Transformer Feature Fusion for Polyp
    Segmentation. MIUA 2022.
"""
# Source: https://github.com/ESandML/FCBFormer

from functools import partial
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


# ── PVTv2 Building Blocks ────────────────────────────────────────────────────

class _DWConv(nn.Module):
    """Depthwise convolution for spatial reduction in PVTv2."""
    def __init__(self, dim, sr_ratio):
        super().__init__()
        self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x_2d = x.transpose(1, 2).view(B, C, H, W)
        x_2d = self.sr(x_2d)
        x_2d = x_2d.flatten(2).transpose(1, 2)
        return self.norm(x_2d)


class _PVTAttention(nn.Module):
    """PVTv2 attention with spatial reduction."""
    def __init__(self, dim, num_heads, sr_ratio):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = _DWConv(dim, sr_ratio)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads)
        q = q.permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_sr = self.sr(x, H, W)
        else:
            x_sr = x

        kv = self.kv(x_sr).reshape(B, -1, 2, self.num_heads,
                                    C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class _PVTBlock(nn.Module):
    """PVTv2 transformer block."""
    def __init__(self, dim, num_heads, mlp_ratio, sr_ratio):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = _PVTAttention(dim, num_heads, sr_ratio)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.mlp(self.norm2(x))
        return x


class _PVTStage(nn.Module):
    """PVTv2 stage with patch embed + transformer blocks."""
    def __init__(self, in_dim, out_dim, depth, num_heads, mlp_ratio,
                 sr_ratio, patch_size=2):
        super().__init__()
        if in_dim > 0:
            self.patch_embed = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, patch_size, patch_size),
                nn.LayerNorm(out_dim, eps=1e-6))
        else:
            self.patch_embed = None
        self.blocks = nn.ModuleList(
            [_PVTBlock(out_dim, num_heads, mlp_ratio, sr_ratio)
             for _ in range(depth)])
        self.norm = nn.LayerNorm(out_dim, eps=1e-6)

    def forward(self, x):
        if self.patch_embed is not None:
            if isinstance(x, tuple):
                x, _, _ = x
            x = self.patch_embed[0](x)  # conv
            B, C, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)
            x = self.patch_embed[1](x)  # layernorm
        else:
            x, H, W = x
            B = x.shape[0]

        H_out = H if self.patch_embed is None else x.shape[1] // 1
        if self.patch_embed is not None:
            # Recompute H, W from sequence length
            N = x.shape[1]
            H = W = int(N ** 0.5)

        for blk in self.blocks:
            x = blk(x, H, W)

        x = self.norm(x)
        B = x.shape[0]
        x_2d = x.transpose(1, 2).view(B, -1, H, W).contiguous()
        return x_2d, H, W


class _PVTv2Backbone(nn.Module):
    """PVTv2-B3 backbone (self-contained, no pretrained weights).

    Config: embed_dims=[64,128,320,512], depths=[3,4,18,3],
            num_heads=[1,2,5,8], mlp_ratios=[8,8,4,4], sr_ratios=[8,4,2,1]
    """
    def __init__(self, in_channels=3):
        super().__init__()
        embed_dims = [64, 128, 320, 512]
        depths = [3, 4, 18, 3]
        num_heads = [1, 2, 5, 8]
        mlp_ratios = [8, 8, 4, 4]
        sr_ratios = [8, 4, 2, 1]

        # Stage 0: patch embed from image
        self.patch_embed1 = nn.Sequential(
            nn.Conv2d(in_channels, embed_dims[0], 4, 4),
            nn.LayerNorm(embed_dims[0], eps=1e-6))

        self.stages = nn.ModuleList()
        for i in range(4):
            in_d = embed_dims[i - 1] if i > 0 else 0
            patch_sz = 2 if i > 0 else 4
            self.stages.append(_PVTStage(
                in_d, embed_dims[i], depths[i], num_heads[i],
                mlp_ratios[i], sr_ratios[i], patch_sz))

    def forward(self, x):
        B = x.shape[0]
        # Stage 0: initial patch embedding
        x_pe = self.patch_embed1[0](x)  # conv
        H, W = x_pe.shape[2], x_pe.shape[3]
        x_seq = x_pe.flatten(2).transpose(1, 2)
        x_seq = self.patch_embed1[1](x_seq)  # layernorm

        pyramid = []
        inp = (x_seq, H, W)

        for i, stage in enumerate(self.stages):
            out, H, W = stage(inp)
            pyramid.append(out)
            # Prepare input for next stage
            inp = out

        return pyramid


# ── FCBFormer Building Blocks (from official) ────────────────────────────────

class RB(nn.Module):
    """ResBlock with GroupNorm(32) + SiLU (official implementation)."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        if out_channels == in_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        h = self.in_layers(x)
        h = self.out_layers(h)
        return h + self.skip(x)


class FCB(nn.Module):
    """Fully Convolutional Branch (official implementation).

    6-level CNN with GroupNorm+SiLU+RB, min_level_channels=32,
    channel_mults=[1,1,2,2,4,4].
    """
    def __init__(
        self,
        in_channels=3,
        min_level_channels=32,
        min_channel_mults=[1, 1, 2, 2, 4, 4],
        n_levels_down=6,
        n_levels_up=6,
        n_RBs=2,
        in_resolution=352,
    ):
        super().__init__()
        self.enc_blocks = nn.ModuleList(
            [nn.Conv2d(in_channels, min_level_channels, kernel_size=3,
                       padding=1)])
        ch = min_level_channels
        enc_block_chans = [min_level_channels]
        for level in range(n_levels_down):
            min_channel_mult = min_channel_mults[level]
            for block in range(n_RBs):
                self.enc_blocks.append(
                    nn.Sequential(RB(ch, min_channel_mult * min_level_channels))
                )
                ch = min_channel_mult * min_level_channels
                enc_block_chans.append(ch)
            if level != n_levels_down - 1:
                self.enc_blocks.append(
                    nn.Sequential(nn.Conv2d(ch, ch, kernel_size=3, padding=1,
                                            stride=2)))
                enc_block_chans.append(ch)

        self.middle_block = nn.Sequential(RB(ch, ch), RB(ch, ch))

        self.dec_blocks = nn.ModuleList([])
        for level in range(n_levels_up):
            min_channel_mult = min_channel_mults[::-1][level]
            for block in range(n_RBs + 1):
                layers = [
                    RB(
                        ch + enc_block_chans.pop(),
                        min_channel_mult * min_level_channels,
                    )
                ]
                ch = min_channel_mult * min_level_channels
                if level < n_levels_up - 1 and block == n_RBs:
                    layers.append(
                        nn.Sequential(
                            nn.Upsample(scale_factor=2, mode="nearest"),
                            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
                        )
                    )
                self.dec_blocks.append(nn.Sequential(*layers))

    def forward(self, x):
        hs = []
        h = x
        for module in self.enc_blocks:
            h = module(h)
            hs.append(h)
        h = self.middle_block(h)
        for module in self.dec_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h = module(cat_in)
        return h


class TB(nn.Module):
    """Transformer Branch (official implementation).

    Uses PVTv2 backbone with Local Enhancement (LE) and
    Scale Feature Aggregation (SFA) modules.
    """
    def __init__(self, in_channels=3):
        super().__init__()
        self.backbone = _PVTv2Backbone(in_channels)

        # Local Enhancement modules - upsample each pyramid level to fixed size
        self.LE = nn.ModuleList([])
        for i in range(4):
            self.LE.append(
                nn.Sequential(
                    RB([64, 128, 320, 512][i], 64), RB(64, 64),
                )
            )
        # Store target upsample size (computed dynamically in forward)
        self._target_size = None

        # Scale Feature Aggregation
        self.SFA = nn.ModuleList([])
        for i in range(3):
            self.SFA.append(nn.Sequential(RB(128, 64), RB(64, 64)))

    def forward(self, x):
        B, _, H_in, W_in = x.shape
        pyramid = self.backbone(x)

        # Compute target size for LE upsample (H_in/4 to match FCB output)
        target_h = H_in // 4
        target_w = W_in // 4

        pyramid_emph = []
        for i, level in enumerate(pyramid):
            le = self.LE[i](level)
            le = F.interpolate(le, size=(target_h, target_w), mode="bilinear",
                               align_corners=False)
            pyramid_emph.append(le)

        l_i = pyramid_emph[-1]
        for i in range(2, -1, -1):
            l = torch.cat((pyramid_emph[i], l_i), dim=1)
            l = self.SFA[i](l)
            l_i = l

        return l


class FCBFormer(nn.Module):
    """FCBFormer: FCN-Transformer Feature Fusion (official architecture).

    TB (PVTv2) + FCB (GroupNorm+SiLU CNN) -> concat -> PH (RB+RB+conv).

    Args:
        in_channels: Input channels (default 3).
        num_classes: Output segmentation classes (default 2).
        img_size: Input spatial size (default 224).
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        img_size: int = 224,
        **kwargs,
    ):
        super().__init__()
        self.TB = TB(in_channels)
        self.FCB = FCB(in_channels=in_channels, in_resolution=img_size)
        self.PH = nn.Sequential(
            RB(64 + 32, 64), RB(64, 64),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )
        self.img_size = img_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H_in, W_in = x.shape[2:]
        x1 = self.TB(x)
        x2 = self.FCB(x)
        # Upsample TB output to match FCB output size (full resolution)
        x1 = F.interpolate(x1, size=(H_in, W_in), mode="bilinear",
                           align_corners=False)
        x_cat = torch.cat((x1, x2), dim=1)
        out = self.PH(x_cat)
        return out
