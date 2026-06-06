"""MUCM-Net: Mamba-powered UCM-Net for Skin Lesion Segmentation.

Reference:
    Yuan et al., "MUCM-Net: A Mamba Powered UCM-Net for Skin Lesion
    Segmentation", Exploration of Engineering Materials 2024.
    https://github.com/chunyuyuan/MUCM-Net

Architecture:
    * 6-stage UNet with very small channel dims [8,16,24,32,48,64,3].
    * Encoder: OverlapPatchEmbed downsample + UCMBlock (MLP + shifted-MLP)
      at each stage.
    * UCMBlock = MambaLayer (SSM) + shifted-window MLP for local context.
    * Decoder: OverlapPatchEmbed upsample + UCMBlock.
    * 1x1 conv final head.

Constructor:
    MUCMNet(in_channels=3, num_classes=2, img_size=256, **kwargs)
"""
# Source: https://github.com/chunyuyuan/MUCM-Net

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_, DropPath
from einops import rearrange

from medseg.models.encoders.vmunet_encoder import SS2D


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class _OverlapPatchEmbed(nn.Module):
    """Overlapping patch embedding (3x3 conv stride + LayerNorm)."""
    def __init__(self, img_size=256, patch_size=3, stride=2,
                 in_chans=3, embed_dim=8):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, patch_size,
                              stride=stride, padding=patch_size // 2)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm(x)
        return x, H, W


class _MambaBlock(nn.Module):
    """Single-directional Mamba block using project SS2D."""
    def __init__(self, dim, d_state=16, d_conv=3, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ss2d = SS2D(d_model=dim, d_state=d_state,
                         d_conv=d_conv, expand=expand)

    def forward(self, x):
        # x: (B, N, C)
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        # SS2D expects (B, H, W, C)
        x_hw = x.view(B, H, W, C)
        out = self.ss2d(x_hw)
        return out.view(B, N, C)


class _ShiftedMLP(nn.Module):
    """Lightweight shifted MLP for local context (UCM-style)."""
    def __init__(self, dim, mlp_ratio=1):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class UCMBlock(nn.Module):
    """UCM Block: Mamba + shifted MLP with residual."""
    def __init__(self, dim, num_heads=1, mlp_ratio=1,
                 drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, sr_ratio=8,
                 d_state=16, d_conv=3, expand=2):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.mamba = _MambaBlock(dim, d_state=d_state,
                                 d_conv=d_conv, expand=expand)
        self.norm2 = norm_layer(dim)
        self.mlp = _ShiftedMLP(dim, mlp_ratio=mlp_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.mamba(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# MUCM-Net
# ---------------------------------------------------------------------------

class MUCMNet(nn.Module):
    """MUCM-Net for skin lesion segmentation.

    Default embed_dims=[8,16,24,32,48,64,3] matches the original paper.
    """
    def __init__(self, in_channels=3, num_classes=2, img_size=256,
                 embed_dims=None, num_heads=None, mlp_ratios=None,
                 qkv_bias=False, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=None, sr_ratios=None, **kwargs):
        super().__init__()
        if embed_dims is None:
            embed_dims = [8, 16, 24, 32, 48, 64, 3]
        if depths is None:
            depths = [1, 1, 1]
        if sr_ratios is None:
            sr_ratios = [8, 4, 2, 1]
        if num_heads is None:
            num_heads = [1]
        if mlp_ratios is None:
            mlp_ratios = [4, 4, 4, 4]

        self.num_classes = num_classes
        self.img_size = img_size

        # -- Encoder conv --
        self.encoder1 = nn.Conv2d(in_channels, embed_dims[0], 3, 1, 1)
        self.ebn1 = nn.GroupNorm(4, embed_dims[0])

        # -- Norms --
        self.norm1 = norm_layer(embed_dims[1])
        self.norm2 = norm_layer(embed_dims[2])
        self.norm3 = norm_layer(embed_dims[3])
        self.norm4 = norm_layer(embed_dims[4])
        self.norm5 = norm_layer(embed_dims[5])

        self.dnorm2 = norm_layer(embed_dims[4])
        self.dnorm3 = norm_layer(embed_dims[3])
        self.dnorm4 = norm_layer(embed_dims[2])
        self.dnorm5 = norm_layer(embed_dims[1])
        self.dnorm6 = norm_layer(embed_dims[0])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, max(sum(depths), 1))]

        # -- Encoder UCM blocks --
        self.block_0_1 = nn.ModuleList([UCMBlock(
            dim=embed_dims[1], drop_path=dpr[0], sr_ratio=sr_ratios[0])])
        self.block0 = nn.ModuleList([UCMBlock(
            dim=embed_dims[2], drop_path=dpr[0], sr_ratio=sr_ratios[0])])
        self.block1 = nn.ModuleList([UCMBlock(
            dim=embed_dims[3], drop_path=dpr[0], sr_ratio=sr_ratios[0])])
        self.block2 = nn.ModuleList([UCMBlock(
            dim=embed_dims[4], drop_path=dpr[min(1, len(dpr)-1)], sr_ratio=sr_ratios[0])])
        self.block3 = nn.ModuleList([UCMBlock(
            dim=embed_dims[5], drop_path=dpr[min(1, len(dpr)-1)], sr_ratio=sr_ratios[0])])

        # -- Decoder UCM blocks --
        self.dblock0 = nn.ModuleList([UCMBlock(
            dim=embed_dims[4], drop_path=dpr[0], sr_ratio=sr_ratios[0])])
        self.dblock1 = nn.ModuleList([UCMBlock(
            dim=embed_dims[3], drop_path=dpr[0], sr_ratio=sr_ratios[0])])
        self.dblock2 = nn.ModuleList([UCMBlock(
            dim=embed_dims[2], drop_path=dpr[min(1, len(dpr)-1)], sr_ratio=sr_ratios[0])])
        self.dblock3 = nn.ModuleList([UCMBlock(
            dim=embed_dims[1], drop_path=dpr[min(1, len(dpr)-1)], sr_ratio=sr_ratios[0])])
        self.dblock4 = nn.ModuleList([UCMBlock(
            dim=embed_dims[0], drop_path=dpr[min(1, len(dpr)-1)], sr_ratio=sr_ratios[0])])

        # -- Patch embed (downsample) --
        self.patch_embed1 = _OverlapPatchEmbed(img_size, 3, 2, embed_dims[0], embed_dims[1])
        self.patch_embed2 = _OverlapPatchEmbed(img_size // 2, 3, 2, embed_dims[1], embed_dims[2])
        self.patch_embed3 = _OverlapPatchEmbed(img_size // 4, 3, 2, embed_dims[2], embed_dims[3])
        self.patch_embed4 = _OverlapPatchEmbed(img_size // 8, 3, 2, embed_dims[3], embed_dims[4])
        self.patch_embed5 = _OverlapPatchEmbed(img_size // 16, 3, 2, embed_dims[4], embed_dims[5])

        # -- Decoder 1x1 conv --
        self.decoder0 = nn.Conv2d(embed_dims[5], embed_dims[4], 1)
        self.decoder1 = nn.Conv2d(embed_dims[4], embed_dims[3], 1)
        self.decoder2 = nn.Conv2d(embed_dims[3], embed_dims[2], 1)
        self.decoder3 = nn.Conv2d(embed_dims[2], embed_dims[1], 1)
        self.decoder4 = nn.Conv2d(embed_dims[1], embed_dims[0], 1)
        self.decoder5 = nn.Conv2d(embed_dims[0], embed_dims[-1], 1)

        # -- Decoder BN --
        self.dbn0 = nn.GroupNorm(4, embed_dims[4])
        self.dbn1 = nn.GroupNorm(4, embed_dims[3])
        self.dbn2 = nn.GroupNorm(4, embed_dims[2])
        self.dbn3 = nn.GroupNorm(4, embed_dims[1])
        self.dbn4 = nn.GroupNorm(4, embed_dims[0])

        # -- Final head --
        self.final = nn.Conv2d(embed_dims[-1], num_classes, 1)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _run_blocks(self, blocks, x, H, W):
        for blk in blocks:
            x = blk(x)
        return x

    def forward(self, x_in):
        B = x_in.shape[0]
        H, W = x_in.shape[2:]

        # Encoder
        e1 = F.relu(F.group_norm(self.encoder1(x_in), 4))
        e1p, H1, W1 = self.patch_embed1(e1)
        e1p = self._run_blocks(self.block_0_1, e1p, H1, W1)

        e2p, H2, W2 = self.patch_embed2(e1p.view(B, H1, W1, -1).permute(0,3,1,2))
        e2p = self._run_blocks(self.block0, e2p, H2, W2)

        e3p, H3, W3 = self.patch_embed3(e2p.view(B, H2, W2, -1).permute(0,3,1,2))
        e3p = self._run_blocks(self.block1, e3p, H3, W3)

        e4p, H4, W4 = self.patch_embed4(e3p.view(B, H3, W3, -1).permute(0,3,1,2))
        e4p = self._run_blocks(self.block2, e4p, H4, W4)

        e5p, H5, W5 = self.patch_embed5(e4p.view(B, H4, W4, -1).permute(0,3,1,2))
        e5p = self._run_blocks(self.block3, e5p, H5, W5)

        # Bottleneck
        d5 = e5p.view(B, H5, W5, -1).permute(0,3,1,2)  # BCHW
        d5 = self.decoder0(d5)  # reduce channels
        d5 = F.relu(F.group_norm(d5, 4))
        d5 = F.interpolate(d5, scale_factor=2, mode='bilinear', align_corners=False)

        # Decoder stage 4
        skip4 = e4p.view(B, H4, W4, -1).permute(0,3,1,2)
        d4 = d5 + skip4
        d4_flat = rearrange(d4, 'b c h w -> b (h w) c')
        d4_flat = self._run_blocks(self.dblock0, d4_flat, H4, W4)
        d4 = d4_flat.view(B, H4, W4, -1).permute(0,3,1,2)
        d4 = F.relu(self.dbn0(d4))
        d4 = self.decoder1(d4)
        d4 = F.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=False)

        # Decoder stage 3
        skip3 = e3p.view(B, H3, W3, -1).permute(0,3,1,2)
        d3 = d4 + skip3
        d3_flat = rearrange(d3, 'b c h w -> b (h w) c')
        d3_flat = self._run_blocks(self.dblock1, d3_flat, H3, W3)
        d3 = d3_flat.view(B, H3, W3, -1).permute(0,3,1,2)
        d3 = F.relu(self.dbn1(d3))
        d3 = self.decoder2(d3)
        d3 = F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=False)

        # Decoder stage 2
        skip2 = e2p.view(B, H2, W2, -1).permute(0,3,1,2)
        d2 = d3 + skip2
        d2_flat = rearrange(d2, 'b c h w -> b (h w) c')
        d2_flat = self._run_blocks(self.dblock2, d2_flat, H2, W2)
        d2 = d2_flat.view(B, H2, W2, -1).permute(0,3,1,2)
        d2 = F.relu(self.dbn2(d2))
        d2 = self.decoder3(d2)
        d2 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False)

        # Decoder stage 1
        skip1 = e1p.view(B, H1, W1, -1).permute(0,3,1,2)
        d1 = d2 + skip1
        d1_flat = rearrange(d1, 'b c h w -> b (h w) c')
        d1_flat = self._run_blocks(self.dblock3, d1_flat, H1, W1)
        d1 = d1_flat.view(B, H1, W1, -1).permute(0,3,1,2)
        d1 = F.relu(self.dbn3(d1))
        d1 = self.decoder4(d1)
        d1 = F.interpolate(d1, scale_factor=2, mode='bilinear', align_corners=False)

        # Final decoder
        d0 = d1 + e1
        d0_flat = rearrange(d0, 'b c h w -> b (h w) c')
        d0_flat = self._run_blocks(self.dblock4, d0_flat, H, W)
        d0 = d0_flat.view(B, H, W, -1).permute(0,3,1,2)
        d0 = F.relu(self.dbn4(d0))
        d0 = self.decoder5(d0)

        out = self.final(d0)
        if out.shape[2:] != x_in.shape[2:]:
            out = F.interpolate(out, size=x_in.shape[2:],
                                mode='bilinear', align_corners=False)
        return out
