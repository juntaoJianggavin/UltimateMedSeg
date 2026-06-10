"""HSNet: Hybrid Semantic Network for Polyp Segmentation.

Reference:
    Wenchao Zhang et al. "HSNet: A hybrid semantic network for polyp segmentation."
    Computers in Biology and Medicine / MICCAI 2022.
    Upstream code: https://github.com/baiboat/HSNet (lib/pvt.py)

Architecture:
  - Encoder: PVTv2-B2 (channels [64, 128, 320, 512])
  - Decoder: HybridSemantic (OverlapPatchEmbed + Block/PVT-Attention + ECA + SpatialAttn
    + Bottleneck + upsample) cascaded with HybridAttention (ECA + SpatialAttn)
  - Multi-scale gating: pool→fc1→relu→fc2→sigmoid→weighted sum of 4 predictions
"""
# Source: https://github.com/baiboat/HSNet

from __future__ import annotations

import os
import math

os.environ.setdefault('HF_HUB_ETAG_TIMEOUT', '3')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '5')

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm
from medseg.utils.timm_compat import DropPath, to_2tuple, trunc_normal_

from medseg.models.networks.sam.sam_base import load_with_ssl_fallback


# ---------------------------------------------------------------------------
# PVT-style building blocks (from official lib/pvt.py)
# ---------------------------------------------------------------------------
class BasicConv2d(nn.Module):
    """Conv + BN (no ReLU - matches official implementation)."""

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.bn(self.conv(x))


class OverlapPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class DWConv_Mulit(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv_Mulit(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """PVT-style attention with spatial reduction. Returns (x, q, k)."""

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, q, k


class Block(nn.Module):
    """PVT transformer block. Returns (x, q, k)."""

    def __init__(self, dim, num_heads=1, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 sr_ratio=0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
                              sr_ratio=sr_ratio)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        msa, q, k = self.attn(self.norm1(x), H, W)
        x = x + msa
        x = x + self.mlp(self.norm2(x), H, W)
        return x, q, k


# ---------------------------------------------------------------------------
# HSNet-specific blocks
# ---------------------------------------------------------------------------
class Bottleneck(nn.Module):
    """1x1→3x3→1x1 with 1x1 shortcut, BN, ReLU at the end."""

    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.map = nn.Conv2d(in_planes, out_planes, kernel_size=1, padding=0, bias=False)
        self.conv0 = nn.Conv2d(in_planes, out_planes // 4, kernel_size=1, padding=0, bias=False)
        self.conv1 = nn.Conv2d(out_planes // 4, out_planes // 4, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_planes // 4, out_planes, kernel_size=1, padding=0, bias=False)
        self.bn0 = nn.BatchNorm2d(out_planes // 4)
        self.bn1 = nn.BatchNorm2d(out_planes // 4)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.bn_map = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_ = self.bn_map(self.map(x))
        x = self.relu(self.bn0(self.conv0(x)))
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(x_ + self.bn2(self.conv2(x)))
        return x


class Linear_Eca_block(nn.Module):
    """ECA: GAP -> 1D conv -> sigmoid -> expand to input shape."""

    def __init__(self):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=5, padding=5 // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avgpool(x)
        y = self.conv1d(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return y.expand_as(x)


class HybridAttention(nn.Module):
    """Split channels: ECA on first half, spatial attention on second half, concat."""

    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.eca = Linear_Eca_block()
        self.conv = BasicConv2d(in_planes // 2, out_planes // 2, 3, 1, 1)
        self.down_c = BasicConv2d(out_planes // 2, 1, 3, 1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        c = x.shape[1]
        x_t, x_c = torch.split(x, c // 2, dim=1)
        sa = self.sigmoid(self.down_c(x_c))
        gc = self.eca(x_t)
        x_c = self.conv(x_c)
        x_c = x_c * gc
        x_t = x_t * sa
        x = torch.cat((x_t, x_c), 1)
        return x


class HybridSenmentic(nn.Module):
    """Transformer-based hybrid semantic block.
    OverlapPatchEmbed → Block (PVT attention) → norm → q*k attention →
    ECA gate + Spatial gate → weighted → Bottleneck(input) → multiply → upsample 2x
    """

    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.patch_embed = OverlapPatchEmbed(
            img_size=224 // 4, patch_size=3, stride=1,
            in_chans=in_planes, embed_dim=out_planes)
        self.block = Block(dim=out_planes)
        self.norm = nn.LayerNorm(out_planes)
        self.gc = Linear_Eca_block()
        self.conv = Bottleneck(in_planes, out_planes)
        self.down_c = BasicConv2d(out_planes, 1, 3, 1, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        B = x.shape[0]
        x_t, H, W = self.patch_embed(x)
        x_t, q, k = self.block(x_t, H, W)
        x_t = self.norm(x_t)
        x_t = x_t.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        q = q.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        k = q.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        atten = q * k
        atten_c = self.gc(atten)
        atten_s = self.sigmoid(self.down_c(atten))
        x_t = x_t * atten_c * atten_s
        x_c = self.conv(x)
        x = x_t * x_c
        x = self.upsample(x)
        return x


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------
class Decoder(nn.Module):
    """Official HSNet decoder with HybridSemantic + HybridAttention cascade."""

    def __init__(self, embed_dims=[512, 320, 128, 64]):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.transferlayer = BasicConv2d(embed_dims[0], embed_dims[0], 1, padding=0)
        self.hs0 = HybridSenmentic(embed_dims[0], embed_dims[1])  # 512 -> 320
        self.hs1 = HybridSenmentic(embed_dims[1], embed_dims[2])  # 320 -> 128
        self.hs2 = HybridSenmentic(embed_dims[2], embed_dims[3])  # 128 -> 64
        self.hb0 = HybridAttention(embed_dims[0], embed_dims[0])  # 512
        self.hb1 = HybridAttention(embed_dims[1], embed_dims[1])  # 320
        self.hb2 = HybridAttention(embed_dims[2], embed_dims[2])  # 128
        self.hb3 = HybridAttention(embed_dims[3], embed_dims[3])  # 64

    def forward(self, pvt):
        x1 = pvt[0]  # 64 channels, stride 4
        x2 = pvt[1]  # 128 channels, stride 8
        x3 = pvt[2]  # 320 channels, stride 16
        x4 = pvt[3]  # 512 channels, stride 32
        x_4 = self.transferlayer(x4)
        x = self.hs0(x_4)          # 512->320, upsampled to stride 16
        x = x * self.hb1(x3)       # modulated by x3
        x_3 = x
        x = self.hs1(x)            # 320->128, upsampled to stride 8
        x = x * self.hb2(x2)       # modulated by x2
        x_2 = x
        x = self.hs2(x)            # 128->64, upsampled to stride 4
        x_1 = x * self.hb3(x1)     # modulated by x1
        return x_1, x_2, x_3, x_4


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------
class HSNet(nn.Module):
    """Hybrid Semantic Network (HSNet) for polyp segmentation.

    Args:
        in_channels: number of input image channels.
        num_classes: number of output classes.
        img_size: nominal input spatial size.
    """

    def __init__(self, in_channels=3, num_classes=2, img_size=224, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.img_size = img_size

        # Backbone (PVTv2-B2)
        self.backbone = load_with_ssl_fallback(
            timm.create_model, 'pvt_v2_b2',
            features_only=True, pretrained=True, in_chans=in_channels)

        enc_channels = self.backbone.feature_info.channels()
        if len(enc_channels) != 4:
            raise RuntimeError(
                'HSNet expects 4 encoder stages, got %d: %s' %
                (len(enc_channels), enc_channels))
        c1, c2, c3, c4 = enc_channels  # (64, 128, 320, 512)

        self.decoder = Decoder(embed_dims=[c4, c3, c2, c1])

        # Output heads
        self.out1 = nn.Conv2d(c1, num_classes, 1)
        self.out2 = nn.Conv2d(c2, num_classes, 1)
        self.out3 = nn.Conv2d(c3, num_classes, 1)
        self.out4 = nn.Conv2d(c4, num_classes, 1)

        # Multi-scale gating
        self.pooling = nn.AdaptiveAvgPool2d(1)
        gate_c = 64
        self.dc1 = nn.Conv2d(c1, gate_c, 1)
        self.dc2 = nn.Conv2d(c2, gate_c, 1)
        self.dc3 = nn.Conv2d(c3, gate_c, 1)
        self.dc4 = nn.Conv2d(c4, gate_c, 1)
        self.bn_dc1 = nn.BatchNorm2d(gate_c)
        self.bn_dc2 = nn.BatchNorm2d(gate_c)
        self.bn_dc3 = nn.BatchNorm2d(gate_c)
        self.bn_dc4 = nn.BatchNorm2d(gate_c)
        self.fc1 = nn.Linear(gate_c, gate_c)
        self.fc2 = nn.Linear(gate_c, 4)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        H_orig, W_orig = x.shape[-2:]
        # Encode
        pvt = self.backbone(x)
        # Decode
        x1, x2, x3, x4 = self.decoder(pvt)
        B = x1.shape[0]
        # Multi-scale gating
        y1 = self.pooling(self.bn_dc1(self.dc1(x1)))
        y2 = self.pooling(self.bn_dc2(self.dc2(x2)))
        y3 = self.pooling(self.bn_dc3(self.dc3(x3)))
        y4 = self.pooling(self.bn_dc4(self.dc4(x4)))
        y = y1 + y2 + y3 + y4
        coeff = self.sigmoid(self.fc2(self.relu(self.fc1(y.reshape(B, -1)))))
        # Per-scale predictions
        prediction1 = self.out1(x1) * coeff[:, 0].reshape(B, 1, 1, 1)
        prediction2 = self.out2(x2) * coeff[:, 1].reshape(B, 1, 1, 1)
        prediction3 = self.out3(x3) * coeff[:, 2].reshape(B, 1, 1, 1)
        prediction4 = self.out4(x4) * coeff[:, 3].reshape(B, 1, 1, 1)
        # Upsample to original resolution
        target = (H_orig, W_orig)
        p1 = F.interpolate(prediction1, size=target, mode='bilinear', align_corners=True)
        p2 = F.interpolate(prediction2, size=target, mode='bilinear', align_corners=True)
        p3 = F.interpolate(prediction3, size=target, mode='bilinear', align_corners=True)
        p4 = F.interpolate(prediction4, size=target, mode='bilinear', align_corners=True)
        return p1 + p2 + p3 + p4


__all__ = ['HSNet']
