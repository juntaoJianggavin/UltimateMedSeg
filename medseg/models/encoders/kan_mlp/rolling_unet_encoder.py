"""Rolling-UNet encoder (AAAI 2024).

Extracted from ``medseg.models.networks.kan_mlp.rolling_unet``. Provides the
hybrid Conv + DOR-MLP encoder up to the deepest bottleneck feature.

Architecture (default ``embed_dims=[16, 32, 64, 128, 256]``):

    x  (B, 3, H, W)
      conv1 (DoubleConv 3 -> 16)                       -> t1  (16,  H,    W)
      pool1 -> conv2 (DoubleConv 16 -> 32)             -> t2  (32,  H/2,  W/2)
      pool2 -> conv3 (DoubleConv 32 -> 64)             -> t3  (64,  H/4,  W/4)
      pool3 -> FIBlock1 (3x3 s=1 pad=1) + Lo2Block     -> t4  (128, H/8,  W/8)
      pool4 -> FIBlock2 (3x3 s=1 pad=1) + Lo2Block     -> b   (256, H/16, W/16)

Returns five multi-scale features with the deepest LAST, per framework
convention.
"""
# Source: https://github.com/Jiaoyang45/Rolling-Unet

import math
from typing import List

import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from medseg.registry import ENCODER_REGISTRY


# ---------------------------------------------------------------------------
# Helper modules (inlined from medseg.models.networks.kan_mlp.rolling_unet)
# ---------------------------------------------------------------------------

class _DWConv(nn.Module):
    """Depthwise convolution + pointwise convolution."""

    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.point_conv = nn.Conv2d(dim, dim, 1, 1, 0, bias=True, groups=1)

    def forward(self, x, H, W):
        x = self.dwconv(x)
        x = self.point_conv(x)
        return x


class _Lo2(nn.Module):
    """DOR-MLP: Dual-direction OR-MLP + DSC (Depthwise Separable Conv)."""

    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0., shift_size=5):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(in_features, hidden_features)
        self.fc3 = nn.Linear(in_features, hidden_features)
        self.fc4 = nn.Linear(in_features, hidden_features)
        self.fc5 = nn.Linear(in_features * 2, hidden_features)
        self.fc6 = nn.Linear(hidden_features * 2, out_features)
        self.drop = nn.Dropout(drop)
        self.dwconv = _DWConv(hidden_features)
        self.act1 = act_layer()
        self.act2 = nn.ReLU()
        self.norm1 = nn.LayerNorm(hidden_features * 2)
        self.norm2 = nn.BatchNorm2d(hidden_features)
        self.shift_size = shift_size
        self.pad = shift_size // 2
        self.apply(self._init_weights)

    def _init_weights(self, m):
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

    def forward(self, x, H, W):
        B, N, C = x.shape

        # --- OR-MLP branch 1 (row-shift -> col-shift) ---
        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xs = torch.chunk(xn, C, 1)
        x_shift = [torch.roll(x_c, shift, 2)
                   for x_c, shift in zip(xs, range(0, C))]
        x_cat = torch.cat(x_shift, 1)
        x_s = x_cat.reshape(B, C, H * W).contiguous()
        x_shift_r = x_s.transpose(1, 2)
        x_shift_r = self.fc1(x_shift_r)
        x_shift_r = self.act1(x_shift_r)
        x_shift_r = self.drop(x_shift_r)
        xn = x_shift_r.transpose(1, 2).view(B, C, H, W).contiguous()
        xs = torch.chunk(xn, C, 1)
        x_shift = [torch.roll(x_c, shift, 3)
                   for x_c, shift in zip(xs, range(0, C))]
        x_cat = torch.cat(x_shift, 1)
        x_s = x_cat.reshape(B, C, H * W).contiguous()
        x_shift_c = x_s.transpose(1, 2)
        x_shift_c = self.fc2(x_shift_c)
        x_1 = self.drop(x_shift_c)

        # --- OR-MLP branch 2 (col-shift -> row-shift, opposite direction) ---
        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xs = torch.chunk(xn, C, 1)
        x_shift = [torch.roll(x_c, -shift, 3)
                   for x_c, shift in zip(xs, range(0, C))]
        x_cat = torch.cat(x_shift, 1)
        x_s = x_cat.reshape(B, C, H * W).contiguous()
        x_shift_c = x_s.transpose(1, 2)
        x_shift_c = self.fc3(x_shift_c)
        x_shift_c = self.act1(x_shift_c)
        x_shift_c = self.drop(x_shift_c)
        xn = x_shift_c.transpose(1, 2).view(B, C, H, W).contiguous()
        xs = torch.chunk(xn, C, 1)
        x_shift = [torch.roll(x_c, shift, 2)
                   for x_c, shift in zip(xs, range(0, C))]
        x_cat = torch.cat(x_shift, 1)
        x_s = x_cat.reshape(B, C, H * W).contiguous()
        x_shift_r = x_s.transpose(1, 2)
        x_shift_r = self.fc4(x_shift_r)
        x_2 = self.drop(x_shift_r)

        # Merge two OR-MLP branches
        x_1 = torch.add(x_1, x)
        x_2 = torch.add(x_2, x)
        x1 = torch.cat([x_1, x_2], dim=2)
        x1 = self.norm1(x1)
        x1 = self.fc5(x1)
        x1 = self.drop(x1)
        x1 = torch.add(x1, x)

        # --- DSC branch ---
        x2 = x.transpose(1, 2).view(B, C, H, W)
        x2 = self.dwconv(x2, H, W)
        x2 = self.act2(x2)
        x2 = self.norm2(x2)
        x2 = x2.flatten(2).transpose(1, 2)

        # Merge DOR-MLP + DSC
        x3 = torch.cat([x1, x2], dim=2)
        x3 = self.fc6(x3)
        x3 = self.drop(x3)
        return x3


class _Lo2Block(nn.Module):
    """Wrapper: LayerNorm -> Lo2 with DropPath."""

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = _Lo2(in_features=dim, hidden_features=mlp_hidden_dim,
                        act_layer=act_layer, drop=drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
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

    def forward(self, x, H, W):
        x = self.drop_path(self.mlp(x, H, W))
        return x


class _FeatureIncentiveBlock(nn.Module):
    """Patch embedding with GELU activation (Feature Incentive Block).

    Resolution-friendly: forward derives H, W from the runtime tensor shape
    so the ``img_size`` argument is informational only.
    """

    def __init__(self, img_size=224, patch_size=7, stride=4,
                 in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = (img_size[0] // patch_size[0],
                          img_size[1] // patch_size[1])
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=stride,
                              padding=(patch_size[0] // 2,
                                       patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)
        self.act = nn.GELU()
        self.apply(self._init_weights)

    def _init_weights(self, m):
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

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.act(x)
        x = self.norm(x)
        return x, H, W


class _DoubleConv(nn.Module):
    """Two Conv3x3-BN-ReLU blocks (encoder conv stage)."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

@ENCODER_REGISTRY.register("rolling_unet")
class RollingUNetEncoder(nn.Module):
    """Rolling-UNet encoder.

    3 stage stack of ``DoubleConv + MaxPool`` followed by 2 Lo2 (DOR-MLP)
    stages wrapped by ``Feature_Incentive_Block``. Returns five multi-scale
    feature maps with the deepest LAST.

    Args:
        in_channels: Input image channels. If not 3, a 1x1 conv stem is
            prepended to project to 3 channels.
        img_size: Reference input resolution. Used only to seed the
            ``Feature_Incentive_Block`` metadata; actual spatial dims are
            derived from the runtime tensor shape, so any resolution
            divisible by 16 works.
        pretrained: Unused (no public pretrained weights for Rolling-UNet).
        embed_dims: Channel dims [c1, c2, c3, c4, c5]; defaults to the
            Rolling-UNet-S configuration ``[16, 32, 64, 128, 256]``.
        depths: Number of Lo2Blocks per Lo2 stage; defaults to ``[1, 1]``.
        drop_rate: Dropout rate inside Lo2 blocks.
        drop_path_rate: Stochastic depth max rate (linearly scaled).
        block2_drop_extra: Additional dropout applied to the deepest
            (bottleneck) Lo2 block.
    """

    def __init__(self, in_channels: int = 3, img_size: int = 224,
                 pretrained: bool = False,
                 embed_dims: List[int] = None,
                 depths: List[int] = None,
                 drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 block2_drop_extra: float = 0.1,
                 **kwargs):
        super().__init__()
        if embed_dims is None:
            embed_dims = [16, 32, 64, 128, 256]
        if depths is None:
            depths = [1, 1]
        assert len(embed_dims) == 5, "embed_dims must have 5 entries"
        assert len(depths) == 2, "depths must have 2 entries (Lo2 stages)"

        # Optional 1x1 stem when in_channels != 3 — matches the framework
        # convention used by peer encoders.
        if in_channels != 3:
            self.stem = nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)
        else:
            self.stem = nn.Identity()

        self.embed_dims = embed_dims
        self.out_channels = list(embed_dims)  # [c1, c2, c3, c4, c5]

        num_heads = [1, 2, 4, 8]
        norm_layer = nn.LayerNorm
        sr_ratios = [8, 4, 2, 1]

        # ---- Conv encoder stages ----
        self.conv1 = _DoubleConv(3, embed_dims[0])
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = _DoubleConv(embed_dims[0], embed_dims[1])
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = _DoubleConv(embed_dims[1], embed_dims[2])
        self.pool3 = nn.MaxPool2d(2)
        self.pool4 = nn.MaxPool2d(2)

        # ---- Lo2 encoder / bottleneck stages ----
        # Feature_Incentive_Block uses stride=1, padding=1 so spatial size
        # is preserved; ``img_size`` is only used for stored metadata.
        self.FIBlock1 = _FeatureIncentiveBlock(
            img_size=max(img_size // 4, 1), patch_size=3, stride=1,
            in_chans=embed_dims[2], embed_dim=embed_dims[3])
        self.FIBlock2 = _FeatureIncentiveBlock(
            img_size=max(img_size // 8, 1), patch_size=3, stride=1,
            in_chans=embed_dims[3], embed_dim=embed_dims[4])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.block1 = nn.ModuleList([
            _Lo2Block(dim=embed_dims[3], num_heads=num_heads[0], mlp_ratio=1,
                      drop=drop_rate, drop_path=dpr[i],
                      norm_layer=norm_layer, sr_ratio=sr_ratios[0])
            for i in range(depths[0])
        ])
        self.block2 = nn.ModuleList([
            _Lo2Block(dim=embed_dims[4], num_heads=num_heads[0], mlp_ratio=1,
                      drop=drop_rate + block2_drop_extra,
                      drop_path=dpr[depths[0] + i],
                      norm_layer=norm_layer, sr_ratio=sr_ratios[0])
            for i in range(depths[1])
        ])

        self.norm1 = norm_layer(embed_dims[3])
        self.norm2 = norm_layer(embed_dims[4])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        B = x.shape[0]

        # Stage 1: conv (full res)
        out = self.conv1(x)
        t1 = out
        out = self.pool1(out)

        # Stage 2: conv (1/2)
        out = self.conv2(out)
        t2 = out
        out = self.pool2(out)

        # Stage 3: conv (1/4)
        out = self.conv3(out)
        t3 = out
        out = self.pool3(out)

        # Stage 4: Lo2 (1/8)
        out, H, W = self.FIBlock1(out)
        for blk in self.block1:
            out = blk(out, H, W)
        out = self.norm1(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out
        out = self.pool4(out)

        # Stage 5 (bottleneck): Lo2 (1/16)
        out, H, W = self.FIBlock2(out)
        for blk in self.block2:
            out = blk(out, H, W)
        out = self.norm2(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t5 = out

        return [t1, t2, t3, t4, t5]
