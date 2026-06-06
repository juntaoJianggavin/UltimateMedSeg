"""Rolling-UNet: Revitalizing MLP's Ability to Efficiently Extract
Long-Distance Dependencies for Medical Image Segmentation (AAAI 2024).

Faithful reimplementation from:
  https://github.com/Jiaoyang45/Rolling-Unet

Provides Rolling_Unet_S / M / L.
Default RollingUNet = Rolling_Unet_S.
"""
# Source: https://github.com/Jiaoyang45/Rolling-Unet

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


# ---------------------------------------------------------------------------
# Helper modules (from original repo)
# ---------------------------------------------------------------------------

class DWConv(nn.Module):
    """Depthwise convolution + pointwise convolution."""
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.point_conv = nn.Conv2d(dim, dim, 1, 1, 0, bias=True, groups=1)

    def forward(self, x, H, W):
        x = self.dwconv(x)
        x = self.point_conv(x)
        return x


class Lo2(nn.Module):
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
        self.dwconv = DWConv(hidden_features)
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

        # --- OR-MLP branch 1 (row-shift → col-shift) ---
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

        # --- OR-MLP branch 2 (col-shift → row-shift, opposite direction) ---
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


class Lo2Block(nn.Module):
    """Wrapper: LayerNorm → Lo2 with DropPath."""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Lo2(in_features=dim, hidden_features=mlp_hidden_dim,
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


class Feature_Incentive_Block(nn.Module):
    """Patch embedding with GELU activation (Feature Incentive Block)."""
    def __init__(self, img_size=224, patch_size=7, stride=4,
                 in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
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


class DoubleConv(nn.Module):
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


class D_DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


# ---------------------------------------------------------------------------
# Rolling-UNet  (S / M / L)
# ---------------------------------------------------------------------------

def _build_rolling_unet(in_channels, num_classes, img_size, embed_dims,
                        block2_drop_extra, final_ch,
                        drop_rate, drop_path_rate, depths):
    """Factory that builds a Rolling-UNet with the given hyperparameters."""
    return _RollingUNetBase(
        in_channels=in_channels, num_classes=num_classes, img_size=img_size,
        embed_dims=embed_dims, block2_drop_extra=block2_drop_extra,
        final_ch=final_ch, drop_rate=drop_rate,
        drop_path_rate=drop_path_rate, depths=depths)


class _RollingUNetBase(nn.Module):
    """Rolling-UNet base: hybrid Conv + Lo2-MLP architecture (AAAI 2024).

    Args:
        in_channels: Input image channels.
        num_classes: Number of segmentation classes.
        img_size: Input spatial resolution.
        embed_dims: Channel dims [enc1, enc2, enc3, Lo2-enc, bottleneck].
        block2_drop_extra: Extra dropout added to bottleneck Lo2Block.
        final_ch: Output channels of last decoder conv (before 1x1 head).
    """
    def __init__(self, in_channels=3, num_classes=2, img_size=224,
                 embed_dims=None, block2_drop_extra=0.1, final_ch=8,
                 drop_rate=0., drop_path_rate=0.,
                 depths=None, deep_supervision=False, **kwargs):
        super().__init__()
        if embed_dims is None:
            embed_dims = [16, 32, 64, 128, 256]
        if depths is None:
            depths = [1, 1, 1]
        self.deep_supervision = deep_supervision
        self.embed_dims = embed_dims
        num_heads = [1, 2, 4, 8]
        norm_layer = nn.LayerNorm
        sr_ratios = [8, 4, 2, 1]

        # ---- Encoder (Conv stages) ----
        self.conv1 = DoubleConv(in_channels, embed_dims[0])
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(embed_dims[0], embed_dims[1])
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(embed_dims[1], embed_dims[2])
        self.pool3 = nn.MaxPool2d(2)
        self.pool4 = nn.MaxPool2d(2)

        # ---- Encoder / Bottleneck (Lo2 stages) ----
        self.FIBlock1 = Feature_Incentive_Block(
            img_size=img_size // 4, patch_size=3, stride=1,
            in_chans=embed_dims[2], embed_dim=embed_dims[3])
        self.FIBlock2 = Feature_Incentive_Block(
            img_size=img_size // 8, patch_size=3, stride=1,
            in_chans=embed_dims[3], embed_dim=embed_dims[4])
        self.FIBlock3 = Feature_Incentive_Block(
            img_size=img_size // 8, patch_size=3, stride=1,
            in_chans=embed_dims[4], embed_dim=embed_dims[3])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.block1 = nn.ModuleList([Lo2Block(
            dim=embed_dims[3], num_heads=num_heads[0], mlp_ratio=1,
            drop=drop_rate, drop_path=dpr[0],
            norm_layer=norm_layer, sr_ratio=sr_ratios[0])])
        self.block2 = nn.ModuleList([Lo2Block(
            dim=embed_dims[4], num_heads=num_heads[0], mlp_ratio=1,
            drop=drop_rate + block2_drop_extra, drop_path=dpr[1],
            norm_layer=norm_layer, sr_ratio=sr_ratios[0])])
        self.block3 = nn.ModuleList([Lo2Block(
            dim=embed_dims[3], num_heads=num_heads[0], mlp_ratio=1,
            drop=drop_rate, drop_path=dpr[0],
            norm_layer=norm_layer, sr_ratio=sr_ratios[0])])

        self.norm1 = norm_layer(embed_dims[3])
        self.norm2 = norm_layer(embed_dims[4])
        self.norm3 = norm_layer(embed_dims[3])

        # ---- Decoder ----
        self.FIBlock4 = nn.Conv2d(embed_dims[3], embed_dims[2], 3, stride=1, padding=1)
        self.dbn4 = nn.BatchNorm2d(embed_dims[2])
        self.decoder3 = D_DoubleConv(embed_dims[2], embed_dims[1])
        self.decoder2 = D_DoubleConv(embed_dims[1], embed_dims[0])
        self.decoder1 = D_DoubleConv(embed_dims[0], final_ch)

        self.final = nn.Conv2d(final_ch, num_classes, kernel_size=1)

        if deep_supervision:
            self.ds_heads = nn.ModuleList([
                nn.Conv2d(embed_dims[2], num_classes, 1),
                nn.Conv2d(embed_dims[1], num_classes, 1),
                nn.Conv2d(embed_dims[0], num_classes, 1),
            ])

    def forward(self, x):
        B = x.shape[0]

        # ---- Conv Encoder ----
        out = self.conv1(x)
        t1 = out
        out = self.pool1(out)
        out = self.conv2(out)
        t2 = out
        out = self.pool2(out)
        out = self.conv3(out)
        t3 = out
        out = self.pool3(out)

        # ---- Lo2 Encoder Stage 4 ----
        out, H, W = self.FIBlock1(out)
        for blk in self.block1:
            out = blk(out, H, W)
        out = self.norm1(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out
        out = self.pool4(out)

        # ---- Bottleneck ----
        out, H, W = self.FIBlock2(out)
        for blk in self.block2:
            out = blk(out, H, W)
        out = self.norm2(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out, H, W = self.FIBlock3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.interpolate(out, scale_factor=(2, 2), mode='bilinear')

        # ---- Lo2 Decoder Stage 4 ----
        out = torch.add(out, t4)
        out = out.flatten(2).transpose(1, 2)
        for blk in self.block3:
            out = blk(out, H * 2, W * 2)
        out = self.norm3(out)
        out = out.reshape(B, H * 2, W * 2, -1).permute(0, 3, 1, 2).contiguous()
        out = F.interpolate(
            F.relu(self.dbn4(self.FIBlock4(out))),
            scale_factor=(2, 2), mode='bilinear')

        # ---- Conv Decoder ----
        ds_collect = self.training and self.deep_supervision
        intermediates = []

        out = torch.add(out, t3)
        if ds_collect:
            intermediates.append(out)
        out = F.interpolate(self.decoder3(out), scale_factor=(2, 2), mode='bilinear')
        out = torch.add(out, t2)
        if ds_collect:
            intermediates.append(out)
        out = F.interpolate(self.decoder2(out), scale_factor=(2, 2), mode='bilinear')
        out = torch.add(out, t1)
        if ds_collect:
            intermediates.append(out)
        out = self.decoder1(out)

        out = self.final(out)

        if ds_collect:
            input_size = out.shape[2:]
            aux = []
            for feat, head in zip(intermediates, self.ds_heads):
                a = head(feat)
                if a.shape[2:] != input_size:
                    a = F.interpolate(a, size=input_size, mode='bilinear',
                                      align_corners=False)
                aux.append(a)
            return [out] + aux
        return out


# ---------------------------------------------------------------------------
# Concrete size variants (matching original repo exactly)
# ---------------------------------------------------------------------------

class RollingUNet(_RollingUNetBase):
    """Rolling-UNet-S (Small): embed_dims=[16,32,64,128,256], final=8."""
    def __init__(self, in_channels=3, num_classes=2, img_size=224, **kwargs):
        super().__init__(
            in_channels=in_channels, num_classes=num_classes, img_size=img_size,
            embed_dims=[16, 32, 64, 128, 256],
            block2_drop_extra=0.1, final_ch=8, **kwargs)


class RollingUNet_M(_RollingUNetBase):
    """Rolling-UNet-M (Medium): embed_dims=[32,64,128,256,512], final=16."""
    def __init__(self, in_channels=3, num_classes=2, img_size=224, **kwargs):
        super().__init__(
            in_channels=in_channels, num_classes=num_classes, img_size=img_size,
            embed_dims=[32, 64, 128, 256, 512],
            block2_drop_extra=0.2, final_ch=16, **kwargs)


class RollingUNet_L(_RollingUNetBase):
    """Rolling-UNet-L (Large): embed_dims=[64,128,256,512,1024], final=32."""
    def __init__(self, in_channels=3, num_classes=2, img_size=224, **kwargs):
        super().__init__(
            in_channels=in_channels, num_classes=num_classes, img_size=img_size,
            embed_dims=[64, 128, 256, 512, 1024],
            block2_drop_extra=0.3, final_ch=32, **kwargs)
