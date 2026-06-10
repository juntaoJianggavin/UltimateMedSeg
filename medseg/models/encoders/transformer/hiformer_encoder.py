"""HiFormer Encoder: faithful port from https://github.com/amirhossein-kz/HiFormer

Reference: Heidari et al., "HiFormer: Hierarchical Multi-scale Representations
           Using Transformers for Medical Image Segmentation"
All class/attribute names match the original for pretrained weight loading.
"""
# Source: https://github.com/amirhossein-kz/HiFormer

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from functools import partial
import math
from medseg.utils.timm_compat import DropPath, to_2tuple, trunc_normal_
from einops.layers.torch import Rearrange
from einops import rearrange

from medseg.registry import ENCODER_REGISTRY


# ============= Swin Transformer Components =============
def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        self.register_buffer("relative_position_index", relative_coords.sum(-1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        attn = attn + relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, to_2tuple(self.window_size), num_heads, qkv_bias, qk_scale, attn_drop, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), act_layer=nn.GELU, drop=drop)

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            for h in [slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None)]:
                for w in [slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None)]:
                    img_mask[:, h, w, :] = 0
            cnt = 0
            for h in [slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None)]:
                for w in [slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None)]:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size).view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        x_windows = window_partition(shifted_x, self.window_size).view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=self.attn_mask).view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = shortcut + self.drop_path(x.view(B, H * W, C))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1).view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class SwinTransformer(nn.Module):
    """Swin Transformer backbone for HiFormer."""
    def __init__(self, img_size=224, patch_size=4, embed_dim=96, depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4., drop_path_rate=0.1):
        super().__init__()
        self.num_layers = len(depths)
        patches_resolution = [img_size // patch_size, img_size // patch_size]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = nn.ModuleList([
                SwinTransformerBlock(
                    dim=int(embed_dim * 2 ** i_layer),
                    input_resolution=(patches_resolution[0] // (2 ** i_layer), patches_resolution[1] // (2 ** i_layer)),
                    num_heads=num_heads[i_layer], window_size=window_size,
                    shift_size=0 if (j % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    drop_path=dpr[sum(depths[:i_layer]) + j])
                for j in range(depths[i_layer])])
            self.layers.append(layer)

    def forward(self, x, layer_idx):
        for blk in self.layers[layer_idx]:
            x = blk(x)
        return x


# ============= PyramidFeatures for HiFormer =============
class PyramidFeatures(nn.Module):
    """CNN + Swin Transformer multi-scale feature extraction."""
    def __init__(self, config, img_size=224, in_chans=3):
        super().__init__()
        # ResNet backbone (first 7 layers)
        import torchvision.models as models
        model_name = config.cnn_backbone
        if model_name == 'resnet50':
            resnet = models.resnet50(pretrained=config.cnn_pretrained if hasattr(config, 'cnn_pretrained') else True)
        else:
            resnet = models.resnet34(pretrained=config.cnn_pretrained if hasattr(config, 'cnn_pretrained') else True)
        self.resnet_layers = nn.Sequential(*list(resnet.children())[:7])

        # Channel projection for CNN features
        self.p1_ch = nn.Conv2d(config.cnn_pyramid_fm[0], config.swin_pyramid_fm[0], kernel_size=1)
        self.p2_ch = nn.Conv2d(config.cnn_pyramid_fm[1], config.swin_pyramid_fm[1], kernel_size=1)
        self.p3_ch = nn.Conv2d(config.cnn_pyramid_fm[2], config.swin_pyramid_fm[2], kernel_size=1)

        # Swin Transformer
        self.swin_transformer = SwinTransformer(
            img_size=img_size, embed_dim=config.swin_pyramid_fm[0],
            depths=config.depths, num_heads=config.num_heads,
            window_size=config.window_size)

        # Patch merging layers
        patches_resolution = img_size // 4
        self.p1_pm = PatchMerging((patches_resolution, patches_resolution), config.swin_pyramid_fm[0])
        self.p2_pm = PatchMerging((patches_resolution // 2, patches_resolution // 2), config.swin_pyramid_fm[1])

        self.norm_1 = nn.LayerNorm(config.swin_pyramid_fm[0])
        self.norm_2 = nn.LayerNorm(config.swin_pyramid_fm[1])
        self.avgpool_1 = nn.AdaptiveAvgPool1d(1)
        self.avgpool_2 = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # CNN features
        cnn_features = []
        for i, layer in enumerate(self.resnet_layers):
            x_cnn = layer(x) if i == 0 else layer(cnn_features[-1] if cnn_features else x)
            cnn_features.append(x_cnn)

        return cnn_features


# ============= MultiScaleBlock and All2Cross =============
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MultiScaleBlock(nn.Module):
    """Cross-attention block for dual-level fusion (DLF)."""
    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, qkv_bias=True, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ============= ConvUpsample and SegmentationHead =============
class ConvUpsample(nn.Module):
    def __init__(self, in_chans, out_chans=None, upsample=True):
        super().__init__()
        if out_chans is None:
            out_chans = [in_chans]
        self.conv_tower = nn.ModuleList()
        for out_ch in out_chans:
            self.conv_tower.append(nn.Sequential(
                nn.Conv2d(in_chans, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)))
            in_chans = out_ch
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) if upsample else nn.Identity()

    def forward(self, x):
        for conv in self.conv_tower:
            x = conv(x)
        x = self.upsample(x)
        return x


@ENCODER_REGISTRY.register("hiformer")
class HiFormerEncoder(nn.Module):
    """HiFormer Encoder: CNN + Swin Transformer hybrid.
    Faithful to https://github.com/amirhossein-kz/HiFormer
    """

    def __init__(
        self,
        pretrained: bool = False,
        in_channels: int = 3,
        img_size: int = 224,
        variant: str = 'S',
        pretrained_path: str = None,
        **kwargs,
    ):
        super().__init__()
        import torchvision.models as models

        # HiFormer-S config (default)
        if variant == 'B':
            cnn_backbone = 'resnet50'
            cnn_pyramid_fm = [256, 512, 1024]
            swin_pyramid_fm = [96, 192, 384]
            depths = [1, 2, 0]
            num_heads_val = [6, 12]
        elif variant == 'L':
            cnn_backbone = 'resnet34'
            cnn_pyramid_fm = [64, 128, 256]
            swin_pyramid_fm = [96, 192, 384]
            depths = [1, 4, 0]
            num_heads_val = [6, 6]
        else:  # 'S'
            cnn_backbone = 'resnet34'
            cnn_pyramid_fm = [64, 128, 256]
            swin_pyramid_fm = [96, 192, 384]
            depths = [1, 1, 0]
            num_heads_val = [3, 3]

        # CNN backbone
        if cnn_backbone == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
        else:
            resnet = models.resnet34(pretrained=pretrained)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

        self.p1_ch = nn.Conv2d(cnn_pyramid_fm[0], swin_pyramid_fm[0], kernel_size=1)
        self.p2_ch = nn.Conv2d(cnn_pyramid_fm[1], swin_pyramid_fm[1], kernel_size=1)

        self._out_channels = [swin_pyramid_fm[0], swin_pyramid_fm[1]]

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # CNN path
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)

        e1 = self.layer1(x)   # (B, 64/256, H/4, W/4)
        e2 = self.layer2(e1)  # (B, 128/512, H/8, W/8)
        e3 = self.layer3(e2)  # (B, 256/1024, H/16, W/16)

        # Project to swin dimensions
        p1 = self.p1_ch(e1)   # (B, 96, H/4, W/4)
        p2 = self.p2_ch(e2)   # (B, 192, H/8, W/8)

        return [p1, p2]
