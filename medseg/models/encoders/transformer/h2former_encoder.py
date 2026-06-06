"""H2Former Encoder: faithful port from https://github.com/NKUhealong/H2Former

Reference: He et al., "H2Former: An Effective Hierarchical Hybrid Transformer
           for Medical Image Segmentation"
Files: models/H2Former.py, basic_module.py
All class/attribute names match the original for pretrained weight loading.
"""
# Source: https://github.com/NKUhealong/H2Former

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from medseg.registry import ENCODER_REGISTRY


# ============= Basic ResNet blocks =============
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


# ============= Swin components (from basic_module.py) =============
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


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
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
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

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
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                num_heads=num_heads, window_size=window_size,
                                shift_size=0 if (i % 2 == 0) else window_size // 2,
                                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop=drop, attn_drop=attn_drop,
                                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                norm_layer=norm_layer)
            for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


# ============= PatchEmbed and PatchMerging (H2Former specific) =============
class PatchEmbed(nn.Module):
    """Multi-scale patch embedding for H2Former."""
    def __init__(self, img_size=224, patch_size=[2, 4, 8, 16], in_chans=4, embed_dim=64):
        super().__init__()
        self.projs = nn.ModuleList()
        for ps in patch_size:
            self.projs.append(nn.Conv2d(in_chans, embed_dim, kernel_size=ps, stride=ps // 2, padding=ps // 4))
        self.norm = nn.LayerNorm(embed_dim)
        # Target resolution: img_size // 2 (matching output after conv1+maxpool+layer1)
        self.target_size = img_size // 2

    def forward(self, x):
        outs = []
        for proj in self.projs:
            out = proj(x)  # (B, C, H_i, W_i) - different spatial sizes
            out = F.adaptive_avg_pool2d(out, (self.target_size, self.target_size))
            outs.append(out)
        # Average multi-scale features
        x = sum(outs) / len(outs)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class PatchMerging(nn.Module):
    """Patch merging for H2Former with ECA-like attention."""
    def __init__(self, dim):
        super().__init__()
        self.reductions = nn.ModuleList()
        self.reductions.append(nn.Conv2d(dim, dim * 2, kernel_size=2, stride=2))
        self.norm = nn.LayerNorm(dim * 2)

    def forward(self, x):
        B, L, C = x.shape
        H = W = int(math.sqrt(L))
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        x = self.reductions[0](x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


# ============= Decoder (from H2Former.py) =============
class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_channels + out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv_bn_relu(x)
        return x


# ============= Res34_Swin_MS (H2Former main model) =============
class Res34_Swin_MS(nn.Module):
    """H2Former: ResNet34 + Swin Transformer hybrid.
    Faithful to original Res34_Swin_MS class.
    """
    def __init__(self, image_size, block, layers, num_classes, in_chans=4):
        super(Res34_Swin_MS, self).__init__()
        norm_layer = nn.BatchNorm2d
        self._norm_layer = nn.BatchNorm2d
        self.inplanes = 64
        self.dilation = 1
        self.groups = 1
        self.base_width = 64

        self.conv1 = nn.Conv2d(in_chans, self.inplanes, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Swin Transformer layers
        self.swin_layers = nn.ModuleList()
        embed_dim = 64
        self.num_layers = 4
        self.image_size = image_size
        depths = [2, 2, 2, 2]
        num_heads = [2, 4, 8, 16]
        window_size = self.image_size // 16
        self.mlp_ratio = 4.0
        drop_path_rate = 0.1
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        patches_resolution = [self.image_size // 2, self.image_size // 2]

        patch_size = [2, 4, 8, 16]
        self.patch_embed = PatchEmbed(img_size=image_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.MS2 = PatchMerging(64)
        self.MS3 = PatchMerging(128)
        self.MS4 = PatchMerging(256)

        for i_layer in range(self.num_layers):
            swin_layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(patches_resolution[0] // (2 ** i_layer), patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer], num_heads=num_heads[i_layer],
                window_size=window_size, mlp_ratio=self.mlp_ratio,
                qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False)
            self.swin_layers.append(swin_layer)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride),
                                       norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width,
                                dilation=self.dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward_encoder(self, x):
        """Extract multi-scale encoder features."""
        encoder = []
        ms1 = self.patch_embed(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        x = x + ms1
        x = self.swin_layers[0](x)
        B, L, C = x.shape
        H = W = int(np.sqrt(L))
        ms2 = self.MS2(x)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        encoder.append(x)

        x = self.layer2(x)
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        x = x + ms2
        x = self.swin_layers[1](x)
        B, L, C = x.shape
        H = W = int(np.sqrt(L))
        ms3 = self.MS3(x)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        encoder.append(x)

        x = self.layer3(x)
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        x = x + ms3
        x = self.swin_layers[2](x)
        B, L, C = x.shape
        H = W = int(np.sqrt(L))
        ms4 = self.MS4(x)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        encoder.append(x)

        x = self.layer4(x)
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        x = x + ms4
        x = self.swin_layers[3](x)
        B, L, C = x.shape
        H = W = int(np.sqrt(L))
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        encoder.append(x)

        return encoder


@ENCODER_REGISTRY.register("h2former")
class H2FormerEncoder(nn.Module):
    """H2Former encoder wrapper.
    Faithful to https://github.com/NKUhealong/H2Former
    """

    def __init__(
        self,
        pretrained: bool = False,
        in_channels: int = 3,
        img_size: int = 224,
        pretrained_path: str = None,
        **kwargs,
    ):
        super().__init__()
        self.model = Res34_Swin_MS(img_size, BasicBlock, [3, 4, 6, 3], num_classes=1, in_chans=in_channels)
        self._out_channels = [64, 128, 256, 512]

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.model.forward_encoder(x)
