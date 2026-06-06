"""H2Former – self-contained port from github.com/NKUhealong/H2Former.

H2Former: Hybrid Hierarchical Transformer for Medical Image Segmentation.
Architecture: ResNet-34 + Swin Transformer dual-branch with multi-scale
              patch embedding/merging (ECA-enhanced) + UNet decoder.
"""
# Source: https://github.com/NKUhealong/H2Former

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# ECA layer
# ---------------------------------------------------------------------------
class _ECA(nn.Module):
    def __init__(self, channel, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, k_size, padding=(k_size - 1) // 2,
                              bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y) + x


# ---------------------------------------------------------------------------
# Channel attention (for Swin blocks)
# ---------------------------------------------------------------------------
class _ChannelAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                   C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        k = k * self.scale
        attention = (k.transpose(-1, -2) @ v).softmax(dim=-1)
        x = (attention @ q.transpose(-1, -2)).transpose(-1, -2)
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class _ChannelBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _ChannelAttention(dim, num_heads, qkv_bias)
        self.drop_path = nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = _MlpWithECA(dim, mlp_hidden)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))


# ---------------------------------------------------------------------------
# MLP with ECA
# ---------------------------------------------------------------------------
class _MlpWithECA(nn.Module):
    def __init__(self, in_features, hidden_features=None, drop=0.):
        super().__init__()
        hidden = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, in_features)
        self.drop = nn.Dropout(drop)
        self.eca = _ECA(in_features)

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        x = self.act(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        x = x.view(B, C, H, W)
        x = self.eca(x)
        return x.flatten(2).transpose(1, 2)


# ---------------------------------------------------------------------------
# Window attention (Swin-style)
# ---------------------------------------------------------------------------
def _window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size,
               W // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(
        -1, window_size, window_size, C)


def _window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size,
                     window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


class _WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                        num_heads))
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flat = torch.flatten(coords, 1)
        relative_coords = coords_flat[:, :, None] - coords_flat[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        self.register_buffer("relative_position_index",
                             relative_coords.sum(-1))
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads,
                                   C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)
        rpb = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1], -1)
        attn = attn + rpb.permute(2, 0, 1).contiguous().unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(x)


class _SwinBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7,
                 shift_size=0, mlp_ratio=4., qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size
        H, W = input_resolution
        if min(H, W) <= window_size:
            self.shift_size = 0
            self.window_size = min(H, W)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _WindowAttention(dim, (self.window_size, self.window_size),
                                     num_heads, qkv_bias)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = _MlpWithECA(dim, mlp_hidden)
        if self.shift_size > 0:
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = h_slices
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = _window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size ** 2)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0)
            attn_mask = attn_mask.masked_fill(attn_mask == 0, 0.0)
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size),
                                   dims=(1, 2))
        else:
            shifted_x = x
        x_windows = _window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size ** 2, C)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size,
                                         self.window_size, C)
        shifted_x = _window_reverse(attn_windows, self.window_size, H, W)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size),
                           dims=(1, 2))
        else:
            x = shifted_x
        x = shortcut + x.view(B, H * W, C)
        return x + self.mlp(self.norm2(x))


class _BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size=7,
                 mlp_ratio=4., qkv_bias=True):
        super().__init__()
        self.blocks = nn.ModuleList([
            _SwinBlock(dim, input_resolution, num_heads, window_size,
                       0 if (i % 2 == 0) else window_size // 2,
                       mlp_ratio, qkv_bias)
            for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


# ---------------------------------------------------------------------------
# ResNet BasicBlock
# ---------------------------------------------------------------------------
def _conv3x3(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)


def _conv1x1(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, 1, stride, bias=False)


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = _conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


# ---------------------------------------------------------------------------
# Multi-scale patch embedding / merging (ECA-enhanced)
# ---------------------------------------------------------------------------
class _PatchEmbed(nn.Module):
    """Multi-scale patch embedding with ECA."""

    def __init__(self, img_size=224, patch_size=None, in_chans=3,
                 embed_dim=64):
        super().__init__()
        if patch_size is None:
            patch_size = [2, 4, 8, 16]
        self.projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_chans, embed_dim, ps, ps, 0, bias=False),
                _ECA(embed_dim))
            for ps in patch_size])

    def forward(self, x):
        outs = [proj(x) for proj in self.projs]
        # Average multi-scale embeddings
        out = outs[0]
        for o in outs[1:]:
            if o.shape != out.shape:
                o = F.interpolate(o, size=out.shape[-2:], mode='bilinear',
                                  align_corners=True)
            out = out + o
        B, C, H, W = out.shape
        return out.flatten(2).transpose(1, 2)


class _PatchMerging(nn.Module):
    """ECA-enhanced patch merging."""

    def __init__(self, dim):
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)
        self.eca = _ECA(2 * dim)

    def forward(self, x):
        B, L, C = x.shape
        H = W = int(np.sqrt(L))
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.reduction(self.norm(x))
        # Apply ECA
        B, N, C2 = x.shape
        h = w = int(np.sqrt(N))
        x = x.view(B, h, w, C2).permute(0, 3, 1, 2)
        x = self.eca(x)
        return x.flatten(2).transpose(1, 2)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------
class _Decoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear',
                              align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + out_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear',
                              align_corners=True)
        return self.conv(torch.cat([x, skip], dim=1))


# ---------------------------------------------------------------------------
# H2Former
# ---------------------------------------------------------------------------
class H2Former(nn.Module):
    """H2Former: ResNet-34 + Swin Transformer with multi-scale fusion.

    Args:
        in_channels: Number of input image channels (default 3).
        num_classes: Number of output segmentation classes (default 2).
        img_size: Expected input spatial resolution (default 224).
    """

    def __init__(self, in_channels=3, num_classes=2, img_size=224, **kwargs):
        super().__init__()
        norm_layer = nn.BatchNorm2d
        self.inplanes = 64
        embed_dim = 64
        depths = [2, 2, 2, 2]
        num_heads = [2, 4, 8, 16]
        window_size = img_size // 16
        mlp_ratio = 4.0
        drop_path_rate = 0.1
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        patches_resolution = [img_size // 2, img_size // 2]
        patch_size = [2, 4, 8, 16]
        # ResNet encoder
        self.conv1 = nn.Conv2d(in_channels, 64, 7, 1, 3, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make_layer(64, 64, 3, norm_layer)
        self.layer2 = self._make_layer(64, 128, 4, norm_layer, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, norm_layer, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, norm_layer, stride=2)
        # Multi-scale patch embedding
        self.patch_embed = _PatchEmbed(img_size, patch_size, in_channels,
                                       embed_dim)
        # Multi-scale merging
        self.MS2 = _PatchMerging(64)
        self.MS3 = _PatchMerging(128)
        self.MS4 = _PatchMerging(256)
        # Swin layers
        self.swin_layers = nn.ModuleList()
        for i in range(4):
            dim_i = embed_dim * (2 ** i)
            res_i = (patches_resolution[0] // (2 ** i),
                     patches_resolution[1] // (2 ** i))
            self.swin_layers.append(_BasicLayer(
                dim_i, res_i, depths[i], num_heads[i], window_size,
                mlp_ratio, True))
        # Decoder
        channels = [64, 128, 256, 512]
        self.decode4 = _Decoder(channels[3], channels[2])
        self.decode3 = _Decoder(channels[2], channels[1])
        self.decode2 = _Decoder(channels[1], channels[0])
        self.decode0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(channels[0], num_classes, 1, bias=False))

    def _make_layer(self, in_ch, planes, blocks, norm_layer, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                _conv1x1(self.inplanes, planes, stride),
                norm_layer(planes))
        layers = [_BasicBlock(self.inplanes, planes, stride, downsample,
                              norm_layer=norm_layer)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(_BasicBlock(planes, planes, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Multi-scale patch embedding
        ms1 = self.patch_embed(x)
        # ResNet path
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        B, C, H, W = x.shape  # e.g. (B, 64, 112, 112)
        # Fuse ResNet + Swin at level 1
        x_flat = x.flatten(2).transpose(1, 2)
        x_flat = x_flat + ms1
        x_flat = self.swin_layers[0](x_flat)
        ms2 = self.MS2(x_flat)  # halves spatial, doubles channels
        H2, W2 = H // 2, W // 2
        enc1 = x_flat.view(B, H, W, -1).permute(0, 3, 1, 2)
        # Level 2
        x = self.layer2(enc1) + ms2.view(B, H2, W2, -1).permute(0, 3, 1, 2)
        x_flat = x.flatten(2).transpose(1, 2)
        x_flat = self.swin_layers[1](x_flat)
        ms3 = self.MS3(x_flat)
        H3, W3 = H2 // 2, W2 // 2
        enc2 = x
        # Level 3
        x = self.layer3(enc2)
        x_flat = x.flatten(2).transpose(1, 2) + ms3
        x_flat = self.swin_layers[2](x_flat)
        ms4 = self.MS4(x_flat)
        H4, W4 = H3 // 2, W3 // 2
        enc3 = x
        # Level 4
        x = self.layer4(enc3)
        x_flat = x.flatten(2).transpose(1, 2) + ms4
        x_flat = self.swin_layers[3](x_flat)
        enc4 = x_flat.view(B, H4, W4, -1).permute(0, 3, 1, 2)
        # Decode
        d4 = self.decode4(enc4, enc3)
        d3 = self.decode3(d4, enc2)
        d2 = self.decode2(d3, enc1)
        return self.decode0(d2)
