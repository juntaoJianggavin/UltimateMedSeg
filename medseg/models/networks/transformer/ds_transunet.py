"""DS-TransUNet: Dual Swin Transformer U-Net for Medical Image Segmentation
Source: https://github.com/TianBaoGe/DS-TransUNet (master branch)
Official file: lib/DS_TransUNet.py (1020 lines)

Architecture Overview:
  - Encoder1: SwinTransformer (embed_dim=128, depths=[2,2,18,2], num_heads=[4,8,16,32], patch_size=4)
  - Encoder2: SwinTransformer (embed_dim=96, depths=[2,2,6,2], num_heads=[3,6,12,24], patch_size=8)
  - Cross_Att: Transformer-based cross-attention (PreNorm + Attention + FeedForward) at each stage
  - Decoder: Swin_Decoder (stages 1-3) + Decoder (stages 4-5) with skip connections
  - Shallow branch: down1 (1x1 conv) + down2 (conv_block with MaxPool)
  - Deep supervision: loss1 (from bottleneck e4), loss2 (from d3), final output
  - All normalization: GroupNorm(32) throughout (except LayerNorm in transformer blocks)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

groups = 32


# ==================== Transformer components for Cross_Att ====================
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(t.shape[0], t.shape[1], self.heads, -1).permute(0, 2, 1, 3), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(out.shape[0], out.shape[2], -1)
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


# ==================== Swin building blocks ====================
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
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


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
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
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
    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=to_2tuple(self.window_size),
                                    num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W
        x = x.view(B, H, W, C)
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))
        x = self.proj(x)
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)
        return x


# ==================== BasicLayer (Swin stage) ====================
class BasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size=7, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None,
                 use_checkpoint=False, up=True):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.up = up
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer) for i in range(depth)])
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
        h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size).view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            if self.up:
                Wh, Ww = (H + 1) // 2, (W + 1) // 2
            else:
                Wh, Ww = H * 2, W * 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W


# ==================== Swin Encoder ====================
class SwinTransformer(nn.Module):
    """Swin Transformer backbone. Returns multi-scale features as (B, C, H, W)."""

    def __init__(self, pretrain_img_size=224, patch_size=4, in_chans=3, embed_dim=128,
                 depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32], window_size=7,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.5, norm_layer=nn.LayerNorm,
                 ape=False, patch_norm=True, out_indices=(0, 1, 2, 3),
                 frozen_stages=-1, use_checkpoint=False):
        super().__init__()
        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        if self.ape:
            pretrain_img_size = to_2tuple(pretrain_img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1]]
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1]))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer), depth=depths[i_layer],
                num_heads=num_heads[i_layer], window_size=window_size,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)
        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False
        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patch_embed(x)
        Wh, Ww = x.size(2), x.size(3)
        if self.ape:
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)
        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)
                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
        return outs

    def train(self, mode=True):
        super(SwinTransformer, self).train(mode)
        self._freeze_stages()


# ==================== Decoder blocks ====================
class up_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(num_channels=out_ch, num_groups=groups),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.up(x)


class Decoder(nn.Module):
    """Simple decoder: upsample + concat + conv_relu."""

    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.up = up_conv(in_channels, out_channels)
        self.conv_relu = nn.Sequential(
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat((x2, x1), dim=1)
        x1 = self.conv_relu(x1)
        return x1


class conv_block(nn.Module):
    """Convolution block with MaxPool (used in shallow branch)."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(num_channels=out_ch, num_groups=groups),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(num_channels=out_ch, num_groups=groups),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


class Conv_block(nn.Module):
    """Convolution block without MaxPool (used for skip fusion)."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(num_channels=out_ch, num_groups=groups),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(num_channels=out_ch, num_groups=groups),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


# ==================== Swin Decoder ====================
class SwinDecoder(nn.Module):
    """Inner Swin-based decoder: upsample -> BasicLayer (Swin blocks) -> conv_reduce."""

    def __init__(self, embed_dim, patch_size=4, depths=2, num_heads=6, window_size=7,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.2, norm_layer=nn.LayerNorm,
                 patch_norm=True, use_checkpoint=False):
        super().__init__()
        self.patch_norm = patch_norm
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]
        self.layer = BasicLayer(
            dim=embed_dim // 2, depth=depths, num_heads=num_heads,
            window_size=window_size, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr, norm_layer=norm_layer,
            downsample=None, use_checkpoint=use_checkpoint)
        self.up = up_conv(embed_dim, embed_dim // 2)
        self.conv_relu = nn.Sequential(
            nn.Conv2d(embed_dim // 2, embed_dim // 4, kernel_size=1, stride=1, padding=0),
            nn.ReLU())

    def forward(self, x):
        identity = x
        B, C, H, W = x.shape
        x = self.up(x)  # B, C//2, 2H, 2W
        x = x.reshape(B, C // 2, H * W * 4)
        x = x.permute(0, 2, 1)
        x_out, H, W, x, Wh, Ww = self.layer(x, H * 2, W * 2)
        x = x.permute(0, 2, 1)
        x = x.reshape(B, C // 2, H, W)
        x = self.conv_relu(x)
        return x


class Swin_Decoder(nn.Module):
    """Swin_Decoder wrapper: SwinDecoder(x1) + conv_reduce(x2) -> concat -> conv."""

    def __init__(self, in_channels, depths, num_heads):
        super().__init__()
        self.up = SwinDecoder(in_channels, depths=depths, num_heads=num_heads)
        self.conv_relu = nn.Sequential(
            nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels // 2, in_channels // 4, kernel_size=1, stride=1, padding=0),
            nn.ReLU())

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x2 = self.conv2(x2)
        x1 = torch.cat((x2, x1), dim=1)
        out = self.conv_relu(x1)
        return out


# ==================== Cross Attention ====================
class Cross_Att(nn.Module):
    """Cross attention between encoder1 (large) and encoder2 (small) features.
    Uses Transformer blocks with pooled token injection for bidirectional fusion."""

    def __init__(self, dim_s, dim_l):
        super().__init__()
        self.transformer_s = Transformer(dim=dim_s, depth=1, heads=3, dim_head=32, mlp_dim=128)
        self.transformer_l = Transformer(dim=dim_l, depth=1, heads=1, dim_head=64, mlp_dim=256)
        self.norm_s = nn.LayerNorm(dim_s)
        self.norm_l = nn.LayerNorm(dim_l)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.linear_s = nn.Linear(dim_s, dim_l)
        self.linear_l = nn.Linear(dim_l, dim_s)

    def forward(self, e, r):
        # e: encoder1 features (B, dim_l, H_e, W_e) -- large encoder
        # r: encoder2 features (B, dim_s, H_r, W_r) -- small encoder
        b_e, c_e, h_e, w_e = e.shape
        e = e.reshape(b_e, c_e, -1).permute(0, 2, 1)  # B, HW, C_l
        b_r, c_r, h_r, w_r = r.shape
        r = r.reshape(b_r, c_r, -1).permute(0, 2, 1)  # B, HW, C_s
        # Pool to get global tokens
        e_t = torch.flatten(self.avgpool(self.norm_l(e).transpose(1, 2)), 1)  # B, C_l
        r_t = torch.flatten(self.avgpool(self.norm_s(r).transpose(1, 2)), 1)  # B, C_s
        # Project to other dimension
        e_t = self.linear_l(e_t).unsqueeze(1)  # B, 1, C_s
        r_t = self.linear_s(r_t).unsqueeze(1)  # B, 1, C_l
        # Inject token and run transformer, then remove injected token
        r = self.transformer_s(torch.cat([e_t, r], dim=1))[:, 1:, :]  # B, HW, C_s
        e = self.transformer_l(torch.cat([r_t, e], dim=1))[:, 1:, :]  # B, HW, C_l
        e = e.permute(0, 2, 1).reshape(b_e, c_e, h_e, w_e)
        r = r.permute(0, 2, 1).reshape(b_r, c_r, h_r, w_r)
        return e, r


# ==================== Main Model ====================
class DS_TransUNet(nn.Module):
    """
    DS-TransUNet: Dual Swin Transformer U-Net for Medical Image Segmentation.
    Official: https://github.com/TianBaoGe/DS-TransUNet

    Architecture:
      - Encoder1: SwinTransformer (embed_dim=128, depths=[2,2,18,2], patch_size=4)
      - Encoder2: SwinTransformer (embed_dim=96, depths=[2,2,6,2], patch_size=8)
      - Cross_Att at each of 4 stages for bidirectional fusion
      - change1-4: Conv_block to fuse concatenated encoder features
      - Swin_Decoder stages 1-3, Decoder stages 4-5
      - Shallow branch: down1 (1x1 conv) + down2 (conv_block with MaxPool)
      - Deep supervision: loss1 (from e4), loss2 (from d3), final (from d5)
    """

    def __init__(self, in_channels, num_classes, img_size, **kwargs):
        super().__init__()
        dim = kwargs.get('dim', 128)
        self.num_classes = num_classes

        # Encoder 1: Swin-B-like
        self.encoder = SwinTransformer(
            depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32],
            drop_path_rate=0.5, embed_dim=128, patch_size=4, in_chans=in_channels,
            window_size=kwargs.get('window_size', 7))
        # Encoder 2: Swin-T-like with patch_size=8
        self.encoder2 = SwinTransformer(
            depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
            drop_path_rate=0.2, patch_size=8, embed_dim=96, in_chans=in_channels,
            window_size=kwargs.get('window_size', 7))

        # Cross attention modules (dim_s * scale, dim_l * scale)
        # enc2 channels: [96, 192, 384, 768], enc1 channels: [128, 256, 512, 1024]
        self.cross_att_1 = Cross_Att(96 * 1, 128 * 1)
        self.cross_att_2 = Cross_Att(96 * 2, 128 * 2)
        self.cross_att_3 = Cross_Att(96 * 4, 128 * 4)
        self.cross_att_4 = Cross_Att(96 * 8, 128 * 8)

        # Skip fusion conv blocks (concat enc1 + upsampled enc2 -> conv)
        dim_s = 96
        dim_l = 128
        tb = dim_s + dim_l  # 224
        self.change1 = Conv_block(tb, dim)          # 224 -> 128
        self.change2 = Conv_block(tb * 2, dim * 2)  # 448 -> 256
        self.change3 = Conv_block(tb * 4, dim * 4)  # 896 -> 512
        self.change4 = Conv_block(tb * 8, dim * 8)  # 1792 -> 1024

        # Upsample for enc2 -> enc1 resolution alignment
        self.m1 = nn.Upsample(scale_factor=2)

        # Swin decoders (stages 1-3)
        self.layer1 = Swin_Decoder(8 * dim, 2, 8)   # 1024 -> 512, then conv to 256
        self.layer2 = Swin_Decoder(4 * dim, 2, 4)   # 512 -> 256, then conv to 128
        self.layer3 = Swin_Decoder(2 * dim, 2, 2)   # 256 -> 128, then conv to 64

        # Simple decoders (stages 4-5, fusing with shallow branch)
        self.layer4 = Decoder(dim, dim, dim // 2)    # 128 -> 64 (skip from ds2: dim//2=64)
        self.layer5 = Decoder(dim // 2, dim // 2, dim // 4)  # 64 -> 32 (skip from ds1: dim//4=32)

        # Shallow CNN branch
        self.down1 = nn.Conv2d(in_channels, dim // 4, kernel_size=1, stride=1, padding=0)  # in_ch -> 32
        self.down2 = conv_block(dim // 4, dim // 2)  # 32 -> 64 (with MaxPool2d)

        # Deep supervision heads
        self.loss1 = nn.Sequential(
            nn.Conv2d(dim * 8, num_classes, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Upsample(scale_factor=32))
        self.loss2 = nn.Sequential(
            nn.Conv2d(dim, num_classes, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Upsample(scale_factor=4))

        # Final output
        self.final = nn.Conv2d(dim // 4, num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Encode
        out = self.encoder(x)    # list of (B, C, H, W)
        out2 = self.encoder2(x)  # list of (B, C, H, W)
        e1, e2, e3, e4 = out[0], out[1], out[2], out[3]
        r1, r2, r3, r4 = out2[0], out2[1], out2[2], out2[3]

        # Cross attention fusion at each stage
        e1, r1 = self.cross_att_1(e1, r1)
        e2, r2 = self.cross_att_2(e2, r2)
        e3, r3 = self.cross_att_3(e3, r3)
        e4, r4 = self.cross_att_4(e4, r4)

        # Concatenate and fuse (with spatial alignment for arbitrary input sizes)
        def _align_cat(e, r):
            r_up = self.m1(r)
            if r_up.shape[2:] != e.shape[2:]:
                r_up = F.interpolate(r_up, size=e.shape[2:], mode='bilinear', align_corners=True)
            return torch.cat([e, r_up], 1)

        e1 = _align_cat(e1, r1)
        e2 = _align_cat(e2, r2)
        e3 = _align_cat(e3, r3)
        e4 = _align_cat(e4, r4)
        e1 = self.change1(e1)  # -> dim (128)
        e2 = self.change2(e2)  # -> dim*2 (256)
        e3 = self.change3(e3)  # -> dim*4 (512)
        e4 = self.change4(e4)  # -> dim*8 (1024)

        # Deep supervision loss from bottleneck
        loss1 = self.loss1(e4)

        # Shallow branch
        ds1 = self.down1(x)    # B, 32, H, W
        ds2 = self.down2(ds1)  # B, 64, H/2, W/2

        # Decode
        d1 = self.layer1(e4, e3)   # Swin_Decoder: 1024->512->256, skip e3
        d2 = self.layer2(d1, e2)   # Swin_Decoder: 512->256->128, skip e2
        d3 = self.layer3(d2, e1)   # Swin_Decoder: 256->128->64, skip e1

        # Deep supervision loss from d3
        loss2 = self.loss2(d3)

        d4 = self.layer4(d3, ds2)  # Decoder: 128->64, skip ds2
        d5 = self.layer5(d4, ds1)  # Decoder: 64->32, skip ds1
        o = self.final(d5)

        if self.training:
            return o, loss1, loss2
        return o


# Alias for registry
DSTransUNet = DS_TransUNet
