"""TransNuSeg – pure Swin-Transformer nuclei segmentation.

Ported from: https://github.com/zhenqi-he/transnuseg
Paper: TransNuSeg: A Lightweight Multi-Task Transformer for Nuclei
       Segmentation (MICCAI 2023)

Architecture highlights
-----------------------
* First entirely Swin-Transformer driven architecture for nuclei
* Multi-task decoder with shared attention across branches
* Bottleneck uses shifted MLP block

Adapted for the project's standard interface:
    TransNuSeg(in_channels, num_classes, img_size, pretrained, ...)
"""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.layers import DropPath, to_2tuple, trunc_normal_


# ── helpers ──────────────────────────────────────────────────────────────────

def _conv1x1(in_p, out_p, stride=1):
    return nn.Conv2d(in_p, out_p, 1, stride, bias=False)


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


# ── MLP variants ─────────────────────────────────────────────────────────────

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.drop(self.fc1(x)))))


class ShiftMLP(nn.Module):
    """Shifted MLP for bottleneck block."""
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0., shift_size=5):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.shift_size = shift_size
        self.pad = shift_size // 2

    def forward(self, x, H, W):
        B, N, C = x.shape
        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(xc, s, 2) for xc, s in zip(xs, range(-self.pad, self.pad + 1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_s = torch.narrow(x_cat, 3, self.pad, W)
        x_s = x_s.reshape(B, C, H * W).contiguous().transpose(1, 2)
        x = self.drop(self.act(self.dwconv(self.fc1(x_s), H, W)))
        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(xc, s, 3) for xc, s in zip(xs, range(-self.pad, self.pad + 1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_s = torch.narrow(x_cat, 3, self.pad, W)
        return self.drop(self.fc2(x_s.reshape(B, C, H * W).contiguous().transpose(1, 2)))


class ShiftedBlock(nn.Module):
    """Bottleneck block with shifted MLP."""
    def __init__(self, dim, num_heads, mlp_ratio=1., qkv_bias=False,
                 qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, sr_ratio=1, input_resolution=(1, 1)):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.H, self.W = input_resolution
        self.mlp = ShiftMLP(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x):
        return x + self.drop_path(self.mlp(self.norm2(x), self.H, self.W))


# ── Attention modules ────────────────────────────────────────────────────────

class _BaseWindowAttn(nn.Module):
    """Base class for window attention with relative position bias."""

    def __init__(self, dim, window_size, num_heads, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = qk_scale or (dim // num_heads) ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flat = torch.flatten(coords, 1)
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]
        rel = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += window_size[0] - 1
        rel[:, :, 1] += window_size[1] - 1
        rel[:, :, 0] *= 2 * window_size[1] - 1
        self.register_buffer("relative_position_index", rel.sum(-1))
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)

    def _get_rpb(self):
        ws = self.window_size
        rpb = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(ws[0] * ws[1], ws[0] * ws[1], -1)
        return rpb.permute(2, 0, 1).contiguous()

    def _attn_forward(self, qkv, B_, N, C, mask=None):
        qkv = qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1) + self._get_rpb().unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.attn_drop(F.softmax(attn, dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


class WindowAttention(_BaseWindowAttn):
    """Standard window attention (encoder blocks)."""
    def __init__(self, dim, window_size, num_heads, qkv_bias=True,
                 qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__(dim, window_size, num_heads, qk_scale, attn_drop, proj_drop)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

    def forward(self, x, mask=None):
        return self._attn_forward(self.qkv(x), x.shape[0], x.shape[1], x.shape[2], mask)


class WindowAttentionUp(_BaseWindowAttn):
    """Window attention for decoder (external qkv)."""
    def __init__(self, dim, window_size, num_heads, qkv,
                 qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__(dim, window_size, num_heads, qk_scale, attn_drop, proj_drop)
        self.qkv = qkv

    def forward(self, x, mask=None):
        return self._attn_forward(self.qkv(x), x.shape[0], x.shape[1], x.shape[2], mask)


class SharedWindowAttention(_BaseWindowAttn):
    """Window attention with shared QKV heads across decoder branches."""
    def __init__(self, dim, window_size, num_heads, qkv, shared_qkv,
                 qk_scale=None, attn_drop=0., proj_drop=0., shared_ratio=0.5):
        super().__init__(dim, window_size, num_heads, qk_scale, attn_drop, proj_drop)
        self.qkv = qkv
        self.shared_qkv = shared_qkv
        self.shared_ratio = shared_ratio

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        ss = int(N * self.shared_ratio)
        qkv1 = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv2 = self.shared_qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q = torch.cat((qkv2[0][:ss], qkv1[0][ss:]), 0)
        k = torch.cat((qkv2[1][:ss], qkv1[1][ss:]), 0)
        v = torch.cat((qkv2[2][:ss], qkv1[2][ss:]), 0)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1) + self._get_rpb().unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.attn_drop(F.softmax(attn, dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


# ── Swin Transformer blocks ─────────────────────────────────────────────────

def _make_attn_mask(input_resolution, window_size, shift_size):
    if shift_size <= 0:
        return None
    H, W = input_resolution
    mask = torch.zeros((1, H, W, 1))
    cnt = 0
    for h in (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None)):
        for w in (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None)):
            mask[:, h, w, :] = cnt
            cnt += 1
    mw = window_partition(mask, window_size).view(-1, window_size * window_size)
    am = mw.unsqueeze(1) - mw.unsqueeze(2)
    return am.masked_fill(am != 0, -100.0).masked_fill(am == 0, 0.0)


class SwinBlock(nn.Module):
    """Standard Swin Transformer block (encoder)."""
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        ws = min(input_resolution) if min(input_resolution) <= window_size else window_size
        self.window_size = ws
        self.shift_size = 0 if min(input_resolution) <= window_size else shift_size
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, to_2tuple(ws), num_heads, qkv_bias, qk_scale, attn_drop, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)
        self.register_buffer("attn_mask", _make_attn_mask(input_resolution, ws, self.shift_size))

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)
        if self.shift_size > 0:
            x = torch.roll(x, (-self.shift_size, -self.shift_size), (1, 2))
        xw = window_partition(x, self.window_size).view(-1, self.window_size ** 2, C)
        xw = self.attn(xw, self.attn_mask).view(-1, self.window_size, self.window_size, C)
        x = window_reverse(xw, self.window_size, H, W)
        if self.shift_size > 0:
            x = torch.roll(x, (self.shift_size, self.shift_size), (1, 2))
        x = shortcut + self.drop_path(x.view(B, H * W, C))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class SwinBlockUp(nn.Module):
    """Swin block with external qkv (decoder)."""
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv=None, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        ws = min(input_resolution) if min(input_resolution) <= window_size else window_size
        self.window_size = ws
        self.shift_size = 0 if min(input_resolution) <= window_size else shift_size
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttentionUp(dim, to_2tuple(ws), num_heads, qkv, qk_scale, attn_drop, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)
        self.register_buffer("attn_mask", _make_attn_mask(input_resolution, ws, self.shift_size))

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)
        if self.shift_size > 0:
            x = torch.roll(x, (-self.shift_size, -self.shift_size), (1, 2))
        xw = window_partition(x, self.window_size).view(-1, self.window_size ** 2, C)
        xw = self.attn(xw, self.attn_mask).view(-1, self.window_size, self.window_size, C)
        x = window_reverse(xw, self.window_size, H, W)
        if self.shift_size > 0:
            x = torch.roll(x, (self.shift_size, self.shift_size), (1, 2))
        x = shortcut + self.drop_path(x.view(B, H * W, C))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class SharedSwinBlock(nn.Module):
    """Swin block with shared QKV attention for multi-task decoder."""
    def __init__(self, dim, input_resolution, num_heads, qkv, shared_qkv,
                 window_size=7, shift_size=0, mlp_ratio=4., qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm,
                 shared_ratio=0.5):
        super().__init__()
        self.input_resolution = input_resolution
        ws = min(input_resolution) if min(input_resolution) <= window_size else window_size
        self.window_size = ws
        self.shift_size = 0 if min(input_resolution) <= window_size else shift_size
        self.norm1 = norm_layer(dim)
        self.attn = SharedWindowAttention(dim, to_2tuple(ws), num_heads, qkv, shared_qkv,
                                          qk_scale, attn_drop, drop, shared_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)
        self.register_buffer("attn_mask", _make_attn_mask(input_resolution, ws, self.shift_size))

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)
        if self.shift_size > 0:
            x = torch.roll(x, (-self.shift_size, -self.shift_size), (1, 2))
        xw = window_partition(x, self.window_size).view(-1, self.window_size ** 2, C)
        xw = self.attn(xw, self.attn_mask).view(-1, self.window_size, self.window_size, C)
        x = window_reverse(xw, self.window_size, H, W)
        if self.shift_size > 0:
            x = torch.roll(x, (self.shift_size, self.shift_size), (1, 2))
        x = shortcut + self.drop_path(x.view(B, H * W, C))
        return x + self.drop_path(self.mlp(self.norm2(x)))


# ── Patch operations ─────────────────────────────────────────────────────────

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x = torch.cat([x[:, 0::2, 0::2, :], x[:, 1::2, 0::2, :],
                        x[:, 0::2, 1::2, :], x[:, 1::2, 1::2, :]], -1)
        return self.reduction(self.norm(x.view(B, -1, 4 * C)))


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        return self.norm(x.view(B, -1, C // 4))


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(dim)

    def forward(self, x):
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        ds = self.dim_scale
        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=ds, p2=ds, c=C // (ds ** 2))
        return self.norm(x.view(B, -1, self.output_dim))


# ── Layer modules ────────────────────────────────────────────────────────────

class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinBlock(dim, input_resolution, num_heads, window_size,
                      0 if (i % 2 == 0) else window_size // 2, mlp_ratio,
                      qkv_bias, qk_scale, drop, attn_drop,
                      drop_path[i] if isinstance(drop_path, list) else drop_path,
                      norm_layer) for i in range(depth)])
        self.downsample = downsample(input_resolution, dim, norm_layer) if downsample else None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample:
            x = self.downsample(x)
        return x


class BasicLayerUp(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_lists=None, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinBlockUp(dim, input_resolution, num_heads, window_size,
                        0 if (i % 2 == 0) else window_size // 2, mlp_ratio,
                        qkv_lists[i] if qkv_lists else None, qk_scale, drop, attn_drop,
                        drop_path[i] if isinstance(drop_path, list) else drop_path,
                        norm_layer) for i in range(depth)])
        self.upsample = PatchExpand(input_resolution, dim, 2, norm_layer) if upsample else None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.upsample:
            x = self.upsample(x)
        return x


class SharedBasicLayerUp(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., shared_qkv_lists=None, qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm,
                 upsample=None, shared_ratio=0.5):
        super().__init__()
        blocks = []
        for i in range(depth):
            shared_qkv = shared_qkv_lists[i] if shared_qkv_lists else None
            blocks.append(SharedSwinBlock(
                dim, input_resolution, num_heads,
                nn.Linear(dim, dim * 3, bias=qkv_bias), shared_qkv,
                window_size, 0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio, qk_scale, drop, attn_drop,
                drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer, shared_ratio))
        self.blocks = nn.ModuleList(blocks)
        self.upsample = PatchExpand(input_resolution, dim, 2, norm_layer) if upsample else None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.upsample:
            x = self.upsample(x)
        return x


# ── Main model ───────────────────────────────────────────────────────────────

class TransNuSeg(nn.Module):
    """TransNuSeg: pure Swin-Transformer nuclei segmentation.

    Parameters
    ----------
    in_channels : int
        Number of input channels (default 3).
    num_classes : int
        Number of output segmentation classes.
    img_size : int
        Input image size (must be divisible by patch_size × 2^num_layers).
    pretrained : bool
        Not used (kept for interface compatibility).
    embed_dim : int
        Patch embedding dimension.
    depths : list[int]
        Depth of each encoder stage.
    num_heads : list[int]
        Number of attention heads per stage.
    window_size : int
        Window size for Swin attention.
    shared_ratio : float
        Ratio of shared attention heads between decoder branches.
    """

    def __init__(self, in_channels=3, num_classes=2, img_size=512,
                 pretrained=True, embed_dim=96, depths=(2, 2, 2, 2),
                 depths_decoder=(1, 2, 2, 2), num_heads=(3, 6, 12, 24),
                 window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_size=4, ape=False,
                 patch_norm=True, shared_ratio=0.5, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio

        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim,
                                      norm_layer if patch_norm else None)
        pr = self.patch_embed.patches_resolution
        self.patches_resolution = pr

        if ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, self.patch_embed.num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Encoder + bottleneck
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            if i < self.num_layers - 1:
                layer = BasicLayer(
                    int(embed_dim * 2 ** i),
                    (pr[0] // (2 ** i), pr[1] // (2 ** i)),
                    depths[i], num_heads[i], window_size, mlp_ratio,
                    qkv_bias, qk_scale, drop_rate, attn_drop_rate,
                    dpr[sum(depths[:i]):sum(depths[:i + 1])],
                    norm_layer, PatchMerging if i < self.num_layers - 1 else None)
            else:
                layer = ShiftedBlock(
                    int(embed_dim * 2 ** i), num_heads[i], mlp_ratio=1,
                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])][0],
                    norm_layer=norm_layer, sr_ratio=8,
                    input_resolution=(pr[0] // (2 ** i), pr[1] // (2 ** i)))
            self.layers.append(layer)

        # Decoder branches
        self.layers_up = nn.ModuleList()
        self.layers_up2 = nn.ModuleList()
        self.layers_up3 = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        self.concat_back_dim2 = nn.ModuleList()
        self.concat_back_dim3 = nn.ModuleList()

        for i in range(self.num_layers):
            qkv_lists = []
            for j in range(depths[i]):
                d = int(embed_dim * 2 ** (self.num_layers - 1 - i))
                qkv_lists.append(nn.Linear(d, d * 3, bias=qkv_bias))

            concat = nn.Linear(2 * int(embed_dim * 2 ** (self.num_layers - 1 - i)),
                               int(embed_dim * 2 ** (self.num_layers - 1 - i))
                               ) if i > 0 else nn.Identity()

            if i == 0:
                lu = PatchExpand(
                    (pr[0] // (2 ** (self.num_layers - 1)), pr[1] // (2 ** (self.num_layers - 1))),
                    int(embed_dim * 2 ** (self.num_layers - 1)), 2, norm_layer)
            else:
                lu = BasicLayerUp(
                    int(embed_dim * 2 ** (self.num_layers - 1 - i)),
                    (pr[0] // (2 ** (self.num_layers - 1 - i)), pr[1] // (2 ** (self.num_layers - 1 - i))),
                    depths[self.num_layers - 1 - i], num_heads[self.num_layers - 1 - i],
                    window_size, mlp_ratio, qkv_lists, qk_scale,
                    drop_rate, attn_drop_rate,
                    dpr[sum(depths[:self.num_layers - 1 - i]):sum(depths[:self.num_layers - 1 - i + 1])],
                    norm_layer,
                    PatchExpand if i < self.num_layers - 1 else None)
            self.layers_up.append(lu)
            self.concat_back_dim.append(concat)

            if i < self.num_layers - 1:
                if i == 0:
                    lu2 = PatchExpand(
                        (pr[0] // (2 ** (self.num_layers - 1)), pr[1] // (2 ** (self.num_layers - 1))),
                        int(embed_dim * 2 ** (self.num_layers - 1)), 2, norm_layer)
                else:
                    lu2 = SharedBasicLayerUp(
                        int(embed_dim * 2 ** (self.num_layers - 1 - i)),
                        (pr[0] // (2 ** (self.num_layers - 1 - i)), pr[1] // (2 ** (self.num_layers - 1 - i))),
                        depths[self.num_layers - 1 - i], num_heads[self.num_layers - 1 - i],
                        window_size, mlp_ratio, qkv_lists, qkv_bias, qk_scale,
                        drop_rate, attn_drop_rate,
                        dpr[sum(depths[:self.num_layers - 1 - i]):sum(depths[:self.num_layers - 1 - i + 1])],
                        norm_layer,
                        PatchExpand if i < self.num_layers - 1 else None,
                        shared_ratio)
                concat2 = nn.Linear(2 * int(embed_dim * 2 ** (self.num_layers - 1 - i)),
                                    int(embed_dim * 2 ** (self.num_layers - 1 - i))
                                    ) if i > 0 else nn.Identity()
                self.layers_up2.append(lu2)
                self.layers_up3.append(lu2)
                self.concat_back_dim2.append(concat2)
                self.concat_back_dim3.append(concat2)
            else:
                concat2 = nn.Linear(
                    2 * int(embed_dim * 2 ** 0), int(embed_dim * 2 ** 0))
                concat3 = nn.Linear(
                    2 * int(embed_dim * 2 ** 0), int(embed_dim * 2 ** 0))
                lu2 = SharedBasicLayerUp(
                    int(embed_dim * 2 ** 0), (pr[0], pr[1]),
                    depths[0], num_heads[0], window_size, mlp_ratio,
                    qkv_lists, qkv_bias, qk_scale, drop_rate, attn_drop_rate,
                    dpr[sum(depths[:0]):sum(depths[:1])], norm_layer,
                    None, shared_ratio)
                lu3 = SharedBasicLayerUp(
                    int(embed_dim * 2 ** 0), (pr[0], pr[1]),
                    depths[0], num_heads[0], window_size, mlp_ratio,
                    qkv_lists, qkv_bias, qk_scale, drop_rate, attn_drop_rate,
                    dpr[sum(depths[:0]):sum(depths[:1])], norm_layer,
                    None, shared_ratio)
                self.layers_up2.append(lu2)
                self.layers_up3.append(lu3)
                self.concat_back_dim2.append(concat2)
                self.concat_back_dim3.append(concat3)

        self.norm = norm_layer(self.num_features)
        self.norm_up = norm_layer(embed_dim)
        self.norm2 = norm_layer(self.num_features)
        self.norm_up2 = norm_layer(embed_dim)
        self.norm3 = norm_layer(self.num_features)
        self.norm_up3 = norm_layer(embed_dim)

        # Final 4× upsample + output heads
        self.up = FinalPatchExpand_X4((pr[0], pr[1]), embed_dim, 4, norm_layer)
        self.output = nn.Conv2d(embed_dim, num_classes, 1, bias=False)
        self.up2 = FinalPatchExpand_X4((pr[0], pr[1]), embed_dim, 4, norm_layer)
        self.output2 = nn.Conv2d(embed_dim, num_classes, 1, bias=False)
        self.up3 = FinalPatchExpand_X4((pr[0], pr[1]), embed_dim, 4, norm_layer)
        self.output3 = nn.Conv2d(embed_dim, num_classes, 1, bias=False)

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

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        seg = self.pos_drop(x)
        edge = self.pos_drop(x)
        seg_down, edge_down = [], []
        for layer in self.layers:
            edge_down.append(edge)
            seg_down.append(seg)
            seg = layer(seg)
            edge = layer(edge)
        return self.norm(seg), self.norm(edge), seg_down, edge_down

    def forward_up_features(self, seg, edge, seg_down, edge_down):
        for i in range(len(self.layers_up)):
            if i == 0:
                seg = self.layers_up[i](seg)
                cluster = self.layers_up3[i](edge)
                edge = self.layers_up2[i](edge)
            else:
                seg = self.concat_back_dim[i](
                    torch.cat([seg, seg_down[self.num_layers - 1 - i]], -1))
                seg = self.layers_up[i](seg)

                edge = self.concat_back_dim2[i](
                    torch.cat([edge, edge_down[self.num_layers - 1 - i]], -1))
                edge = self.layers_up2[i](edge)

                cluster = self.concat_back_dim3[i](
                    torch.cat([cluster, edge_down[self.num_layers - 1 - i]], -1))
                cluster = self.layers_up3[i](cluster)

        return self.norm_up(seg), self.norm_up2(edge), self.norm_up3(cluster)

    def forward(self, x):
        seg, edge, seg_d, edge_d = self.forward_features(x)
        seg, edge, cluster = self.forward_up_features(seg, edge, seg_d, edge_d)
        H, W = self.patches_resolution
        B = seg.shape[0]
        # Primary output (nuclei segmentation)
        s = self.up(seg).view(B, 4 * H, 4 * W, -1).permute(0, 3, 1, 2)
        return self.output(s)
