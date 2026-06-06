"""Zig-RiR (Zigzag RWKV-in-RWKV) Encoder.

Faithful port of the **2D** model from the official repository:
    https://github.com/txchen-USTC/Zig-RiR  (file: Zig_RiR2d.py)

Reference:
    Chen et al., "Zig-RiR: Zigzag RWKV-in-RWKV for Efficient Medical Image
    Segmentation", IEEE TMI 2025.

Key components reproduced 1:1 from the official source:
    - q_shift                       (5-way channel shift)
    - VRWKV_SpatialMix              (zigzag scan + recurrence=2 WKV attention)
    - VRWKV_ChannelMix              (relu^2 gated FFN with q_shift)
    - Block                         (RWKV-in-RWKV: inner + outer + projection)
    - PatchMerging2D_sentence / _word
    - Stem                          (1/8 dual-tower stem producing sentences + words)
    - Stage                         (stack of Block, first/second block keep inner)
    - UpsampleBlock                 (per-stage 2x upsample head)
    - PyramidRiR_enc                (4-stage hierarchical encoder)

Differences from the upstream file (kept minimal):
    1. The CUDA-only ``RUN_CUDA(WKV.apply)`` is replaced by a CUDA/CPU
       dispatcher backed by :mod:`medseg.kernels.wkv`. The CUDA path uses the
       byte-identical official ``cuda/wkv_op.cpp`` + ``cuda/wkv_cuda.cu``
       sources (now shipped under ``medseg/kernels/wkv/``) and is JIT-compiled
       on first use; the CPU fallback is a vectorised PyTorch implementation
       differentiated by autograd.
    2. ``timm.models.layers`` is replaced with local helpers (``DropPath``,
       ``trunc_normal_`` from ``torch.nn.init``) to avoid an extra dependency.
    3. The encoder is wrapped as a registered module exposing
       ``out_channels`` and returning a list of pyramid feature maps, matching
       the project's encoder interface; no algorithmic changes.
"""
# Source: https://github.com/txchen-USTC/Zig-RiR

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.registry import ENCODER_REGISTRY
# WKV kernel dispatcher: JIT-compiles the official Vision-RWKV CUDA op when
# available (with analytic backward) and otherwise falls back to a vectorised
# PyTorch implementation that PyTorch autograd differentiates automatically.
from medseg.kernels.wkv import load_wkv_cuda, is_cuda_available, run_wkv
from .rwkv_encoder import DropPath  # noqa: E402


# ---------------------------------------------------------------------------
# WKV kernel dispatcher
# ---------------------------------------------------------------------------

def _try_load_cuda_wkv(t_max: int = 8192):
    """Lazily JIT-compile the official Vision-RWKV CUDA op.

    Returns the compiled module, or ``None`` when CUDA / nvcc / a matching
    GPU is unavailable. The compilation result is cached inside
    :mod:`medseg.kernels.wkv`, so callers may invoke this function as often
    as needed without paying the compile cost again.
    """
    return load_wkv_cuda(t_max=t_max)


def RUN_CUDA(B, T, C, w, u, k, v):
    """Dispatcher matching the official ``RUN_CUDA`` signature.

    Identical to ``WKV.apply(B, T, C, w, u, k, v)`` from the paper. The
    underlying :func:`run_wkv` is differentiable on both code paths, so the
    caller does not need to wrap the result in another autograd function.
    """
    return run_wkv(B, T, C, w.float(), u.float(), k.float(), v.float())


# ---------------------------------------------------------------------------
# q_shift (verbatim from Zig_RiR2d.py)
# ---------------------------------------------------------------------------

def q_shift(input, shift_pixel=1, gamma=1 / 4):
    assert gamma <= 1 / 4
    B, C, H, W = input.shape
    output = torch.zeros_like(input)
    output[:, 0:int(C * gamma), :, shift_pixel:W] = input[:, 0:int(C * gamma), :, 0:W - shift_pixel]
    output[:, int(C * gamma):int(C * gamma * 2), :, 0:W - shift_pixel] = input[:, int(C * gamma):int(C * gamma * 2), :,
                                                                         shift_pixel:W]
    output[:, int(C * gamma * 2):int(C * gamma * 3), shift_pixel:H, :] = input[:, int(C * gamma * 2):int(C * gamma * 3),
                                                                         0:H - shift_pixel, :]
    output[:, int(C * gamma * 3):int(C * gamma * 4), 0:H - shift_pixel, :] = input[:,
                                                                             int(C * gamma * 3):int(C * gamma * 4),
                                                                             shift_pixel:H, :]
    output[:, int(C * gamma * 4):, ...] = input[:, int(C * gamma * 4):, ...]
    return output


# ---------------------------------------------------------------------------
# Helpers (replace timm/einops dependencies)
# ---------------------------------------------------------------------------

def to_2tuple(x):
    return x if isinstance(x, (tuple, list)) else (x, x)


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)


def _bnc_to_bchw(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """(B, H*W, C) -> (B, C, H, W)"""
    B, _, C = x.shape
    return x.transpose(1, 2).reshape(B, C, h, w)


def _bchw_to_bnc(x: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) -> (B, H*W, C)"""
    B, C, H, W = x.shape
    return x.reshape(B, C, H * W).transpose(1, 2)


# ---------------------------------------------------------------------------
# VRWKV Spatial Mix (zigzag scan + recurrence=2)
# ---------------------------------------------------------------------------

class VRWKV_SpatialMix(nn.Module):
    """Verbatim from Zig_RiR2d.py."""

    def __init__(self, n_embd, n_layer, layer_id, init_mode='fancy', key_norm=False,
                 scan_schemes=None):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        attn_sz = n_embd
        self.device = None
        self.recurrence = 2
        self.scan_schemes = scan_schemes or [('top-left', 'horizontal'), ('bottom-right', 'vertical')]
        self.dwconv = nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1, groups=n_embd, bias=False)
        self.key = nn.Linear(n_embd, attn_sz, bias=False)
        self.value = nn.Linear(n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(n_embd, attn_sz, bias=False)
        if key_norm:
            self.key_norm = nn.LayerNorm(n_embd)
        else:
            self.key_norm = None
        self.output = nn.Linear(attn_sz, n_embd, bias=False)
        self.spatial_decay = nn.Parameter(torch.randn((self.recurrence, self.n_embd)))
        self.spatial_first = nn.Parameter(torch.randn((self.recurrence, self.n_embd)))

    def get_zigzag_indices(self, h, w, start='top-left', direction='horizontal'):
        indices = []
        if start == 'top-left':
            row_start = 0
            col_start = 0
            row_step = 1
            col_step = 1 if direction == 'horizontal' else 1
        elif start == 'top-right':
            row_start = 0
            col_start = w - 1
            row_step = 1
            col_step = -1 if direction == 'horizontal' else -1
        elif start == 'bottom-left':
            row_start = h - 1
            col_start = 0
            row_step = -1
            col_step = 1 if direction == 'horizontal' else 1
        elif start == 'bottom-right':
            row_start = h - 1
            col_start = w - 1
            row_step = -1
            col_step = -1 if direction == 'horizontal' else -1

        for i in range(h):
            current_row = row_start + row_step * i
            if direction == 'horizontal':
                if current_row % 2 == 0:
                    cols = list(range(w))
                else:
                    cols = list(range(w - 1, -1, -1))
                for col in cols:
                    indices.append(current_row * w + col)
            elif direction == 'vertical':
                if (col_start + col_step * i) % 2 == 0:
                    rows = list(range(h))
                else:
                    rows = list(range(h - 1, -1, -1))
                for row in rows:
                    indices.append(row * w + (col_start + col_step * i))
        return torch.tensor(indices, dtype=torch.long, device=self.device)

    def jit_func(self, x, resolution, scan_scheme):
        h, w = resolution
        start, direction = scan_scheme
        zigzag_order = self.get_zigzag_indices(h, w, start=start, direction=direction)

        # x: (B, h*w, C) -> (B, C, h, w)
        x = _bnc_to_bchw(x, h, w)
        x = q_shift(x)

        # zigzag flatten
        B, C, _, _ = x.shape
        x = x.reshape(B, C, h * w)
        x = x[..., zigzag_order]
        x = x.transpose(1, 2)  # (B, h*w, C)

        k = self.key(x)
        v = self.value(x)
        r = self.receptance(x)
        sr = torch.sigmoid(r)
        return sr, k, v

    def forward(self, x, resolution):
        B, T, C = x.size()
        self.device = x.device

        selected_scheme = self.scan_schemes[self.layer_id % len(self.scan_schemes)]
        sr, k, v = self.jit_func(x, resolution, selected_scheme)

        for j in range(self.recurrence):
            if j % 2 == 0:
                v = RUN_CUDA(B, T, C, self.spatial_decay[j] / T, self.spatial_first[j] / T, k, v)
            else:
                h, w = resolution
                new_h, new_w = (h, w) if selected_scheme[1] == 'horizontal' else (w, h)
                zigzag_order = self.get_zigzag_indices(new_h, new_w, start=selected_scheme[0],
                                                       direction=selected_scheme[1])
                k = _bnc_to_bchw(k, h, w)
                k = k.reshape(B, C, h * w)[..., zigzag_order]
                k = k.transpose(1, 2)  # (B, new_h*new_w, C) == (B, h*w, C)

                v = _bnc_to_bchw(v, h, w)
                v = v.reshape(B, C, h * w)[..., zigzag_order]
                v = v.transpose(1, 2)

                v = RUN_CUDA(B, T, C, self.spatial_decay[j] / T, self.spatial_first[j] / T, k, v)
                # restore canonical shape; algebraically equivalent to upstream
                # rearrange chain since (h, w) == (new_h, new_w) in token count.
                k = k.reshape(B, T, C)
                v = v.reshape(B, T, C)

        x = v
        if self.key_norm is not None:
            x = self.key_norm(x)
        x = sr * x
        x = self.output(x)
        return x


# ---------------------------------------------------------------------------
# VRWKV Channel Mix
# ---------------------------------------------------------------------------

class VRWKV_ChannelMix(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, hidden_rate=4, init_mode='fancy', key_norm=False):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        hidden_sz = int(hidden_rate * n_embd)
        self.key = nn.Linear(n_embd, hidden_sz, bias=False)
        if key_norm:
            self.key_norm = nn.LayerNorm(hidden_sz)
        else:
            self.key_norm = None
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)

    def forward(self, x, resolution):
        h, w = resolution
        x = _bnc_to_bchw(x, h, w)
        x = q_shift(x)
        x = _bchw_to_bnc(x)
        k = self.key(x)
        k = torch.square(torch.relu(k))
        if self.key_norm is not None:
            k = self.key_norm(k)
        kv = self.value(k)
        x = torch.sigmoid(self.receptance(x)) * kv
        return x


# ---------------------------------------------------------------------------
# RWKV-in-RWKV Block (the core of Zig-RiR)
# ---------------------------------------------------------------------------

class Block(nn.Module):
    """Outer (sentence) + Inner (word) nested RWKV block.

    Matches Zig_RiR2d.py:Block. When ``inner_dim <= 0`` the block degenerates
    to outer-only (used for blocks beyond the first/second one in each stage).
    """

    def __init__(self, outer_dim, inner_dim, layer_id, outer_head, inner_head, num_words, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, se=0, sr_ratio=1):
        super().__init__()
        self.has_inner = inner_dim > 0
        if self.has_inner:
            self.inner_norm1 = norm_layer(num_words * inner_dim)
            self.inner_attn = VRWKV_SpatialMix(n_embd=inner_dim, n_layer=None, layer_id=layer_id)
            self.inner_norm2 = norm_layer(num_words * inner_dim)
            self.inner_ffn = VRWKV_ChannelMix(n_embd=inner_dim, n_layer=None, layer_id=None)
            self.proj_norm1 = norm_layer(num_words * inner_dim)
            self.proj = nn.Linear(num_words * inner_dim, outer_dim, bias=False)
            self.proj_norm2 = norm_layer(outer_dim)

        self.outer_norm1 = norm_layer(outer_dim)
        self.outer_attn = VRWKV_SpatialMix(n_embd=outer_dim, n_layer=None, layer_id=layer_id)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.outer_norm2 = norm_layer(outer_dim)
        self.outer_ffn = VRWKV_ChannelMix(n_embd=outer_dim, n_layer=None, layer_id=1)

    def forward(self, x, outer_tokens, H_out, W_out, H_in, W_in):
        B, N, C = outer_tokens.size()
        if self.has_inner:
            inner_patch_resolution = [H_in, W_in]
            x = x + self.drop_path(self.inner_attn(
                self.inner_norm1(x.reshape(B, N, -1)).reshape(B * N, H_in * W_in, -1),
                inner_patch_resolution))
            x = x + self.drop_path(self.inner_ffn(
                self.inner_norm2(x.reshape(B, N, -1)).reshape(B * N, H_in * W_in, -1),
                inner_patch_resolution))
            outer_tokens = outer_tokens + self.proj_norm2(
                self.proj(self.proj_norm1(x.reshape(B, N, -1))))
        outer_patch_resolution = [H_out, W_out]
        outer_tokens = outer_tokens + self.drop_path(
            self.outer_attn(self.outer_norm1(outer_tokens), outer_patch_resolution))
        outer_tokens = outer_tokens + self.drop_path(
            self.outer_ffn(self.outer_norm2(outer_tokens), outer_patch_resolution))
        return x, outer_tokens


# ---------------------------------------------------------------------------
# Patch merging for sentences and words
# ---------------------------------------------------------------------------

class PatchMerging2D_sentence(nn.Module):
    def __init__(self, dim_in, dim_out, stride=2):
        super().__init__()
        self.stride = stride
        self.norm = nn.LayerNorm(dim_in)
        self.conv = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=2 * stride - 1, padding=stride - 1, stride=stride),
        )

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = self.norm(x)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.conv(x)
        H, W = math.ceil(H / self.stride), math.ceil(W / self.stride)
        x = x.reshape(B, -1, H * W).transpose(1, 2)
        return x, H, W


class PatchMerging2D_word(nn.Module):
    def __init__(self, dim_in, dim_out, stride=2):
        super().__init__()
        self.stride = stride
        self.dim_out = dim_out
        self.norm = nn.LayerNorm(dim_in)
        self.conv = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=2 * stride - 1, padding=stride - 1, stride=stride),
        )

    def forward(self, x, H_out, W_out, H_in, W_in):
        B_N, M, C = x.shape
        x = self.norm(x)
        x = x.reshape(-1, H_out, W_out, H_in, W_in, C)

        pad_input = (H_out % 2 == 1) or (W_out % 2 == 1)
        if pad_input:
            x = F.pad(x.permute(0, 3, 4, 5, 1, 2), (0, W_out % 2, 0, H_out % 2))
            x = x.permute(0, 4, 5, 1, 2, 3)

        x1 = x[:, 0::2, 0::2, :, :, :]
        x2 = x[:, 1::2, 0::2, :, :, :]
        x3 = x[:, 0::2, 1::2, :, :, :]
        x4 = x[:, 1::2, 1::2, :, :, :]
        x = torch.cat([torch.cat([x1, x2], 3), torch.cat([x3, x4], 3)], 4)
        x = x.reshape(-1, 2 * H_in, 2 * W_in, C).permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.reshape(-1, self.dim_out, M).transpose(1, 2)
        return x


# ---------------------------------------------------------------------------
# Stem
# ---------------------------------------------------------------------------

class Stem(nn.Module):
    def __init__(self, img_size=224, in_chans=1, outer_dim=768, inner_dim=24):
        super().__init__()
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.inner_dim = inner_dim
        self.num_patches = img_size[0] // 8 * img_size[1] // 8
        self.num_words = 16

        self.common_conv = nn.Sequential(
            nn.Conv2d(in_chans, inner_dim * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(inner_dim * 2),
            nn.ReLU(inplace=True),
        )
        self.inner_convs = nn.Sequential(
            nn.Conv2d(inner_dim * 2, inner_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(inner_dim),
            nn.ReLU(inplace=False),
        )
        self.outer_convs = nn.Sequential(
            nn.Conv2d(inner_dim * 2, inner_dim * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(inner_dim * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_dim * 4, inner_dim * 8, 3, stride=2, padding=1),
            nn.BatchNorm2d(inner_dim * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_dim * 8, outer_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(outer_dim),
            nn.ReLU(inplace=False),
        )
        self.unfold = nn.Unfold(kernel_size=4, padding=0, stride=4)

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.common_conv(x)

        H_out, W_out = H // 8, W // 8
        H_in, W_in = 4, 4

        inner_tokens = self.inner_convs(x)
        inner_tokens = self.unfold(inner_tokens).transpose(1, 2)
        inner_tokens = inner_tokens.reshape(B * H_out * W_out, self.inner_dim, H_in * W_in).transpose(1, 2)

        outer_tokens = self.outer_convs(x)
        outer_tokens = outer_tokens.permute(0, 2, 3, 1).reshape(B, H_out * W_out, -1)
        return inner_tokens, outer_tokens, (H_out, W_out), (H_in, W_in)


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------

class Stage(nn.Module):
    def __init__(self, num_blocks, outer_dim, inner_dim, outer_head, inner_head, num_patches, num_words, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, se=0, sr_ratio=1):
        super().__init__()
        blocks = []
        drop_path = drop_path if isinstance(drop_path, list) else [drop_path] * num_blocks

        for j in range(num_blocks):
            if j == 0:
                _inner_dim = inner_dim
            elif j == 1 and num_blocks > 6:
                _inner_dim = inner_dim
            else:
                _inner_dim = -1
            blocks.append(Block(
                outer_dim, _inner_dim, layer_id=j, outer_head=outer_head, inner_head=inner_head,
                num_words=num_words, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop,
                attn_drop=attn_drop, drop_path=drop_path[j], act_layer=act_layer, norm_layer=norm_layer,
                se=se, sr_ratio=sr_ratio))

        self.blocks = nn.ModuleList(blocks)

    def forward(self, inner_tokens, outer_tokens, H_out, W_out, H_in, W_in):
        for blk in self.blocks:
            inner_tokens, outer_tokens = blk(inner_tokens, outer_tokens, H_out, W_out, H_in, W_in)
        return inner_tokens, outer_tokens


# ---------------------------------------------------------------------------
# Per-stage upsample head
# ---------------------------------------------------------------------------

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.transposed_conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2, padding=0
        )
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.gelu1 = nn.GELU()
        self.conv = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.gelu2 = nn.GELU()

    def forward(self, x):
        x = self.transposed_conv(x)
        x = self.batch_norm1(x)
        x = self.gelu1(x)
        x = self.conv(x)
        x = self.batch_norm2(x)
        x = self.gelu2(x)
        return x


# ---------------------------------------------------------------------------
# 4-stage hierarchical encoder
# ---------------------------------------------------------------------------

class PyramidRiR_enc(nn.Module):
    """Verbatim from Zig_RiR2d.py.

    Produces 4 pyramid feature maps with strides 4 / 8 / 16 / 32 (after the
    per-stage UpsampleBlock).
    """

    def __init__(self, img_size=512, outer_dims=None, in_chans=1, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, se=0):
        super().__init__()
        if outer_dims is None:
            outer_dims = [96, 192, 384, 768]
        depths = [2, 4, 9, 2]
        inner_dims = [4, 4 * 2, 4 * 4, 4 * 8]
        outer_heads = [2, 2 * 2, 2 * 4, 2 * 8]
        inner_heads = [1, 1 * 2, 1 * 4, 1 * 8]
        sr_ratios = [4, 2, 1, 1]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.num_features = outer_dims[-1]

        self.patch_embed = Stem(img_size=img_size, in_chans=in_chans, outer_dim=outer_dims[0], inner_dim=inner_dims[0])
        num_patches = self.patch_embed.num_patches
        num_words = self.patch_embed.num_words
        self.pos_embed_sentence = nn.Parameter(torch.zeros(1, num_patches, outer_dims[0]))
        self.pos_embed_word = nn.Parameter(torch.zeros(1, num_words, inner_dims[0]))
        self.interpolate_mode = 'bicubic'

        depth = 0
        self.word_merges = nn.ModuleList([])
        self.sentence_merges = nn.ModuleList([])
        self.stages = nn.ModuleList([])
        for i in range(4):
            if i > 0:
                self.word_merges.append(PatchMerging2D_word(inner_dims[i - 1], inner_dims[i]))
                self.sentence_merges.append(PatchMerging2D_sentence(outer_dims[i - 1], outer_dims[i]))
            self.stages.append(Stage(depths[i], outer_dim=outer_dims[i], inner_dim=inner_dims[i],
                                     outer_head=outer_heads[i], inner_head=inner_heads[i],
                                     num_patches=num_patches // (2 ** i) // (2 ** i), num_words=num_words,
                                     mlp_ratio=mlp_ratio,
                                     qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                                     drop_path=dpr[depth:depth + depths[i]], norm_layer=norm_layer, se=se,
                                     sr_ratio=sr_ratios[i])
                               )
            depth += depths[i]

        self.up_blocks = nn.ModuleList([])
        for i in range(4):
            self.up_blocks.append(UpsampleBlock(outer_dims[i], outer_dims[i]))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'outer_pos', 'inner_pos'}

    def forward_features(self, x):
        inner_tokens, outer_tokens, (H_out, W_out), (H_in, W_in) = self.patch_embed(x)
        outputs = []

        for i in range(4):
            if i > 0:
                inner_tokens = self.word_merges[i - 1](inner_tokens, H_out, W_out, H_in, W_in)
                outer_tokens, H_out, W_out = self.sentence_merges[i - 1](outer_tokens, H_out, W_out)
            inner_tokens, outer_tokens = self.stages[i](inner_tokens, outer_tokens, H_out, W_out, H_in, W_in)
            b, l, m = outer_tokens.shape
            # outer_tokens grid may be non-square when img_size is not a power-of-2 multiple;
            # use the tracked H_out / W_out instead of int(sqrt(l)) for correctness.
            mid_out = outer_tokens.reshape(b, H_out, W_out, m).permute(0, 3, 1, 2)
            mid_out = self.up_blocks[i](mid_out)
            outputs.append(mid_out)
        return outputs

    def forward(self, x):
        return self.forward_features(x)


# ---------------------------------------------------------------------------
# Project-side wrapper: register under "rir_zigzag"
# ---------------------------------------------------------------------------

@ENCODER_REGISTRY.register("rir_zigzag")
class RIRZigzagEncoder(nn.Module):
    """Zig-RiR (RWKV-in-RWKV) hierarchical encoder.

    Thin wrapper exposing the standard project encoder interface:
        - ``out_channels``: list of per-stage channel widths
        - ``forward(x) -> List[Tensor]``: 4 pyramid feature maps

    Parameters
    ----------
    pretrained : bool
        Unused (upstream provides no public ImageNet checkpoint).
    in_channels : int
        Input channels (default 3 to match the project convention).
    img_size : int
        Spatial size of the input. Should be a multiple of 8 (Stem stride);
        a multiple of 64 is recommended so that all 4 stages produce
        non-fractional spatial sizes.
    outer_dims : tuple
        Per-stage outer-RWKV (sentence-level) channel widths. Default
        ``[96, 192, 384, 768]`` matches the default yaml in
        ``configs/{synapse,acdc,binary}/rir_zigzag.yaml``.
    drop_path_rate : float
        Stochastic depth rate.
    """

    def __init__(
        self,
        pretrained: bool = False,
        in_channels: int = 3,
        img_size: int = 224,
        outer_dims: Tuple[int, int, int, int] = (96, 192, 384, 768),
        drop_path_rate: float = 0.1,
        pretrained_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        outer_dims = list(outer_dims)
        self.backbone = PyramidRiR_enc(
            img_size=img_size,
            outer_dims=outer_dims,
            in_chans=in_channels,
            drop_path_rate=drop_path_rate,
        )
        self.out_channels = outer_dims

        if pretrained and pretrained_path:
            self._load_pretrained(pretrained_path)

    def _load_pretrained(self, path: str) -> None:
        state = torch.load(path, map_location='cpu')
        if isinstance(state, dict):
            if 'model' in state:
                state = state['model']
            elif 'state_dict' in state:
                state = state['state_dict']
        msg = self.load_state_dict(state, strict=False)
        print(f"[RIRZigzagEncoder] pretrained loaded: {msg}")

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.backbone(x)


# Public alias matching the paper's class name.
ZigRiREncoder = RIRZigzagEncoder
