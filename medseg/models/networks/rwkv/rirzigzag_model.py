"""RIRZigzag (Zig-RiR) – self-contained port from github.com/txchen-USTC/Zig-RiR.

Zigzag RWKV-in-RWKV for Efficient Medical Image Segmentation (TMI 2025).

Architecture: Hierarchical encoder with nested RWKV-in-RWKV blocks
              (Outer + Inner Zigzag RWKV) + UNet-style decoder.

NOTE: The original uses a custom CUDA kernel for WKV computation.
      This port provides a pure-PyTorch fallback that works on any device.
"""
# Source: https://github.com/txchen-USTC/Zig-RiR

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Try to load CUDA WKV kernel; raise if unavailable on CUDA device
# ---------------------------------------------------------------------------
_WKV_CUDA_AVAILABLE = False
_wkv_cuda = None
try:
    from torch.utils.cpp_extension import load as _load_ext
    _wkv_cuda = _load_ext(
        name="wkv",
        sources=["./cuda/wkv_op.cpp", "./cuda/wkv_cuda.cu"],
        verbose=False,
        extra_cuda_cflags=['-res-usage', '--maxrregcount 60', '-DTmax=32768'])
    _WKV_CUDA_AVAILABLE = True
except Exception as _e:
    import warnings
    warnings.warn(
        f"RWKV WKV CUDA kernel compilation failed ({type(_e).__name__}: {_e}). "
        "The model will fall back to the pure-PyTorch WKV implementation "
        "which is significantly slower. To use the fast CUDA kernel, ensure "
        "a working CUDA toolkit and C++ compiler are available."
    )


def _wkv_pure_pytorch(B, T, C, w, u, k, v):
    """Pure-PyTorch WKV computation (differentiable, O(T) sequential)."""
    y = torch.empty(B, T, C, device=k.device, dtype=k.dtype)
    aa = torch.zeros(B, C, device=k.device, dtype=torch.float32)
    bb = torch.zeros(B, C, device=k.device, dtype=torch.float32)
    pp = torch.full((B, C), -1e30, device=k.device, dtype=torch.float32)
    ww = u.float()
    for i in range(T):
        kk = k[:, i, :].float()
        vv = v[:, i, :].float()
        ww_new = torch.maximum(pp, kk)
        e1 = torch.exp(pp - ww_new)
        e2 = torch.exp(kk - ww_new)
        y[:, i, :] = ((e1 * aa + e2 * vv) / (e1 * bb + e2)).to(k.dtype)
        ww2 = torch.maximum(pp + w.float(), kk)
        e1 = torch.exp(pp + w.float() - ww2)
        e2 = torch.exp(kk - ww2)
        aa = e1 * aa + e2 * vv
        bb = e1 * bb + e2
        pp = ww2
    return y


def _RUN_WKV(B, T, C, w, u, k, v):
    """Run WKV: uses CUDA kernel if available, else pure PyTorch."""
    if _WKV_CUDA_AVAILABLE and k.is_cuda:
        return _WKVCUDA.apply(B, T, C, w.cuda(), u.cuda(), k.cuda(), v.cuda())
    return _wkv_pure_pytorch(B, T, C, w, u, k, v)


class _WKVCUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v):
        ctx.B, ctx.T, ctx.C = B, T, C
        ctx.save_for_backward(w, u, k, v)
        wf = w.float().contiguous()
        uf = u.float().contiguous()
        kf = k.float().contiguous()
        vf = v.float().contiguous()
        y = torch.empty(B, T, C, device=k.device, dtype=k.dtype)
        _wkv_cuda.forward(B, T, C, wf, uf, kf, vf, y)
        if k.dtype == torch.half:
            y = y.half()
        return y

    @staticmethod
    def backward(ctx, gy):
        B, T, C = ctx.B, ctx.T, ctx.C
        w, u, k, v = ctx.saved_tensors
        gw = torch.zeros(B, C, device=k.device).contiguous()
        gu = torch.zeros(B, C, device=k.device).contiguous()
        gk = torch.zeros(B, T, C, device=k.device).contiguous()
        gv = torch.zeros(B, T, C, device=k.device).contiguous()
        _wkv_cuda.backward(B, T, C, w.float().contiguous(), u.float().contiguous(),
                           k.float().contiguous(), v.float().contiguous(),
                           gy.float().contiguous(), gw, gu, gk, gv)
        return None, None, None, gw.sum(0), gu.sum(0), gk, gv


# ---------------------------------------------------------------------------
# Spatial shift
# ---------------------------------------------------------------------------
def q_shift(input, shift_pixel=1, gamma=1 / 4):
    B, C, H, W = input.shape
    output = torch.zeros_like(input)
    g = int(C * gamma)
    output[:, :g, :, shift_pixel:W] = input[:, :g, :, :W - shift_pixel]
    output[:, g:2 * g, :, :W - shift_pixel] = input[:, g:2 * g, :, shift_pixel:W]
    output[:, 2 * g:3 * g, shift_pixel:H, :] = input[:, 2 * g:3 * g, :H - shift_pixel, :]
    output[:, 3 * g:4 * g, :H - shift_pixel, :] = input[:, 3 * g:4 * g, shift_pixel:H, :]
    if 4 * g < C:
        output[:, 4 * g:, ...] = input[:, 4 * g:, ...]
    return output


# ---------------------------------------------------------------------------
# VRWKV Spatial Mix (Zigzag RWKV attention)
# ---------------------------------------------------------------------------
class _VRWKVSpatialMix(nn.Module):
    def __init__(self, n_embd, layer_id, key_norm=False, scan_schemes=None):
        super().__init__()
        self.n_embd = n_embd
        self.layer_id = layer_id
        self.recurrence = 2
        self.scan_schemes = scan_schemes or [
            ('top-left', 'horizontal'), ('bottom-right', 'vertical')]
        self.dwconv = nn.Conv2d(n_embd, n_embd, 3, 1, 1, groups=n_embd, bias=False)
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.key_norm = nn.LayerNorm(n_embd) if key_norm else None
        self.output = nn.Linear(n_embd, n_embd, bias=False)
        self.spatial_decay = nn.Parameter(torch.randn(self.recurrence, n_embd))
        self.spatial_first = nn.Parameter(torch.randn(self.recurrence, n_embd))

    @staticmethod
    def _get_zigzag_indices(h, w, start, direction, device):
        indices = []
        if start == 'top-left':
            row_start, col_start, row_step, col_step = 0, 0, 1, 1
        elif start == 'top-right':
            row_start, col_start, row_step, col_step = 0, w - 1, 1, -1
        elif start == 'bottom-left':
            row_start, col_start, row_step, col_step = h - 1, 0, -1, 1
        else:
            row_start, col_start, row_step, col_step = h - 1, w - 1, -1, -1
        for i in range(h):
            cr = row_start + row_step * i
            if direction == 'horizontal':
                cols = list(range(w)) if cr % 2 == 0 else list(range(w - 1, -1, -1))
                for col in cols:
                    indices.append(cr * w + col)
            else:
                cc = col_start + col_step * i
                rows = list(range(h)) if cc % 2 == 0 else list(range(h - 1, -1, -1))
                for row in rows:
                    indices.append(row * w + cc)
        return torch.tensor(indices, dtype=torch.long, device=device)

    def forward(self, x, resolution):
        B, T, C = x.size()
        h, w = resolution
        scheme = self.scan_schemes[self.layer_id % len(self.scan_schemes)]
        start, direction = scheme
        zigzag = self._get_zigzag_indices(h, w, start, direction, x.device)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = q_shift(x)
        x = rearrange(x, 'b c h w -> b c (h w)')
        x = x[..., zigzag]
        x = rearrange(x, 'b c (h w) -> b (h w) c', h=h, w=w)
        k = self.key(x)
        v = self.value(x)
        r = torch.sigmoid(self.receptance(x))
        for j in range(self.recurrence):
            v = _RUN_WKV(B, T, C, self.spatial_decay[j] / T,
                         self.spatial_first[j] / T, k, v)
        x = v
        if self.key_norm is not None:
            x = self.key_norm(x)
        return self.output(r * x)


# ---------------------------------------------------------------------------
# VRWKV Channel Mix (FFN)
# ---------------------------------------------------------------------------
class _VRWKVChannelMix(nn.Module):
    def __init__(self, n_embd, hidden_rate=4, key_norm=False):
        super().__init__()
        hidden = int(hidden_rate * n_embd)
        self.key = nn.Linear(n_embd, hidden, bias=False)
        self.key_norm = nn.LayerNorm(hidden) if key_norm else None
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden, n_embd, bias=False)

    def forward(self, x, resolution):
        h, w = resolution
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = q_shift(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        k = torch.square(torch.relu(self.key(x)))
        if self.key_norm is not None:
            k = self.key_norm(k)
        return torch.sigmoid(self.receptance(x)) * self.value(k)


# ---------------------------------------------------------------------------
# Zig-RiR Block (nested RWKV-in-RWKV)
# ---------------------------------------------------------------------------
class _Block(nn.Module):
    def __init__(self, outer_dim, inner_dim, layer_id, num_words,
                 mlp_ratio=4., drop_path=0.):
        super().__init__()
        self.has_inner = inner_dim > 0
        if self.has_inner:
            self.inner_norm1 = nn.LayerNorm(num_words * inner_dim)
            self.inner_attn = _VRWKVSpatialMix(inner_dim, layer_id)
            self.inner_norm2 = nn.LayerNorm(num_words * inner_dim)
            self.inner_ffn = _VRWKVChannelMix(inner_dim, mlp_ratio)
            self.proj_norm1 = nn.LayerNorm(num_words * inner_dim)
            self.proj = nn.Linear(num_words * inner_dim, outer_dim, bias=False)
            self.proj_norm2 = nn.LayerNorm(outer_dim)
        self.outer_norm1 = nn.LayerNorm(outer_dim)
        self.outer_attn = _VRWKVSpatialMix(outer_dim, layer_id)
        self.drop_path = nn.Identity()
        if drop_path > 0.:
            from timm.models.layers import DropPath
            self.drop_path = DropPath(drop_path)
        self.outer_norm2 = nn.LayerNorm(outer_dim)
        self.outer_ffn = _VRWKVChannelMix(outer_dim, mlp_ratio)

    def forward(self, x, outer_tokens, H_out, W_out, H_in, W_in):
        B, N, C = outer_tokens.size()
        if self.has_inner:
            inner_res = [H_in, W_in]
            x_flat = self.inner_norm1(x.reshape(B, N, -1))
            x_in = x_flat.reshape(B * N, H_in * W_in, -1)
            x = x + self.drop_path(self.inner_attn(x_in, inner_res))
            x_flat = self.inner_norm2(x.reshape(B, N, -1))
            x_in = x_flat.reshape(B * N, H_in * W_in, -1)
            x = x + self.drop_path(self.inner_ffn(x_in, inner_res))
            outer_tokens = outer_tokens + self.proj_norm2(
                self.proj(self.proj_norm1(x.reshape(B, N, -1))))
        outer_res = [H_out, W_out]
        outer_tokens = outer_tokens + self.drop_path(
            self.outer_attn(self.outer_norm1(outer_tokens), outer_res))
        outer_tokens = outer_tokens + self.drop_path(
            self.outer_ffn(self.outer_norm2(outer_tokens), outer_res))
        return x, outer_tokens


# ---------------------------------------------------------------------------
# Patch Merging
# ---------------------------------------------------------------------------
class _PatchMerging2DSentence(nn.Module):
    def __init__(self, dim_in, dim_out, stride=2):
        super().__init__()
        self.stride = stride
        self.norm = nn.LayerNorm(dim_in)
        self.conv = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 2 * stride - 1, padding=stride - 1,
                      stride=stride))

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = self.norm(x).transpose(1, 2).reshape(B, C, H, W)
        x = self.conv(x)
        H, W = math.ceil(H / self.stride), math.ceil(W / self.stride)
        return x.reshape(B, -1, H * W).transpose(1, 2), H, W


class _PatchMerging2DWord(nn.Module):
    def __init__(self, dim_in, dim_out, stride=2):
        super().__init__()
        self.stride = stride
        self.dim_out = dim_out
        self.norm = nn.LayerNorm(dim_in)
        self.conv = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 2 * stride - 1, padding=stride - 1,
                      stride=stride))

    def forward(self, x, H_out, W_out, H_in, W_in):
        B_N, M, C = x.shape
        x = self.norm(x).reshape(-1, H_out, W_out, H_in, W_in, C)
        pad = (H_out % 2 == 1) or (W_out % 2 == 1)
        if pad:
            x = F.pad(x.permute(0, 3, 4, 5, 1, 2), (0, W_out % 2, 0, H_out % 2))
            x = x.permute(0, 4, 5, 1, 2, 3)
        x1 = x[:, 0::2, 0::2, :, :, :]
        x2 = x[:, 1::2, 0::2, :, :, :]
        x3 = x[:, 0::2, 1::2, :, :, :]
        x4 = x[:, 1::2, 1::2, :, :, :]
        x = torch.cat([torch.cat([x1, x2], 3), torch.cat([x3, x4], 3)], 4)
        x = x.reshape(-1, 2 * H_in, 2 * W_in, C).permute(0, 3, 1, 2)
        x = self.conv(x)
        return x.reshape(-1, self.dim_out, M).transpose(1, 2)


# ---------------------------------------------------------------------------
# Stem
# ---------------------------------------------------------------------------
class _Stem(nn.Module):
    def __init__(self, img_size=224, in_chans=3, outer_dim=768, inner_dim=24):
        super().__init__()
        self.inner_dim = inner_dim
        self.num_patches = (img_size // 8) ** 2
        self.num_words = 16
        self.common_conv = nn.Sequential(
            nn.Conv2d(in_chans, inner_dim * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(inner_dim * 2), nn.ReLU(inplace=True))
        self.inner_convs = nn.Sequential(
            nn.Conv2d(inner_dim * 2, inner_dim, 3, 1, 1),
            nn.BatchNorm2d(inner_dim), nn.ReLU(inplace=False))
        self.outer_convs = nn.Sequential(
            nn.Conv2d(inner_dim * 2, inner_dim * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(inner_dim * 4), nn.ReLU(inplace=True),
            nn.Conv2d(inner_dim * 4, inner_dim * 8, 3, stride=2, padding=1),
            nn.BatchNorm2d(inner_dim * 8), nn.ReLU(inplace=True),
            nn.Conv2d(inner_dim * 8, outer_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(outer_dim), nn.ReLU(inplace=False))
        self.unfold = nn.Unfold(kernel_size=4, padding=0, stride=4)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.common_conv(x)
        H_out, W_out = H // 8, W // 8
        H_in, W_in = 4, 4
        inner_tokens = self.inner_convs(x)
        inner_tokens = self.unfold(inner_tokens).transpose(1, 2)
        inner_tokens = inner_tokens.reshape(
            B * H_out * W_out, self.inner_dim, H_in * W_in).transpose(1, 2)
        outer_tokens = self.outer_convs(x)
        outer_tokens = outer_tokens.permute(0, 2, 3, 1).reshape(
            B, H_out * W_out, -1)
        return inner_tokens, outer_tokens, (H_out, W_out), (H_in, W_in)


# ---------------------------------------------------------------------------
# Stage + UpsampleBlock
# ---------------------------------------------------------------------------
class _Stage(nn.Module):
    def __init__(self, num_blocks, outer_dim, inner_dim, layer_offset,
                 num_patches, num_words, drop_path=0.):
        super().__init__()
        blocks = []
        dp = drop_path if isinstance(drop_path, list) else [drop_path] * num_blocks
        for j in range(num_blocks):
            _inner = inner_dim if j == 0 else -1
            blocks.append(_Block(outer_dim, _inner, layer_offset + j,
                                 num_words, drop_path=dp[j]))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, inner_tokens, outer_tokens, H_out, W_out, H_in, W_in):
        for blk in self.blocks:
            inner_tokens, outer_tokens = blk(
                inner_tokens, outer_tokens, H_out, W_out, H_in, W_in)
        return inner_tokens, outer_tokens


class _UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2),
            nn.BatchNorm2d(out_channels), nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels), nn.GELU())

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Pyramid Encoder
# ---------------------------------------------------------------------------
class _PyramidRiREnc(nn.Module):
    def __init__(self, img_size=224, outer_dims=None, in_chans=3):
        super().__init__()
        depths = [2, 4, 9, 2]
        inner_dims = [4, 8, 16, 32]
        self.word_merges = nn.ModuleList()
        self.sentence_merges = nn.ModuleList()
        self.stages = nn.ModuleList()
        self.patch_embed = _Stem(img_size, in_chans, outer_dims[0], inner_dims[0])
        num_patches = self.patch_embed.num_patches
        num_words = self.patch_embed.num_words
        depth = 0
        for i in range(4):
            if i > 0:
                self.word_merges.append(
                    _PatchMerging2DWord(inner_dims[i - 1], inner_dims[i]))
                self.sentence_merges.append(
                    _PatchMerging2DSentence(outer_dims[i - 1], outer_dims[i]))
            self.stages.append(_Stage(
                depths[i], outer_dims[i], inner_dims[i], depth,
                num_patches // (4 ** i), num_words, drop_path=0.1))
            depth += depths[i]
        self.up_blocks = nn.ModuleList([
            _UpsampleBlock(d, d) for d in outer_dims])

    def forward_features(self, x):
        inner_tokens, outer_tokens, (Ho, Wo), (Hi, Wi) = self.patch_embed(x)
        outputs = []
        for i in range(4):
            if i > 0:
                inner_tokens = self.word_merges[i - 1](
                    inner_tokens, Ho, Wo, Hi, Wi)
                outer_tokens, Ho, Wo = self.sentence_merges[i - 1](
                    outer_tokens, Ho, Wo)
            inner_tokens, outer_tokens = self.stages[i](
                inner_tokens, outer_tokens, Ho, Wo, Hi, Wi)
            b, l, m = outer_tokens.shape
            mid = outer_tokens.reshape(
                b, int(math.sqrt(l)), int(math.sqrt(l)), m).permute(0, 3, 1, 2)
            mid = self.up_blocks[i](mid)
            outputs.append(mid)
        return outputs


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------
class _Decoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(2 * out_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Interpolate if spatial sizes don't match
        if x1.shape[-2:] != x2.shape[-2:]:
            x1 = F.interpolate(x1, size=x2.shape[-2:], mode='bilinear',
                               align_corners=True)
        return self.conv_bn_relu(torch.cat((x1, x2), dim=1))


# ---------------------------------------------------------------------------
# RIRZigzag
# ---------------------------------------------------------------------------
class RIRZigzag(nn.Module):
    """Zigzag RWKV-in-RWKV segmentation model.

    Args:
        in_channels: Number of input image channels (default 3).
        num_classes: Number of output segmentation classes (default 2).
        img_size: Expected input spatial resolution (default 224).
    """

    def __init__(self, in_channels=3, num_classes=2, img_size=224, **kwargs):
        super().__init__()
        channels = [96, 192, 384, 768]
        self.backbone = _PyramidRiREnc(img_size, channels, in_channels)
        self.decode4 = _Decoder(channels[3], channels[2])
        self.decode3 = _Decoder(channels[2], channels[1])
        self.decode2 = _Decoder(channels[1], channels[0])
        self.decode0 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(channels[0], num_classes, 1, bias=False))

    def forward(self, x):
        outputs = self.backbone.forward_features(x)
        t1, t2, t3, t4 = outputs
        d4 = self.decode4(t4, t3)
        d3 = self.decode3(d4, t2)
        d2 = self.decode2(d3, t1)
        return self.decode0(d2)
