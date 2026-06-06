"""RWKV-UNet Encoder: 1:1 faithful port of the official implementation.

Reference: "RWKV-UNet: Improving UNet with Linear Complexity for Medical Image
Segmentation" — https://github.com/juntaoJianggavin/RWKV-UNet

This module mirrors the official ``rwkv_unet.py`` (RWKV_UNet_encoder + GLSP +
VRWKV_SpatialMix) byte-for-byte in semantics, including the exact T / S / B
preset hyper-parameters published in the repository:

    Tiny  (t):  depths=[2, 2, 4, 2]  stem=24  embed=[32, 48, 96, 160]
    Small (s):  depths=[3, 3, 6, 3]  stem=24  embed=[32, 64, 128, 192]
    Base  (b):  depths=[3, 3, 6, 3]  stem=24  embed=[48, 72, 144, 240]

All three variants share dw_kss=[5, 5, 5, 5], attn_ss=[F, F, T, T],
norm_layers=[bn_2d, bn_2d, ln_2d, ln_2d], act_layers=[silu, silu, gelu, gelu]
and channel_gamma=1/4.

WKV implementation lives in :mod:`medseg.kernels.wkv`; this module re-exports
the legacy ``WKV`` / ``RUN_WKV`` symbols so that ``rir_zigzag_encoder.py`` and
any external code keep working unchanged.
"""
# Source: https://github.com/juntaoJianggavin/RWKV-UNet

import math
from functools import partial
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from medseg.registry import ENCODER_REGISTRY
from medseg.kernels.wkv import (  # noqa: F401  (re-exported)
    load_wkv_cuda,
    is_cuda_available,
    run_wkv,
    wkv_pytorch,
)


# ---------- WKV kernel dispatcher (backward-compatible shim) ----------


class WKV:
    """Backwards-compatible shim mimicking ``torch.autograd.Function.apply``.

    External code (notably ``rir_zigzag_encoder.py``) calls
    ``WKV.apply(B, T, C, w, u, k, v)`` exactly as in the upstream Vision-RWKV
    reference. We forward straight to :func:`run_wkv`, which is itself
    differentiable (CUDA path uses the analytic kernel, CPU path uses
    autograd), so callers get correct gradients.
    """

    @staticmethod
    def apply(B, T, C, w, u, k, v):
        return run_wkv(B, T, C, w, u, k, v)


def RUN_WKV(B, T, C, w, u, k, v):
    """Public WKV entry-point used inside this file (matches official RUN_CUDA)."""
    return run_wkv(B, T, C, w.float(), u.float(), k.float(), v.float())


# Legacy alias kept for any external import.
wkv_pytorch_fast = wkv_pytorch


# ---------- DropPath (timm-style) ----------


class DropPath(nn.Module):
    """Stochastic Depth per sample (matches timm.DropPath)."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rand = torch.rand(shape, dtype=x.dtype, device=x.device)
        rand = rand + keep_prob
        rand = rand.floor_()
        return x.div(keep_prob) * rand


# ---------- norm / act helpers (1:1 with module/basic_modules.py) ----------


class LayerNorm2d(nn.Module):
    def __init__(self, normalized_shape, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, eps, elementwise_affine)

    def forward(self, x):
        x = rearrange(x, "b c h w -> b h w c").contiguous()
        x = self.norm(x)
        x = rearrange(x, "b h w c -> b c h w").contiguous()
        return x


def get_norm(norm_layer: str = "bn_2d"):
    eps = 1e-6
    table = {
        "none": nn.Identity,
        "in_1d": partial(nn.InstanceNorm1d, eps=eps),
        "in_2d": partial(nn.InstanceNorm2d, eps=eps),
        "in_3d": partial(nn.InstanceNorm3d, eps=eps),
        "bn_1d": partial(nn.BatchNorm1d, eps=eps),
        "bn_2d": partial(nn.BatchNorm2d, eps=eps),
        "bn_3d": partial(nn.BatchNorm3d, eps=eps),
        "gn": partial(nn.GroupNorm, eps=eps),
        "ln_1d": partial(nn.LayerNorm, eps=eps),
        "ln_2d": partial(LayerNorm2d, eps=eps),
    }
    return table[norm_layer]


def get_act(act_layer: str = "relu"):
    table = {
        "none": nn.Identity,
        "relu": nn.ReLU,
        "relu6": nn.ReLU6,
        "silu": nn.SiLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
    }
    return table[act_layer]


class ConvNormAct(nn.Module):
    """Conv -> Norm -> Act (1:1 with module.basic_modules.ConvNormAct)."""

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        skip: bool = False,
        norm_layer: str = "bn_2d",
        act_layer: str = "relu",
        inplace: bool = True,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.has_skip = skip and dim_in == dim_out
        padding = math.ceil((kernel_size - stride) / 2)
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size, stride, padding, dilation, groups, bias)
        self.norm = get_norm(norm_layer)(dim_out)
        act_cls = get_act(act_layer)
        # GELU/Identity don't accept inplace; nn.ReLU/SiLU/ReLU6 do.
        if act_cls in (nn.GELU, nn.Identity, nn.Tanh, nn.Sigmoid):
            self.act = act_cls()
        else:
            self.act = act_cls(inplace=inplace)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x


class SE(nn.Module):
    """Squeeze-and-Excitation (matches timm SE used by the official repo)."""

    def __init__(self, in_chs: int, rd_ratio: float = 0.25, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        rd_channels = max(1, round(in_chs * rd_ratio))
        self.conv_reduce = nn.Conv2d(in_chs, rd_channels, 1, bias=True)
        self.act1 = act_layer(inplace=True) if act_layer in (nn.ReLU, nn.SiLU, nn.ReLU6) else act_layer()
        self.conv_expand = nn.Conv2d(rd_channels, in_chs, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


# ---------- q_shift (1:1 with the official 5-direction shift) ----------


def q_shift(input, shift_pixel: int = 1, gamma: float = 1 / 4, patch_resolution=None):
    """Bidirectional spatial q-shift used by VRWKV.

    Splits channels into 5 groups; first 4 groups are shifted in the four
    cardinal directions, the remaining channels stay put. Matches the official
    code byte-for-byte (cuda/wkv_op.cpp paired with VRWKV_SpatialMix).
    """
    assert gamma <= 1 / 4
    B, N, C = input.shape
    H, W = patch_resolution
    input = input.transpose(1, 2).reshape(B, C, H, W)
    out = torch.zeros_like(input)
    g = int(C * gamma)
    out[:, 0 * g:1 * g, :, shift_pixel:W] = input[:, 0 * g:1 * g, :, 0:W - shift_pixel]
    out[:, 1 * g:2 * g, :, 0:W - shift_pixel] = input[:, 1 * g:2 * g, :, shift_pixel:W]
    out[:, 2 * g:3 * g, shift_pixel:H, :] = input[:, 2 * g:3 * g, 0:H - shift_pixel, :]
    out[:, 3 * g:4 * g, 0:H - shift_pixel, :] = input[:, 3 * g:4 * g, shift_pixel:H, :]
    out[:, 4 * g:, ...] = input[:, 4 * g:, ...]
    return out.flatten(2).transpose(1, 2)


# ---------- Vision-RWKV blocks (1:1 with rwkv_unet.py / ccm.py) ----------


class VRWKV_SpatialMix(nn.Module):
    """Vision-RWKV spatial mixing (matches the official VRWKV_SpatialMix)."""

    def __init__(self, n_embd: int, channel_gamma: float = 1 / 4, shift_pixel: int = 1):
        super().__init__()
        self.n_embd = n_embd
        attn_sz = n_embd
        self.shift_pixel = shift_pixel
        self.channel_gamma = channel_gamma if shift_pixel > 0 else None

        # Learnable parameters (initialised to zero like the official repo).
        self.spatial_decay = nn.Parameter(torch.zeros(n_embd))
        self.spatial_first = nn.Parameter(torch.zeros(n_embd))
        self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, n_embd]) * 0.5)
        self.spatial_mix_v = nn.Parameter(torch.ones([1, 1, n_embd]) * 0.5)
        self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, n_embd]) * 0.5)

        self.key = nn.Linear(n_embd, attn_sz, bias=False)
        self.value = nn.Linear(n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(n_embd, attn_sz, bias=False)
        self.key_norm = nn.LayerNorm(n_embd)
        self.output = nn.Linear(attn_sz, n_embd, bias=False)

    def jit_func(self, x, patch_resolution):
        if self.shift_pixel > 0:
            xx = q_shift(x, self.shift_pixel, self.channel_gamma, patch_resolution)
            xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
            xv = x * self.spatial_mix_v + xx * (1 - self.spatial_mix_v)
            xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
        else:
            xk = xv = xr = x
        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        sr = torch.sigmoid(r)
        return sr, k, v

    def forward(self, x, patch_resolution=None):
        B, T, C = x.size()
        sr, k, v = self.jit_func(x, patch_resolution)
        # Note: official code passes spatial_decay/T and spatial_first/T to RUN_CUDA.
        x = RUN_WKV(B, T, C, self.spatial_decay / T, self.spatial_first / T, k, v)
        x = self.key_norm(x)
        x = sr * x
        x = self.output(x)
        return x


class VRWKV_ChannelMix(nn.Module):
    """Vision-RWKV channel mixing (used by CCMix in the decoder)."""

    def __init__(self, n_embd: int, channel_gamma: float = 1 / 4, shift_pixel: int = 1,
                 hidden_rate: int = 2, key_norm: bool = True):
        super().__init__()
        self.n_embd = n_embd
        self.shift_pixel = shift_pixel
        self.channel_gamma = channel_gamma if shift_pixel > 0 else None

        self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, n_embd]) * 0.5)
        self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, n_embd]) * 0.5)

        hidden_sz = hidden_rate * n_embd
        self.key = nn.Linear(n_embd, hidden_sz, bias=False)
        self.key_norm = nn.LayerNorm(hidden_sz) if key_norm else None
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)

    def forward(self, x, patch_resolution=None):
        if self.shift_pixel > 0:
            xx = q_shift(x, self.shift_pixel, self.channel_gamma, patch_resolution)
            xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
            xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
        else:
            xk = xr = x
        k = self.key(xk)
        k = torch.square(torch.relu(k))
        if self.key_norm is not None:
            k = self.key_norm(k)
        kv = self.value(k)
        x = torch.sigmoid(self.receptance(xr)) * kv
        return x


# ---------- GLSP (1:1 with the official Global-Local Spatial Perception) ----------


class GLSP(nn.Module):
    """Global-Local Spatial Perception block from RWKV-UNet (faithful port)."""

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        norm_in: bool = True,
        has_skip: bool = True,
        exp_ratio: float = 1.0,
        norm_layer: str = "bn_2d",
        act_layer: str = "relu",
        dw_ks: int = 3,
        stride: int = 1,
        dilation: int = 1,
        se_ratio: float = 0.0,
        attn_s: bool = True,
        drop_path: float = 0.0,
        drop: float = 0.0,
        img_size: int = 224,
        channel_gamma: float = 1 / 4,
        shift_pixel: int = 1,
    ):
        super().__init__()
        self.norm = get_norm(norm_layer)(dim_in) if norm_in else nn.Identity()
        dim_mid = int(dim_in * exp_ratio)
        self.ln1 = nn.LayerNorm(dim_mid)
        self.conv = ConvNormAct(dim_in, dim_mid, kernel_size=1)
        self.has_skip = (dim_in == dim_out and stride == 1) and has_skip
        self.attn_s = attn_s
        self.att = VRWKV_SpatialMix(dim_mid, channel_gamma, shift_pixel) if attn_s else None
        if se_ratio > 0.0:
            self.se = SE(dim_mid, rd_ratio=se_ratio, act_layer=get_act(act_layer))
        else:
            self.se = nn.Identity()
        self.proj_drop = nn.Dropout(drop)
        # Final 1x1 projection (no norm/act, matching the official block).
        self.proj = ConvNormAct(dim_mid, dim_out, kernel_size=1, norm_layer="none", act_layer="none")
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()
        # Local depthwise branch (k=dw_ks, with stride for downsampling).
        self.conv_local = ConvNormAct(
            dim_mid, dim_mid,
            kernel_size=dw_ks, stride=stride, dilation=dilation, groups=dim_mid,
            norm_layer="bn_2d", act_layer="silu",
        )

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        x = self.conv(x)
        if self.attn_s:
            B, hidden, H, W = x.size()
            patch_resolution = (H, W)
            x_seq = x.view(B, hidden, -1).permute(0, 2, 1)  # (B, N, C)
            x_seq = x_seq + self.drop_path(self.ln1(self.att(x_seq, patch_resolution)))
            B, n_patch, hidden = x_seq.size()
            h = w = int(math.sqrt(n_patch))
            x = x_seq.permute(0, 2, 1).contiguous().view(B, hidden, h, w)
        x = x + self.se(self.conv_local(x)) if self.has_skip else self.se(self.conv_local(x))
        x = self.proj_drop(x)
        x = self.proj(x)
        x = (shortcut + self.drop_path(x)) if self.has_skip else x
        return x


# ---------- RWKV-UNet Encoder (T / S / B) ----------


_VARIANT_PRESETS = {
    "t": dict(
        depths=[2, 2, 4, 2], stem_dim=24, embed_dims=[32, 48, 96, 160],
        exp_ratios=[2.0, 2.5, 3.0, 3.5],
    ),
    "s": dict(
        depths=[3, 3, 6, 3], stem_dim=24, embed_dims=[32, 64, 128, 192],
        exp_ratios=[2.0, 2.5, 3.0, 4.0],
    ),
    "b": dict(
        depths=[3, 3, 6, 3], stem_dim=24, embed_dims=[48, 72, 144, 240],
        exp_ratios=[2.0, 2.5, 4.0, 4.0],
    ),
}

# Hyper-parameters that are identical across all three variants.
_VARIANT_SHARED = dict(
    norm_layers=["bn_2d", "bn_2d", "ln_2d", "ln_2d"],
    act_layers=["silu", "silu", "gelu", "gelu"],
    dw_kss=[5, 5, 5, 5],
    se_ratios=[0.0, 0.0, 0.0, 0.0],
    attn_ss=[False, False, True, True],
    channel_gamma=1 / 4,
    shift_pixel=1,
)


@ENCODER_REGISTRY.register("rwkv_unet")
class RWKVUNetEncoder(nn.Module):
    """Faithful 1:1 port of the official RWKV_UNet_encoder (T / S / B variants).

    Forward returns ``[enc1, enc2, enc3, stage4_out]`` where ``stage4_out`` is
    the deepest feature consumed by the bottleneck and ``enc1..enc3`` are the
    skip features consumed by :class:`RWKVUNetDecoder`. The stem-stage output
    (24-d, full-resolution) is intentionally not exposed because the official
    decoder does not consume it either.
    """

    def __init__(
        self,
        pretrained: bool = False,
        in_channels: int = 3,
        img_size: int = 224,
        variant: str = "b",
        drop: float = 0.0,
        drop_path: float = 0.05,
        pretrained_path: str = None,
        # Optional manual overrides (e.g. for ablation); leave at defaults to
        # use the official preset for the chosen variant.
        depths=None,
        stem_dim=None,
        embed_dims=None,
        exp_ratios=None,
        norm_layers=None,
        act_layers=None,
        dw_kss=None,
        se_ratios=None,
        attn_ss=None,
        channel_gamma=None,
        shift_pixel=None,
        **kwargs,
    ):
        super().__init__()
        variant = variant.lower()
        if variant not in _VARIANT_PRESETS:
            raise ValueError(
                f"Unknown RWKV-UNet variant '{variant}'. Choose from 't', 's', 'b'."
            )
        preset = dict(_VARIANT_PRESETS[variant])
        shared = dict(_VARIANT_SHARED)
        # Apply manual overrides if provided.
        for name, val in [
            ("depths", depths), ("stem_dim", stem_dim), ("embed_dims", embed_dims),
            ("exp_ratios", exp_ratios),
        ]:
            if val is not None:
                preset[name] = list(val) if isinstance(val, (list, tuple)) else val
        for name, val in [
            ("norm_layers", norm_layers), ("act_layers", act_layers),
            ("dw_kss", dw_kss), ("se_ratios", se_ratios), ("attn_ss", attn_ss),
            ("channel_gamma", channel_gamma), ("shift_pixel", shift_pixel),
        ]:
            if val is not None:
                shared[name] = list(val) if isinstance(val, (list, tuple)) and name != "channel_gamma" and name != "shift_pixel" else val

        depths = preset["depths"]
        stem_dim = preset["stem_dim"]
        embed_dims = preset["embed_dims"]
        exp_ratios = preset["exp_ratios"]
        norm_layers = shared["norm_layers"]
        act_layers = shared["act_layers"]
        dw_kss = shared["dw_kss"]
        se_ratios = shared["se_ratios"]
        attn_ss = shared["attn_ss"]
        channel_gamma = shared["channel_gamma"]
        shift_pixel = shared["shift_pixel"]

        self.variant = variant
        self.in_channels = in_channels
        self.img_size = img_size
        self.embed_dims = embed_dims
        self.stem_dim = stem_dim
        self.depths = depths

        # ---- stage 0: stem GLSP (no spatial attn, stride=1) ----
        self.stage0 = nn.ModuleList([
            GLSP(
                in_channels, stem_dim,
                norm_in=False, has_skip=False, exp_ratio=1.0,
                norm_layer=norm_layers[0], act_layer=act_layers[0], dw_ks=dw_kss[0],
                stride=1, dilation=1, se_ratio=1.0, attn_s=False,
                drop_path=0.0, drop=0.0, img_size=img_size,
                channel_gamma=channel_gamma, shift_pixel=shift_pixel,
            )
        ])

        # Build remaining 4 stages.
        dprs = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]
        emb_dim_pre = stem_dim
        cur_img_size = img_size  # tracked solely to keep parity with official ctor

        for i in range(len(depths)):
            layers = []
            stage_dpr = dprs[sum(depths[:i]):sum(depths[:i + 1])]
            for j in range(depths[i]):
                if j == 0:
                    stride, has_skip, attn_s = 2, False, False
                    exp_ratio = exp_ratios[i] * 2
                    cur_img_size = cur_img_size // 2
                else:
                    stride, has_skip, attn_s = 1, True, attn_ss[i]
                    exp_ratio = exp_ratios[i]
                layers.append(GLSP(
                    emb_dim_pre, embed_dims[i],
                    norm_in=True, has_skip=has_skip, exp_ratio=exp_ratio,
                    norm_layer=norm_layers[i], act_layer=act_layers[i], dw_ks=dw_kss[i],
                    stride=stride, dilation=1, se_ratio=se_ratios[i], attn_s=attn_s,
                    drop_path=stage_dpr[j], drop=drop, img_size=cur_img_size,
                    channel_gamma=channel_gamma, shift_pixel=shift_pixel,
                ))
                emb_dim_pre = embed_dims[i]
            self.__setattr__(f"stage{i + 1}", nn.ModuleList(layers))

        # The last layer of the official encoder body (used as bottleneck input).
        self.norm = get_norm(norm_layers[-1])(embed_dims[-1])

        # Channels exposed to the model builder:
        #   - skip features = enc1, enc2, enc3 (embed_dims[0..2])
        #   - deepest feature (bottleneck input) = stage4 output (embed_dims[3])
        self.out_channels = list(embed_dims)

        self.apply(self._init_weights)

        if pretrained and pretrained_path:
            self.load_pretrained(pretrained_path)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if m.bias is not None:
                nn.init.zeros_(m.bias)
            if m.weight is not None:
                nn.init.ones_(m.weight)

    def load_pretrained(self, path: str):
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict):
            for key in ("model", "state_dict"):
                if key in state:
                    state = state[key]
                    break
        msg = self.load_state_dict(state, strict=False)
        print(f"[RWKVUNetEncoder] loaded pretrained '{path}': {msg}")

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # If the network was configured for 3-channel input but the user passed
        # a 1-channel tensor (mirroring the official RWKV_UNet.forward), repeat
        # along the channel axis automatically.
        if x.shape[1] == 1 and self.in_channels == 3:
            x = x.repeat(1, 3, 1, 1)

        for blk in self.stage0:
            x = blk(x)
        # enc0 is intentionally not exposed (decoder does not consume it).

        for blk in self.stage1:
            x = blk(x)
        enc1 = x

        for blk in self.stage2:
            x = blk(x)
        enc2 = x

        for blk in self.stage3:
            x = blk(x)
        enc3 = x

        for blk in self.stage4:
            x = blk(x)
        # x is the deepest feature; the bottleneck stage will run a no-op or
        # custom transformation on it.
        return [enc1, enc2, enc3, x]
