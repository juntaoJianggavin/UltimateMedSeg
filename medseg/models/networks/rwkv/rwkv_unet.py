"""RWKV-UNet: Improving UNet with Long-Range Cooperation for Medical Image Segmentation.

Faithful self-contained port of:
  https://github.com/juntaoJianggavin/RWKV-UNet  (arxiv 2025)

Key components (1:1 with the official repo):
  - VRWKV_SpatialMix: Vision-RWKV spatial mixing with q_shift + WKV attention
  - VRWKV_ChannelMix: Vision-RWKV channel mixing with ReLU^2 gating
  - GLSP: Global-Local Spatial Perception block (conv + optional VRWKV + SE)
  - CCMix: cross-channel mixer fusing [enc3, enc2, enc1] via VRWKV_ChannelMix
  - UpBlock: decoder block (conv → DWConv k=9 → 1x1 proj → bilinear 2x up)
  - T / S / B encoder presets

WKV is dispatched via medseg.kernels.wkv (CUDA on GPU, PyTorch on CPU).
All building blocks are inlined — no external encoder/decoder imports.
"""
# Source: https://github.com/juntaoJianggavin/RWKV-UNet

import math
from functools import partial
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce

from medseg.kernels.wkv import run_wkv as _run_wkv


# ---------------------------------------------------------------------------
# WKV helper
# ---------------------------------------------------------------------------

def _RUN_WKV(B, T, C, w, u, k, v):
    """Matches official ``RUN_CUDA`` (float32 WKV dispatch)."""
    return _run_wkv(B, T, C, w.float(), u.float(), k.float(), v.float())


# ---------------------------------------------------------------------------
# DropPath / Norm / Act helpers (inlined)
# ---------------------------------------------------------------------------

class DropPath(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.drop_prob = p

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rnd = (torch.rand(shape, dtype=x.dtype, device=x.device) + keep).floor_()
        return x.div(keep) * rnd


class LayerNorm2d(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()


def _get_norm(name):
    return {"none": nn.Identity, "bn_2d": partial(nn.BatchNorm2d, eps=1e-6),
            "ln_2d": partial(LayerNorm2d, eps=1e-6),
            "ln_1d": partial(nn.LayerNorm, eps=1e-6)}[name]


def _make_act(name, inplace=True):
    t = {"none": nn.Identity, "relu": nn.ReLU, "silu": nn.SiLU,
         "gelu": nn.GELU, "sigmoid": nn.Sigmoid}
    cls = t[name]
    return cls(inplace=inplace) if cls in (nn.ReLU, nn.SiLU) else cls()


class ConvNormAct(nn.Module):
    def __init__(self, din, dout, ks, stride=1, dilation=1, groups=1,
                 bias=False, norm_layer="bn_2d", act_layer="relu", inplace=True):
        super().__init__()
        pad = math.ceil((ks - stride) / 2)
        self.conv = nn.Conv2d(din, dout, ks, stride, pad, dilation, groups, bias)
        self.norm = _get_norm(norm_layer)(dout)
        self.act = _make_act(act_layer, inplace)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


# ---------------------------------------------------------------------------
# SE (Squeeze-and-Excitation)
# ---------------------------------------------------------------------------

class SE(nn.Module):
    def __init__(self, in_chs, rd_ratio=0.25, act_layer=nn.ReLU,
                 gate_layer=nn.Sigmoid):
        super().__init__()
        rd = max(1, round(in_chs * rd_ratio))
        self.conv_reduce = nn.Conv2d(in_chs, rd, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(rd, in_chs, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x):
        s = self.gate(self.conv_expand(self.act1(self.conv_reduce(
            x.mean((2, 3), keepdim=True)))))
        return x * s


# ---------------------------------------------------------------------------
# q_shift (1:1 with the official 5-direction shift)
# ---------------------------------------------------------------------------

def q_shift(input, shift_pixel=1, gamma=1 / 4, patch_resolution=None):
    """Bidirectional spatial q-shift used by VRWKV_SpatialMix."""
    assert gamma <= 1 / 4
    B, N, C = input.shape
    H, W = patch_resolution
    input = input.transpose(1, 2).reshape(B, C, H, W)
    out = torch.zeros_like(input)
    g = int(C * gamma)
    out[:, 0:g, :, shift_pixel:W] = input[:, 0:g, :, 0:W - shift_pixel]
    out[:, g:2 * g, :, 0:W - shift_pixel] = input[:, g:2 * g, :, shift_pixel:W]
    out[:, 2 * g:3 * g, shift_pixel:H, :] = input[:, 2 * g:3 * g, 0:H - shift_pixel, :]
    out[:, 3 * g:4 * g, 0:H - shift_pixel, :] = input[:, 3 * g:4 * g, shift_pixel:H, :]
    out[:, 4 * g:, ...] = input[:, 4 * g:, ...]
    return out.flatten(2).transpose(1, 2)


# ---------------------------------------------------------------------------
# VRWKV_SpatialMix (1:1 with the official rwkv_unet.py)
# ---------------------------------------------------------------------------

class VRWKV_SpatialMix(nn.Module):
    def __init__(self, n_embd, channel_gamma=1 / 4, shift_pixel=1):
        super().__init__()
        self.n_embd = n_embd
        self.shift_pixel = shift_pixel
        self.channel_gamma = channel_gamma if shift_pixel > 0 else None
        self._init_weights()

        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.key_norm = nn.LayerNorm(n_embd)
        self.output = nn.Linear(n_embd, n_embd, bias=False)

        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

    def _init_weights(self):
        self.spatial_decay = nn.Parameter(torch.zeros(self.n_embd))
        self.spatial_first = nn.Parameter(torch.zeros(self.n_embd))
        self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
        self.spatial_mix_v = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
        self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)

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
        x = _RUN_WKV(B, T, C, self.spatial_decay / T,
                      self.spatial_first / T, k, v)
        x = self.key_norm(x)
        x = sr * x
        return self.output(x)


# ---------------------------------------------------------------------------
# VRWKV_ChannelMix (used by CCMix)
# ---------------------------------------------------------------------------

class VRWKV_ChannelMix(nn.Module):
    def __init__(self, n_embd, channel_gamma=1 / 4, shift_pixel=1,
                 hidden_rate=2):
        super().__init__()
        self.channel_gamma = channel_gamma if shift_pixel > 0 else None
        self.shift_pixel = shift_pixel
        self.mix_k = nn.Parameter(torch.ones(1, 1, n_embd) * 0.5)
        self.mix_r = nn.Parameter(torch.ones(1, 1, n_embd) * 0.5)
        hidden = hidden_rate * n_embd
        self.key = nn.Linear(n_embd, hidden, bias=False)
        self.key_norm = nn.LayerNorm(hidden)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden, n_embd, bias=False)

    def forward(self, x, patch_resolution=None):
        xx = x
        xk = x * self.mix_k + xx * (1 - self.mix_k)
        xr = x * self.mix_r + xx * (1 - self.mix_r)
        k = torch.square(torch.relu(self.key(xk)))
        k = self.key_norm(k)
        return torch.sigmoid(self.receptance(xr)) * self.value(k)


# ---------------------------------------------------------------------------
# GLSP (Global-Local Spatial Perception, 1:1 with official)
# ---------------------------------------------------------------------------

class GLSP(nn.Module):
    def __init__(self, dim_in, dim_out, norm_in=True, has_skip=True,
                 exp_ratio=1.0, norm_layer="bn_2d", act_layer="relu",
                 dw_ks=3, stride=1, dilation=1, se_ratio=0.0, attn_s=True,
                 drop_path=0.0, drop=0.0, img_size=224, channel_gamma=1 / 4,
                 shift_pixel=1):
        super().__init__()
        self.norm = _get_norm(norm_layer)(dim_in) if norm_in else nn.Identity()
        dim_mid = int(dim_in * exp_ratio)
        self.ln1 = nn.LayerNorm(dim_mid)
        self.conv = ConvNormAct(dim_in, dim_mid, 1)
        self.has_skip = (dim_in == dim_out and stride == 1) and has_skip
        self.attn_s = attn_s
        self.att = VRWKV_SpatialMix(dim_mid, channel_gamma, shift_pixel) if attn_s else None
        self.se = SE(dim_mid, rd_ratio=se_ratio) if se_ratio > 0.0 else nn.Identity()
        self.proj_drop = nn.Dropout(drop) if drop > 0 else nn.Identity()
        self.proj = ConvNormAct(dim_mid, dim_out, 1, norm_layer="none", act_layer="none")
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.conv_local = ConvNormAct(dim_mid, dim_mid, dw_ks, stride, dilation,
                                      dim_mid, norm_layer="bn_2d", act_layer="silu")

    def forward(self, x):
        shortcut = x
        x = self.conv(self.norm(x))
        if self.attn_s:
            B, hidden, H, W = x.size()
            pr = (H, W)
            seq = x.view(B, hidden, -1).permute(0, 2, 1)
            seq = seq + self.drop_path(self.ln1(self.att(seq, pr)))
            x = seq.permute(0, 2, 1).view(B, hidden, H, W)
        x = x + self.se(self.conv_local(x)) if self.has_skip else self.se(self.conv_local(x))
        x = self.proj(self.proj_drop(x))
        return (shortcut + self.drop_path(x)) if self.has_skip else x


# ---------------------------------------------------------------------------
# UpBlock (decoder, 1:1 with official — NO SKAttention)
# ---------------------------------------------------------------------------

class UpBlock(nn.Module):
    def __init__(self, dim_in, dim_out, norm_in=False, has_skip=False,
                 exp_ratio=1.0, norm_layer="bn_2d", dw_ks=9, stride=1,
                 dilation=1, se_ratio=0.0, drop_path=0.0, drop=0.0):
        super().__init__()
        self.has_skip = has_skip
        self.norm = _get_norm(norm_layer)(dim_in) if norm_in else nn.Identity()
        dim_mid = int(dim_in * exp_ratio)
        self.ln1 = nn.LayerNorm(dim_mid)
        self.conv = ConvNormAct(dim_in, dim_mid, 1)
        self.se = SE(dim_mid, rd_ratio=se_ratio) if se_ratio > 0.0 else nn.Identity()
        self.proj_drop = nn.Dropout(drop) if drop > 0 else nn.Identity()
        self.proj = ConvNormAct(dim_mid, dim_out, 1, norm_layer="bn_2d", act_layer="relu")
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.conv_local = ConvNormAct(dim_mid, dim_mid, dw_ks, stride, dilation,
                                      dim_mid, norm_layer="bn_2d", act_layer="silu")
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")

    def forward(self, x):
        x = self.conv(self.norm(x))
        if self.has_skip:
            x = x + self.se(self.conv_local(x))
        else:
            x = self.se(self.conv_local(x))
        x = self.proj(self.proj_drop(x))
        return self.upsample(x)


# ---------------------------------------------------------------------------
# CCMix (cross-channel mixer, 1:1 with official ccm/ccm.py)
# ---------------------------------------------------------------------------

class CCMix(nn.Module):
    def __init__(self, in_dims, target_dim, target_size):
        super().__init__()
        self.in_dims = list(in_dims)
        self.target_dim = target_dim
        # `target_size` is kept only for backwards compat / introspection;
        # actual interpolation targets are derived per-forward from the
        # runtime skip tensors so the module works for any input resolution.
        self.target_size = target_size
        self.projections = nn.ModuleList(
            [nn.Conv2d(c, target_dim, 1) for c in self.in_dims])
        self.ln1 = nn.LayerNorm(target_dim * 3)
        self.drop_path = DropPath(0.05)
        self.channel = VRWKV_ChannelMix(
            target_dim * 3, channel_gamma=1 / 4, shift_pixel=1, hidden_rate=2)
        self.final_projections = nn.ModuleList(
            [nn.Conv2d(target_dim, c, 1) for c in self.in_dims])

    def forward(self, features):
        # Capture each skip's original H/W at runtime so chunks can be
        # restored to the exact shape the decoder expects to concatenate with.
        original_sizes = [tuple(feat.shape[-2:]) for feat in features]
        # Use the largest (last / shallowest) skip as the common interpolation
        # target — matches the original behaviour where target_size mirrored
        # the shallowest encoder feature (img_size // 2 for the default stem).
        target_hw = original_sizes[-1]

        upsampled = []
        for i, feat in enumerate(features):
            feat = F.interpolate(feat, size=target_hw, mode="bilinear",
                                 align_corners=False)
            upsampled.append(self.projections[i](feat))

        cat = torch.cat(upsampled, dim=1)
        B, C, H, W = cat.shape
        seq = cat.view(B, C, -1).permute(0, 2, 1).contiguous()
        attn = seq + self.drop_path(self.ln1(
            self.channel(seq, (H, W))))

        B2, N, hidden = attn.shape
        attn = attn.permute(0, 2, 1).contiguous().view(B2, hidden, H, W)

        chunks = torch.split(attn, self.target_dim, dim=1)
        outputs = []
        for i, chunk in enumerate(chunks):
            chunk = self.final_projections[i](chunk)
            chunk = F.interpolate(chunk, size=original_sizes[i],
                                  mode="bilinear", align_corners=False)
            outputs.append(chunk)
        return outputs


# ---------------------------------------------------------------------------
# Encoder (T / S / B presets, 1:1 with official RWKV_UNet_encoder)
# ---------------------------------------------------------------------------

_PRESETS = {
    "t": dict(depths=[2, 2, 4, 2], stem_dim=24,
              embed_dims=[32, 48, 96, 160], exp_ratios=[2., 2.5, 3., 3.5]),
    "s": dict(depths=[3, 3, 6, 3], stem_dim=24,
              embed_dims=[32, 64, 128, 192], exp_ratios=[2., 2.5, 3., 4.]),
    "b": dict(depths=[3, 3, 6, 3], stem_dim=24,
              embed_dims=[48, 72, 144, 240], exp_ratios=[2., 2.5, 4., 4.]),
}

_SHARED = dict(
    norm_layers=["bn_2d", "bn_2d", "ln_2d", "ln_2d"],
    act_layers=["silu", "silu", "gelu", "gelu"],
    dw_kss=[5, 5, 5, 5], attn_ss=[False, False, True, True],
)


class _Encoder(nn.Module):
    def __init__(self, variant="b", img_size=224, drop_path=0.05):
        super().__init__()
        p = dict(_PRESETS[variant])
        s = dict(_SHARED)
        depths, stem_dim = p["depths"], p["stem_dim"]
        embed_dims, exp_ratios = p["embed_dims"], p["exp_ratios"]
        norm_layers, act_layers = s["norm_layers"], s["act_layers"]
        dw_kss, attn_ss = s["dw_kss"], s["attn_ss"]

        self.embed_dims = embed_dims
        dprs = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]

        # Stage 0: stem GLSP (no spatial attn, se_ratio=1)
        self.stage0 = nn.ModuleList([GLSP(
            3, stem_dim, norm_in=False, has_skip=False, exp_ratio=1,
            norm_layer=norm_layers[0], act_layer=act_layers[0], dw_ks=dw_kss[0],
            stride=1, se_ratio=1.0, attn_s=False)])

        emb_pre = stem_dim
        for i in range(len(depths)):
            layers = nn.ModuleList()
            dpr = dprs[sum(depths[:i]):sum(depths[:i + 1])]
            for j in range(depths[i]):
                if j == 0:
                    stride, has_skip, attn_s = 2, False, False
                    exp_r = exp_ratios[i] * 2
                else:
                    stride, has_skip, attn_s = 1, True, attn_ss[i]
                    exp_r = exp_ratios[i]
                layers.append(GLSP(
                    emb_pre, embed_dims[i], norm_in=True, has_skip=has_skip,
                    exp_ratio=exp_r, norm_layer=norm_layers[i],
                    act_layer=act_layers[i], dw_ks=dw_kss[i], stride=stride,
                    se_ratio=0.0, attn_s=attn_s, drop_path=dpr[j]))
                emb_pre = embed_dims[i]
            self.__setattr__(f"stage{i + 1}", layers)

        self.norm = _get_norm(norm_layers[-1])(embed_dims[-1])
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d,
                            nn.BatchNorm2d, nn.BatchNorm3d)):
            if m.bias is not None:
                nn.init.zeros_(m.bias)
            if hasattr(m, "weight") and m.weight is not None:
                nn.init.ones_(m.weight)

    def forward(self, x):
        for blk in self.stage0:
            x = blk(x)
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
        return x, enc3, enc2, enc1


# ---------------------------------------------------------------------------
# RWKV_UNet (full model, 1:1 with official)
# ---------------------------------------------------------------------------

class _RWKVUNet(nn.Module):
    def __init__(self, variant="b", in_channels=1, num_classes=1,
                 img_size=224):
        super().__init__()
        self.in_channels = in_channels
        self.encoder = _Encoder(variant=variant, img_size=img_size)
        ed = self.encoder.embed_dims  # [48, 72, 144, 240] for B

        self.ccm = CCMix([ed[2], ed[1], ed[0]], ed[0], img_size // 2)

        self.decoder1 = UpBlock(ed[3], ed[2], dw_ks=9)
        self.decoder2 = UpBlock(ed[2] * 2, ed[1], dw_ks=9)
        self.decoder3 = UpBlock(ed[1] * 2, ed[0], dw_ks=9)
        self.decoder4 = UpBlock(ed[0] * 2, 24, dw_ks=9)
        self.final_conv = nn.Conv2d(24, num_classes, 1)

    def forward(self, x):
        # Project to 3 channels
        if x.shape[1] == 1:
            x3 = x.repeat(1, 3, 1, 1)
        elif x.shape[1] == 3:
            x3 = x
        else:
            x3 = x[:, :3].contiguous() if x.shape[1] >= 3 else F.pad(
                x, (0, 0, 0, 0, 0, 3 - x.shape[1]))

        bottleneck, enc3, enc2, enc1 = self.encoder(x3)

        # CCMix: [deep, mid, shallow] -> fused
        enc3_m, enc2_m, enc1_m = self.ccm([enc3, enc2, enc1])

        dec3 = self.decoder1(bottleneck)
        dec2 = self.decoder2(torch.cat([dec3, enc3_m], dim=1))
        dec1 = self.decoder3(torch.cat([dec2, enc2_m], dim=1))
        dec0 = self.decoder4(torch.cat([dec1, enc1_m], dim=1))

        return self.final_conv(dec0)


# ---------------------------------------------------------------------------
# Public wrapper with standard interface
# ---------------------------------------------------------------------------

class RWKVUNet(nn.Module):
    """RWKV-UNet (arxiv 2025) with standard segmentation interface.

    Args:
        in_channels: Input channels (1 or 3 typical).
        num_classes: Output segmentation classes.
        img_size: Input spatial size.
        variant: Encoder variant ('t', 's', or 'b').
    """

    def __init__(self, in_channels=3, num_classes=2, img_size=224,
                 variant="b", **kwargs):
        super().__init__()
        self.model = _RWKVUNet(
            variant=variant, in_channels=in_channels,
            num_classes=num_classes, img_size=img_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H_in, W_in = x.shape[2:]
        out = self.model(x)
        if out.shape[2:] != (H_in, W_in):
            out = F.interpolate(out, size=(H_in, W_in), mode="bilinear",
                                align_corners=False)
        return out
