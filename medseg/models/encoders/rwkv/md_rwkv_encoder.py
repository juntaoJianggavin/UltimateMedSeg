"""MD-RWKV-UNet Encoder (standalone).

Extracted from :mod:`medseg.models.networks.rwkv.md_rwkv_unet` so the encoder can be
re-used with arbitrary decoders / bottlenecks. Mirrors the official MD-RWKV
encoder body (DeformableShift + VRWKV_SpatialMix + SKAttention inside the
``iR_RWKV`` blocks) for the T / S / B presets, while exposing the standard
:class:`medseg.models.encoders` interface:

    forward(x) -> List[Tensor]   # deepest LAST
    self.out_channels: List[int]

WKV is dispatched via :mod:`medseg.kernels.wkv` (CUDA on GPU, PyTorch on CPU).
"""
# Source: https://github.com/fzy-eng/MD-RWKV-UNet

import math
from functools import partial
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.registry import ENCODER_REGISTRY
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

class _DropPath(nn.Module):
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


class _LayerNorm2d(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()


def _get_norm(name):
    return {"none": nn.Identity, "bn_2d": partial(nn.BatchNorm2d, eps=1e-6),
            "ln_2d": partial(_LayerNorm2d, eps=1e-6),
            "ln_1d": partial(nn.LayerNorm, eps=1e-6)}[name]


def _make_act(name, inplace=True):
    t = {"none": nn.Identity, "relu": nn.ReLU, "silu": nn.SiLU,
         "gelu": nn.GELU, "sigmoid": nn.Sigmoid}
    cls = t[name]
    return cls(inplace=inplace) if cls in (nn.ReLU, nn.SiLU) else cls()


class _ConvNormAct(nn.Module):
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

class _SE(nn.Module):
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
# DeformableShift (learnable spatial shift via grid_sample)
# ---------------------------------------------------------------------------

class _DeformableShift(nn.Module):
    def __init__(self, channels, channel_gamma=1 / 4, num_parts=4):
        super().__init__()
        self.channel_gamma = channel_gamma
        self.num_parts = num_parts
        self.offset_conv = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 3, padding=1, groups=4),
            nn.GELU(),
            nn.Conv2d(channels // 4, num_parts * 2, 3, padding=1),
        )
        nn.init.constant_(self.offset_conv[-1].weight, 0)
        nn.init.constant_(self.offset_conv[-1].bias, 0)

    def forward(self, x):
        B, C, H, W = x.shape
        offsets = torch.tanh(self.offset_conv(x)).view(B, self.num_parts, 2, H, W)

        ps = int(C * self.channel_gamma)
        parts = [x[:, i * ps:(i + 1) * ps] for i in range(self.num_parts)]
        rest = x[:, self.num_parts * ps:]

        gy, gx = torch.meshgrid(
            torch.arange(H, device=x.device, dtype=torch.float32),
            torch.arange(W, device=x.device, dtype=torch.float32), indexing="ij")
        grid = torch.stack((gx, gy), -1).unsqueeze(0).expand(B, -1, -1, -1)

        warped = []
        for i in range(self.num_parts):
            g = grid.clone()
            g[..., 0] = 2.0 * (g[..., 0] + offsets[:, i, 0]) / max(W - 1, 1) - 1.0
            g[..., 1] = 2.0 * (g[..., 1] + offsets[:, i, 1]) / max(H - 1, 1) - 1.0
            warped.append(F.grid_sample(parts[i], g, padding_mode="zeros",
                                        align_corners=True))
        return torch.cat(warped + [rest], dim=1)


# ---------------------------------------------------------------------------
# VRWKV_SpatialMix (1:1 with official MD-RWKV-UNet)
# ---------------------------------------------------------------------------

class _VRWKV_SpatialMix(nn.Module):
    def __init__(self, n_embd, channel_gamma=1 / 4, num_deform_groups=4):
        super().__init__()
        self.n_embd = n_embd
        self.deform_shift = _DeformableShift(
            n_embd, channel_gamma=channel_gamma, num_parts=num_deform_groups)
        self.spatial_decay = nn.Parameter(torch.zeros(n_embd))
        self.spatial_first = nn.Parameter(torch.zeros(n_embd))
        self.mix_k = nn.Parameter(torch.ones(1, 1, n_embd) * 0.5)
        self.mix_v = nn.Parameter(torch.ones(1, 1, n_embd) * 0.5)
        self.mix_r = nn.Parameter(torch.ones(1, 1, n_embd) * 0.5)

        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.key_norm = nn.LayerNorm(n_embd)
        self.output = nn.Linear(n_embd, n_embd, bias=False)

        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

    def _def_shift(self, x, patch_resolution):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, *patch_resolution)
        x = self.deform_shift(x)
        return x.flatten(2).transpose(1, 2)

    def forward(self, x, patch_resolution=None):
        B, T, C = x.size()
        xx = self._def_shift(x, patch_resolution)
        xk = x * self.mix_k + xx * (1 - self.mix_k)
        xv = x * self.mix_v + xx * (1 - self.mix_v)
        xr = x * self.mix_r + xx * (1 - self.mix_r)
        k = self.key(xk)
        v = self.value(xv)
        r = torch.sigmoid(self.receptance(xr))
        x = _RUN_WKV(B, T, C, self.spatial_decay / T,
                     self.spatial_first / T, k, v)
        x = self.key_norm(x)
        return self.output(r * x)


# ---------------------------------------------------------------------------
# SKAttention (Selective Kernel Attention)
# ---------------------------------------------------------------------------

class _SKAttention(nn.Module):
    def __init__(self, channels, reduction=16, num_paths=2):
        super().__init__()
        self.num_paths = num_paths
        mid = max(channels // reduction, 32)
        self.conv_paths = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3 + 2 * i, 1, 1 + i,
                          groups=channels, bias=False),
                nn.BatchNorm2d(channels),
                nn.Conv2d(channels, channels, 1, bias=False),
                nn.BatchNorm2d(channels),
            ) for i in range(num_paths)
        ])
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, 1), nn.ReLU(inplace=True),
            nn.Conv2d(mid, num_paths, 1),
        )
        self.softmax = nn.Softmax(dim=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        feats = torch.stack([p(x) for p in self.conv_paths], dim=1)  # (B, P, C, H, W)
        attn = self.softmax(self.fc(x))  # (B, P, 1, 1)
        return (attn.unsqueeze(2) * feats).sum(dim=1)


# ---------------------------------------------------------------------------
# iR_RWKV (main encoder block: parallel SK + VRWKV branches)
# ---------------------------------------------------------------------------

class _iR_RWKV(nn.Module):
    def __init__(self, dim_in, dim_out, norm_in=True, has_skip=True,
                 exp_ratio=1.0, norm_layer="bn_2d", act_layer="relu",
                 dw_ks=3, stride=1, dilation=1, se_ratio=0.0, attn_s=True,
                 drop_path=0.0, drop=0.0, img_size=224, channel_gamma=1 / 4,
                 shift_pixel=1, use_sk=True, sk_reduction=16):
        super().__init__()
        self.norm = _get_norm(norm_layer)(dim_in) if norm_in else nn.Identity()
        dim_mid = int(dim_in * exp_ratio)
        self.ln1 = nn.LayerNorm(dim_mid)
        self.conv = _ConvNormAct(dim_in, dim_mid, 1)
        self.has_skip = (dim_in == dim_out and stride == 1) and has_skip
        self.use_sk = use_sk
        self.attn_s = attn_s

        self.sk = _SKAttention(dim_mid, reduction=sk_reduction) if use_sk else None
        self.att = _VRWKV_SpatialMix(dim_mid, channel_gamma, shift_pixel) if attn_s else None

        if se_ratio > 0.0:
            self.se = _SE(dim_mid, rd_ratio=se_ratio)
        else:
            self.se = nn.Identity()
        self.proj_drop = nn.Dropout(drop) if drop > 0 else nn.Identity()
        self.proj = _ConvNormAct(dim_mid, dim_out, 1, norm_layer="none",
                                 act_layer="none")
        self.drop_path = _DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.conv_local = _ConvNormAct(dim_mid, dim_mid, dw_ks, stride, dilation,
                                       dim_mid, norm_layer="bn_2d", act_layer="silu")
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.ones_(m.weight)

    def forward(self, x):
        shortcut = x
        x = self.conv(self.norm(x))

        if self.attn_s or self.use_sk:
            B, hidden, H, W = x.size()
            pr = (H, W)
            att_feat = x
            if self.att is not None:
                seq = x.view(B, hidden, -1).permute(0, 2, 1)
                seq = seq + self.drop_path(self.ln1(self.att(seq, pr)))
                att_feat = seq.permute(0, 2, 1).view(B, hidden, H, W)
            sk_feat = self.sk(x) if self.sk is not None else x
            if self.att is not None and self.sk is not None:
                x = 0.5 * att_feat + 0.5 * sk_feat
            elif self.att is not None:
                x = att_feat
            else:
                x = sk_feat

        x = x + self.se(self.conv_local(x)) if self.has_skip else self.se(self.conv_local(x))
        x = self.proj(self.proj_drop(x))
        return (shortcut + self.drop_path(x)) if self.has_skip else x


# ---------------------------------------------------------------------------
# MD-RWKV-UNet Encoder presets
# ---------------------------------------------------------------------------

_VARIANT_PRESETS = {
    "t": dict(depths=[2, 2, 4, 2], stem_dim=24,
              embed_dims=[32, 48, 96, 160], exp_ratios=[2., 2.5, 3., 3.5]),
    "s": dict(depths=[3, 3, 6, 3], stem_dim=24,
              embed_dims=[32, 64, 128, 192], exp_ratios=[2., 2.5, 3., 4.]),
    "b": dict(depths=[3, 3, 6, 3], stem_dim=24,
              embed_dims=[48, 72, 144, 240], exp_ratios=[2., 2.5, 4., 4.]),
}

_VARIANT_SHARED = dict(
    norm_layers=["bn_2d", "bn_2d", "ln_2d", "ln_2d"],
    act_layers=["silu", "silu", "gelu", "gelu"],
    dw_kss=[5, 5, 5, 5], attn_ss=[False, False, True, True],
)


def _load_with_ssl_fallback(load_fn, *args, **kwargs):
    import ssl, warnings
    try:
        return load_fn(*args, **kwargs)
    except Exception as e1:
        prev = ssl._create_default_https_context
        try:
            ssl._create_default_https_context = ssl._create_unverified_context
            return load_fn(*args, **kwargs)
        except Exception as e2:
            warnings.warn(f'Pretrained download failed ({e2}); using random init.')
            return load_fn(*args, **{**kwargs, 'pretrained': False})
        finally:
            ssl._create_default_https_context = prev


@ENCODER_REGISTRY.register("md_rwkv")
class MDRWKVUNetEncoder(nn.Module):
    """MD-RWKV-UNet encoder (T / S / B variants).

    Extracted from the official MD-RWKV-UNet (DeformableShift + SKAttention +
    iR_RWKV blocks). Returns 4 multi-scale skip features with the deepest
    feature LAST (framework convention):

        ``[enc1, enc2, enc3, bottleneck]``

    The stem-stage output (full-resolution, ``stem_dim`` channels) is not
    exposed because the official decoder does not consume it.

    Args:
        in_channels: input channels (if not 3, a 1x1 projection is prepended).
        img_size: nominal input spatial size (used only for parity-style state).
        pretrained: kept for API symmetry; the official MD-RWKV repo ships no
            pretrained backbone, so this is a no-op.
        variant: ``'t'``, ``'s'``, or ``'b'`` (default ``'b'``).
        drop_path: maximum stochastic depth rate.
        sk_use: whether to use SKAttention inside iR_RWKV blocks.
        sk_reduction: SK reduction ratio.
    """

    def __init__(self, in_channels: int = 3, img_size: int = 224,
                 pretrained: bool = False, variant: str = "b",
                 drop_path: float = 0.05, sk_use: bool = True,
                 sk_reduction: int = 16, **kwargs):
        super().__init__()
        variant = variant.lower()
        if variant not in _VARIANT_PRESETS:
            raise ValueError(
                f"Unknown MD-RWKV-UNet variant '{variant}'. "
                f"Choose from 't', 's', 'b'.")

        p = dict(_VARIANT_PRESETS[variant])
        s = dict(_VARIANT_SHARED)
        depths = p["depths"]
        stem_dim = p["stem_dim"]
        embed_dims = p["embed_dims"]
        exp_ratios = p["exp_ratios"]
        norm_layers = s["norm_layers"]
        act_layers = s["act_layers"]
        dw_kss = s["dw_kss"]
        attn_ss = s["attn_ss"]

        self.variant = variant
        self.in_channels = in_channels
        self.img_size = img_size
        self.embed_dims = embed_dims
        self.stem_dim = stem_dim
        self.depths = depths

        # Optional projection to 3 channels (encoder body is wired for 3-ch input).
        if in_channels != 3:
            self.input_proj = nn.Conv2d(in_channels, 3, 1, bias=False)
        else:
            self.input_proj = nn.Identity()

        dprs = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]

        # Stage 0: stem (no spatial attn, no SK, se_ratio=1)
        self.stage0 = nn.ModuleList([_iR_RWKV(
            3, stem_dim, norm_in=False, has_skip=False, exp_ratio=1,
            norm_layer=norm_layers[0], act_layer=act_layers[0], dw_ks=dw_kss[0],
            stride=1, se_ratio=1.0, attn_s=False, use_sk=False)])

        emb_pre = stem_dim
        cur_sz = img_size
        for i in range(len(depths)):
            layers = nn.ModuleList()
            dpr = dprs[sum(depths[:i]):sum(depths[:i + 1])]
            for j in range(depths[i]):
                if j == 0:
                    stride, has_skip, attn_s = 2, False, False
                    exp_r = exp_ratios[i] * 2
                    cur_sz = cur_sz // 2
                    use_sk_j = False
                else:
                    stride, has_skip, attn_s = 1, True, attn_ss[i]
                    exp_r = exp_ratios[i]
                    use_sk_j = sk_use
                layers.append(_iR_RWKV(
                    emb_pre, embed_dims[i], norm_in=True, has_skip=has_skip,
                    exp_ratio=exp_r, norm_layer=norm_layers[i],
                    act_layer=act_layers[i], dw_ks=dw_kss[i], stride=stride,
                    se_ratio=0.0, attn_s=attn_s, drop_path=dpr[j],
                    img_size=cur_sz, use_sk=use_sk_j,
                    sk_reduction=sk_reduction))
                emb_pre = embed_dims[i]
            self.__setattr__(f"stage{i + 1}", layers)

        self.norm = _get_norm(norm_layers[-1])(embed_dims[-1])

        # Channels exposed to the model builder: deepest LAST.
        self.out_channels: List[int] = list(embed_dims)

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

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # Auto-repeat 1-channel input when configured for 3 channels.
        if x.shape[1] == 1 and self.in_channels == 3:
            x = x.repeat(1, 3, 1, 1)

        x = self.input_proj(x)

        for blk in self.stage0:
            x = blk(x)
        # enc0 (stem output) is intentionally not exposed.

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
        # x is the deepest feature consumed by the bottleneck.
        return [enc1, enc2, enc3, x]
