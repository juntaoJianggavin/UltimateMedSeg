"""MD-RWKV-UNet: Multi-Directional RWKV UNet for Medical Image Segmentation.

Faithful self-contained port of:
  https://github.com/fzy-eng/MD-RWKV-UNet  (md_rwkv_unet.py)

Key components (1:1 with the official repo):
  - DeformableShift: learnable spatial shift via grid_sample
  - SKAttention: Selective Kernel attention (multi-path DWConv + softmax)
  - VRWKV_SpatialMix: Vision-RWKV with DeformableShift + WKV attention
  - iR_RWKV: main encoder block (parallel SK + VRWKV branches, fused 50/50)
  - UpBlock: decoder block with SKAttention
  - CCMix: cross-channel mixer fusing [enc3, enc2, enc1] via VRWKV_ChannelMix
  - MD_RWKV_UNet_encoder_B / S / T presets

WKV is dispatched via medseg.kernels.wkv (CUDA on GPU, PyTorch on CPU).
"""
# Source: https://github.com/fzy-eng/MD-RWKV-UNet

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
# DeformableShift (learnable spatial shift via grid_sample)
# ---------------------------------------------------------------------------

class DeformableShift(nn.Module):
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

class VRWKV_SpatialMix(nn.Module):
    def __init__(self, n_embd, channel_gamma=1 / 4, num_deform_groups=4):
        super().__init__()
        self.n_embd = n_embd
        self.deform_shift = DeformableShift(
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
        xx = x  # no q_shift in channel mix for MD-RWKV
        xk = x * self.mix_k + xx * (1 - self.mix_k)
        xr = x * self.mix_r + xx * (1 - self.mix_r)
        k = torch.square(torch.relu(self.key(xk)))
        k = self.key_norm(k)
        return torch.sigmoid(self.receptance(xr)) * self.value(k)


# ---------------------------------------------------------------------------
# SKAttention (Selective Kernel Attention)
# ---------------------------------------------------------------------------

class SKAttention(nn.Module):
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

class iR_RWKV(nn.Module):
    def __init__(self, dim_in, dim_out, norm_in=True, has_skip=True,
                 exp_ratio=1.0, norm_layer="bn_2d", act_layer="relu",
                 dw_ks=3, stride=1, dilation=1, se_ratio=0.0, attn_s=True,
                 drop_path=0.0, drop=0.0, img_size=224, channel_gamma=1 / 4,
                 shift_pixel=1, use_sk=True, sk_reduction=16):
        super().__init__()
        self.norm = _get_norm(norm_layer)(dim_in) if norm_in else nn.Identity()
        dim_mid = int(dim_in * exp_ratio)
        self.ln1 = nn.LayerNorm(dim_mid)
        self.conv = ConvNormAct(dim_in, dim_mid, 1)
        self.has_skip = (dim_in == dim_out and stride == 1) and has_skip
        self.use_sk = use_sk
        self.attn_s = attn_s

        self.sk = SKAttention(dim_mid, reduction=sk_reduction) if use_sk else None
        self.att = VRWKV_SpatialMix(dim_mid, channel_gamma, shift_pixel) if attn_s else None

        if se_ratio > 0.0:
            self.se = SE(dim_mid, rd_ratio=se_ratio)
        else:
            self.se = nn.Identity()
        self.proj_drop = nn.Dropout(drop) if drop > 0 else nn.Identity()
        self.proj = ConvNormAct(dim_mid, dim_out, 1, norm_layer="none",
                                act_layer="none")
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.conv_local = ConvNormAct(dim_mid, dim_mid, dw_ks, stride, dilation,
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
# UpBlock (decoder block with SK attention)
# ---------------------------------------------------------------------------

class UpBlock(nn.Module):
    def __init__(self, dim_in, dim_out, norm_in=False, has_skip=False,
                 exp_ratio=1.0, norm_layer="bn_2d", dw_ks=3, stride=1,
                 dilation=1, se_ratio=0.0, drop_path=0.0, drop=0.0,
                 use_sk=True, sk_reduction=16):
        super().__init__()
        self.has_skip = has_skip
        self.norm = _get_norm(norm_layer)(dim_in) if norm_in else nn.Identity()
        dim_mid = int(dim_in * exp_ratio)
        self.ln1 = nn.LayerNorm(dim_mid)
        self.conv = ConvNormAct(dim_in, dim_mid, 1)
        self.sk = SKAttention(dim_mid, reduction=sk_reduction) if use_sk else None
        if se_ratio > 0.0:
            self.se = SE(dim_mid, rd_ratio=se_ratio)
        else:
            self.se = nn.Identity()
        self.proj_drop = nn.Dropout(drop) if drop > 0 else nn.Identity()
        self.proj = ConvNormAct(dim_mid, dim_out, 1, norm_layer="bn_2d",
                                act_layer="relu")
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.conv_local = ConvNormAct(dim_mid, dim_mid, dw_ks, stride, dilation,
                                      dim_mid, norm_layer="bn_2d", act_layer="silu")
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")

    def forward(self, x):
        x = self.conv(self.norm(x))
        if self.sk is not None:
            x = self.sk(x)
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
        self.target_size = target_size
        self.projections = nn.ModuleList(
            [nn.Conv2d(c, target_dim, 1) for c in self.in_dims])
        self.ln1 = nn.LayerNorm(target_dim * 3)
        self.drop_path = DropPath(0.05)
        self.channel = VRWKV_ChannelMix(
            target_dim * 3, channel_gamma=1 / 4, shift_pixel=1, hidden_rate=2)
        self.final_projections = nn.ModuleList(
            [nn.Conv2d(target_dim, c, 1) for c in self.in_dims])
        self.original_sizes = [target_size // 4, target_size // 2, target_size]

    def forward(self, features):
        # Derive shapes from the actual skip tensors at forward time so the
        # module works for any input resolution regardless of what
        # ``target_size`` was passed at construction time.
        original_sizes = [tuple(f.shape[-2:]) for f in features]
        # The shared mixing grid is the largest skip (last entry == enc1).
        target_h, target_w = original_sizes[-1]

        upsampled = []
        for i, feat in enumerate(features):
            feat = F.interpolate(feat, size=(target_h, target_w),
                                 mode="bilinear", align_corners=False)
            upsampled.append(self.projections[i](feat))

        cat = torch.cat(upsampled, dim=1)
        B, C, H, W = cat.shape
        seq = cat.view(B, C, -1).permute(0, 2, 1).contiguous()
        attn = seq + self.drop_path(self.ln1(self.channel(seq, (H, W))))

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
# MD_RWKV_UNet Encoder (T / S / B presets, 1:1 with official)
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
    """MD_RWKV_UNet_encoder (B/S/T variant)."""

    def __init__(self, variant="b", img_size=224, drop_path=0.05,
                 sk_use=True, sk_reduction=16):
        super().__init__()
        p = dict(_PRESETS[variant])
        s = dict(_SHARED)
        depths, stem_dim = p["depths"], p["stem_dim"]
        embed_dims, exp_ratios = p["embed_dims"], p["exp_ratios"]
        norm_layers, act_layers = s["norm_layers"], s["act_layers"]
        dw_kss, attn_ss = s["dw_kss"], s["attn_ss"]

        self.embed_dims = embed_dims
        self.stem_dim = stem_dim
        dprs = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]

        # Stage 0: stem (no spatial attn, no SK, se_ratio=1)
        self.stage0 = nn.ModuleList([iR_RWKV(
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
                    use_sk = False
                else:
                    stride, has_skip, attn_s = 1, True, attn_ss[i]
                    exp_r = exp_ratios[i]
                    use_sk = sk_use
                layers.append(iR_RWKV(
                    emb_pre, embed_dims[i], norm_in=True, has_skip=has_skip,
                    exp_ratio=exp_r, norm_layer=norm_layers[i],
                    act_layer=act_layers[i], dw_ks=dw_kss[i], stride=stride,
                    se_ratio=0.0, attn_s=attn_s, drop_path=dpr[j],
                    img_size=cur_sz, use_sk=use_sk,
                    sk_reduction=sk_reduction))
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
        enc0 = x
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
        return x, enc3, enc2, enc1, enc0


# ---------------------------------------------------------------------------
# MD_RWKV_UNet (full segmentation model, 1:1 with official)
# ---------------------------------------------------------------------------

class _MDRWKVUNet(nn.Module):
    """Official MD_RWKV_UNet / RWKV_UNet_S / RWKV_UNet_T."""

    def __init__(self, variant="b", in_channels=1, num_classes=1,
                 img_size=224):
        super().__init__()
        self.in_channels = in_channels
        self.encoder = _Encoder(variant=variant, img_size=img_size)
        ed = self.encoder.embed_dims  # [48, 72, 144, 240] for B
        self.embed_dims = ed

        self.ccm = CCMix([ed[2], ed[1], ed[0]], ed[0], img_size // 2)

        self.decoder1 = UpBlock(ed[3], ed[2], dw_ks=9)
        self.decoder2 = UpBlock(ed[2] * 2, ed[1], dw_ks=9)
        self.decoder3 = UpBlock(ed[1] * 2, ed[0], dw_ks=9)
        self.decoder4 = UpBlock(ed[0] * 2, 24, dw_ks=9)
        self.final_conv = nn.Conv2d(24, num_classes, 1)

    def forward(self, x):
        # Project to 3 channels for encoder
        if x.shape[1] == 1:
            x3 = x.repeat(1, 3, 1, 1)
        elif x.shape[1] == 3:
            x3 = x
        else:
            x3 = x[:, :3].contiguous() if x.shape[1] >= 3 else F.pad(x, (0, 0, 0, 0, 0, 3 - x.shape[1]))

        bottleneck, enc3, enc2, enc1, enc0 = self.encoder(x3)

        # CCMix: [deep, mid, shallow] -> [deep', mid', shallow']
        enc3_m, enc2_m, enc1_m = self.ccm([enc3, enc2, enc1])

        dec3 = self.decoder1(bottleneck)
        dec2 = self.decoder2(torch.cat([dec3, enc3_m], dim=1))
        dec1 = self.decoder3(torch.cat([dec2, enc2_m], dim=1))
        dec0 = self.decoder4(torch.cat([dec1, enc1_m], dim=1))

        return self.final_conv(dec0)


# ---------------------------------------------------------------------------
# Public wrapper with standard interface
# ---------------------------------------------------------------------------

class MDRWKVUNet(nn.Module):
    """MD-RWKV-UNet with standard segmentation interface.

    Args:
        in_channels: Input channels (1 or 3 typical).
        num_classes: Output segmentation classes.
        img_size: Input spatial size.
        variant: Encoder variant ('t', 's', or 'b').
    """

    def __init__(self, in_channels=3, num_classes=2, img_size=224,
                 variant="b", **kwargs):
        super().__init__()
        self.model = _MDRWKVUNet(
            variant=variant, in_channels=in_channels,
            num_classes=num_classes, img_size=img_size)
        self._in_channels = in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H_in, W_in = x.shape[2:]
        out = self.model(x)
        if out.shape[2:] != (H_in, W_in):
            out = F.interpolate(out, size=(H_in, W_in), mode="bilinear",
                                align_corners=False)
        return out
