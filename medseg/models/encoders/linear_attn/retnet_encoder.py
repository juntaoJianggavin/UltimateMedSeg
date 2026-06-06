"""RetNet encoder.

Standalone hierarchical retention encoder extracted from
``medseg/networks/other/retnet_unet.py``. Implements the multi-scale
retention backbone from Sun et al., "Retentive Network: A Successor to
Transformer for Large Language Models" (Microsoft, 2023).

Architecture
------------
* Stride-4 patch-embed stem (4x4 conv stride 4 + LayerNorm)
* 4 hierarchical stages with windowed parallel-form retention blocks
  (per-head learnable decay scalar + head-wise GroupNorm)
* dims      = [64, 128, 256, 512]
* depths    = [2,   2,   4,   2]
* num_heads = [2,   4,   8,  16]
* Strides   = [4,   8,  16,  32] (one stem + three 2x downsamples)

forward(x) returns a list of 4 BCHW feature maps, deepest LAST,
matching the standard encoder contract used by the basic / timm encoders.
"""
# Source: UNCHECKED — please verify

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.registry import ENCODER_REGISTRY


# ---------------------------------------------------------------------------
# Windowed multi-scale retention (parallel form)
# ---------------------------------------------------------------------------


class _Retention(nn.Module):
    """Multi-scale retention with per-head learnable decay scalar."""

    def __init__(self, dim, num_heads, window_size=8):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        h_idx = torch.arange(num_heads, dtype=torch.float32)
        gammas = 1.0 - torch.pow(2.0, -5.0 - h_idx)
        gammas = gammas.clamp(1e-4, 1 - 1e-4)
        self.gamma_logit = nn.Parameter(torch.log(gammas / (1.0 - gammas)))

        self.gn = nn.GroupNorm(num_heads, dim, eps=1e-5)
        self.proj = nn.Linear(dim, dim)

    @staticmethod
    def _window_partition(x, ws):
        B, H, W, C = x.shape
        x = x.view(B, H // ws, ws, W // ws, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        return x.view(-1, ws * ws, C)

    @staticmethod
    def _window_reverse(w, ws, B, H, W):
        C = w.shape[-1]
        x = w.view(B, H // ws, W // ws, ws, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        return x.view(B, H, W, C)

    def _build_decay(self, S, device, dtype):
        gamma = torch.sigmoid(self.gamma_logit).to(dtype=dtype, device=device)
        idx = torch.arange(S, device=device, dtype=dtype)
        rel = idx.unsqueeze(1) - idx.unsqueeze(0)
        causal = (rel >= 0).to(dtype)
        rel = rel.clamp(min=0)
        log_gamma = torch.log(gamma).view(-1, 1, 1)
        D = torch.exp(rel.unsqueeze(0) * log_gamma) * causal.unsqueeze(0)
        return D.unsqueeze(0)  # (1, NH, S, S)

    def forward(self, x, H, W):
        B, N, C = x.shape
        assert N == H * W, f"Token count {N} != H*W {H*W}"
        ws = self.window_size

        x = x.view(B, H, W, C)
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = H + pad_h, W + pad_w

        x_w = self._window_partition(x, ws)  # (BW, S, C)
        BW, S, _ = x_w.shape

        qkv = self.qkv(x_w).view(BW, S, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        D = self._build_decay(S, x.device, x.dtype)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores * D
        out = torch.matmul(scores, v)

        out = out.transpose(1, 2).contiguous().view(BW, S, C)
        out = self.gn(out.transpose(1, 2)).transpose(1, 2)
        out = self.proj(out)

        out = self._window_reverse(out, ws, B, Hp, Wp)
        if pad_h or pad_w:
            out = out[:, :H, :W, :].contiguous()
        return out.view(B, H * W, C)


# ---------------------------------------------------------------------------
# MLP + retention block
# ---------------------------------------------------------------------------


class _MLP(nn.Module):
    def __init__(self, dim, ratio=4):
        super().__init__()
        hidden = int(dim * ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class _RetBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, window_size=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.ret = _Retention(dim, num_heads, window_size=window_size)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _MLP(dim, ratio=mlp_ratio)

    def forward(self, x, H, W):
        x = x + self.ret(self.norm1(x), H, W)
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Patch embedding + 2x down-sampling
# ---------------------------------------------------------------------------


class _PatchEmbed(nn.Module):
    """Stride-4 stem: 4x4 conv (stride 4) + LayerNorm."""

    def __init__(self, in_channels, dim):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=4, stride=4)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class _PatchMerging(nn.Module):
    """2x downsample via 2x2 stride-2 conv + LayerNorm."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.proj(x)
        H2, W2 = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H2, W2


# ---------------------------------------------------------------------------
# RetNet encoder
# ---------------------------------------------------------------------------


@ENCODER_REGISTRY.register("retnet")
class RetNetEncoder(nn.Module):
    """Hierarchical RetNet (multi-scale retention) encoder.

    Parameters
    ----------
    in_channels : int
        Number of input image channels. If != 3, a 1x1 conv stem maps to 3
        channels before the retention backbone.
    img_size : int
        Reference input spatial size; the encoder itself derives shapes at
        runtime, so other sizes work too (224 / 256 / 512 all supported).
    pretrained : bool
        Accepted for API compatibility; no public pretrained RetNet vision
        backbone is bundled, so this is a no-op.
    """

    def __init__(self, in_channels: int = 3, img_size: int = 224,
                 pretrained: bool = False, **kwargs):
        super().__init__()

        dims = list(kwargs.get("dims", [64, 128, 256, 512]))
        depths = list(kwargs.get("depths", [2, 2, 4, 2]))
        heads = list(kwargs.get("heads", [2, 4, 8, 16]))
        window_size = int(kwargs.get("window_size", 8))
        mlp_ratio = int(kwargs.get("mlp_ratio", 4))

        assert len(dims) == len(depths) == len(heads), \
            "dims / depths / heads must have matching length"

        self.dims = dims
        self.depths = depths
        self.heads = heads
        self.img_size = img_size
        self.pretrained = pretrained
        self.out_channels: List[int] = list(dims)

        # optional channel-adapter for in_channels != 3
        if in_channels != 3:
            self.input_proj = nn.Conv2d(in_channels, 3, kernel_size=1)
            stem_in = 3
        else:
            self.input_proj = None
            stem_in = in_channels

        # stem
        self.patch_embed = _PatchEmbed(stem_in, dims[0])

        # encoder stages + inter-stage downsamples
        self.enc_stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i in range(len(dims)):
            stage = nn.ModuleList([
                _RetBlock(dims[i], heads[i], mlp_ratio=mlp_ratio,
                          window_size=window_size)
                for _ in range(depths[i])
            ])
            self.enc_stages.append(stage)
            if i < len(dims) - 1:
                self.downsamples.append(_PatchMerging(dims[i], dims[i + 1]))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        if self.input_proj is not None:
            x = self.input_proj(x)

        B = x.shape[0]
        x, h, w = self.patch_embed(x)

        features: List[torch.Tensor] = []
        for i, stage in enumerate(self.enc_stages):
            for blk in stage:
                x = blk(x, h, w)
            # collect this stage's feature in BCHW form
            C = x.shape[-1]
            feat = x.transpose(1, 2).contiguous().view(B, C, h, w)
            features.append(feat)
            if i < len(self.enc_stages) - 1:
                x, h, w = self.downsamples[i](x, h, w)

        return features
