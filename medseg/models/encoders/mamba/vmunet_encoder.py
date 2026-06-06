"""VM-UNet Encoder: faithful port from https://github.com/JCruan519/VM-UNet

Reference: "VM-UNet: Vision Mamba UNet for Medical Image Segmentation"
Key components: SS2D (4-direction selective scan), VSSBlock, VSSM backbone.

Requires ``mamba_ssm`` — the official VM-UNet source hard-depends on
``pip install mamba_ssm==1.0.1``.
"""
# Source: https://github.com/JCruan519/VM-UNet

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from functools import partial
from einops import rearrange

from medseg.registry import ENCODER_REGISTRY

# ---------- Selective Scan (hard dependency on mamba_ssm) ----------

def _get_selective_scan_fn():
    """Get selective scan function from mamba_ssm (hard dependency).

    The official VM-UNet source (https://github.com/JCruan519/VM-UNet)
    requires ``pip install mamba_ssm==1.0.1``. No pure-PyTorch fallback
    is provided.
    """
    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
        return selective_scan_fn
    except ImportError:
        pass
    try:
        from selective_scan import selective_scan_fn
        return selective_scan_fn
    except ImportError:
        pass
    raise RuntimeError(
        "VM-UNet encoder requires the `mamba_ssm` CUDA package. "
        "Install: pip install mamba-ssm. "
        "The official VM-UNet source hard-depends on mamba_ssm; "
        "no pure-PyTorch fallback is provided."
    )



# Lazy-loaded cache: resolved on first use, not at import time
_selective_scan_fn_cache = None


def _lazy_selective_scan_fn():
    global _selective_scan_fn_cache
    if _selective_scan_fn_cache is None:
        _selective_scan_fn_cache = _get_selective_scan_fn()
    return _selective_scan_fn_cache


# ---------- SS2D: 4-direction Selective Scan 2D ----------

class SS2D(nn.Module):
    """Selective Scan 2D: scans image in 4 directions with Mamba SSM."""

    def __init__(self, d_model, d_state=16, d_conv=3, expand=2, dropout=0.0, dt_rank="auto"):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        d_inner = int(expand * d_model)
        self.d_inner = d_inner

        if dt_rank == "auto":
            dt_rank = math.ceil(d_model / 16)
        self.dt_rank = dt_rank

        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv2d = nn.Conv2d(d_inner, d_inner, d_conv, padding=d_conv//2, groups=d_inner, bias=True)
        self.act = nn.SiLU()

        # SSM parameters for 4 directions
        # x_proj produces [dt_rank*4, d_state*4, d_state*4] interleaved
        self.x_proj_weight = nn.Parameter(torch.empty(4, d_inner, dt_rank + d_state * 2))
        nn.init.normal_(self.x_proj_weight, std=0.02)
        self.dt_projs = nn.ModuleList([
            nn.Linear(dt_rank, d_inner, bias=True) for _ in range(4)
        ])

        # A, D parameters
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))

        self.out_norm = nn.LayerNorm(d_inner)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def _scan_four_directions(self, x):
        """Create 4 scan sequences from 2D feature map.
        x: (B, C, H, W) -> 4 x (B, C, L)
        """
        B, C, H, W = x.shape
        L = H * W
        # Direction 1: left-to-right, top-to-bottom (row-major)
        x1 = x.reshape(B, C, L)
        # Direction 2: right-to-left, bottom-to-top (reverse)
        x2 = x1.flip(dims=[-1])
        # Direction 3: top-to-bottom, left-to-right (column-major)
        x3 = x.permute(0, 1, 3, 2).reshape(B, C, L)
        # Direction 4: bottom-to-top, right-to-left (reverse column-major)
        x4 = x3.flip(dims=[-1])
        return [x1, x2, x3, x4]

    def _unscan_four_directions(self, ys, H, W):
        """Merge 4 direction outputs back to 2D."""
        B, C, L = ys[0].shape
        y1 = ys[0]
        y2 = ys[1].flip(dims=[-1])
        y3 = ys[2].reshape(B, C, W, H).permute(0, 1, 3, 2).reshape(B, C, L)
        y4 = ys[3].flip(dims=[-1]).reshape(B, C, W, H).permute(0, 1, 3, 2).reshape(B, C, L)
        return y1 + y2 + y3 + y4

    def forward(self, x):
        """x: (B, H, W, C) -> (B, H, W, C)"""
        B, H, W, C = x.shape

        xz = self.in_proj(x)  # (B, H, W, 2*d_inner)
        x_inner, z = xz.chunk(2, dim=-1)

        x_inner = x_inner.permute(0, 3, 1, 2).contiguous()  # (B, d_inner, H, W)
        x_inner = self.act(self.conv2d(x_inner))  # (B, d_inner, H, W)

        # 4-direction scan
        xs = self._scan_four_directions(x_inner)  # 4 x (B, d_inner, L)

        A = -torch.exp(self.A_log)  # (d_inner, d_state)
        L = H * W

        ys = []
        for i in range(4):
            xi = xs[i]  # (B, d_inner, L)
            # Per-direction projection: (B, L, d_inner) @ (d_inner, dt_rank+2*d_state)
            x_dbl = torch.einsum('bld,de->ble', xi.transpose(1, 2), self.x_proj_weight[i])  # (B, L, dt_rank+2*d_state)

            dt_i = x_dbl[:, :, :self.dt_rank]  # (B, L, dt_rank)
            B_i = x_dbl[:, :, self.dt_rank:self.dt_rank + self.d_state]  # (B, L, d_state)
            C_i = x_dbl[:, :, self.dt_rank + self.d_state:]  # (B, L, d_state)

            dt_i = self.dt_projs[i](dt_i).transpose(1, 2)  # (B, d_inner, L)
            B_i = B_i.transpose(1, 2)  # (B, d_state, L)
            C_i = C_i.transpose(1, 2)  # (B, d_state, L)

            y_i = _lazy_selective_scan_fn()(
                xi.contiguous(), dt_i.contiguous(), A, B_i.contiguous(), C_i.contiguous(),
                D=self.D, delta_softplus=True
            )
            ys.append(y_i)

        y = self._unscan_four_directions(ys, H, W)  # (B, d_inner, L)
        y = y.transpose(1, 2).reshape(B, H, W, self.d_inner)
        y = self.out_norm(y)

        # Gate with z
        y = y * F.silu(z)
        y = self.out_proj(y)
        y = self.dropout(y)
        return y


# ---------- VSS Block ----------

class VSSBlock(nn.Module):
    """Visual State Space Block from VM-UNet."""

    def __init__(self, hidden_dim, d_state=16, d_conv=3, expand=2, drop_path=0.0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.self_attention = SS2D(hidden_dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, input):
        """input: (B, H, W, C) -> (B, H, W, C)"""
        x = self.ln_1(input)
        x = self.self_attention(x)
        x = input + self.drop_path(x)
        return x


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        output = x / keep_prob * (random_tensor >= self.drop_prob).float()
        return output


# ---------- Patch Embed & Merge ----------

class PatchEmbed2D(nn.Module):
    """Image to Patch Embedding."""
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=nn.LayerNorm):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)  # (B, C, H, W)
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.norm(x)
        return x


class PatchMerging2D(nn.Module):
    """Patch Merging (2x downsampling)."""
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """x: (B, H, W, C) -> (B, H/2, W/2, 2C)"""
        B, H, W, C = x.shape
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)  # (B, H/2, W/2, 4C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


# ---------- VSSLayer (one stage) ----------

class VSSLayer(nn.Module):
    """One stage of VSS blocks + optional downsampling."""
    def __init__(self, dim, depth, d_state=16, d_conv=3, expand=2,
                 drop_path=0.0, downsample=None):
        super().__init__()
        if isinstance(drop_path, (list, tuple)):
            dp_rates = drop_path
        else:
            dp_rates = [drop_path] * depth

        self.blocks = nn.ModuleList([
            VSSBlock(dim, d_state=d_state, d_conv=d_conv, expand=expand, drop_path=dp_rates[i])
            for i in range(depth)
        ])
        self.downsample = downsample(dim) if downsample else None

    def forward(self, x):
        """x: (B, H, W, C)"""
        for blk in self.blocks:
            x = blk(x)
        x_out = x
        if self.downsample is not None:
            x = self.downsample(x)
        return x_out, x


# ---------- VM-UNet Encoder ----------

@ENCODER_REGISTRY.register("vmunet")
class VMUNetEncoder(nn.Module):
    """VM-UNet Encoder (VSSM backbone).

    Faithful to https://github.com/JCruan519/VM-UNet
    4-stage hierarchical encoder with SS2D selective scan.
    """

    def __init__(
        self,
        pretrained: bool = False,
        in_channels: int = 3,
        img_size: int = 224,
        patch_size: int = 4,
        embed_dim: int = 96,
        depths: tuple = (2, 2, 9, 2),
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 2,
        drop_path_rate: float = 0.2,
        pretrained_path: str = None,
        **kwargs,
    ):
        super().__init__()
        self.num_stages = 4
        dims = [embed_dim * (2 ** i) for i in range(4)]
        self.dims = dims

        # Patch embedding
        self.patch_embed = PatchEmbed2D(patch_size, in_channels, embed_dim)

        # Build stages
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i in range(4):
            layer = VSSLayer(
                dim=dims[i],
                depth=depths[i],
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i+1])],
                downsample=PatchMerging2D if i < 3 else None,
            )
            self.layers.append(layer)

        # Norms for each stage output
        self.norms = nn.ModuleList([nn.LayerNorm(dims[i]) for i in range(4)])

        self.out_channels = dims

        if pretrained and pretrained_path:
            self.load_pretrained(pretrained_path)

    def load_pretrained(self, path):
        state = torch.load(path, map_location='cpu')
        if 'model' in state:
            state = state['model']
        if 'state_dict' in state:
            state = state['state_dict']
        encoder_state = {}
        for k, v in state.items():
            if k.startswith('encoder.'):
                encoder_state[k.replace('encoder.', '')] = v
            elif not k.startswith('decoder') and not k.startswith('head') and not k.startswith('final'):
                encoder_state[k] = v
        msg = self.load_state_dict(encoder_state, strict=False)
        print(f"VM-UNet encoder loaded: {msg}")

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.patch_embed(x)  # (B, H/4, W/4, C)
        features = []
        for i, layer in enumerate(self.layers):
            x_out, x = layer(x)
            x_out = self.norms[i](x_out)
            # Convert to (B, C, H, W) format
            feat = x_out.permute(0, 3, 1, 2).contiguous()
            features.append(feat)
        return features
