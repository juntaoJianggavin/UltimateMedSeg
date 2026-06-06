"""U-KAN Encoder (AAAI 2025).

Standalone encoder extracted from ``medseg.models.networks.kan_mlp.ukan.UKAN``.

Pipeline (default ``embed_dims=[256, 320, 512]``):
    in -> [1x1 conv if in_channels != 3] ->
    encoder1 (DoubleConv,  in -> 32)  -> MaxPool/2 -> t1   (C=32,  H/2)
    encoder2 (DoubleConv,  32 -> 64)  -> MaxPool/2 -> t2   (C=64,  H/4)
    encoder3 (DoubleConv,  64 -> 256) -> MaxPool/2 -> t3   (C=256, H/8)
    patch_embed3 (stride 2) + KANBlock + LayerNorm ->        t4 (C=320, H/16)
    patch_embed4 (stride 2) + KANBlock + LayerNorm ->        t5 (C=512, H/32)

Returns 5 multi-scale features ordered shallow->deep, deepest LAST.
Inputs need to be divisible by 16 (16 = 2^3 strided pool * 2 PatchEmbeds).
"""
# Source: https://github.com/CUHK-AIM-Group/U-KAN

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from medseg.registry import ENCODER_REGISTRY


# ---------------------------------------------------------------------------
# KANLinear (copied verbatim from the source network).
# ---------------------------------------------------------------------------

class _KANLinear(nn.Module):
    """KAN linear layer with B-spline learnable activation functions."""

    def __init__(self, in_features, out_features, grid_size=5, spline_order=3,
                 scale_noise=0.1, scale_base=1.0, scale_spline=1.0,
                 enable_standalone_scale_spline=True,
                 base_activation=nn.SiLU, grid_eps=0.02, grid_range=(-1, 1)):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (torch.arange(-spline_order, grid_size + spline_order + 1) * h
             + grid_range[0])
            .expand(in_features, -1).contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order))
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(
                torch.Tensor(out_features, in_features))

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                 - 0.5) * self.scale_noise / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline
                 if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order:-self.spline_order], noise)
            )
            if self.enable_standalone_scale_spline:
                nn.init.kaiming_uniform_(
                    self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, :-(k + 1)])
                / (grid[:, k:-1] - grid[:, :-(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1:] - x)
                / (grid[:, k + 1:] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )
        return bases.contiguous()

    def curve2coeff(self, x, y):
        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        return solution.permute(2, 0, 1).contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline else 1.0)

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1))
        return base_output + spline_output


# ---------------------------------------------------------------------------
# Building blocks (mirrors of the network-side helpers).
# ---------------------------------------------------------------------------

class _DW_bn_relu(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class _KANLayer(nn.Module):
    """Tokenized KAN layer: fc1 -> dw -> fc2 -> dw -> fc3 -> dw."""

    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0., no_kan=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features

        if not no_kan:
            kan_kw = dict(grid_size=5, spline_order=3, scale_noise=0.1,
                          scale_base=1.0, scale_spline=1.0,
                          base_activation=nn.SiLU, grid_eps=0.02,
                          grid_range=[-1, 1])
            self.fc1 = _KANLinear(in_features, hidden_features, **kan_kw)
            self.fc2 = _KANLinear(hidden_features, out_features, **kan_kw)
            self.fc3 = _KANLinear(hidden_features, out_features, **kan_kw)
        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)
            self.fc3 = nn.Linear(hidden_features, out_features)

        self.dwconv_1 = _DW_bn_relu(hidden_features)
        self.dwconv_2 = _DW_bn_relu(hidden_features)
        self.dwconv_3 = _DW_bn_relu(hidden_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = self.fc1(x.reshape(B * N, C))
        x = x.reshape(B, N, C).contiguous()
        x = self.dwconv_1(x, H, W)
        x = self.fc2(x.reshape(B * N, C))
        x = x.reshape(B, N, C).contiguous()
        x = self.dwconv_2(x, H, W)
        x = self.fc3(x.reshape(B * N, C))
        x = x.reshape(B, N, C).contiguous()
        x = self.dwconv_3(x, H, W)
        return x


class _KANBlock(nn.Module):
    """KAN block: LayerNorm -> KANLayer with residual + DropPath."""

    def __init__(self, dim, drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, no_kan=False):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim)
        self.layer = _KANLayer(in_features=dim, hidden_features=mlp_hidden_dim,
                               act_layer=act_layer, drop=drop, no_kan=no_kan)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.layer(self.norm2(x), H, W))
        return x


class _PatchEmbed(nn.Module):
    """Image to Patch Embedding (overlap, stride configurable).

    Spatial output size is read at runtime from ``proj`` output, so this
    module works for any input resolution.
    """

    def __init__(self, img_size=224, patch_size=7, stride=4,
                 in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        # H/W kept for reference only; runtime forward uses tensor shape.
        self.H = img_size[0] // patch_size[0]
        self.W = img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class _ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


# ---------------------------------------------------------------------------
# Pretrained-load helper (kept for parity with other encoders; UKAN ships
# no public pretrained weights, so by default this is a no-op).
# ---------------------------------------------------------------------------

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
            warnings.warn(
                f'Pretrained download failed ({e2}); using random init.')
            return load_fn(*args, **{**kwargs, 'pretrained': False})
        finally:
            ssl._create_default_https_context = prev


# ---------------------------------------------------------------------------
# Top-level encoder.
# ---------------------------------------------------------------------------

@ENCODER_REGISTRY.register("ukan")
class UKANEncoder(nn.Module):
    """U-KAN encoder.

    Returns 5 multi-scale features (shallow -> deep, deepest LAST):
        [C0 @ H/2, C1 @ H/4, C2 @ H/8, C3 @ H/16, C4 @ H/32]
    where for the default ``embed_dims=[256, 320, 512]`` this is
        [32, 64, 256, 320, 512].

    Args:
        in_channels: Number of input channels. If != 3, a 1x1 stem maps to 3.
        img_size: Reference spatial resolution (only used to seed PatchEmbed
            metadata; the forward path reads true H/W from the tensor).
        pretrained: Unused for U-KAN (no public weights); kept for the
            standard encoder interface.
        embed_dims: Channel dims for the 3 KAN-side stages (kan_input, mid,
            bottleneck). Defaults to ``[256, 320, 512]``.
        no_kan: If True, replace KAN layers with plain MLP (ablation).
        drop_rate / drop_path_rate / depths: Forwarded to the KAN blocks.
    """

    def __init__(self, in_channels: int = 3, img_size: int = 224,
                 pretrained: bool = False,
                 embed_dims: List[int] = None, no_kan: bool = False,
                 drop_rate: float = 0., drop_path_rate: float = 0.,
                 depths: List[int] = None, **kwargs):
        super().__init__()

        if embed_dims is None:
            embed_dims = [256, 320, 512]
        if depths is None:
            depths = [1, 1, 1]

        self._embed_dims = list(embed_dims)
        kan_input_dim = embed_dims[0]
        norm_layer = nn.LayerNorm

        # Optional 1x1 stem for non-RGB inputs (e.g. grayscale CT/MR).
        if in_channels != 3:
            self.in_proj = nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)
            stem_in = 3
        else:
            self.in_proj = nn.Identity()
            stem_in = in_channels

        # -- Conv encoder stages --
        self.encoder1 = _ConvLayer(stem_in, kan_input_dim // 8)
        self.encoder2 = _ConvLayer(kan_input_dim // 8, kan_input_dim // 4)
        self.encoder3 = _ConvLayer(kan_input_dim // 4, kan_input_dim)

        # -- KAN-side norms --
        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        # -- KAN blocks (encoder-side only) --
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.block1 = nn.ModuleList([_KANBlock(
            dim=embed_dims[1], drop=drop_rate, drop_path=dpr[0],
            norm_layer=norm_layer, no_kan=no_kan)])
        self.block2 = nn.ModuleList([_KANBlock(
            dim=embed_dims[2], drop=drop_rate, drop_path=dpr[1],
            norm_layer=norm_layer, no_kan=no_kan)])

        # -- Patch embeddings (KAN stage + bottleneck) --
        self.patch_embed3 = _PatchEmbed(
            img_size=img_size // 4, patch_size=3, stride=2,
            in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed4 = _PatchEmbed(
            img_size=img_size // 8, patch_size=3, stride=2,
            in_chans=embed_dims[1], embed_dim=embed_dims[2])

        # Channel list for each returned feature (deepest LAST).
        self.out_channels: List[int] = [
            kan_input_dim // 8,   # t1  H/2
            kan_input_dim // 4,   # t2  H/4
            kan_input_dim,        # t3  H/8
            embed_dims[1],        # t4  H/16
            embed_dims[2],        # t5  H/32
        ]

        # ``pretrained`` is accepted for interface parity but unused
        # (no canonical public U-KAN weights). The fallback helper is
        # exposed for use by subclasses if needed.
        self._pretrained_requested = bool(pretrained)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.in_proj(x)
        B = x.shape[0]

        # Stage 1: Conv + MaxPool -> H/2
        out = F.relu(F.max_pool2d(self.encoder1(x), 2, 2))
        t1 = out

        # Stage 2: Conv + MaxPool -> H/4
        out = F.relu(F.max_pool2d(self.encoder2(out), 2, 2))
        t2 = out

        # Stage 3: Conv + MaxPool -> H/8
        out = F.relu(F.max_pool2d(self.encoder3(out), 2, 2))
        t3 = out

        # Stage 4: tokenized KAN stage -> H/16
        out, H, W = self.patch_embed3(out)
        for blk in self.block1:
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out

        # Stage 5 (bottleneck): tokenized KAN stage -> H/32
        out, H, W = self.patch_embed4(out)
        for blk in self.block2:
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t5 = out

        return [t1, t2, t3, t4, t5]
