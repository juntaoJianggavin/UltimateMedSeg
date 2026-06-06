"""U-VixLSTM — 2-D Vision-xLSTM segmentation network.

Ported from:
    https://github.com/willxxy/2D-U-VixLSTM  (twoDUVixLSTM.py + vLSTM.py)

Reference:
    "U-VixLSTM: A 2D Vision-xLSTM Based Framework for Medical Image
    Segmentation" (2024).

NOTE: All VisionLSTM primitives (ViLBlock, ViLLayer, MatrixLSTMCell, etc.)
are inlined from the upstream ``vLSTM.py`` to avoid external dependencies.
"""

import math
from enum import Enum

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# VisionLSTM primitives (from upstream vLSTM.py)
# ============================================================================


class _SequenceTraversal(Enum):
    ROWWISE_FROM_TOP_LEFT = "rowwise_from_top_left"
    ROWWISE_FROM_BOT_RIGHT = "rowwise_from_bot_right"


def _bias_linspace_init_(param: torch.Tensor, start: float = 3.4, end: float = 6.0):
    n_dims = param.shape[0]
    init_vals = torch.linspace(start, end, n_dims)
    with torch.no_grad():
        param.copy_(init_vals)
    return param


def _small_init_(param: torch.Tensor, dim: int):
    std = math.sqrt(2 / (5 * dim))
    torch.nn.init.normal_(param, mean=0.0, std=std)
    return param


def _wang_init_(param: torch.Tensor, dim: int, num_blocks: int):
    std = 2 / num_blocks / math.sqrt(dim)
    torch.nn.init.normal_(param, mean=0.0, std=std)
    return param


class _DropPath(nn.Sequential):
    """Stochastic depth per-sample drop-path."""

    def __init__(self, *args, drop_prob: float = 0., scale_by_keep: bool = True,
                 stochastic_drop_prob: bool = False):
        super().__init__(*args)
        self._drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
        self.stochastic_drop_prob = stochastic_drop_prob

    @property
    def drop_prob(self):
        return self._drop_prob

    @property
    def keep_prob(self):
        return 1. - self.drop_prob

    def forward(self, x, residual_path=None, residual_path_kwargs=None):
        residual_path_kwargs = residual_path_kwargs or {}
        if self.drop_prob == 0. or not self.training:
            if residual_path is None:
                return x + super().forward(x, **residual_path_kwargs)
            else:
                return x + residual_path(x, **residual_path_kwargs)
        bs = len(x)
        if self.stochastic_drop_prob:
            perm = torch.empty(bs, device=x.device).bernoulli_(self.keep_prob).nonzero().squeeze(1)
            scale = 1 / self.keep_prob
        else:
            keep_count = max(int(bs * self.keep_prob), 1)
            scale = bs / keep_count
            perm = torch.randperm(bs, device=x.device)[:keep_count]
        residual_path_kwargs = {
            key: value[perm] if torch.is_tensor(value) else value
            for key, value in residual_path_kwargs.items()
        }
        if residual_path is None:
            residual = super().forward(x[perm], **residual_path_kwargs)
        else:
            residual = residual_path(x[perm], **residual_path_kwargs)
        return torch.index_add(
            x.flatten(start_dim=1), dim=0, index=perm,
            source=residual.to(x.dtype).flatten(start_dim=1),
            alpha=scale if self.scale_by_keep else 1.,
        ).view_as(x)


def _parallel_stabilized_simple(queries, keys, values, igate_preact, fgate_preact,
                                lower_triangular_matrix=None, stabilize_rowwise=True,
                                eps=1e-6):
    """mLSTM cell in parallel form (stabilized)."""
    B, NH, S, DH = queries.shape
    _dtype, _device = queries.dtype, queries.device

    log_fgates = F.logsigmoid(fgate_preact)
    if lower_triangular_matrix is None or S < lower_triangular_matrix.size(-1):
        ltr = torch.tril(torch.ones((S, S), dtype=torch.bool, device=_device))
    else:
        ltr = lower_triangular_matrix

    log_fgates_cumsum = torch.cat([
        torch.zeros((B, NH, 1, 1), dtype=_dtype, device=_device),
        torch.cumsum(log_fgates, dim=-2),
    ], dim=-2)
    rep = log_fgates_cumsum.repeat(1, 1, 1, S + 1)
    _log_fg_matrix = rep - rep.transpose(-2, -1)
    log_fg_matrix = torch.where(ltr, _log_fg_matrix[:, :, 1:, 1:], -float("inf"))

    log_D_matrix = log_fg_matrix + igate_preact.transpose(-2, -1)
    if stabilize_rowwise:
        max_log_D, _ = torch.max(log_D_matrix, dim=-1, keepdim=True)
    else:
        max_log_D = torch.max(log_D_matrix.view(B, NH, -1), dim=-1, keepdim=True)[0].unsqueeze(-1)
    log_D_stab = log_D_matrix - max_log_D
    D_matrix = torch.exp(log_D_stab)

    keys_scaled = keys / math.sqrt(DH)
    qk = queries @ keys_scaled.transpose(-2, -1)
    C = qk * D_matrix
    normalizer = torch.maximum(C.sum(dim=-1, keepdim=True).abs(), torch.exp(-max_log_D))
    C_norm = C / (normalizer + eps)
    return C_norm @ values


class _LinearHeadwiseExpand(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        dim_per_head = dim // num_heads
        self.weight = nn.Parameter(torch.empty(num_heads, dim_per_head, dim_per_head))
        self.bias = nn.Parameter(torch.empty(dim)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight.data, mean=0.0, std=math.sqrt(2 / 5 / self.weight.shape[-1]))
        if self.bias is not None:
            nn.init.zeros_(self.bias.data)

    def forward(self, x):
        x = einops.rearrange(x, "... (nh d) -> ... nh d", nh=self.num_heads)
        x = einops.einsum(x, self.weight, "... nh d, nh out_d d -> ... nh out_d")
        x = einops.rearrange(x, "... nh out_d -> ... (nh out_d)")
        if self.bias is not None:
            x = x + self.bias
        return x


class _CausalConv1d(nn.Module):
    def __init__(self, dim, kernel_size=4, bias=True):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(dim, dim, kernel_size=kernel_size,
                              padding=self.pad, groups=dim, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x):
        x = einops.rearrange(x, "b l d -> b d l")
        x = self.conv(x)[:, :, :-self.pad]
        return einops.rearrange(x, "b d l -> b l d")


class _LayerNorm(nn.Module):
    def __init__(self, ndim=-1, weight=True, bias=False, eps=1e-5, residual_weight=True):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(ndim)) if weight else None
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = eps
        self.residual_weight = residual_weight
        self.ndim = ndim
        self.reset_parameters()

    @property
    def weight_proxy(self):
        if self.weight is None:
            return None
        return 1.0 + self.weight if self.residual_weight else self.weight

    def forward(self, x):
        return F.layer_norm(x, normalized_shape=(self.ndim,),
                            weight=self.weight_proxy, bias=self.bias, eps=self.eps)

    def reset_parameters(self):
        if self.weight_proxy is not None:
            if self.residual_weight:
                nn.init.zeros_(self.weight)
            else:
                nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class _MultiHeadLayerNorm(_LayerNorm):
    def forward(self, x):
        B, NH, S, DH = x.shape
        gn_in = x.transpose(1, 2).reshape(B * S, NH * DH)
        out = F.group_norm(gn_in, num_groups=NH, weight=self.weight_proxy,
                           bias=self.bias, eps=self.eps)
        return out.view(B, S, NH, DH).transpose(1, 2)


class _MatrixLSTMCell(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.igate = nn.Linear(3 * dim, num_heads)
        self.fgate = nn.Linear(3 * dim, num_heads)
        self.outnorm = _MultiHeadLayerNorm(ndim=dim, weight=True, bias=False)
        self.causal_mask_cache = {}
        self.reset_parameters()

    def forward(self, q, k, v):
        B, S, _ = q.shape
        if_gate_input = torch.cat([q, k, v], dim=-1)
        q = q.view(B, S, self.num_heads, -1).transpose(1, 2)
        k = k.view(B, S, self.num_heads, -1).transpose(1, 2)
        v = v.view(B, S, self.num_heads, -1).transpose(1, 2)

        igate_preact = self.igate(if_gate_input).transpose(-1, -2).unsqueeze(-1)
        fgate_preact = self.fgate(if_gate_input).transpose(-1, -2).unsqueeze(-1)

        if S in self.causal_mask_cache:
            causal_mask = self.causal_mask_cache[(S, str(q.device))]
        else:
            causal_mask = torch.tril(torch.ones(S, S, dtype=torch.bool, device=q.device))
            self.causal_mask_cache[(S, str(q.device))] = causal_mask

        h_state = _parallel_stabilized_simple(q, k, v, igate_preact, fgate_preact,
                                              lower_triangular_matrix=causal_mask)
        h_norm = self.outnorm(h_state)
        return h_norm.transpose(1, 2).reshape(B, S, -1)

    def reset_parameters(self):
        self.outnorm.reset_parameters()
        torch.nn.init.zeros_(self.fgate.weight)
        _bias_linspace_init_(self.fgate.bias, start=3.0, end=6.0)
        torch.nn.init.zeros_(self.igate.weight)
        torch.nn.init.normal_(self.igate.bias, mean=0.0, std=0.1)


class _ViLLayer(nn.Module):
    def __init__(self, dim, direction, expansion=2, qkv_block_size=4,
                 proj_bias=False, conv_bias=True, kernel_size=4):
        super().__init__()
        self.dim = dim
        self.direction = direction
        inner_dim = expansion * dim
        num_heads = inner_dim // qkv_block_size

        self.proj_up = nn.Linear(dim, 2 * inner_dim, bias=proj_bias)
        self.q_proj = _LinearHeadwiseExpand(inner_dim, num_heads, bias=proj_bias)
        self.k_proj = _LinearHeadwiseExpand(inner_dim, num_heads, bias=proj_bias)
        self.v_proj = _LinearHeadwiseExpand(inner_dim, num_heads, bias=proj_bias)
        self.conv1d = _CausalConv1d(inner_dim, kernel_size, bias=conv_bias)
        self.mlstm_cell = _MatrixLSTMCell(inner_dim, num_heads=qkv_block_size)
        self.learnable_skip = nn.Parameter(torch.ones(inner_dim))
        self.proj_down = nn.Linear(inner_dim, dim, bias=proj_bias)
        self.reset_parameters()

    def forward(self, x):
        if self.direction == _SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
            x = x.flip(dims=[1])

        x_inner = self.proj_up(x)
        x_mlstm, z = torch.chunk(x_inner, chunks=2, dim=-1)

        x_conv = self.conv1d(x_mlstm)
        x_conv_act = F.silu(x_conv)
        q = self.q_proj(x_conv_act)
        k = self.k_proj(x_conv_act)
        v = self.v_proj(x_mlstm)
        h = self.mlstm_cell(q=q, k=k, v=v)
        h_skip = h + (self.learnable_skip * x_conv_act)
        h_state = h_skip * F.silu(z)
        x = self.proj_down(h_state)

        if self.direction == _SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
            x = x.flip(dims=[1])
        return x

    def reset_parameters(self):
        _small_init_(self.proj_up.weight, dim=self.dim)
        if self.proj_up.bias is not None:
            nn.init.zeros_(self.proj_up.bias)
        _wang_init_(self.proj_down.weight, dim=self.dim, num_blocks=1)
        if self.proj_down.bias is not None:
            nn.init.zeros_(self.proj_down.bias)
        nn.init.ones_(self.learnable_skip)
        for proj in (self.q_proj, self.k_proj, self.v_proj):
            _small_init_(proj.weight, dim=self.dim)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)
        self.mlstm_cell.reset_parameters()


class _ViLBlock(nn.Module):
    def __init__(self, dim, direction, drop_path=0.1, norm_bias=False):
        super().__init__()
        self.drop_path = _DropPath(drop_prob=drop_path)
        self.norm = _LayerNorm(ndim=dim, weight=True, bias=norm_bias)
        self.layer = _ViLLayer(dim=dim, direction=direction)
        self.reset_parameters()

    def _forward_path(self, x):
        return self.layer(self.norm(x))

    def forward(self, x):
        return self.drop_path(x, self._forward_path)

    def reset_parameters(self):
        self.layer.reset_parameters()
        self.norm.reset_parameters()


# ============================================================================
# Encoder / Decoder (from upstream twoDUVixLSTM.py)
# ============================================================================


class _EncoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, base_width=64):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        width = int(out_channels * (base_width / 64))
        self.conv1 = nn.Conv2d(in_channels, width, 1, stride=1, bias=False)
        self.norm1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, 3, stride=2, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, out_channels, 1, stride=1, bias=False)
        self.norm3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.downsample(x)
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.relu(self.norm2(self.conv2(out)))
        out = self.norm3(self.conv3(out))
        return self.relu(out + identity)


class _Encoder(nn.Module):
    """ResNet-stem + VisionLSTM blocks encoder."""

    def __init__(self, img_dim, in_channels, out_channels,
                 depth=24, dim=1024, drop_path_rate=0.0,
                 alternation="bidirectional"):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 7, stride=2, padding=3, bias=False)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.encoder1 = _EncoderBottleneck(out_channels, out_channels * 2)
        self.encoder2 = _EncoderBottleneck(out_channels * 2, out_channels * 4)
        self.encoder3 = _EncoderBottleneck(out_channels * 4, out_channels * 8)

        # Feed encoder output directly to ViL blocks (no patch_embed)
        # to preserve channel count (out_channels * 8) == dim.
        self.conv2 = nn.Conv2d(out_channels * 8, dim, 3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(dim)
        self.alternation = alternation
        if drop_path_rate > 0.:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        else:
            dpr = [0.0] * depth

        directions = []
        for i in range(depth):
            if alternation == "bidirectional":
                directions.append(
                    _SequenceTraversal.ROWWISE_FROM_TOP_LEFT if i % 2 == 0
                    else _SequenceTraversal.ROWWISE_FROM_BOT_RIGHT
                )
            else:
                raise NotImplementedError(f"invalid alternation '{alternation}'")

        self.blocks = nn.ModuleList([
            _ViLBlock(dim=dim, drop_path=dpr[i], direction=directions[i])
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        # Spatial resolution after encoder: img_dim / 16
        self.spatial_size_div = 16
        self.feat_dim = dim

    def forward(self, x):
        x = self.relu(self.norm1(self.conv1(x)))
        x1 = x
        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x = self.encoder3(x3)
        # Project to dim channels, then reshape to sequence for ViL blocks
        x = self.relu(self.norm2(self.conv2(x)))
        H = x.shape[2]
        x = einops.rearrange(x, "b c h w -> b (h w) c")
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = einops.rearrange(x, "b (h w) c -> b c h w", h=H, w=H)
        return x, x1, x2, x3


class _DecoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )

    def forward(self, x, x_concat=None):
        if x_concat is not None:
            x = F.interpolate(x, size=x_concat.shape[2:],
                              mode="bilinear", align_corners=True)
            x = torch.cat([x_concat, x], dim=1)
        else:
            x = F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=True)
        return self.layer(x)


class _Decoder(nn.Module):
    def __init__(self, out_channels, num_classes, dim=None):
        super().__init__()
        if dim is None:
            dim = out_channels * 8
        # ViL output: dim channels; skip x3: out_channels*4 channels
        # After concat: dim + out_channels*4
        self.decoder1 = _DecoderBottleneck(dim + out_channels * 4, out_channels * 2)
        self.decoder2 = _DecoderBottleneck(out_channels * 4, out_channels)
        self.decoder3 = _DecoderBottleneck(out_channels * 2, out_channels // 2)
        self.decoder4 = _DecoderBottleneck(out_channels // 2, out_channels // 8)
        self.conv_out = nn.Conv2d(out_channels // 8, num_classes, 1, stride=2)

    def forward(self, x, x1, x2, x3):
        x = self.decoder1(x, x3)
        x = self.decoder2(x, x2)
        x = self.decoder3(x, x1)
        x = self.decoder4(x)
        return self.conv_out(x)


# ============================================================================
# Public model
# ============================================================================


class UVixLSTM(nn.Module):
    """2-D U-VixLSTM segmentation network.

    Ported from ``willxxy/2D-U-VixLSTM`` (twoDUVixLSTM.py + vLSTM.py).
    Uses a ResNet-stem encoder followed by VisionLSTM (mLSTM) blocks,
    with a UNet-style skip-connection decoder.

    Constructor signature follows project convention:
        ``UVixLSTM(in_channels=3, num_classes=2, img_size=224, ...)``
    """

    def __init__(self, in_channels=3, num_classes=2, img_size=224,
                 out_channels=64, depth=12, dim=None,
                 drop_path_rate=0.0, **kwargs):
        super().__init__()
        # dim must equal out_channels*8 to match encoder output channels
        if dim is None:
            dim = out_channels * 8
        self.encoder = _Encoder(
            img_dim=img_size, in_channels=in_channels,
            out_channels=out_channels, depth=depth, dim=dim,
            drop_path_rate=drop_path_rate,
        )
        self.decoder = _Decoder(out_channels=out_channels, num_classes=num_classes, dim=dim)

    def forward(self, x):
        x, x1, x2, x3 = self.encoder(x)
        return self.decoder(x, x1, x2, x3)
