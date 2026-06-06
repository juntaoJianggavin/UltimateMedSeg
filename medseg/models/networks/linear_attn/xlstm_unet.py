"""xLSTM-UNet 2D: UNet with Vision-LSTM (xLSTM) for Medical Image Segmentation.

Faithful reimplementation from:
  https://github.com/tianrun-chen/xLSTM-UNet-PyTorch  (arXiv 2407.01530)

Architecture follows the original nnU-Net-based implementation:
  - Encoder: stem (stride-1 stage) + strided stages
  - Decoder: UpsampleLayer + concat skip + BasicResBlock, seg_layers for deep supervision
  - XLSTMUNetBot: UNetResEncoder + xLSTMLayer at bottleneck + UNetResDecoder
  - XLSTMUNetEnc: ResidualXLSTMEncoder (alternating xLSTMLayers) + UNetResDecoder

Core mLSTM cell from Vision-LSTM (ViL):
  https://github.com/NX-AI/vision-lstm (AGPL-3.0, NXAI GmbH)
"""
# Source: https://github.com/tianrun-chen/xLSTM-UNet-PyTorch

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union
from enum import Enum

try:
    import einops
    _HAS_EINOPS = True
except ImportError:
    _HAS_EINOPS = False

# Reuse shared building blocks from U-Mamba
from medseg.models.networks.mamba.umamba import (
    BasicResBlock, BasicBlockD, UpsampleLayer,
    UNetResEncoder, UNetResDecoder,
)


# ---------------------------------------------------------------------------
# mLSTM core components (from vision_lstm.py, AGPL-3.0 NXAI GmbH)
# ---------------------------------------------------------------------------

class SequenceTraversal(Enum):
    ROWWISE_FROM_TOP_LEFT = "rowwise_from_top_left"
    ROWWISE_FROM_BOT_RIGHT = "rowwise_from_bot_right"


def bias_linspace_init_(param: torch.Tensor, start: float = 3.4, end: float = 6.0):
    n_dims = param.shape[0]
    init_vals = torch.linspace(start, end, n_dims)
    with torch.no_grad():
        param.copy_(init_vals)
    return param


def small_init_(param: torch.Tensor, dim: int):
    std = math.sqrt(2 / (5 * dim))
    torch.nn.init.normal_(param, mean=0.0, std=std)
    return param


def wang_init_(param: torch.Tensor, dim: int, num_blocks: int):
    std = 2 / num_blocks / math.sqrt(dim)
    torch.nn.init.normal_(param, mean=0.0, std=std)
    return param


def parallel_stabilized_simple(
        queries, keys, values,
        igate_preact, fgate_preact,
        lower_triangular_matrix=None,
        stabilize_rowwise=True, eps=1e-6):
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
    rep_log_fgates_cumsum = log_fgates_cumsum.repeat(1, 1, 1, S + 1)
    _log_fg_matrix = rep_log_fgates_cumsum - rep_log_fgates_cumsum.transpose(-2, -1)
    log_fg_matrix = torch.where(ltr, _log_fg_matrix[:, :, 1:, 1:], -float("inf"))

    log_D_matrix = log_fg_matrix + igate_preact.transpose(-2, -1)
    if stabilize_rowwise:
        max_log_D, _ = torch.max(log_D_matrix, dim=-1, keepdim=True)
    else:
        max_log_D = torch.max(log_D_matrix.view(B, NH, -1), dim=-1, keepdim=True)[0].unsqueeze(-1)
    log_D_matrix_stabilized = log_D_matrix - max_log_D
    D_matrix = torch.exp(log_D_matrix_stabilized)

    keys_scaled = keys / math.sqrt(DH)
    qk_matrix = queries @ keys_scaled.transpose(-2, -1)
    C_matrix = qk_matrix * D_matrix
    normalizer = torch.maximum(C_matrix.sum(dim=-1, keepdim=True).abs(), torch.exp(-max_log_D))
    C_matrix_normalized = C_matrix / (normalizer + eps)
    h_tilde_state = C_matrix_normalized @ values
    return h_tilde_state


class _LayerNorm(nn.Module):
    """LayerNorm with optional bias and residual weight (from ViL)."""
    def __init__(self, ndim, weight=True, bias=False, eps=1e-5, residual_weight=True):
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
        return F.layer_norm(x, (self.ndim,), self.weight_proxy, self.bias, self.eps)

    def reset_parameters(self):
        if self.weight is not None:
            nn.init.zeros_(self.weight) if self.residual_weight else nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class MultiHeadLayerNorm(_LayerNorm):
    def forward(self, x):
        B, NH, S, DH = x.shape
        gn_in = x.transpose(1, 2).reshape(B * S, NH * DH)
        out = F.group_norm(gn_in, num_groups=NH,
                           weight=self.weight_proxy, bias=self.bias, eps=self.eps)
        return out.view(B, S, NH, DH).transpose(1, 2)


class LinearHeadwiseExpand(nn.Module):
    """Structured per-head linear projection (from ViL)."""
    def __init__(self, dim, num_heads, bias=False):
        super().__init__()
        assert dim % num_heads == 0
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
        # x: (..., dim) -> rearrange to (..., num_heads, dim_per_head)
        shape = x.shape[:-1]
        nh = self.num_heads
        dh = x.shape[-1] // nh
        x = x.view(*shape, nh, dh)
        # Per-head matmul
        x = torch.einsum('...hd,hod->...ho', x, self.weight)
        x = x.reshape(*shape, nh * dh)
        if self.bias is not None:
            x = x + self.bias
        return x


class CausalConv1d(nn.Module):
    """Causal depthwise 1D convolution (from ViL)."""
    def __init__(self, dim, kernel_size=4, bias=True):
        super().__init__()
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(dim, dim, kernel_size, padding=self.pad, groups=dim, bias=bias)

    def forward(self, x):
        # x: (B, L, D)
        x = x.transpose(1, 2)
        x = self.conv(x)[:, :, :-self.pad]
        return x.transpose(1, 2)


class MatrixLSTMCell(nn.Module):
    """mLSTM cell with multi-head structure (from ViL)."""
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.igate = nn.Linear(3 * dim, num_heads)
        self.fgate = nn.Linear(3 * dim, num_heads)
        self.outnorm = MultiHeadLayerNorm(ndim=dim, weight=True, bias=False)
        self.causal_mask_cache = {}
        self.reset_parameters()

    def forward(self, q, k, v):
        B, S, _ = q.shape
        NH = self.num_heads
        DH = q.shape[-1] // NH

        if_gate_input = torch.cat([q, k, v], dim=-1)
        q = q.view(B, S, NH, DH).transpose(1, 2)
        k = k.view(B, S, NH, DH).transpose(1, 2)
        v = v.view(B, S, NH, DH).transpose(1, 2)

        igate_preact = self.igate(if_gate_input).transpose(-1, -2).unsqueeze(-1)
        fgate_preact = self.fgate(if_gate_input).transpose(-1, -2).unsqueeze(-1)

        cache_key = (S, str(q.device))
        if cache_key not in self.causal_mask_cache:
            self.causal_mask_cache[cache_key] = torch.tril(
                torch.ones(S, S, dtype=torch.bool, device=q.device))
        causal_mask = self.causal_mask_cache[cache_key]

        h_state = parallel_stabilized_simple(
            q, k, v, igate_preact, fgate_preact, causal_mask)
        h_state = self.outnorm(h_state)
        return h_state.transpose(1, 2).reshape(B, S, -1)

    def reset_parameters(self):
        self.outnorm.reset_parameters()
        nn.init.zeros_(self.fgate.weight)
        bias_linspace_init_(self.fgate.bias, start=3.0, end=6.0)
        nn.init.zeros_(self.igate.weight)
        nn.init.normal_(self.igate.bias, mean=0.0, std=0.1)


class ViLLayerInner(nn.Module):
    """Inner ViL layer: up-proj -> conv1d -> mLSTM -> down-proj (from ViL)."""
    def __init__(self, dim, direction, expansion=2, qkv_block_size=4,
                 proj_bias=False, conv_bias=True, kernel_size=4):
        super().__init__()
        if dim % qkv_block_size != 0:
            qkv_block_size = 2
        self.dim = dim
        self.direction = direction
        inner_dim = expansion * dim
        num_heads = inner_dim // qkv_block_size

        self.proj_up = nn.Linear(dim, 2 * inner_dim, bias=proj_bias)
        self.q_proj = LinearHeadwiseExpand(inner_dim, num_heads, bias=proj_bias)
        self.k_proj = LinearHeadwiseExpand(inner_dim, num_heads, bias=proj_bias)
        self.v_proj = LinearHeadwiseExpand(inner_dim, num_heads, bias=proj_bias)
        self.conv1d = CausalConv1d(inner_dim, kernel_size, conv_bias)
        self.mlstm_cell = MatrixLSTMCell(inner_dim, qkv_block_size)
        self.learnable_skip = nn.Parameter(torch.ones(inner_dim))
        self.proj_down = nn.Linear(inner_dim, dim, bias=proj_bias)
        self.reset_parameters()

    def forward(self, x):
        B, S, _ = x.shape
        if self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
            x = x.flip(dims=[1])

        x_inner = self.proj_up(x)
        x_mlstm, z = torch.chunk(x_inner, 2, dim=-1)

        x_conv = F.silu(self.conv1d(x_mlstm))
        q = self.q_proj(x_conv)
        k = self.k_proj(x_conv)
        v = self.v_proj(x_mlstm)
        h = self.mlstm_cell(q, k, v)
        h = (h + self.learnable_skip * x_conv) * F.silu(z)
        x = self.proj_down(h)

        if self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
            x = x.flip(dims=[1])
        return x

    def reset_parameters(self):
        small_init_(self.proj_up.weight, dim=self.dim)
        if self.proj_up.bias is not None:
            nn.init.zeros_(self.proj_up.bias)
        wang_init_(self.proj_down.weight, dim=self.dim, num_blocks=1)
        if self.proj_down.bias is not None:
            nn.init.zeros_(self.proj_down.bias)
        nn.init.ones_(self.learnable_skip)
        for proj in [self.q_proj, self.k_proj, self.v_proj]:
            small_init_(proj.weight, dim=self.dim)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)
        self.mlstm_cell.reset_parameters()


class ViLBlock(nn.Module):
    """Vision-LSTM block: LayerNorm -> ViLLayer + residual (from ViL)."""
    def __init__(self, dim, direction=SequenceTraversal.ROWWISE_FROM_TOP_LEFT):
        super().__init__()
        self.norm = _LayerNorm(ndim=dim, weight=True, bias=False)
        self.layer = ViLLayerInner(dim=dim, direction=direction)

    def forward(self, x):
        return x + self.layer(self.norm(x))


# ---------------------------------------------------------------------------
# xLSTMLayer: wraps ViLBlock for 2D feature maps (analogous to MambaLayer)
# ---------------------------------------------------------------------------

class XLSTMLayer(nn.Module):
    """xLSTM layer for 2D features with optional channel_token mode.

    Faithful to xLSTM-UNet source:
      - forward_patch_token: spatial tokens (H*W) with channel features (default)
      - forward_channel_token: channel tokens (C) with spatial features (H*W)

    For very high-resolution feature maps (>max_tokens spatial tokens),
    the layer is skipped (identity) to avoid prohibitive O(n^2) memory.
    """
    _MAX_TOKENS = 8192  # max spatial tokens before skipping

    def __init__(self, dim, channel_token=False, max_tokens=None):
        super().__init__()
        self.dim = dim
        self.vil = ViLBlock(dim=dim, direction=SequenceTraversal.ROWWISE_FROM_TOP_LEFT)
        self.channel_token = channel_token
        if max_tokens is not None:
            self._MAX_TOKENS = max_tokens

    def forward_patch_token(self, x):
        B, C = x.shape[:2]
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)  # (B, N, C)
        x_vil = self.vil(x_flat)
        return x_vil.transpose(-1, -2).reshape(B, C, *img_dims)

    def forward_channel_token(self, x):
        B, n_tokens = x.shape[:2]
        d_model = x.shape[2:].numel()
        assert d_model == self.dim
        img_dims = x.shape[2:]
        x_flat = x.flatten(2)  # (B, C, H*W)
        x_vil = self.vil(x_flat)
        return x_vil.reshape(B, n_tokens, *img_dims)

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.float()
        if self.channel_token:
            # channel_token mode: check spatial token count
            n_tokens = x.shape[2:].numel()
            if n_tokens > self._MAX_TOKENS:
                return x  # skip to avoid O(n^2) memory
            return self.forward_channel_token(x)
        # patch_token mode: check spatial token count
        n_tokens = x.shape[2:].numel()
        if n_tokens > self._MAX_TOKENS:
            return x  # skip to avoid O(n^2) memory
        return self.forward_patch_token(x)


# ---------------------------------------------------------------------------
# ResidualXLSTMEncoder (faithful to source: alternating xLSTMLayers)
# ---------------------------------------------------------------------------

class ResidualXLSTMEncoder(nn.Module):
    """Encoder with alternating xLSTM layers at each stage (faithful to source).

    Same alternating pattern as U-Mamba Enc but with xLSTM instead of Mamba.
    """
    def __init__(self, input_size, input_channels, n_stages, features_per_stage,
                 kernel_sizes=None, strides=None, n_blocks_per_stage=None,
                 conv_bias=False):
        super().__init__()
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if kernel_sizes is None:
            kernel_sizes = [3] * n_stages
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if strides is None:
            strides = [1] + [2] * (n_stages - 1)
        if isinstance(strides, int):
            strides = [strides] * n_stages
        if n_blocks_per_stage is None:
            n_blocks_per_stage = [2] * n_stages
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(input_size, int):
            input_size = (input_size, input_size)

        self.output_channels = list(features_per_stage)
        self.strides = list(strides)
        self.kernel_sizes = list(kernel_sizes)
        self.conv_pad_sizes = [k // 2 for k in kernel_sizes]
        self.conv_bias = conv_bias

        # Compute feature map sizes and channel_token flags
        do_channel_token = [False] * n_stages
        feature_map_sizes = []
        feature_map_size = list(input_size)
        for s in range(n_stages):
            feature_map_size = [sz // strides[s] for sz in feature_map_size]
            feature_map_sizes.append(list(feature_map_size))
            if int(np.prod(feature_map_size)) <= features_per_stage[s]:
                do_channel_token[s] = True

        # Stem (stage 0)
        stem_ch = features_per_stage[0]
        ks0 = kernel_sizes[0]
        pad0 = ks0 // 2
        self.stem = nn.Sequential(
            BasicResBlock(input_channels, stem_ch, ks0, pad0,
                          stride=1, use_1x1conv=True),
            *[BasicBlockD(stem_ch, ks0, conv_bias)
              for _ in range(n_blocks_per_stage[0] - 1)]
        )

        # Alternating xLSTM layers for ALL stages
        # Pattern: bool(s%2) ^ bool(n_stages%2) guarantees last stage has xLSTM
        xlstm_layers = []
        for s in range(n_stages):
            if bool(s % 2) ^ bool(n_stages % 2):
                xlstm_dim = (int(np.prod(feature_map_sizes[s]))
                             if do_channel_token[s]
                             else features_per_stage[s])
                xlstm_layers.append(XLSTMLayer(
                    dim=xlstm_dim, channel_token=do_channel_token[s]))
            else:
                xlstm_layers.append(nn.Identity())
        self.xlstm_layers = nn.ModuleList(xlstm_layers)

        # Conv stages 1..n_stages-1
        stages = []
        ch = stem_ch
        for s in range(1, n_stages):
            ks = kernel_sizes[s]
            pad = ks // 2
            stage = nn.Sequential(
                BasicResBlock(ch, features_per_stage[s], ks, pad,
                              stride=strides[s], use_1x1conv=True),
                *[BasicBlockD(features_per_stage[s], ks, conv_bias)
                  for _ in range(n_blocks_per_stage[s] - 1)]
            )
            stages.append(stage)
            ch = features_per_stage[s]
        self.stages = nn.ModuleList(stages)

    def forward(self, x):
        x = self.stem(x)
        x = self.xlstm_layers[0](x)
        ret = []
        for s in range(len(self.stages)):
            x = self.stages[s](x)
            x = self.xlstm_layers[s + 1](x)
            ret.append(x)
        return ret


# ---------------------------------------------------------------------------
# XLSTMUNetBot: UNet encoder + xLSTMLayer at bottleneck + UNet decoder
# ---------------------------------------------------------------------------

class XLSTMUNetBot(nn.Module):
    """xLSTM-UNet Bot: Standard UNet with xLSTM layer at bottleneck.

    Faithful to source: encoder(x) -> xLSTM at bottleneck -> decoder(skips).
    """
    def __init__(self, in_channels=3, num_classes=2, img_size=224,
                 features=None, n_blocks_per_stage=None,
                 strides=None, kernel_sizes=None,
                 n_conv_per_stage_decoder=None,
                 deep_supervision=False, **kwargs):
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256, 512]
        n_stages = len(features)

        if n_blocks_per_stage is None:
            n_blocks_per_stage = [2] * n_stages
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        n_blocks_per_stage = list(n_blocks_per_stage)

        if n_conv_per_stage_decoder is None:
            n_conv_per_stage_decoder = [2] * (n_stages - 1)
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        n_conv_per_stage_decoder = list(n_conv_per_stage_decoder)

        for s in range(math.ceil(n_stages / 2), n_stages):
            n_blocks_per_stage[s] = 1
        for s in range(math.ceil((n_stages - 1) / 2 + 0.5), n_stages - 1):
            n_conv_per_stage_decoder[s] = 1

        self.encoder = UNetResEncoder(
            in_channels, n_stages, features,
            kernel_sizes=kernel_sizes, strides=strides,
            n_blocks_per_stage=n_blocks_per_stage)

        self.xlstm_layer = XLSTMLayer(dim=features[-1])

        self.decoder = UNetResDecoder(
            self.encoder, num_classes,
            n_conv_per_stage=n_conv_per_stage_decoder,
            deep_supervision=deep_supervision)

    def forward(self, x):
        skips = self.encoder(x)
        skips[-1] = self.xlstm_layer(skips[-1])
        return self.decoder(skips)


# ---------------------------------------------------------------------------
# XLSTMUNetEnc: ResidualXLSTMEncoder + UNet decoder
# ---------------------------------------------------------------------------

class XLSTMUNetEnc(nn.Module):
    """xLSTM-UNet Enc: ResidualXLSTMEncoder + UNet decoder.

    Faithful to source: alternating xLSTM layers in encoder, deep supervision in decoder.
    """
    def __init__(self, in_channels=3, num_classes=2, img_size=224,
                 features=None, n_blocks_per_stage=None,
                 strides=None, kernel_sizes=None,
                 n_conv_per_stage_decoder=None,
                 deep_supervision=False, **kwargs):
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256, 512]
        n_stages = len(features)

        if isinstance(img_size, int):
            input_size = (img_size, img_size)
        else:
            input_size = tuple(img_size)

        if n_blocks_per_stage is None:
            n_blocks_per_stage = [2] * n_stages
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        n_blocks_per_stage = list(n_blocks_per_stage)

        if n_conv_per_stage_decoder is None:
            n_conv_per_stage_decoder = [2] * (n_stages - 1)
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        n_conv_per_stage_decoder = list(n_conv_per_stage_decoder)

        for s in range(math.ceil(n_stages / 2), n_stages):
            n_blocks_per_stage[s] = 1
        for s in range(math.ceil((n_stages - 1) / 2 + 0.5), n_stages - 1):
            n_conv_per_stage_decoder[s] = 1

        self.encoder = ResidualXLSTMEncoder(
            input_size, in_channels, n_stages, features,
            kernel_sizes=kernel_sizes, strides=strides,
            n_blocks_per_stage=n_blocks_per_stage)

        self.decoder = UNetResDecoder(
            self.encoder, num_classes,
            n_conv_per_stage=n_conv_per_stage_decoder,
            deep_supervision=deep_supervision)

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)
