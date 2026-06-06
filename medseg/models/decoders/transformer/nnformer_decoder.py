"""nnFormer-style Decoder.

Reference: Zhou et al. "nnFormer: Volumetric Medical Image Segmentation via a
3D Transformer" (https://arxiv.org/abs/2109.03201). This is the 2D adaptation
used as a representative decoder in this framework.

Each decoder stage applies:
    PatchExpand (Linear -> rearrange for 2x spatial upsample)
    -> skip-concat (via external skip_connection module, or plain cat fallback)
    -> 1x1 conv fusion (channel projection)
    -> Swin-style local self-attention blocks (alternating W-MSA / SW-MSA)

The forward pads the bottleneck spatial size up to a multiple of
``window_size`` (so every stage divides evenly into windows) and crops the
output back to the spatial size implied by the original (un-padded) input.
Skip features are also padded to keep spatial alignment with the decoder.

This module mirrors the ``swinunet_decoder`` pattern (PatchExpand + Swin
blocks), but uses external ``skip_connection`` for skip fusion (so
``has_internal_skip = False``) and self-contained windowed attention blocks
to keep this file independent of the swinunet encoder.
"""
# Source: https://github.com/282857341/nnFormer

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from medseg.registry import DECODER_REGISTRY


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def _window_partition(x: torch.Tensor, ws: int) -> torch.Tensor:
    """(B, H, W, C) -> (B*nW, ws, ws, C)."""
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws, ws, C)


def _window_reverse(windows: torch.Tensor, ws: int, H: int, W: int) -> torch.Tensor:
    """(B*nW, ws, ws, C) -> (B, H, W, C)."""
    B = int(windows.shape[0] / (H * W / (ws * ws)))
    x = windows.view(B, H // ws, W // ws, ws, ws, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


# -----------------------------------------------------------------------------
# Building blocks
# -----------------------------------------------------------------------------

class _WindowAttention(nn.Module):
    """Window-based multi-head self-attention with relative position bias."""

    def __init__(self, dim: int, window_size: int, num_heads: int, qkv_bias: bool = True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias table & index (fixed by window_size)
        self.rel_pos_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flat = coords.flatten(1)
        rel_coords = coords_flat[:, :, None] - coords_flat[:, None, :]
        rel_coords = rel_coords.permute(1, 2, 0).contiguous()
        rel_coords[:, :, 0] += window_size - 1
        rel_coords[:, :, 1] += window_size - 1
        rel_coords[:, :, 0] *= 2 * window_size - 1
        rel_pos_index = rel_coords.sum(-1)
        self.register_buffer("rel_pos_index", rel_pos_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        nn.init.trunc_normal_(self.rel_pos_bias_table, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = (self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
                          .permute(2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        rpb = self.rel_pos_bias_table[self.rel_pos_index.view(-1)].view(
            self.window_size * self.window_size,
            self.window_size * self.window_size, -1)
        rpb = rpb.permute(2, 0, 1).contiguous()
        attn = attn + rpb.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(out)


class _SwinBlock(nn.Module):
    """Swin Transformer block: window-attn (optionally shifted) + MLP.

    Operates on channels-first (B, C, H, W) input/output for convenience.
    H and W are assumed to be multiples of ``window_size``.
    """

    def __init__(self, dim: int, num_heads: int, window_size: int,
                 shift_size: int = 0, mlp_ratio: float = 4.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def _build_shift_mask(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        ws = self.window_size
        shift = self.shift_size
        img_mask = torch.zeros((1, H, W, 1), device=device)
        h_slices = (slice(0, -ws), slice(-ws, -shift), slice(-shift, None))
        w_slices = (slice(0, -ws), slice(-ws, -shift), slice(-shift, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mw = _window_partition(img_mask, ws).view(-1, ws * ws)
        attn_mask = mw.unsqueeze(1) - mw.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)) \
                              .masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        ws = self.window_size
        # Disable shift if window can't fit (degenerate small resolutions).
        shift = self.shift_size if (H >= ws and W >= ws and self.shift_size < ws) else 0

        # (B, C, H, W) -> (B, H, W, C)
        x_perm = x.permute(0, 2, 3, 1).contiguous()
        shortcut = x_perm
        x_perm = self.norm1(x_perm)

        if shift > 0:
            x_perm = torch.roll(x_perm, shifts=(-shift, -shift), dims=(1, 2))
            mask = self._build_shift_mask(H, W, x.device)
        else:
            mask = None

        x_windows = _window_partition(x_perm, ws).view(-1, ws * ws, C)
        attn_windows = self.attn(x_windows, mask=mask)
        attn_windows = attn_windows.view(-1, ws, ws, C)
        x_perm = _window_reverse(attn_windows, ws, H, W)

        if shift > 0:
            x_perm = torch.roll(x_perm, shifts=(shift, shift), dims=(1, 2))

        x_perm = shortcut + x_perm
        x_perm = x_perm + self.mlp(self.norm2(x_perm))
        # (B, H, W, C) -> (B, C, H, W)
        return x_perm.permute(0, 3, 1, 2).contiguous()


class _PatchExpand(nn.Module):
    """nnFormer/Swin-UNet PatchExpand: Linear C -> 2C, rearrange to 2x spatial, C/2 channels."""

    def __init__(self, in_dim: int):
        super().__init__()
        assert in_dim % 2 == 0, f"PatchExpand in_dim ({in_dim}) must be even"
        self.in_dim = in_dim
        self.out_dim = in_dim // 2
        self.expand = nn.Linear(in_dim, 2 * in_dim, bias=False)
        self.norm = nn.LayerNorm(self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, H, W, C)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.expand(x)  # (B, H, W, 2C)
        # rearrange (p1=p2=2): (B, H, W, p1*p2*C') -> (B, H*p1, W*p2, C') with C' = 2C / (p1*p2) = C/2
        Cp = self.out_dim
        x = x.view(B, H, W, 2, 2, Cp)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, 2 * H, 2 * W, Cp)
        x = self.norm(x)
        # back to (B, C', 2H, 2W)
        return x.permute(0, 3, 1, 2).contiguous()


# -----------------------------------------------------------------------------
# Decoder
# -----------------------------------------------------------------------------

def _pick_num_heads(dim: int, preferred: int = 4) -> int:
    """Pick a num_heads value that divides dim and is at most ``preferred``."""
    for h in range(min(preferred, dim), 0, -1):
        if dim % h == 0:
            return h
    return 1


@DECODER_REGISTRY.register("nnformer")
class NnFormerDecoder(nn.Module):
    """nnFormer-style decoder using PatchExpand + Swin blocks with external skip fusion.

    Args:
        encoder_channels: Encoder feature channels (shallow -> deep). When the
            framework calls us this is ``encoder.out_channels[:-1]``; in the
            standalone validation it may include the full encoder list.
        bottleneck_channels: Channels of the bottleneck feature handed to
            ``forward``.
        skip_connection: External skip fusion module (e.g. concat/add). When
            ``None``, falls back to plain channel concatenation.
        img_size: Input image size (used only for documentation / API parity).
        window_size: Window size for local self-attention.
        num_heads_preferred: Upper bound on heads per Swin block (actual heads
            are chosen to divide the per-stage channel dim).
        depth_per_stage: Number of Swin blocks per decoder stage.
        mlp_ratio: MLP expansion ratio inside Swin blocks.
        patch_size: Patch size of the upstream encoder (used only for the
            padding-multiple formula and informational purposes).
    """

    has_internal_skip = False

    def __init__(self, encoder_channels: List[int], bottleneck_channels: int,
                 skip_connection: nn.Module = None, img_size: int = 224,
                 window_size: int = 7, num_heads_preferred: int = 4,
                 depth_per_stage: int = 2, mlp_ratio: float = 4.,
                 patch_size: int = 4, **kwargs):
        super().__init__()
        self.skip_connection = skip_connection
        self.img_size = img_size
        self.window_size = window_size
        self.patch_size = patch_size
        self.depth_per_stage = depth_per_stage

        if not encoder_channels:
            raise ValueError("encoder_channels must be a non-empty list")

        skip_channels = list(encoder_channels)  # shallow -> deep
        num_stages = len(skip_channels)
        self.num_stages = num_stages

        # Pad bottleneck spatial to a multiple of window_size. Because
        # subsequent stages just double spatial, every stage is then also a
        # multiple of window_size. Per spec, the effective image-side multiple
        # is patch_size * 2^(stages-1) * window_size.
        self.pad_multiple = window_size
        self.image_pad_multiple = patch_size * (2 ** max(num_stages - 1, 0)) * window_size

        # Build per-stage modules (deep -> shallow).
        self.expands = nn.ModuleList()
        self.fusions = nn.ModuleList()        # 1x1 conv after skip-concat
        self.swin_stages = nn.ModuleList()    # ModuleList[ModuleList[_SwinBlock]]
        # Per-stage skip lookup. Stage 0 has no skip (mirrors swinunet pattern
        # where the first decoder layer is PatchExpand only). Stages 1..N-1
        # consume skip features in deep -> shallow order.
        self._skip_lookup: List[int] = []

        cur_dim = bottleneck_channels
        for i in range(num_stages):
            expand = _PatchExpand(cur_dim)
            expanded_dim = expand.out_dim
            self.expands.append(expand)

            if i == 0:
                # No skip at the deepest decoder stage.
                self.fusions.append(nn.Identity())
                self._skip_lookup.append(-1)
                target_dim = expanded_dim
            else:
                # Skip index walks from the deepest available skip to the
                # shallowest. With len(encoder_channels) = N, stage i (>=1)
                # uses encoder_channels[N - 1 - i].
                skip_idx = num_stages - 1 - i
                skip_idx = max(0, min(skip_idx, len(skip_channels) - 1))
                skip_ch = skip_channels[skip_idx]
                self._skip_lookup.append(skip_idx)
                if skip_connection is not None:
                    fused_ch = skip_connection.get_out_channels(expanded_dim, skip_ch)
                else:
                    fused_ch = expanded_dim + skip_ch
                # Target channel = skip_ch so the decoder ladder mirrors the
                # encoder's channel progression in reverse.
                target_dim = skip_ch
                self.fusions.append(nn.Conv2d(fused_ch, target_dim,
                                              kernel_size=1, bias=False))

            heads = _pick_num_heads(target_dim, num_heads_preferred)
            blocks = nn.ModuleList([
                _SwinBlock(dim=target_dim, num_heads=heads, window_size=window_size,
                           shift_size=0 if (j % 2 == 0) else window_size // 2,
                           mlp_ratio=mlp_ratio)
                for j in range(depth_per_stage)
            ])
            self.swin_stages.append(blocks)
            cur_dim = target_dim

        self._out_channels = cur_dim
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @property
    def out_channels(self) -> int:
        return self._out_channels

    @staticmethod
    def _pad_to_multiple(x: torch.Tensor, multiple: int):
        """Right/bottom-pad spatial dims to multiple of ``multiple``."""
        h, w = x.shape[-2:]
        new_h = int(math.ceil(h / multiple) * multiple)
        new_w = int(math.ceil(w / multiple) * multiple)
        ph, pw = new_h - h, new_w - w
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph))
        return x, ph, pw

    def forward(self, bottleneck_feat: torch.Tensor,
                skip_features: List[torch.Tensor]) -> torch.Tensor:
        # 1. Pad bottleneck so every stage's spatial divides by window_size.
        orig_h, orig_w = bottleneck_feat.shape[-2:]
        x, _, _ = self._pad_to_multiple(bottleneck_feat, self.pad_multiple)

        # 2. Pad each skip independently to the same multiple. Spatial
        #    mismatches with the decoder are reconciled by interpolation
        #    inside the loop (robust to encoders that don't perfectly halve).
        padded_skips = []
        for sf in skip_features:
            sf_padded, _, _ = self._pad_to_multiple(sf, self.pad_multiple)
            padded_skips.append(sf_padded)

        # 3. Decoder ladder.
        for i in range(self.num_stages):
            x = self.expands[i](x)
            skip_idx = self._skip_lookup[i]
            if skip_idx >= 0 and skip_idx < len(padded_skips):
                sf = padded_skips[skip_idx]
                if x.shape[-2:] != sf.shape[-2:]:
                    sf = F.interpolate(sf, size=x.shape[-2:],
                                       mode='bilinear', align_corners=False)
                if self.skip_connection is not None:
                    x = self.skip_connection(x, sf)
                else:
                    x = torch.cat([x, sf], dim=1)
                x = self.fusions[i](x)
            for blk in self.swin_stages[i]:
                x = blk(x)

        # 4. Crop output back to the spatial implied by the original bottleneck.
        out_h = orig_h * (2 ** self.num_stages)
        out_w = orig_w * (2 ** self.num_stages)
        x = x[:, :, :out_h, :out_w].contiguous()
        return x
