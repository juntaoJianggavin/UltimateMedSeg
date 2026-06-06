"""Group Aggregation Bridge (GAB) skip — adapted from EGE-UNet (MICCAI 2023 W).

Original GroupAggregationBridge fuses (xh, xl, mask) — high-res feature,
low-res feature, and a 1-channel segmentation mask used as group-wise
guidance. It splits both features into 4 channel groups, concatenates the
mask with each group, and applies 4 dilated grouped convs (dilations
[1, 2, 5, 7]) before a 1x1 tail conv.

Adapted to the framework's per-pair skip interface:
    - decoder_feat plays the role of xh (deeper / lower-res, but already
      upsampled to skip size by the framework's decoder)
    - skip_feat plays the role of xl (encoder skip at the current scale)
    - mask is OPTIONAL: see `Mask injection` below.

## Mask injection

The standard `model_builder` calls `skip(d, s)` with no mask, so by default
mask=zeros (i.e. no guidance — equivalent to setting all chunks to use the
same neutral prior; behavior matches "ablated GAB without mask"). To
actually use the mask path (EGE-UNet's deep-supervision flow where masks
come from per-stage seg heads), use ONE of the following call patterns:

1. `set_mask(mask)` before each forward, optionally `clear_mask()` after:
   ```python
   skip.set_mask(mask_from_prev_stage)  # (B, 1, H, W) or any (B, k, H, W)
   out = skip(decoder_feat, skip_feat)
   skip.clear_mask()  # optional but recommended
   ```

2. Context manager (auto-clears, safe under exceptions):
   ```python
   with skip.mask_ctx(mask):
       out = skip(decoder_feat, skip_feat)
   ```

3. Direct call (bypass framework interface — useful for custom decoders):
   ```python
   out = skip.forward_with_mask(decoder_feat, skip_feat, mask)
   ```

Both `decoder_feat` and `skip_feat` channel counts can be any value; the
internal Conv2d/LayerNorm are lazily built per `(decoder_ch, skip_ch)` pair
on first forward.

Distinct from `cab_skip` (SE channel attention only): this module uses
*multi-dilation grouped convolutions* for fusion — no channel attention.
"""
# Source: https://github.com/JCruan519/EGE-UNet

from __future__ import annotations
from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from medseg.registry import SKIP_REGISTRY


class _LayerNorm(nn.Module):
    """LayerNorm for channels-first 2D tensors (matches the EGE-UNet helper)."""

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


def _even_split(C: int, n: int) -> list[int]:
    """Split C channels into n groups as evenly as possible (matching torch.chunk semantics)."""
    base = C // n
    rem = C - base * n
    # torch.chunk distributes the remainder into the FIRST `rem` chunks (size base+1)
    # Actually torch.chunk distributes the remainder into the LAST chunk(s) but as one
    # bigger one. For our purposes we mirror that: first n-1 of size base+1 each up to
    # rem, then remaining of size base.
    # To stay 1:1 compatible with torch.chunk, just call torch.chunk and read sizes.
    # Here we just compute matching sizes.
    sizes = [base + (1 if i < rem else 0) for i in range(n)]
    return sizes


@SKIP_REGISTRY.register("gab")
class GABSkip(nn.Module):
    """Group Aggregation Bridge skip (per-pair adaptation of EGE-UNet GAB).

    Output channel count: skip_ch (xl-dim, matching original).
    """

    DILATIONS = (1, 2, 5, 7)
    K_SIZE = 3
    N_CHUNKS = 4

    def __init__(self, dilations=None, n_chunks: int = 4, **kwargs):
        super().__init__()
        self.dilations = tuple(dilations) if dilations is not None else self.DILATIONS
        self.n_chunks = n_chunks
        self._mask: Optional[torch.Tensor] = None  # external injection slot
        self._cache: dict = {}  # (xh_ch, xl_ch) -> nn.ModuleDict

    # ──────────────────────────────────────────────────────────────────────
    # Public mask-injection interface
    # ──────────────────────────────────────────────────────────────────────

    def get_out_channels(self, decoder_ch: int, skip_ch: int) -> int:
        return skip_ch

    def set_mask(self, mask: Optional[torch.Tensor]) -> None:
        """Provide a (B, 1, H, W) or (B, k, H, W) mask for the next forward.
        Pass None to clear. Multi-channel masks are mean-reduced to 1 channel.
        """
        self._mask = mask

    def clear_mask(self) -> None:
        self._mask = None

    @contextmanager
    def mask_ctx(self, mask: Optional[torch.Tensor]):
        """Context manager — sets mask, yields, restores prior state."""
        prev = self._mask
        self._mask = mask
        try:
            yield
        finally:
            self._mask = prev

    def forward_with_mask(self, decoder_feat: torch.Tensor, skip_feat: torch.Tensor,
                          mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Direct 3-arg call (bypasses framework's 2-arg interface)."""
        with self.mask_ctx(mask):
            return self.forward(decoder_feat, skip_feat)

    # ──────────────────────────────────────────────────────────────────────
    # Internal builders
    # ──────────────────────────────────────────────────────────────────────

    def _build_modules(self, xh_ch: int, xl_ch: int, device, chunk_sizes: list[int]) -> nn.ModuleDict:
        """Lazily build conv stack for a given (xh_ch, xl_ch). Built once and cached."""
        key = (xh_ch, xl_ch, str(device))
        if key in self._cache:
            return self._cache[key]

        pre_project = nn.Conv2d(xh_ch, xl_ch, 1).to(device)

        groups = nn.ModuleList()
        # For each chunk i, expected_in = chunk_sizes[i] (from xh) + chunk_sizes[i] (from xl) + 1 (mask)
        # = 2*chunk_sizes[i] + 1
        for i, d in enumerate(self.dilations):
            csize = chunk_sizes[i]
            in_ch = 2 * csize + 1
            pad = (self.K_SIZE + (self.K_SIZE - 1) * (d - 1)) // 2
            groups.append(nn.Sequential(
                _LayerNorm(in_ch),
                nn.Conv2d(in_ch, in_ch, kernel_size=self.K_SIZE,
                          stride=1, padding=pad, dilation=d, groups=in_ch),
            ).to(device))

        tail_in_ch = sum(2 * cs + 1 for cs in chunk_sizes)
        tail = nn.Sequential(
            _LayerNorm(tail_in_ch),
            nn.Conv2d(tail_in_ch, xl_ch, 1),
        ).to(device)

        mod = nn.ModuleDict({
            "pre_project": pre_project,
            "groups": groups,
            "tail": tail,
        })
        # Register as a submodule so parameters() picks them up
        safe_name = f"_gab_{xh_ch}_{xl_ch}_{str(device).replace(':', '_')}"
        setattr(self, safe_name, mod)
        self._cache[key] = mod
        return mod

    # ──────────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────────

    def forward(self, decoder_feat: torch.Tensor, skip_feat: torch.Tensor) -> torch.Tensor:
        xh, xl = decoder_feat, skip_feat
        # Align spatial dims to xl (skip)
        if xh.shape[2:] != xl.shape[2:]:
            xh = F.interpolate(xh, size=xl.shape[2:], mode="bilinear", align_corners=True)

        B, _, H, W = xl.shape
        # Compute the chunk sizes (these must match for xh and xl, because both
        # are projected/native at the same xl_ch count after pre_project).
        xl_ch = xl.shape[1]
        chunk_sizes = _even_split(xl_ch, self.n_chunks)

        mod = self._build_modules(xh.shape[1], xl_ch, xh.device, chunk_sizes)
        xh = mod["pre_project"](xh)  # now xh has xl_ch channels too

        # Build / sanitize the mask. Default is zeros (no guidance).
        if self._mask is not None:
            mask = self._mask
            # Move device if needed
            if mask.device != xl.device:
                mask = mask.to(xl.device)
            # Reduce multi-channel mask to 1 channel
            if mask.dim() == 4 and mask.shape[1] != 1:
                mask = mask.mean(dim=1, keepdim=True)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)
            # Spatial align
            if mask.shape[2:] != xl.shape[2:]:
                mask = F.interpolate(mask, size=xl.shape[2:], mode="bilinear",
                                     align_corners=True)
            # Cast dtype
            mask = mask.to(xl.dtype)
            # Batch broadcasting
            if mask.shape[0] != B:
                if mask.shape[0] == 1:
                    mask = mask.expand(B, -1, -1, -1)
                else:
                    raise RuntimeError(
                        f"GABSkip: mask batch={mask.shape[0]} does not match feature batch={B}"
                    )
        else:
            mask = torch.zeros((B, 1, H, W), device=xl.device, dtype=xl.dtype)

        # Chunk both features (use torch.split with explicit sizes to ensure exact match)
        xh_chunks = list(torch.split(xh, chunk_sizes, dim=1))
        xl_chunks = list(torch.split(xl, chunk_sizes, dim=1))

        outs = []
        for i, g in enumerate(mod["groups"]):
            x_in = torch.cat([xh_chunks[i], xl_chunks[i], mask], dim=1)
            outs.append(g(x_in))

        x = torch.cat(outs, dim=1)
        return mod["tail"](x)
