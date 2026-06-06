"""SK-VM++ Skip — Mamba-assisted skip connection (BSPC 2025, Fang et al.).

Adapted from: https://github.com/wurenkai/SK-VMPlusPlus
Paper: SK-VM++: Mamba assists skip-connections for medical image segmentation
       (Biomedical Signal Processing and Control, 2025)

The core component is **PVMLayer** (Pyramid Vision Mamba Layer):
  1. Splits channels into ``n_chunks`` (default 4) along channel dim
  2. Each chunk passes through an SS2D (4-direction Selective Scan 2D)
     with a learnable skip-scale residual
  3. Re-concatenates, applies LayerNorm + linear projection
  4. Adds residual to the skip feature

Per-pair skip interface:
  1. Project decoder_feat to skip_ch via 1×1 conv
  2. Combine: ``|proj_decoder + skip|``
  3. Refine through PVMLayer
  4. Add residual to skip_feat

Uses the project's ``SS2D`` (from ``vmunet_encoder``) — requires ``mamba_ssm``
for the selective scan CUDA kernel.
"""
# Source: https://github.com/wurenkai/SK-VMPlusPlus

import torch
import torch.nn as nn
import torch.nn.functional as F
from medseg.registry import SKIP_REGISTRY
from medseg.models.encoders.mamba.vmunet_encoder import SS2D, DropPath


class _PVMLayer(nn.Module):
    """Pyramid Vision Mamba Layer — core SSM block of SK-VM++.

    Splits channels into ``n_chunks`` chunks, processes each with an
    independent SS2D block, then reassembles and projects.

    Input / output: ``(B, C, H, W)`` with same channel count.
    Channel count ``C`` must be divisible by ``n_chunks``.
    """

    def __init__(self, channels: int, n_chunks: int = 4,
                 d_state: int = 16, d_conv: int = 4, expand: int = 2,
                 drop_path: float = 0.0):
        super().__init__()
        assert channels % n_chunks == 0, (
            f"PVMLayer requires channels={channels} divisible by "
            f"n_chunks={n_chunks}"
        )
        self.n_chunks = n_chunks
        self.chunk_dim = channels // n_chunks
        self.norm = nn.LayerNorm(channels)
        self.proj = nn.Linear(channels, channels)
        self.skip_scale = nn.Parameter(torch.ones(1))
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # One SS2D per chunk
        self.mambas = nn.ModuleList([
            SS2D(self.chunk_dim, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_chunks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) -> (B, C, H, W)"""
        B, C, H, W = x.shape
        residual = x

        # Flatten -> (B, L, C), L = H * W
        x_flat = x.reshape(B, C, H * W).transpose(1, 2)
        x_norm = self.norm(x_flat)  # (B, L, C)

        # Split into n_chunks along channel dim
        chunks = torch.chunk(x_norm, self.n_chunks, dim=-1)  # n x (B, L, C/n)

        outs = []
        for chunk, mamba in zip(chunks, self.mambas):
            # Reshape to (B, H, W, C/n) for SS2D
            chunk_2d = chunk.transpose(1, 2).reshape(B, H, W, self.chunk_dim)
            # SS2D forward: (B, H, W, C/n) -> (B, H, W, C/n)
            out_2d = mamba(chunk_2d)
            # Back to (B, L, C/n)
            out_flat = out_2d.reshape(B, H * W, self.chunk_dim)
            outs.append(out_flat + self.skip_scale * chunk)

        # Concatenate: (B, L, C)
        out = torch.cat(outs, dim=-1)
        # LayerNorm + project
        out = self.norm(out)
        out = self.proj(out)
        # Reshape back to 2D
        out = out.transpose(1, 2).reshape(B, C, H, W)

        return self.drop_path(out) + residual


@SKIP_REGISTRY.register("skvmpp")
class SKVMPlusPlusSkip(nn.Module):
    """SK-VM++ Mamba-assisted skip connection.

    Fuses decoder and encoder skip features through PVMLayer, then adds
    residual to the skip feature.

    Args:
        n_chunks: Number of channel chunks for parallel SS2D processing.
        d_state: SSM state expansion factor.
        d_conv: Local convolution kernel size in SSM.
        expand: SSM block expansion factor.
        drop_path: Stochastic depth drop rate.
    """

    def __init__(self, n_chunks: int = 4, d_state: int = 16,
                 d_conv: int = 3, expand: int = 2,
                 drop_path: float = 0.0, **kwargs):
        super().__init__()
        self.n_chunks = n_chunks
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.drop_path = drop_path
        # Lazily-built submodules keyed by skip channel count
        self._cache: dict = {}

    def get_out_channels(self, decoder_ch: int, skip_ch: int) -> int:
        return skip_ch

    def _build(self, decoder_ch: int, skip_ch: int, device):
        """Lazily build layers for a (decoder_ch, skip_ch) pair."""
        key = (decoder_ch, skip_ch, str(device))
        if key in self._cache:
            return self._cache[key]

        proj = nn.Conv2d(decoder_ch, skip_ch, 1, bias=False).to(device)
        # Channel count must be divisible by n_chunks
        # If not, adjust n_chunks to the largest divisor <= self.n_chunks
        n_chunks = self.n_chunks
        while n_chunks > 1 and skip_ch % n_chunks != 0:
            n_chunks -= 1

        pvm = _PVMLayer(skip_ch, n_chunks=n_chunks,
                        d_state=self.d_state, d_conv=self.d_conv,
                        expand=self.expand, drop_path=self.drop_path).to(device)
        bn = nn.BatchNorm2d(skip_ch).to(device)

        mod = nn.ModuleDict({
            "proj": proj,
            "pvm": pvm,
            "bn": bn,
        })
        safe_name = f"_skvm_{decoder_ch}_{skip_ch}_{str(device).replace(':', '_')}"
        setattr(self, safe_name, mod)
        self._cache[key] = mod
        return mod

    def forward(self, decoder_feat: torch.Tensor,
                skip_feat: torch.Tensor) -> torch.Tensor:
        # Align spatial dimensions to skip (encoder) size
        if decoder_feat.shape[2:] != skip_feat.shape[2:]:
            decoder_feat = F.interpolate(
                decoder_feat, size=skip_feat.shape[2:],
                mode='bilinear', align_corners=False
            )

        dec_ch = decoder_feat.shape[1]
        skip_ch = skip_feat.shape[1]
        mod = self._build(dec_ch, skip_ch, decoder_feat.device)

        # Project decoder to skip channels
        proj_d = mod["proj"](decoder_feat)

        # Combine with abs (matching original SK-VM++ formulation)
        combined = torch.abs(proj_d + skip_feat)

        # Refine through PVMLayer (includes residual)
        refined = mod["pvm"](combined)

        # BN + ReLU on residual path
        refined = F.relu(mod["bn"](refined))

        # Add to skip feature
        return skip_feat + refined
