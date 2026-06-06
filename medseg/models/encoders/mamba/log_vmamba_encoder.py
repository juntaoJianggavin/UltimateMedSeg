"""LoG-VMamba Encoder.

Standalone encoder extracted from ``medseg.models.networks.mamba.log_vmamba``.

Hierarchical 4-stage backbone:
    - stride-4 patch-embed stem (Conv4x4, stride 4)
    - 4 stages of LoG (Local-Global) blocks
    - 3 strided 3x3 conv downsamples between stages
    - default embed_dim=64 -> out_channels [64, 128, 256, 512]

Returns a multi-scale feature pyramid in (B, C, H, W) format, deepest LAST.
"""
# Source: https://github.com/Oulu-IMEDS/LoG-VMamba

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.registry import ENCODER_REGISTRY


def _load_with_ssl_fallback(load_fn, *args, **kwargs):
    """Try a download/load, falling back to unverified SSL, then random init."""
    import ssl
    import warnings
    try:
        return load_fn(*args, **kwargs)
    except Exception as e1:
        prev = ssl._create_default_https_context
        try:
            ssl._create_default_https_context = ssl._create_unverified_context
            return load_fn(*args, **kwargs)
        except Exception as e2:
            warnings.warn(f"Pretrained download failed ({e2}); using random init.")
            return load_fn(*args, **{**kwargs, 'pretrained': False})
        finally:
            ssl._create_default_https_context = prev


class _LoGBlock(nn.Module):
    """Local-Global block: conv branch + gated SSM-like branch fused.

    Mirrors the block used in the source ``LoGVMamba`` network.
    """

    def __init__(self, dim: int, d_state: int = 16):
        super().__init__()
        # Local branch: depthwise 3x3 + pointwise.
        self.local_norm = nn.BatchNorm2d(dim)
        self.local_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1),
        )
        # Global branch: gated SSM-style projection on tokens.
        self.global_norm = nn.LayerNorm(dim)
        self.global_proj = nn.Linear(dim, dim * 2)
        self.global_gate = nn.Sigmoid()
        self.global_out = nn.Linear(dim, dim)
        # Fusion.
        self.fuse = nn.Conv2d(dim * 2, dim, 1)
        # Token-wise FFN with residual.
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        res = x
        # Local branch
        local_feat = self.local_conv(self.local_norm(x))
        # Global branch
        tokens = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        kv = self.global_proj(self.global_norm(tokens))
        k, v = kv.chunk(2, dim=-1)
        global_out = self.global_out(self.global_gate(k) * v)
        global_feat = global_out.transpose(1, 2).view(B, C, H, W)
        # Fuse local + global
        fused = self.fuse(torch.cat([local_feat, global_feat], dim=1))
        # FFN with residual on tokens
        tokens = fused.flatten(2).transpose(1, 2)
        tokens = tokens + self.ffn(tokens)
        return tokens.transpose(1, 2).view(B, C, H, W) + res


@ENCODER_REGISTRY.register("log_vmamba")
class LoGVMambaEncoder(nn.Module):
    """LoG-VMamba encoder.

    Architecture:
        - Optional 1x1 input stem when ``in_channels != 3``.
        - Conv4x4 stride-4 patch-embed stem -> ``dims[0]`` channels at stride 4.
        - 4 stages of ``_LoGBlock`` with depths ``[2, 2, 6, 2]`` by default.
        - 3 strided 3x3 conv downsamples between stages (strides 8/16/32).

    Returns:
        A list of 4 feature maps in (B, C, H, W) at strides 4/8/16/32.
        The deepest (stride-32) feature is LAST, matching the framework
        convention used by ``BasicEncoder`` and ``MambaPureEncoder``.
    """

    def __init__(
        self,
        in_channels: int = 3,
        img_size: int = 224,
        pretrained: bool = False,
        embed_dim: int = 64,
        depths: Optional[List[int]] = None,
        d_state: int = 16,
        pretrained_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        depths = list(depths) if depths is not None else [2, 2, 6, 2]
        assert len(depths) == 4, "LoGVMambaEncoder expects 4 stages."

        self.in_channels = in_channels
        self.img_size = img_size
        self.depths = tuple(depths)
        dims = [embed_dim * (2 ** i) for i in range(len(depths))]
        self.dims = tuple(dims)
        self.out_channels: List[int] = list(dims)

        # Optional 1x1 stem to coerce arbitrary input channels to 3.
        if in_channels != 3:
            self.input_stem = nn.Conv2d(in_channels, 3, kernel_size=1, bias=True)
            stem_in = 3
        else:
            self.input_stem = nn.Identity()
            stem_in = in_channels

        # Stride-4 patch-embed stem.
        self.stem = nn.Sequential(
            nn.Conv2d(stem_in, dims[0], 4, 4, bias=False),
            nn.BatchNorm2d(dims[0]),
        )

        # Stages + downsamples (3 strided downsamples between 4 stages).
        self.enc = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i in range(len(depths)):
            self.enc.append(
                nn.Sequential(*[_LoGBlock(dims[i], d_state=d_state) for _ in range(depths[i])])
            )
            if i < len(depths) - 1:
                self.downs.append(nn.Conv2d(dims[i], dims[i + 1], 3, 2, 1))

        if pretrained:
            _load_with_ssl_fallback(self._maybe_load_pretrained, pretrained_path)

    # ---- Pretrained loading -------------------------------------------------

    def _maybe_load_pretrained(self, pretrained_path: Optional[str] = None, **_):
        """Best-effort local checkpoint load. No-op without an explicit path."""
        import warnings
        if not pretrained_path:
            warnings.warn(
                "LoGVMambaEncoder: no pretrained_path provided; using random init."
            )
            return
        state = torch.load(pretrained_path, map_location='cpu')
        if isinstance(state, dict):
            if 'model' in state:
                state = state['model']
            if 'state_dict' in state:
                state = state['state_dict']
        cleaned = {}
        for k, v in state.items():
            nk = k
            for pref in ('encoder.', 'backbone.', 'module.'):
                if nk.startswith(pref):
                    nk = nk[len(pref):]
            cleaned[nk] = v
        msg = self.load_state_dict(cleaned, strict=False)
        print(f"LoGVMambaEncoder loaded pretrained from {pretrained_path}: {msg}")

    # ---- Forward ------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Args:
            x: (B, in_channels, H, W).
        Returns:
            List of 4 feature maps in (B, C, H, W). Deepest is LAST.
        """
        x = self.input_stem(x)
        x = self.stem(x)  # stride 4

        features: List[torch.Tensor] = []
        for i, stage in enumerate(self.enc):
            x = stage(x)
            features.append(x)
            if i < len(self.downs):
                x = self.downs[i](x)
        return features
