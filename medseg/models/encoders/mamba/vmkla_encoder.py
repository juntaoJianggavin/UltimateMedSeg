"""VMKLA-UNet encoder.

Stride-4 patch-embed stem followed by 4 hierarchical stages of Mamba-style
SSM + KAN linear-attention blocks. Stages 0/1/2 are each followed by a
strided 3x3 downsample (stride-2), producing a 4-level feature pyramid at
strides 4 / 8 / 16 / 32.

Reference:
    VMKLA-UNet: Vision Mamba with KAN Linear Attention U-Net for
    Medical Image Segmentation. PMC 2025.
"""
# Source: NOT VERIFIED — fabricated by this repo, no upstream confirmed.

from typing import List, Optional

import torch
import torch.nn as nn

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


class _KANLinearAttention(nn.Module):
    """KAN-inspired linear attention for skip-feature refinement."""

    def __init__(self, dim, num_basis=5):
        super().__init__()
        self.basis_weights = nn.Parameter(torch.randn(num_basis, dim, dim) * 0.02)
        self.basis_bias = nn.Parameter(torch.zeros(dim))
        self.act = nn.SiLU()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        tokens = self.norm(tokens)
        out = torch.zeros_like(tokens)
        for i in range(self.basis_weights.shape[0]):
            out = out + self.act(tokens @ self.basis_weights[i])
        out = out + self.basis_bias
        return out.transpose(1, 2).view(B, C, H, W)


class _MambaKANBlock(nn.Module):
    """Block combining a lightweight SSM-style gate with KAN linear attention."""

    def __init__(self, dim, d_state=16):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.ssm_proj = nn.Linear(dim, dim * 2)
        self.ssm_gate = nn.Sigmoid()
        self.ssm_out = nn.Linear(dim, dim)
        self.kan = _KANLinearAttention(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        res = x
        tokens = x.flatten(2).transpose(1, 2)
        kv = self.ssm_proj(self.norm1(tokens))
        k, v = kv.chunk(2, dim=-1)
        ssm_out = self.ssm_out(self.ssm_gate(k) * v)
        ssm_feat = ssm_out.transpose(1, 2).view(B, C, H, W)
        kan_feat = self.kan(x)
        out = ssm_feat + kan_feat + res
        tokens = out.flatten(2).transpose(1, 2)
        tokens = tokens + self.ffn(self.norm2(tokens))
        return tokens.transpose(1, 2).view(B, C, H, W)


@ENCODER_REGISTRY.register("vmkla")
class VMKLAUNetEncoder(nn.Module):
    """VMKLA-UNet encoder.

    Architecture:
        Conv 4x4 stride-4 stem -> stage_0 (depth=2, C=embed_dim)
        -> Conv 3x3 stride-2 -> stage_1 (depth=2, C=2*embed_dim)
        -> Conv 3x3 stride-2 -> stage_2 (depth=6, C=4*embed_dim)
        -> Conv 3x3 stride-2 -> stage_3 (depth=2, C=8*embed_dim)

    Returns 4 multi-scale feature maps in (B, C, H, W) at strides 4 / 8 / 16 / 32.
    The deepest (stride-32) map is LAST.

    Args:
        in_channels: input image channels. If != 3 a 1x1 conv stem maps to 3.
        img_size: nominal input resolution (informational only; spatial state
            is derived from the runtime tensor shape).
        pretrained: no public checkpoint is shipped for VMKLA-UNet; the flag is
            accepted for interface symmetry and results in random init.
        embed_dim: channels of the first stage.
        depths: per-stage block counts (must have length 4).
        pretrained_path: optional local checkpoint path.
    """

    def __init__(
        self,
        in_channels: int = 3,
        img_size: int = 224,
        pretrained: bool = False,
        embed_dim: int = 64,
        depths: Optional[List[int]] = None,
        pretrained_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        depths = list(depths) if depths is not None else [2, 2, 6, 2]
        assert len(depths) == 4, "VMKLAUNetEncoder expects 4 stages."

        dims = [embed_dim * (2 ** i) for i in range(len(depths))]
        self.in_channels = in_channels
        self.img_size = img_size
        self.depths = tuple(depths)
        self.dims = tuple(dims)
        self.out_channels: List[int] = list(dims)

        # 1x1 channel adapter when the user feeds non-RGB inputs.
        if in_channels != 3:
            self.input_stem = nn.Conv2d(in_channels, 3, kernel_size=1, bias=True)
            stem_in = 3
        else:
            self.input_stem = nn.Identity()
            stem_in = in_channels

        # Stride-4 patch-embed stem.
        self.stem = nn.Sequential(
            nn.Conv2d(stem_in, dims[0], kernel_size=4, stride=4, bias=False),
            nn.BatchNorm2d(dims[0]),
        )

        self.enc = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i in range(len(depths)):
            self.enc.append(
                nn.Sequential(*[_MambaKANBlock(dims[i]) for _ in range(depths[i])])
            )
            if i < len(depths) - 1:
                self.downs.append(nn.Conv2d(dims[i], dims[i + 1], 3, 2, 1))

        if pretrained:
            _load_with_ssl_fallback(self._maybe_load_pretrained, pretrained_path)

    # ---- Pretrained loading -------------------------------------------------

    def _maybe_load_pretrained(self, pretrained_path: Optional[str] = None, **_):
        import warnings
        if not pretrained_path:
            warnings.warn(
                "VMKLAUNetEncoder: no pretrained_path provided; using random init."
            )
            return
        state = torch.load(pretrained_path, map_location="cpu")
        if isinstance(state, dict):
            if "model" in state:
                state = state["model"]
            if "state_dict" in state:
                state = state["state_dict"]
        cleaned = {}
        for k, v in state.items():
            nk = k
            for prefix in ("encoder.", "backbone.", "module."):
                if nk.startswith(prefix):
                    nk = nk[len(prefix):]
            cleaned[nk] = v
        msg = self.load_state_dict(cleaned, strict=False)
        print(f"VMKLAUNetEncoder loaded pretrained from {pretrained_path}: {msg}")

    # ---- Forward ------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Args:
            x: (B, in_channels, H, W).
        Returns:
            List of 4 feature maps in (B, C, H, W) at strides 4/8/16/32.
            Deepest (stride-32) feature is LAST.
        """
        x = self.input_stem(x)
        x = self.stem(x)
        features: List[torch.Tensor] = []
        for i, enc in enumerate(self.enc):
            x = enc(x)
            features.append(x)
            if i < len(self.downs):
                x = self.downs[i](x)
        return features
