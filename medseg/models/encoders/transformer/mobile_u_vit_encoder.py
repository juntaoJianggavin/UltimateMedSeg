"""Mobile-U-ViT encoder.

Extracted encoder-only stages from ``medseg.models.networks.transformer.mobile_u_vit``.

Architecture:
    - Large-kernel 7x7 stride-2 conv stem -> /2, embed_dims[0]
    - 4 encoder stages of stacked ``_MobileBlock`` (large-kernel DWConv + FFN)
      with stride-2 3x3 downsamples between stages.
    - ViT bottleneck (2 lightweight ``_ViTBlock`` MHSA blocks) at the deepest
      stage (refines the deepest feature in-place).

Returns 4 multi-scale features (deepest LAST), matching the framework
convention. For ``embed_dims=[32, 64, 128, 256]`` the strides are [2, 4, 8, 16]
and ``self.out_channels = [32, 64, 128, 256]``.

Reference:
    Mobile U-ViT: Revisiting large kernel and U-shaped ViT for efficient
    medical image segmentation. ACM MM 2025.
"""
# Source: https://github.com/FengheTan9/Mobile-U-ViT

import torch
import torch.nn as nn
from typing import List, Optional

from medseg.registry import ENCODER_REGISTRY


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
            warnings.warn(f"Pretrained download failed ({e2}); using random init.")
            return load_fn(*args, **{**kwargs, 'pretrained': False})
        finally:
            ssl._create_default_https_context = prev


class _LargeKernelDWConv(nn.Module):
    """Large kernel depthwise separable convolution (DW 7x7 + PW 1x1)."""

    def __init__(self, dim: int, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        self.dw = nn.Conv2d(dim, dim, kernel_size, 1, pad, groups=dim, bias=False)
        self.pw = nn.Conv2d(dim, dim, 1, bias=False)
        self.norm = nn.BatchNorm2d(dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.pw(self.dw(x))))


class _MobileBlock(nn.Module):
    """Mobile building block: large-kernel DWConv + FFN with residuals."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.lk = _LargeKernelDWConv(dim, kernel_size=7)
        self.norm2 = nn.BatchNorm2d(dim)
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.lk(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class _ViTBlock(nn.Module):
    """Lightweight ViT block (MHSA + FFN) operating on BCHW tensors.

    Spatial dimensions are derived from the runtime tensor shape so any input
    resolution is supported.
    """

    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        q = self.norm1(tokens)
        tokens = tokens + self.attn(q, q, q)[0]
        tokens = tokens + self.ffn(self.norm2(tokens))
        return tokens.transpose(1, 2).view(B, C, H, W)


@ENCODER_REGISTRY.register("mobile_u_vit")
class MobileUViTEncoder(nn.Module):
    """Mobile-U-ViT encoder: LK-DWConv MobileBlocks + ViT bottleneck.

    Args:
        in_channels: Input channels. If != 3, a 1x1 conv projects to 3.
        img_size: Spatial size hint (kept for API uniformity; the module is
            fully resolution-agnostic).
        pretrained: No public Mobile-U-ViT checkpoint is available; kept for
            API uniformity.
        embed_dims: Per-stage channel dims (default ``[32, 64, 128, 256]``).
        depths: Number of ``_MobileBlock`` per stage (default ``[2, 2, 2, 2]``).
        use_vit_bottleneck: If True, refine the deepest feature with two
            ``_ViTBlock`` MHSA blocks (default True).

    Forward returns a list of 4 BCHW tensors at strides [2, 4, 8, 16]
    (high-res first, deepest LAST), with channels equal to ``out_channels``.
    """

    def __init__(
        self,
        in_channels: int = 3,
        img_size: int = 224,
        pretrained: bool = False,
        embed_dims: Optional[List[int]] = None,
        depths: Optional[List[int]] = None,
        use_vit_bottleneck: bool = True,
        **kwargs,
    ):
        super().__init__()
        embed_dims = list(embed_dims) if embed_dims is not None else [32, 64, 128, 256]
        depths = list(depths) if depths is not None else [2, 2, 2, 2]
        assert len(embed_dims) == len(depths), \
            f"embed_dims ({len(embed_dims)}) and depths ({len(depths)}) must match"
        self.img_size = img_size
        self.embed_dims = embed_dims
        self.depths = depths

        # Optional 1x1 stem to project non-RGB inputs to 3 channels so the
        # large-kernel stem operates on the original spec.
        if in_channels != 3:
            self.input_proj = nn.Conv2d(in_channels, 3, kernel_size=1, bias=True)
            stem_in = 3
        else:
            self.input_proj = nn.Identity()
            stem_in = 3

        # Stem: large kernel 7x7 stride-2 conv -> /2
        self.stem = nn.Sequential(
            nn.Conv2d(stem_in, embed_dims[0], 7, 2, 3, bias=False),
            nn.BatchNorm2d(embed_dims[0]),
            nn.GELU(),
        )

        # Encoder stages + (N-1) downsamples between them.
        self.enc_stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i in range(len(embed_dims)):
            blocks = nn.Sequential(*[_MobileBlock(embed_dims[i]) for _ in range(depths[i])])
            self.enc_stages.append(blocks)
            if i < len(embed_dims) - 1:
                self.downsamples.append(nn.Sequential(
                    nn.Conv2d(embed_dims[i], embed_dims[i + 1], 3, 2, 1, bias=False),
                    nn.BatchNorm2d(embed_dims[i + 1]),
                ))

        # Bottleneck applied IN-PLACE on the deepest stage (no extra downsample).
        if use_vit_bottleneck:
            self.bottleneck = nn.Sequential(
                _ViTBlock(embed_dims[-1], num_heads=4),
                _ViTBlock(embed_dims[-1], num_heads=4),
            )
        else:
            self.bottleneck = nn.Sequential(
                _MobileBlock(embed_dims[-1]),
                _MobileBlock(embed_dims[-1]),
            )

        self.out_channels: List[int] = list(embed_dims)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.input_proj(x)
        x = self.stem(x)

        features: List[torch.Tensor] = []
        for i, stage in enumerate(self.enc_stages):
            x = stage(x)
            if i == len(self.enc_stages) - 1:
                # Refine the deepest feature with the ViT bottleneck before
                # exposing it to the decoder.
                x = self.bottleneck(x)
            features.append(x)
            if i < len(self.downsamples):
                x = self.downsamples[i](x)
        # features[0] = highest-res / shallowest, features[-1] = deepest.
        return features
