"""SDI (Scale-Diverse Integration) skip — adapted from U-Net V2 (ISBI 2025).

Adapted from: https://github.com/yaoppeng/U-Net_v2
Paper: U-Net V2: Rethinking the Skip Connections of U-Net for Medical
       Image Segmentation (ISBI 2025, Peng et al.)

The original SDI module takes ALL encoder features {f1, f2, f3, f4} and an
anchor feature, spatial-aligns every encoder feature to the anchor's spatial
size, applies 3×3 conv to each, and fuses them via element-wise
multiplication:

    ans = ones
    for i, fi in enumerate(encoder_features):
        fi_aligned = spatial_align(fi, anchor_size)
        ans = ans * conv3x3(fi_aligned)

This is a **global** operation — it needs all encoder scales at once.

Adapted to the framework's per-pair skip interface:
    1. Project ``decoder_feat`` and ``skip_feat`` to a unified channel dim
       (``max(decoder_ch, skip_ch)``) via 1×1 conv.
    2. Apply **CBAM** (channel + spatial attention) to the skip feature —
       this mirrors the U-Net V2 encoder-side CBAM that preprocesses each
       scale before SDI fusion.
    3. Apply two 3×3 convs (matching SDI's conv structure) to the decoder
       and attended skip, respectively.
    4. **Multiplicative fusion**: ``out = conv_d(d) * conv_s(s_attended)``,
       capturing the cross-scale interaction that is SDI's core innovation.
    5. Final 3×3 conv + BN + ReLU to produce the refined output.

Output channel count: ``max(decoder_ch, skip_ch)`` (unified dimension).
"""
# Source: https://github.com/yaoppeng/U-Net_v2

import torch
import torch.nn as nn
import torch.nn.functional as F
from medseg.registry import SKIP_REGISTRY


class _ChannelAttention(nn.Module):
    """CBAM channel attention: shared MLP on GAP + GMP pools."""

    def __init__(self, channels, ratio=16):
        super().__init__()
        hidden = max(channels // ratio, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class _SpatialAttention(nn.Module):
    """CBAM spatial attention: 7×7 conv on channel-pooled features."""

    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(out))


@SKIP_REGISTRY.register("sdi")
class SDISkip(nn.Module):
    """SDI (Scale-Diverse Integration) skip connection.

    Adapts the cross-scale multiplicative fusion of U-Net V2 to the
    framework's per-pair interface.

    Args:
        ratio: Channel attention reduction ratio.
        spatial_kernel: Spatial attention kernel size (3 or 7).
    """

    def __init__(self, ratio: int = 16, spatial_kernel: int = 7, **kwargs):
        super().__init__()
        self.ratio = ratio
        self.spatial_kernel = spatial_kernel
        # Lazily-built submodules keyed by (decoder_ch, skip_ch)
        self._cache: dict = {}

    def get_out_channels(self, decoder_ch: int, skip_ch: int) -> int:
        return max(decoder_ch, skip_ch)

    def _build(self, decoder_ch: int, skip_ch: int, device):
        """Lazily build layers for a (decoder_ch, skip_ch) pair."""
        key = (decoder_ch, skip_ch, str(device))
        if key in self._cache:
            return self._cache[key]

        unified = max(decoder_ch, skip_ch)

        # Project both to unified channels
        dec_proj = (nn.Conv2d(decoder_ch, unified, 1, bias=False)
                    if decoder_ch != unified else nn.Identity()).to(device)
        skip_proj = (nn.Conv2d(skip_ch, unified, 1, bias=False)
                     if skip_ch != unified else nn.Identity()).to(device)

        # CBAM on skip feature (mirrors U-Net V2's encoder-side attention)
        ca = _ChannelAttention(unified, ratio=self.ratio).to(device)
        sa = _SpatialAttention(kernel_size=self.spatial_kernel).to(device)

        # Two 3×3 convs — matching SDI's conv structure
        conv_d = nn.Sequential(
            nn.Conv2d(unified, unified, 3, 1, 1, bias=False),
            nn.BatchNorm2d(unified),
            nn.ReLU(inplace=True),
        ).to(device)
        conv_s = nn.Sequential(
            nn.Conv2d(unified, unified, 3, 1, 1, bias=False),
            nn.BatchNorm2d(unified),
            nn.ReLU(inplace=True),
        ).to(device)

        # Final 3×3 conv for output refinement
        out_conv = nn.Sequential(
            nn.Conv2d(unified, unified, 3, 1, 1, bias=False),
            nn.BatchNorm2d(unified),
            nn.ReLU(inplace=True),
        ).to(device)

        mod = nn.ModuleDict({
            "dec_proj": dec_proj,
            "skip_proj": skip_proj,
            "ca": ca,
            "sa": sa,
            "conv_d": conv_d,
            "conv_s": conv_s,
            "out_conv": out_conv,
        })
        safe_name = (f"_sdi_{decoder_ch}_{skip_ch}_"
                     f"{str(device).replace(':', '_')}")
        setattr(self, safe_name, mod)
        self._cache[key] = mod
        return mod

    def forward(self, decoder_feat: torch.Tensor,
                skip_feat: torch.Tensor) -> torch.Tensor:
        # Spatial align skip to decoder if needed
        if skip_feat.shape[2:] != decoder_feat.shape[2:]:
            skip_feat = F.interpolate(
                skip_feat, size=decoder_feat.shape[2:],
                mode='bilinear', align_corners=False
            )

        dec_ch = decoder_feat.shape[1]
        skip_ch = skip_feat.shape[1]
        mod = self._build(dec_ch, skip_ch, decoder_feat.device)

        # Project both to unified channels
        d = mod["dec_proj"](decoder_feat)
        s = mod["skip_proj"](skip_feat)

        # CBAM attention on skip (channel then spatial)
        s_attended = mod["ca"](s) * s
        s_attended = mod["sa"](s_attended) * s_attended

        # Multiplicative fusion (core SDI operation)
        # conv_d(d) * conv_s(s_attended) captures cross-scale interaction
        fused = mod["conv_d"](d) * mod["conv_s"](s_attended)

        # Final refinement
        return mod["out_conv"](fused)
