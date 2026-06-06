"""Deformable convolution skip connection."""
# Source: INTERNAL — framework adaptation (this repo).

import torch
import torch.nn as nn
import torch.nn.functional as F
from medseg.registry import SKIP_REGISTRY

try:
    from torchvision.ops import DeformConv2d
    _HAS_DEFORM = True
except Exception:  # pragma: no cover - depends on torchvision availability
    DeformConv2d = None
    _HAS_DEFORM = False


@SKIP_REGISTRY.register("deformable")
class DeformableSkip(nn.Module):
    """Deformable-conv skip.

    Applies a 3x3 deformable convolution (with a sibling offset-prediction
    conv) to the skip features, then concatenates with the decoder features.
    If torchvision's DeformConv2d is unavailable, falls back to a regular
    3x3 Conv2d so the module still functions.
    """

    def __init__(self, kernel_size=3, padding=1, **kwargs):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        # Lazily-built submodules keyed by skip channel count.
        self._offset_convs = nn.ModuleDict()
        self._deform_convs = nn.ModuleDict()
        self._fallback_convs = nn.ModuleDict()

    def get_out_channels(self, decoder_ch, skip_ch):
        return decoder_ch + skip_ch

    def _key(self, c):
        return f"c{c}"

    def _build(self, skip_ch, device):
        key = self._key(skip_ch)
        if key in self._deform_convs or key in self._fallback_convs:
            return
        k = self.kernel_size
        p = self.padding
        if _HAS_DEFORM:
            # Offset conv predicts 2 * k * k offset channels per spatial loc.
            offset_conv = nn.Conv2d(
                skip_ch, 2 * k * k,
                kernel_size=k, padding=p, bias=True,
            )
            # Initialize offsets to zero so initial behaviour matches a
            # plain 3x3 conv on a regular grid.
            nn.init.zeros_(offset_conv.weight)
            if offset_conv.bias is not None:
                nn.init.zeros_(offset_conv.bias)
            deform_conv = DeformConv2d(
                skip_ch, skip_ch,
                kernel_size=k, padding=p, bias=True,
            )
            self._offset_convs[key] = offset_conv.to(device)
            self._deform_convs[key] = deform_conv.to(device)
        else:
            fallback = nn.Conv2d(
                skip_ch, skip_ch,
                kernel_size=k, padding=p, bias=True,
            )
            self._fallback_convs[key] = fallback.to(device)

    def forward(self, decoder_feat, skip_feat):
        # Spatially align skip to decoder if needed.
        if skip_feat.shape[-2:] != decoder_feat.shape[-2:]:
            skip_feat = F.interpolate(
                skip_feat, size=decoder_feat.shape[-2:],
                mode="bilinear", align_corners=False,
            )

        skip_ch = skip_feat.shape[1]
        self._build(skip_ch, skip_feat.device)
        key = self._key(skip_ch)

        if key in self._deform_convs:
            offsets = self._offset_convs[key](skip_feat)
            skip_out = self._deform_convs[key](skip_feat, offsets)
        else:
            skip_out = self._fallback_convs[key](skip_feat)

        return torch.cat([decoder_feat, skip_out], dim=1)
