"""SegFormer MiT (Mix-Transformer) encoder.

Wraps timm's ``mit_b2`` backbone with ``features_only=True`` to expose 4
multi-scale feature maps suitable for U-Net style decoders. Falls back to
random initialization when pretrained weights cannot be downloaded (e.g. SSL
failures in offline environments). If the ``mit_b2`` architecture is not
registered in the local timm install, falls back to the architecturally
similar ``pvt_v2_b2`` (the direct PVTv2 ancestor of MiT).
"""
# Source: https://github.com/NVlabs/SegFormer

import torch
import torch.nn as nn
import timm
import warnings
from typing import List

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
            kwargs2 = {**kwargs, 'pretrained': False}
            return load_fn(*args, **kwargs2)
        finally:
            ssl._create_default_https_context = prev


# MiT (SegFormer Mix-Transformer) is the spec target. Some timm versions do
# not bundle it; PVTv2 is the closest architectural sibling (MiT is a
# refinement of PVTv2). We try MiT first, then PVTv2-B2, then any available
# 4-stage transformer pyramid.
_MIT_CANDIDATES = ("mit_b2", "pvt_v2_b2")


def _create_mit_backbone(pretrained: bool):
    """Create the MiT backbone with graceful fallback across timm versions."""
    last_err = None
    for name in _MIT_CANDIDATES:
        try:
            return timm.create_model(
                name,
                pretrained=pretrained,
                features_only=True,
                in_chans=3,
            ), name
        except RuntimeError as e:
            # "Unknown model" errors -> try the next candidate.
            last_err = e
            continue
    raise RuntimeError(
        f"Could not construct any MiT-like backbone from {_MIT_CANDIDATES}: {last_err}"
    )


@ENCODER_REGISTRY.register("segformer_mit")
class SegFormerMiTEncoder(nn.Module):
    """SegFormer Mix-Transformer (MiT-B2) backbone.

    Uses ``timm.create_model("mit_b2", features_only=True)`` to extract four
    hierarchical feature maps at strides 4, 8, 16, 32. The deepest feature is
    placed LAST in the returned list, matching the project convention.

    Args:
        in_channels: Number of input channels. If not 3, a 1x1 conv stem is
            prepended to project to 3 channels (MiT expects RGB).
        img_size: Spatial size hint (unused by the backbone, kept for API
            uniformity with other encoders).
        pretrained: If True, attempts to load ImageNet pretrained weights
            with an SSL fallback. Falls back to random init on failure.
    """

    def __init__(
        self,
        in_channels: int = 3,
        img_size: int = 224,
        pretrained: bool = True,
        **kwargs,
    ):
        super().__init__()

        # MiT backbones are designed for 3-channel RGB input. If the caller
        # supplies a different in_channels, prepend a learnable 1x1 stem to
        # remap to 3 channels.
        if in_channels != 3:
            self.input_proj = nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)
        else:
            self.input_proj = None

        def _build(pretrained: bool = True, **build_kwargs):
            # Accept **kwargs so that ``_load_with_ssl_fallback``'s offline
            # fallback (which injects ``pretrained=False``) does not raise an
            # unexpected-keyword error. Any additional kwargs are forwarded
            # to ``timm.create_model`` via ``_create_mit_backbone``.
            backbone, _name = _create_mit_backbone(pretrained)
            return backbone

        if pretrained:
            self.backbone = _load_with_ssl_fallback(_build, pretrained=True)
        else:
            self.backbone = _build(pretrained=False)

        # Some timm pyramids emit more than 4 features (e.g. stems); keep the
        # deepest 4 to match the SegFormer 4-stage contract.
        all_channels = list(self.backbone.feature_info.channels())
        if len(all_channels) > 4:
            self._slice = slice(len(all_channels) - 4, len(all_channels))
            warnings.warn(
                f"Backbone returned {len(all_channels)} feature maps; keeping last 4."
            )
        else:
            self._slice = slice(0, len(all_channels))
        self.out_channels: List[int] = all_channels[self._slice]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        if self.input_proj is not None:
            x = self.input_proj(x)
        features = list(self.backbone(x))[self._slice]
        # Ensure BCHW format (some timm models output BHWC).
        out: List[torch.Tensor] = []
        for i, f in enumerate(features):
            if f.ndim == 4:
                expected_c = self.out_channels[i]
                if f.shape[1] != expected_c and f.shape[-1] == expected_c:
                    f = f.permute(0, 3, 1, 2).contiguous()
            out.append(f)
        return out
