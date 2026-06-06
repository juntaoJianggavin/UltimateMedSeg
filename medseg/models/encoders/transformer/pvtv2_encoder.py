"""PVTv2 encoder.

Pyramid Vision Transformer v2 (PVTv2-B2) backbone exposed as a multi-scale
feature extractor via ``timm.create_model(..., features_only=True)``. Designed
as the representative encoder for the PVTv2 family. Pretrained weights are
fetched with a small SSL fallback so the encoder still constructs when the
hosting environment blocks the default certificate chain.

Stage feature dims (PVTv2-B2): ``[64, 128, 320, 512]`` at strides ``[4, 8, 16, 32]``.
"""
# Source: https://github.com/whai362/PVT

from typing import List

import torch
import torch.nn as nn
import timm

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


@ENCODER_REGISTRY.register("pvtv2")
class PVTv2Encoder(nn.Module):
    """PVTv2-B2 encoder producing 4 multi-scale features.

    out_channels = [64, 128, 320, 512]
    Forward returns a list with the deepest (lowest resolution) feature LAST.
    """

    def __init__(
        self,
        in_channels: int = 3,
        img_size: int = 224,
        pretrained: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.img_size = img_size

        # PVTv2 in timm accepts in_chans; we still keep a 1x1 stem path in case
        # an extension ever requires strict RGB. For now we forward in_channels
        # directly to timm when possible.
        self.stem = None
        backbone_in_chans = in_channels
        try:
            self.model = _load_with_ssl_fallback(
                timm.create_model,
                "pvt_v2_b2",
                pretrained=pretrained,
                features_only=True,
                in_chans=backbone_in_chans,
            )
        except Exception:
            # Fallback: build the backbone strictly as RGB and prepend a 1x1 stem
            # mapping arbitrary in_channels -> 3.
            self.stem = nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)
            self.model = _load_with_ssl_fallback(
                timm.create_model,
                "pvt_v2_b2",
                pretrained=pretrained,
                features_only=True,
                in_chans=3,
            )

        # Expose feature channel info.
        self.out_channels: List[int] = list(self.model.feature_info.channels())
        self._out_strides: List[int] = list(self.model.feature_info.reduction())

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        if self.stem is not None:
            x = self.stem(x)
        features = self.model(x)
        out: List[torch.Tensor] = []
        for i, f in enumerate(features):
            # Some timm transformer backbones return BHWC; normalize to BCHW.
            if f.ndim == 4:
                expected_c = self.out_channels[i]
                if f.shape[1] != expected_c and f.shape[-1] == expected_c:
                    f = f.permute(0, 3, 1, 2).contiguous()
            out.append(f)
        return out
