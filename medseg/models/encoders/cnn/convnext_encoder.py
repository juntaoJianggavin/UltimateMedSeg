"""ConvNeXt encoder using timm's `convnext_tiny` with SSL-fallback for pretrained weights."""
# Source: UNCHECKED — please verify

import torch
import torch.nn as nn
import timm
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


@ENCODER_REGISTRY.register("convnext")
class ConvNeXtEncoder(nn.Module):
    """ConvNeXt-Tiny encoder via timm `features_only=True`.

    Returns a list of multi-scale feature maps with the deepest feature LAST.
    out_channels is exposed from timm's `feature_info.channels()`.
    """

    def __init__(self, in_channels: int = 3, img_size: int = 224,
                 pretrained: bool = True, **kwargs):
        super().__init__()

        # If non-RGB input, prepend a 1x1 conv stem to map to 3 channels.
        if in_channels != 3:
            self.stem_adapter = nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)
            backbone_in = 3
        else:
            self.stem_adapter = None
            backbone_in = 3

        self.model = _load_with_ssl_fallback(
            timm.create_model,
            "convnext_tiny",
            pretrained=pretrained,
            features_only=True,
            in_chans=backbone_in,
        )

        self.out_channels: List[int] = list(self.model.feature_info.channels())

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        if self.stem_adapter is not None:
            x = self.stem_adapter(x)
        features = self.model(x)
        # Ensure BCHW (ConvNeXt in timm returns BCHW already, but normalize defensively).
        out = []
        for i, f in enumerate(features):
            if f.ndim == 4:
                expected_c = self.out_channels[i]
                if f.shape[1] != expected_c and f.shape[-1] == expected_c:
                    f = f.permute(0, 3, 1, 2).contiguous()
            out.append(f)
        return out
