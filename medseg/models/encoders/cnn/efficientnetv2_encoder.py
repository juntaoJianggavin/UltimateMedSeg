"""EfficientNetV2 encoder via timm.

Wraps `timm.create_model("tf_efficientnetv2_s", features_only=True)` to expose
multi-scale features. Includes an SSL fallback for pretrained weight downloads.
"""
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


@ENCODER_REGISTRY.register("efficientnetv2")
class EfficientNetV2Encoder(nn.Module):
    """EfficientNetV2-S encoder producing multi-scale features.

    Uses timm's `features_only=True` interface so that the forward pass
    returns a list of feature maps from shallow (high-res) to deep (low-res).
    The deepest feature map is returned LAST in the list.

    Args:
        in_channels: number of input image channels. If != 3, a 1x1 conv stem is
            prepended to project to 3 channels (the backbone is constructed with
            `in_chans=3` to keep weights compatible).
        img_size: input spatial size (informational; backbone is fully conv).
        pretrained: whether to load ImageNet-pretrained weights.
    """

    def __init__(self, in_channels: int = 3, img_size: int = 224,
                 pretrained: bool = True, **kwargs):
        super().__init__()

        self.in_channels = in_channels
        self.img_size = img_size

        # If user passes non-RGB input, prepend a 1x1 conv stem to map to 3 channels.
        if in_channels != 3:
            self.stem = nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)
            backbone_in_chans = 3
        else:
            self.stem = nn.Identity()
            backbone_in_chans = 3

        self.model = _load_with_ssl_fallback(
            timm.create_model,
            "tf_efficientnetv2_s",
            pretrained=pretrained,
            features_only=True,
            in_chans=backbone_in_chans,
        )

        self.out_channels: List[int] = list(self.model.feature_info.channels())
        self._out_strides = list(self.model.feature_info.reduction())

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Return multi-scale features, deepest LAST."""
        x = self.stem(x)
        features = self.model(x)
        out: List[torch.Tensor] = []
        for i, f in enumerate(features):
            if f.ndim == 4:
                expected_c = self.out_channels[i]
                if f.shape[1] != expected_c and f.shape[-1] == expected_c:
                    f = f.permute(0, 3, 1, 2).contiguous()
            out.append(f)
        return out
