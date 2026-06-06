"""MaxViT encoder.

Wraps timm's ``maxvit_tiny_tf_224`` as a multi-scale feature extractor.

Notes:
- MaxViT downsamples by 32x at the deepest stage, so the input spatial dims
  must be divisible by 32. This encoder pads input on the bottom/right if
  needed and the spatial sizes returned are based on the padded input.
- ``maxvit_tiny_tf_224`` strictly expects 3-channel RGB input. When
  ``in_channels != 3`` we prepend a learnable 1x1 conv stem to project to
  3 channels.
- Pretrained download may fail in SSL-restricted environments; a fallback
  retries with unverified SSL context and finally falls back to random init.
"""
# Source: UNCHECKED — please verify

import torch
import torch.nn as nn
import torch.nn.functional as F
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


@ENCODER_REGISTRY.register("maxvit")
class MaxViTEncoder(nn.Module):
    """MaxViT-Tiny encoder via timm (features_only).

    Returns multi-scale features with the deepest (lowest-resolution) feature
    LAST. ``self.out_channels`` lists channel counts in the same order.
    """

    MODEL_NAME = "maxvit_tiny_tf_224"
    STRIDE = 32

    def __init__(self, in_channels: int = 3, img_size: int = 224,
                 pretrained: bool = True, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.img_size = img_size

        # MaxViT strictly needs 3-channel RGB input -- add a projection stem
        # for non-RGB inputs.
        if in_channels != 3:
            self.input_proj = nn.Conv2d(in_channels, 3, kernel_size=1, bias=True)
        else:
            self.input_proj = nn.Identity()

        self.model = _load_with_ssl_fallback(
            timm.create_model,
            self.MODEL_NAME,
            pretrained=pretrained,
            features_only=True,
            in_chans=3,
        )

        # Channel counts for each returned feature map (high-res -> low-res).
        self.out_channels: List[int] = list(self.model.feature_info.channels())
        self._out_strides = list(self.model.feature_info.reduction())

    def _pad_to_multiple(self, x: torch.Tensor):
        """Pad H,W on the bottom/right to the next multiple of ``STRIDE``.

        Returns the padded tensor and (pad_h, pad_w) so callers can crop later
        if needed. Padding is zero-valued.
        """
        _, _, h, w = x.shape
        ph = (self.STRIDE - h % self.STRIDE) % self.STRIDE
        pw = (self.STRIDE - w % self.STRIDE) % self.STRIDE
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph))
        return x, (ph, pw)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.input_proj(x)
        x, _ = self._pad_to_multiple(x)
        features = self.model(x)

        # Ensure all feature maps are BCHW (timm MaxViT outputs BCHW already,
        # but defensively handle BHWC layouts seen in some transformer models).
        out = []
        for i, f in enumerate(features):
            if f.ndim == 4:
                expected_c = self.out_channels[i]
                if f.shape[1] != expected_c and f.shape[-1] == expected_c:
                    f = f.permute(0, 3, 1, 2).contiguous()
            out.append(f)
        return out
