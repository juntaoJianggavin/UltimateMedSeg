"""Timm encoder wrapper — supports ANY model in the timm library as encoder.

In yaml, write ``name: timm_<timm_model_name>`` (no pre-registration needed).

Pretrained weights (``encoder.pretrained: true``):

1. **Local cache** — if weights already exist under
   ``$MEDSEG_WEIGHT_CACHE/timm/`` (or ``~/.cache/medseg/weights/timm/``),
   they are loaded offline.
2. **Hugging Face Hub** (default) — standard timm download path.
3. **ModelScope** (optional) — set ``MEDSEG_TIMM_SOURCE=modelscope`` when the
   ``modelscope`` package is installed.

Pre-download (optional)::

    python scripts/download_timm_pretrained.py resnet50
    python scripts/download_timm_pretrained.py resnet50 --source modelscope

Users may also set ``HF_ENDPOINT`` themselves for any Hugging Face mirror;
the project does not override that automatically.
"""

import logging
import os

import torch
import torch.nn as nn
import timm
from typing import List, Optional

from medseg.registry import ENCODER_REGISTRY
from medseg.utils.timm_pretrained import (
    ensure_timm_pretrained_via_modelscope,
    load_timm_cached_pretrained_kwargs,
    pretrained_kwargs_from_file,
)

logger = logging.getLogger(__name__)


def _pretrained_source() -> str:
    return os.environ.get("MEDSEG_TIMM_SOURCE", "hf").strip().lower()


def _create_timm_features_model(
    create_kwargs: dict,
    img_size: Optional[int],
) -> nn.Module:
    """Create a timm features_only model with cached / HF / optional ModelScope weights."""

    def _build(kwargs: dict) -> nn.Module:
        if img_size is not None:
            try:
                return timm.create_model(img_size=img_size, **kwargs)
            except TypeError:
                return timm.create_model(**kwargs)
        return timm.create_model(**kwargs)

    if not create_kwargs.get("pretrained"):
        return _build(create_kwargs)

    model_name = create_kwargs["model_name"]
    base_kwargs = {k: v for k, v in create_kwargs.items() if k != "pretrained"}

    cached_kwargs = load_timm_cached_pretrained_kwargs(model_name)
    if cached_kwargs is not None:
        try:
            return _build({**base_kwargs, **cached_kwargs})
        except Exception as exc:
            logger.warning("Failed to load cached weights for %s (%s).", model_name, exc)

    source = _pretrained_source()
    if source == "modelscope":
        try:
            weight_path = ensure_timm_pretrained_via_modelscope(model_name)
            return _build({**base_kwargs, **pretrained_kwargs_from_file(weight_path)})
        except Exception as exc:
            logger.warning(
                "ModelScope pretrained load failed for %s (%s). "
                "Falling back to random initialization.",
                model_name,
                exc,
            )
            return _build({**base_kwargs, "pretrained": False})

    try:
        return _build(create_kwargs)
    except Exception as err:
        logger.warning(
            "Could not load pretrained weights for %s from Hugging Face (%s). "
            "Training from random initialization. "
            "Pre-download with: python scripts/download_timm_pretrained.py %s "
            "or set encoder.pretrained: false.",
            model_name,
            err,
            model_name,
        )
        return _build({**base_kwargs, "pretrained": False})


class TimmEncoder(nn.Module):
    """Generic timm model feature extractor using ``features_only=True``."""

    def __init__(
        self,
        model_name: str,
        pretrained: bool = False,
        in_channels: int = 3,
        img_size: int = 224,
        out_indices: Optional[List[int]] = None,
        **kwargs,
    ):
        super().__init__()
        create_kwargs = dict(
            model_name=model_name,
            pretrained=pretrained,
            in_chans=in_channels,
            features_only=True,
        )
        if out_indices is not None:
            create_kwargs["out_indices"] = out_indices
        self.model = _create_timm_features_model(create_kwargs, img_size)
        self.out_channels = self.model.feature_info.channels()
        self._out_strides = self.model.feature_info.reduction()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = self.model(x)
        out = []
        for i, f in enumerate(features):
            if f.ndim == 4:
                expected_c = self.out_channels[i]
                if f.shape[1] != expected_c and f.shape[-1] == expected_c:
                    f = f.permute(0, 3, 1, 2).contiguous()
            out.append(f)
        return out


def _register_timm_encoder(registry_name: str, timm_model_name: str):
    @ENCODER_REGISTRY.register(registry_name)
    class _TimmEnc(TimmEncoder):
        def __init__(self, pretrained=False, in_channels=3, img_size=224, **kwargs):
            super().__init__(
                model_name=timm_model_name,
                pretrained=pretrained,
                in_channels=in_channels,
                img_size=img_size,
                **kwargs,
            )

    _TimmEnc.__name__ = f"Timm_{timm_model_name}"
    _TimmEnc.__qualname__ = f"Timm_{timm_model_name}"
    return _TimmEnc


# ============================================================
# Register popular timm encoders
# ============================================================

_register_timm_encoder("timm_resnet18", "resnet18")
_register_timm_encoder("timm_resnet34", "resnet34")
_register_timm_encoder("timm_resnet50", "resnet50")
_register_timm_encoder("timm_resnet101", "resnet101")
_register_timm_encoder("timm_resnet152", "resnet152")
_register_timm_encoder("timm_resnext50_32x4d", "resnext50_32x4d")
_register_timm_encoder("timm_resnext101_32x8d", "resnext101_32x8d")
_register_timm_encoder("timm_wide_resnet50_2", "wide_resnet50_2")
_register_timm_encoder("timm_wide_resnet101_2", "wide_resnet101_2")
_register_timm_encoder("timm_res2net50_26w_4s", "res2net50_26w_4s")
_register_timm_encoder("timm_vgg16", "vgg16")
_register_timm_encoder("timm_vgg19", "vgg19")
_register_timm_encoder("timm_vgg16_bn", "vgg16_bn")
_register_timm_encoder("timm_vgg19_bn", "vgg19_bn")
_register_timm_encoder("timm_densenet121", "densenet121")
_register_timm_encoder("timm_densenet161", "densenet161")
_register_timm_encoder("timm_densenet169", "densenet169")
_register_timm_encoder("timm_densenet201", "densenet201")
_register_timm_encoder("timm_efficientnet_b0", "efficientnet_b0")
_register_timm_encoder("timm_efficientnet_b1", "efficientnet_b1")
_register_timm_encoder("timm_efficientnet_b2", "efficientnet_b2")
_register_timm_encoder("timm_efficientnet_b3", "efficientnet_b3")
_register_timm_encoder("timm_efficientnet_b4", "efficientnet_b4")
_register_timm_encoder("timm_efficientnet_b5", "efficientnet_b5")
_register_timm_encoder("timm_efficientnetv2_s", "efficientnetv2_s")
_register_timm_encoder("timm_efficientnetv2_m", "efficientnetv2_m")
_register_timm_encoder("timm_mobilenetv2_100", "mobilenetv2_100")
_register_timm_encoder("timm_mobilenetv3_large_100", "mobilenetv3_large_100")
_register_timm_encoder("timm_mobilenetv3_small_100", "mobilenetv3_small_100")
_register_timm_encoder("timm_convnext_tiny", "convnext_tiny")
_register_timm_encoder("timm_convnext_small", "convnext_small")
_register_timm_encoder("timm_convnext_base", "convnext_base")
_register_timm_encoder("timm_convnext_large", "convnext_large")
_register_timm_encoder("timm_convnextv2_tiny", "convnextv2_tiny")
_register_timm_encoder("timm_convnextv2_base", "convnextv2_base")
_register_timm_encoder("timm_swin_tiny_patch4_window7_224", "swin_tiny_patch4_window7_224")
_register_timm_encoder("timm_swin_small_patch4_window7_224", "swin_small_patch4_window7_224")
_register_timm_encoder("timm_swin_base_patch4_window7_224", "swin_base_patch4_window7_224")
_register_timm_encoder("timm_swinv2_tiny_window8_256", "swinv2_tiny_window8_256")
_register_timm_encoder("timm_pvt_v2_b0", "pvt_v2_b0")
_register_timm_encoder("timm_pvt_v2_b1", "pvt_v2_b1")
_register_timm_encoder("timm_pvt_v2_b2", "pvt_v2_b2")
_register_timm_encoder("timm_pvt_v2_b3", "pvt_v2_b3")
_register_timm_encoder("timm_pvt_v2_b4", "pvt_v2_b4")
_register_timm_encoder("timm_mit_b0", "mit_b0")
_register_timm_encoder("timm_mit_b1", "mit_b1")
_register_timm_encoder("timm_mit_b2", "mit_b2")
_register_timm_encoder("timm_mit_b3", "mit_b3")
_register_timm_encoder("timm_mit_b5", "mit_b5")
_register_timm_encoder("timm_maxvit_tiny_tf_224", "maxvit_tiny_tf_224")
_register_timm_encoder("timm_maxvit_small_tf_224", "maxvit_small_tf_224")
_register_timm_encoder("timm_senet154", "senet154")
_register_timm_encoder("timm_seresnet50", "seresnet50")
_register_timm_encoder("timm_inception_v3", "inception_v3")
_register_timm_encoder("timm_ghostnet_100", "ghostnet_100")
_register_timm_encoder("timm_shufflenetv2_x1_0", "shufflenetv2_x1_0")
_register_timm_encoder("timm_poolformer_s12", "poolformer_s12")
_register_timm_encoder("timm_poolformer_s24", "poolformer_s24")
_register_timm_encoder("timm_edgenext_small", "edgenext_small")
_register_timm_encoder("timm_fastvit_t8", "fastvit_t8")
_register_timm_encoder("timm_efficientformer_l1", "efficientformer_l1")
_register_timm_encoder("timm_mobilevit_s", "mobilevit_s")
_register_timm_encoder("timm_coatnet_0_224", "coatnet_0_224")


@ENCODER_REGISTRY.register("timm")
class GenericTimmEncoder(TimmEncoder):
    def __init__(self, pretrained=False, in_channels=3, img_size=224, model_name="resnet50", **kwargs):
        super().__init__(
            model_name=model_name,
            pretrained=pretrained,
            in_channels=in_channels,
            img_size=img_size,
            **kwargs,
        )
