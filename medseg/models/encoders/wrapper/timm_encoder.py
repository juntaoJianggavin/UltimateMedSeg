"""Timm encoder wrapper — 支持 timm 库中所有模型作为 encoder。
Timm encoder wrapper — supports ANY model in the timm library as encoder.

用法 / Usage:
    在 yaml 中只需写 `name: timm_<timm模型名>` 即可，无需预注册。
    In yaml, just write `name: timm_<timm_model_name>`, no pre-registration needed.

    例如 / Examples:
        encoder:
            name: timm_resnet50
            pretrained: true

        encoder:
            name: timm_efficientnet_b7
            pretrained: true

        encoder:
            name: timm_convnextv2_huge
            pretrained: true

        encoder:
            name: timm_swin_base_patch4_window12_384
            pretrained: true

    任何 `timm.list_models()` 中的模型名加上 `timm_` 前缀就能直接使用。
    Any model name from `timm.list_models()` with a `timm_` prefix works directly.
"""

import torch
import torch.nn as nn
import timm
from typing import List, Optional

from medseg.registry import ENCODER_REGISTRY


class TimmEncoder(nn.Module):
    """通用 timm 模型特征提取器 / Generic timm model feature extractor.

    使用 timm 的 `features_only=True` 提取多尺度特征。
    Uses timm's `features_only=True` to extract multi-scale features.
    支持 timm 库中所有支持特征提取的模型（1000+ 种）。
    Supports all timm models that support feature extraction (1000+ models).
    """

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
        # Create timm model with feature extraction
        create_kwargs = dict(
            model_name=model_name,
            pretrained=pretrained,
            in_chans=in_channels,
            features_only=True,
        )
        if out_indices is not None:
            create_kwargs["out_indices"] = out_indices
        # Some timm models (e.g. Swin V2) require matching img_size.
        # Try with img_size first; fall back without it for models that
        # don't accept the argument (e.g. ResNet).
        if img_size is not None:
            try:
                self.model = timm.create_model(img_size=img_size, **create_kwargs)
            except TypeError:
                self.model = timm.create_model(**create_kwargs)
        else:
            self.model = timm.create_model(**create_kwargs)
        # Get output channel info
        self.out_channels = self.model.feature_info.channels()
        self._out_strides = self.model.feature_info.reduction()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale features.

        Returns:
            List of feature tensors from each stage, low-res to high-res order is
            [stage1(highest_res), stage2, ..., stageN(lowest_res)].
        """
        features = self.model(x)
        # Ensure BCHW format (some timm models like Swin output BHWC)
        out = []
        for i, f in enumerate(features):
            if f.ndim == 4:
                expected_c = self.out_channels[i]
                if f.shape[1] != expected_c and f.shape[-1] == expected_c:
                    f = f.permute(0, 3, 1, 2).contiguous()
            out.append(f)
        return out


def _register_timm_encoder(registry_name: str, timm_model_name: str):
    """Helper to register a timm model under a given name."""

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

# --- ResNet family ---
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

# --- VGG family ---
_register_timm_encoder("timm_vgg16", "vgg16")
_register_timm_encoder("timm_vgg19", "vgg19")
_register_timm_encoder("timm_vgg16_bn", "vgg16_bn")
_register_timm_encoder("timm_vgg19_bn", "vgg19_bn")

# --- DenseNet family ---
_register_timm_encoder("timm_densenet121", "densenet121")
_register_timm_encoder("timm_densenet161", "densenet161")
_register_timm_encoder("timm_densenet169", "densenet169")
_register_timm_encoder("timm_densenet201", "densenet201")

# --- EfficientNet family ---
_register_timm_encoder("timm_efficientnet_b0", "efficientnet_b0")
_register_timm_encoder("timm_efficientnet_b1", "efficientnet_b1")
_register_timm_encoder("timm_efficientnet_b2", "efficientnet_b2")
_register_timm_encoder("timm_efficientnet_b3", "efficientnet_b3")
_register_timm_encoder("timm_efficientnet_b4", "efficientnet_b4")
_register_timm_encoder("timm_efficientnet_b5", "efficientnet_b5")
_register_timm_encoder("timm_efficientnetv2_s", "efficientnetv2_s")
_register_timm_encoder("timm_efficientnetv2_m", "efficientnetv2_m")

# --- MobileNet family ---
_register_timm_encoder("timm_mobilenetv2_100", "mobilenetv2_100")
_register_timm_encoder("timm_mobilenetv3_large_100", "mobilenetv3_large_100")
_register_timm_encoder("timm_mobilenetv3_small_100", "mobilenetv3_small_100")

# --- ConvNeXt family ---
_register_timm_encoder("timm_convnext_tiny", "convnext_tiny")
_register_timm_encoder("timm_convnext_small", "convnext_small")
_register_timm_encoder("timm_convnext_base", "convnext_base")
_register_timm_encoder("timm_convnext_large", "convnext_large")
_register_timm_encoder("timm_convnextv2_tiny", "convnextv2_tiny")
_register_timm_encoder("timm_convnextv2_base", "convnextv2_base")

# --- Swin Transformer ---
_register_timm_encoder("timm_swin_tiny_patch4_window7_224", "swin_tiny_patch4_window7_224")
_register_timm_encoder("timm_swin_small_patch4_window7_224", "swin_small_patch4_window7_224")
_register_timm_encoder("timm_swin_base_patch4_window7_224", "swin_base_patch4_window7_224")
_register_timm_encoder("timm_swinv2_tiny_window8_256", "swinv2_tiny_window8_256")

# --- PVT family ---
_register_timm_encoder("timm_pvt_v2_b0", "pvt_v2_b0")
_register_timm_encoder("timm_pvt_v2_b1", "pvt_v2_b1")
_register_timm_encoder("timm_pvt_v2_b2", "pvt_v2_b2")
_register_timm_encoder("timm_pvt_v2_b3", "pvt_v2_b3")
_register_timm_encoder("timm_pvt_v2_b4", "pvt_v2_b4")

# --- SegFormer (MixTransformer) ---
_register_timm_encoder("timm_mit_b0", "mit_b0")
_register_timm_encoder("timm_mit_b1", "mit_b1")
_register_timm_encoder("timm_mit_b2", "mit_b2")
_register_timm_encoder("timm_mit_b3", "mit_b3")
_register_timm_encoder("timm_mit_b5", "mit_b5")

# --- MaxViT ---
_register_timm_encoder("timm_maxvit_tiny_tf_224", "maxvit_tiny_tf_224")
_register_timm_encoder("timm_maxvit_small_tf_224", "maxvit_small_tf_224")

# --- Others ---
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


# Also register a generic "timm" encoder that accepts model_name as param
@ENCODER_REGISTRY.register("timm")
class GenericTimmEncoder(TimmEncoder):
    """Generic timm encoder: specify model_name in params.

    Usage in config:
        encoder:
            name: "timm"
            pretrained: true
            params:
                model_name: "resnet50"
    """

    def __init__(self, pretrained=False, in_channels=3, img_size=224, model_name="resnet50", **kwargs):
        super().__init__(
            model_name=model_name,
            pretrained=pretrained,
            in_channels=in_channels,
            img_size=img_size,
            **kwargs,
        )
