"""ViT Pyramid Adapter: converts single-resolution ViT features to multi-scale pyramid.

For Vision Transformers (DINO, CLIP, SAM, etc.) that output a single spatial resolution,
this wrapper progressively downsamples the features to create a UNet-compatible pyramid.

Example usage:
    encoder:
      name: "vit_pyramid"
      pretrained: true
      params:
        model_name: "vit_base_patch16_224.dino"   # or samvit_base, vit_clip, etc.
        pyramid_scales: 4   # number of pyramid levels to create
"""
# Source: UNCHECKED — please verify

import torch
import torch.nn as nn
import timm
from typing import List, Optional

from medseg.registry import ENCODER_REGISTRY


class ViTPyramidAdapter(nn.Module):
    """Progressive downsampling adapter to convert single-res ViT features to pyramid.

    Takes the ViT output at a single resolution and progressively applies
    stride-2 convolutions to create a feature pyramid at multiple scales.

    Args:
        in_channels: ViT output channel dimension.
        pyramid_scales: Number of pyramid levels (including the input).
        out_channels: Target channel dims for each pyramid level.
            If None, channels double at each level: [C, 2C, 4C, ...].
            If provided, must match pyramid_scales length.
    """

    def __init__(
        self,
        in_channels: int,
        pyramid_scales: int = 4,
        out_channels: Optional[List[int]] = None,
    ):
        super().__init__()
        self.pyramid_scales = pyramid_scales

        # Determine target channels
        if out_channels is not None:
            assert len(out_channels) == pyramid_scales, \
                f"out_channels len={len(out_channels)} != pyramid_scales={pyramid_scales}"
            target_chs = out_channels
        else:
            # Default: double channels at each level
            target_chs = [in_channels * (2 ** i) for i in range(pyramid_scales)]

        # Progressive downsampling layers
        # First layer: channel projection for stage 0 (high-res)
        self.projection = None
        if target_chs[0] != in_channels:
            self.projection = nn.Conv2d(in_channels, target_chs[0], 1, bias=False)

        self.downsample = nn.ModuleList()
        prev_ch = target_chs[0]
        for i in range(pyramid_scales - 1):
            self.downsample.append(nn.Sequential(
                nn.Conv2d(prev_ch, target_chs[i + 1], 3, 2, 1, bias=False),
                nn.BatchNorm2d(target_chs[i + 1]),
                nn.GELU(),
            ))
            prev_ch = target_chs[i + 1]

        # Output channels at each pyramid level
        self._out_channels = [target_chs[0]]
        for i in range(pyramid_scales - 1):
            self._out_channels.append(target_chs[i + 1])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Create pyramid from single-resolution input."""
        # Project channels for stage 0 if needed
        x = self.projection(x) if self.projection is not None else x
        features = [x]
        for ds in self.downsample:
            features.append(ds(features[-1]))
        return features

    @property
    def out_channels(self):
        return self._out_channels


class VitPyramidEncoder(nn.Module):
    """ViT encoder with pyramid adapter for UNet compatibility.

    Combines a timm ViT model (via features_only) with a progressive
    downsampling adapter to produce multi-scale features suitable for
    UNet-style decoders.

    Supports all ViT models: DINO, CLIP, SAM, MAE, etc.

    Args:
        model_name: timm model name (e.g., 'vit_base_patch16_224.dino').
        pretrained: Whether to load pretrained weights.
        in_channels: Input image channels.
        img_size: Input image size.
        pyramid_scales: Number of pyramid levels to create.
        out_channels: Target channel dims for each pyramid level.
            Default: None (channels double at each level).
            Recommended: [128, 256, 512, 1024] for memory efficiency.
        out_indices: Which ViT layers to extract features from.
    """

    def __init__(
        self,
        model_name: str,
        pretrained: bool = False,
        in_channels: int = 3,
        img_size: int = 224,
        pyramid_scales: int = 4,
        out_channels: Optional[List[int]] = None,
        out_indices: Optional[List[int]] = None,
        **kwargs,
    ):
        super().__init__()

        # Create timm ViT with features_only
        create_kwargs = dict(
            model_name=model_name,
            pretrained=pretrained,
            in_chans=in_channels,
            features_only=True,
            pretrained_strict=False,
        )
        if out_indices is not None:
            create_kwargs["out_indices"] = out_indices
        # 为 DINOv2 / DINOv3 / CLIP / MAE / DeiT 等需要非默认分辨率的 ViT 允许动态输入尺寸
        # （SAM 本身默认就是 1024 输入不需要此参数；swin/maxvit 走 TimmEncoder 不走这里）
        if any(k in model_name for k in ("dinov2", "dinov3", "clip", "mae", "deit")):
            create_kwargs["img_size"] = img_size
            create_kwargs["dynamic_img_size"] = True
        # 其他 kwargs 透传（如 patch_size 调整、drop_rate 等）
        for k, v in kwargs.items():
            create_kwargs.setdefault(k, v)

        self.vit = timm.create_model(**create_kwargs)
        vit_channels = self.vit.feature_info.channels()[-1]  # Use last stage channels

        # Pyramid adapter
        self.adapter = ViTPyramidAdapter(vit_channels, pyramid_scales, out_channels)

    @property
    def out_channels(self):
        return self.adapter.out_channels

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale features.

        Returns:
            List of feature tensors from high-res to low-res.
        """
        # Get single-resolution features from ViT
        vit_features = self.vit(x)
        # Use the deepest feature (last in the list)
        deepest = vit_features[-1]

        # Convert to pyramid
        return self.adapter(deepest)


# ============================================================
# Register ViT pyramid encoders for popular pretrained models
# ============================================================

def _register_vit_pyramid(registry_name: str, timm_model_name: str, pyramid_scales: int = 4):
    """Helper to register a ViT pyramid encoder."""

    @ENCODER_REGISTRY.register(registry_name)
    class _VitPyramidEnc(VitPyramidEncoder):
        def __init__(
            self,
            pretrained=False,
            in_channels=3,
            img_size=224,
            pyramid_scales=pyramid_scales,
            out_channels=None,
            **kwargs,
        ):
            super().__init__(
                model_name=timm_model_name,
                pretrained=pretrained,
                in_channels=in_channels,
                img_size=img_size,
                pyramid_scales=pyramid_scales,
                out_channels=out_channels,
                **kwargs,
            )

    _VitPyramidEnc.__name__ = f"VitPyramid_{timm_model_name}"
    _VitPyramidEnc.__qualname__ = f"VitPyramid_{timm_model_name}"
    return _VitPyramidEnc


# --- DINO / DINOv2 / DINOv3 ViT ---
_register_vit_pyramid("timm_vit_dino_base", "vit_base_patch16_224.dino")
_register_vit_pyramid("timm_vit_dino_small", "vit_small_patch16_224.dino")
_register_vit_pyramid("timm_vit_dinov2_base", "vit_base_patch14_dinov2.lvd142m")
_register_vit_pyramid("timm_vit_dinov2_large", "vit_large_patch14_dinov2.lvd142m")
_register_vit_pyramid("timm_vit_dinov2_giant", "vit_giant_patch14_dinov2.lvd142m")

# DINOv3 models (2025)
_register_vit_pyramid("timm_vit_dinov3_small", "vit_small_patch16_dinov3.lvd1689m")
_register_vit_pyramid("timm_vit_dinov3_base", "vit_base_patch16_dinov3.lvd1689m")
_register_vit_pyramid("timm_vit_dinov3_large", "vit_large_patch16_dinov3.lvd1689m")
_register_vit_pyramid("timm_vit_dinov3_huge_plus", "vit_huge_plus_patch16_dinov3.lvd1689m")
_register_vit_pyramid("timm_vit_dinov3_7b", "vit_7b_patch16_dinov3.lvd1689m")

# --- CLIP ViT ---
_register_vit_pyramid("timm_vit_clip_base", "vit_base_patch16_clip_224.laion2b_ft_in1k")
_register_vit_pyramid("timm_vit_clip_large", "vit_large_patch14_clip_224.openai_ft_in12k_in1k")
_register_vit_pyramid("timm_vit_clip_huge", "vit_huge_patch14_clip_224.laion2b_ft_in1k")

# --- SAM (Segment Anything) ViT ---
_register_vit_pyramid("timm_vit_sam_base", "samvit_base_patch16.sa1b")
_register_vit_pyramid("timm_vit_sam_large", "samvit_large_patch16.sa1b")
_register_vit_pyramid("timm_vit_sam_huge", "samvit_huge_patch16.sa1b")

# --- MAE ViT ---
_register_vit_pyramid("timm_vit_mae_base", "vit_base_patch16_224.mae")
_register_vit_pyramid("timm_vit_mae_large", "vit_large_patch16_224.mae")

# --- DeiT ViT ---
# 使用 deit3 的 fb_in1k tag（旧版 fb_deit_in22k_in1k 在新 timm 中已被移除）
_register_vit_pyramid("timm_vit_deit_base", "deit3_base_patch16_224.fb_in1k")
_register_vit_pyramid("timm_vit_deit_large", "deit3_large_patch16_224.fb_in1k")
