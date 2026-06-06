"""CLIP ViT image encoder (foundation-model encoder).

Wraps a timm ViT-CLIP backbone (B/L) and exposes a 4-stage multi-scale
feature pyramid via the SAM-style "DPT-style multi-block" trick: the final ViT
token grid is reshaped to a (B, dim, h, w) feature map and four parallel
spatial-projection branches produce features at progressively coarser
strides with deepest LAST.

Variants (kwarg ``clip_variant``):
    - "base"  -> ``vit_base_patch16_clip_224`` (embed_dim=768, patch=16)
    - "large" -> ``vit_large_patch14_clip_224`` (embed_dim=1024, patch=14)

Output:
    forward(x) -> List[Tensor] of length 4. Approximate spatial strides
    [H/4, H/8, H/16, H/32]; channels [dim/8, dim/4, dim/2, dim].
"""

from __future__ import annotations

import warnings
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.registry import ENCODER_REGISTRY
from medseg.models.encoders.foundation._base import DPTHead, BaseFoundationEncoder, load_with_ssl_fallback


# Default timm model names per variant.
_CLIP_VARIANTS = {
    "base":  {"timm_name": "vit_base_patch16_clip_224", "embed_dim": 768,  "patch_size": 16},
    "large": {"timm_name": "vit_large_patch14_clip_224", "embed_dim": 1024, "patch_size": 14},
}


def _build_timm_clip_vit(variant: str, img_size: int, in_chans: int,
                         pretrained: bool):
    """Create a timm ViT-CLIP model with dynamic-image-size support."""
    import timm  # local import: heavy dependency, lazy-loaded.

    primary = _CLIP_VARIANTS[variant]["timm_name"]
    model = load_with_ssl_fallback(
        timm.create_model, primary,
        pretrained=pretrained, num_classes=0,
        img_size=img_size, dynamic_img_size=True,
    )
    return model, primary


@ENCODER_REGISTRY.register("clip_vit")
class CLIPViTEncoder(BaseFoundationEncoder):
    """CLIP ViT (B/L) image encoder with DPT-style multi-block multi-scale output.

    Constructor follows the BaseFoundationEncoder contract. Additional kwargs:
        clip_variant: "base" (default) or "large".
    """

    def __init__(self, in_channels: int = 3, img_size: int = 224,
                 pretrained: bool = True, pretrained_path: Optional[str] = None,
                 freeze: bool = True, unfreeze_last_n: int = 0,
                 inference_only: bool = False, **kwargs):
        super().__init__(in_channels=in_channels, img_size=img_size,
                         pretrained=pretrained, pretrained_path=pretrained_path,
                         freeze=freeze, unfreeze_last_n=unfreeze_last_n,
                         inference_only=inference_only, **kwargs)

        variant = str(kwargs.get("clip_variant", "base")).lower()
        if variant not in _CLIP_VARIANTS:
            raise ValueError(
                f"[clip_vit] unknown clip_variant='{variant}'. "
                f"Expected one of {sorted(_CLIP_VARIANTS)}.")
        self.clip_variant = variant

        meta = _CLIP_VARIANTS[variant]
        self.embed_dim = meta["embed_dim"]
        self.patch_size = meta["patch_size"]

        # Build the timm ViT-CLIP backbone (with optional pretrained weights).
        self.backbone, self._backbone_name = _build_timm_clip_vit(
            variant=variant, img_size=img_size,
            in_chans=in_channels, pretrained=pretrained,
        )
        self.num_prefix_tokens = int(getattr(self.backbone, "num_prefix_tokens", 1))

        # Adapter for non-RGB inputs (CLIP ViTs expect 3 channels).
        if in_channels != 3:
            self.input_adapter = nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)
        else:
            self.input_adapter = nn.Identity()

        # DPT-style multi-block: project the token grid (B, dim, h, w) to four
        # pyramid stages with channels [dim/8, dim/4, dim/2, dim].
        # Spatial scales (deepest LAST):
        #   stage 0: ConvTranspose 4x  -> ~H/4,  dim/8
        #   stage 1: ConvTranspose 2x  -> ~H/8,  dim/4
        #   stage 2: Identity          -> ~H/16, dim/2
        #   stage 3: MaxPool 2x        -> ~H/32, dim
        dim = self.embed_dim
        # DPT head: 从不同深度 block 构建真正多尺度金字塔
        # DPT head: genuine multi-scale pyramid from different-depth blocks
        self.dpt = DPTHead(
            embed_dim=self.embed_dim,
            num_prefix_tokens=int(self.num_prefix_tokens),
        )
        self.out_channels = self.dpt.out_channels
        self._block_indices = DPTHead.default_block_indices(len(self.backbone.blocks))

        self._maybe_inject_adapters()
        self._apply_freeze_policy()
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.input_adapter(x)
        B, _, H, W = x.shape
        p = self.patch_size

        # 填充到 patch_size 的倍数 / Pad to multiple of patch_size
        pad_h = (p - H % p) % p
        pad_w = (p - W % p) % p
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        Hp, Wp = x.shape[-2], x.shape[-1]

        # 从不同深度 block 提取 token（DPT 核心）
        # Extract tokens from different-depth blocks (DPT core)
        multi_tokens = self.backbone.get_intermediate_layers(
            x, n=self._block_indices,
        )

        h_patches = Hp // p
        w_patches = Wp // p

        return self.dpt(list(multi_tokens), h_patches, w_patches, H, W)
