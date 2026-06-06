"""Derm Foundation dermatology encoder (Google HAI-DEF).

Reference:
    Peng et al., "Derm Foundation", Google Health AI Developer Foundations
    (HAI-DEF), 2024.
    https://research.google/blog/health-specific-embedding-tools-for-dermatology-and-pathology/

Derm Foundation uses a ViT-S/16 architecture (``embed_dim=384``,
``patch_size=16``) trained with Masked Siamese Networks on dermatology
images.  The official weights are hosted at ``google/derm-foundation`` on
HuggingFace Hub.

``pretrained=True`` auto-downloads from HF Hub.
``pretrained=False`` raises ``RuntimeError``.

Registered as ``"derm_foundation"`` in ``ENCODER_REGISTRY``.
"""
# Source: https://huggingface.co/google/derm-foundation
# Paper:  https://research.google/blog/health-specific-embedding-tools-for-dermatology-and-pathology/

from __future__ import annotations

import warnings
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.registry import ENCODER_REGISTRY
from medseg.models.encoders.foundation._base import (BaseFoundationEncoder, load_hf_vit,
                     hf_hub_download_vision_weights, HuggingFaceViTWrapper, DPTHead)


_PRIMARY_HF_NAME = "google/derm-foundation"
PRIMARY_BACKBONE_NAME = _PRIMARY_HF_NAME
_EMBED_DIM = 384
_PATCH_SIZE = 16


@ENCODER_REGISTRY.register("derm_foundation")
class DermFoundationEncoder(BaseFoundationEncoder):
    """Derm Foundation (dermatology ViT-S/16, MSN) encoder with DPT-style multi-block pyramid.

    The backbone is a ViT-Small/16 (``embed_dim=384``, ``patch_size=16``)
    pretrained with Masked Siamese Networks on dermatology images.

    ``out_channels = [dim/8, dim/4, dim/2, dim]`` (deepest LAST).
    """

    native_img_size: int = 224
    PATCH_SIZE = _PATCH_SIZE
    EMBED_DIM = _EMBED_DIM

    def __init__(self, in_channels: int = 3, img_size: Optional[int] = None,
                 pretrained: bool = True, pretrained_path: Optional[str] = None,
                 freeze: bool = True, unfreeze_last_n: int = 0,
                 inference_only: bool = False, **kwargs):
        resolved_img_size = int(img_size) if img_size is not None else self.native_img_size
        super().__init__(in_channels=in_channels, img_size=resolved_img_size,
                         pretrained=pretrained, pretrained_path=pretrained_path,
                         freeze=freeze, unfreeze_last_n=unfreeze_last_n,
                         inference_only=inference_only, **kwargs)

        if in_channels != 3:
            self.input_adapter: nn.Module = nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)
        else:
            self.input_adapter = nn.Identity()

        # Backbone — try standard ViTModel first, then fallback.
        if pretrained:
            try:
                self.backbone = load_hf_vit(
                    hf_name=_PRIMARY_HF_NAME,
                    pretrained_path=pretrained_path,
                    model_cls_name="ViTModel",
                )
            except Exception:
                import transformers
                _skel = transformers.ViTModel(transformers.ViTConfig(
                    hidden_size=384, num_hidden_layers=12,
                    num_attention_heads=6, intermediate_size=1536,
                    patch_size=16, image_size=224,
                ))
                _state = hf_hub_download_vision_weights(
                    repo_id=_PRIMARY_HF_NAME,
                    prefix_strip=("vision_tower.", "vision_model.",
                                  "image_encoder.", "vit.",
                                  "encoder.", "backbone.",
                                  "trunk.", "module."),
                )
                _msg = _skel.load_state_dict(_state, strict=False)
                warnings.warn(f"[derm_foundation] loaded via hf_hub_download: {_msg}")
                self.backbone = HuggingFaceViTWrapper(_skel)
        else:
            raise RuntimeError(
                "DermFoundationEncoder does not support pretrained=False. "
                "This encoder requires pretrained weights from "
                "'google/derm-foundation'. Pass pretrained=True to "
                "auto-download, or provide a local checkpoint via pretrained_path."
            )
        self._backbone_name = PRIMARY_BACKBONE_NAME

        # Introspection.
        self.patch_size = int(self.backbone.patch_embed.patch_size)
        self.embed_dim = int(self.backbone.embed_dim)
        self.num_prefix_tokens = int(self.backbone.num_prefix_tokens)

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
