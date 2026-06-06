"""PanDerm dermatology foundation-model encoder.

Reference:
    Yan et al., "A multimodal vision foundation model for clinical dermatology",
    Nature Medicine, 2025.  arXiv:2410.15038.

PanDerm is a self-supervised ViT-Large/16 (``embed_dim=1024``,
``patch_size=16``) pretrained on over 2 million real-world skin disease
images.  The official code is at ``SiyuanYan1/PanDerm`` on GitHub.

Weights are loaded via ``hf_hub_download`` from the PanDerm release or
from a local checkpoint.  ``pretrained=True`` auto-downloads;
``pretrained=False`` raises ``RuntimeError``.

Registered as ``"panderm"`` in ``ENCODER_REGISTRY``.
"""
# Source: https://github.com/SiyuanYan1/PanDerm
# Paper:  https://arxiv.org/abs/2410.15038  (Nature Medicine 2025)

from __future__ import annotations

import warnings
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.registry import ENCODER_REGISTRY
from medseg.models.encoders.foundation._base import (BaseFoundationEncoder, load_hf_vit,
                     hf_hub_download_vision_weights, HuggingFaceViTWrapper, DPTHead)


_PRIMARY_HF_NAME = "SiyuanYan1/PanDerm"
PRIMARY_BACKBONE_NAME = _PRIMARY_HF_NAME
_EMBED_DIM = 1024
_PATCH_SIZE = 16


@ENCODER_REGISTRY.register("panderm")
class PanDermEncoder(BaseFoundationEncoder):
    """PanDerm (dermatology ViT-L/16) encoder with DPT-style multi-block pyramid.

    The backbone is a self-supervised ViT-Large/16 (``embed_dim=1024``,
    ``patch_size=16``) pretrained on 2M+ dermatology images.

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

        # Backbone — try HF ViTModel first, then fallback to manual weight loading.
        if pretrained:
            try:
                self.backbone = load_hf_vit(
                    hf_name=_PRIMARY_HF_NAME,
                    pretrained_path=pretrained_path,
                    model_cls_name="ViTModel",
                    trust_remote_code=True,
                )
            except Exception:
                # Fallback: build ViT-L/16 skeleton and download weights.
                import transformers
                _skel = transformers.ViTModel(transformers.ViTConfig(
                    hidden_size=1024, num_hidden_layers=24,
                    num_attention_heads=16, intermediate_size=4096,
                    patch_size=16, image_size=224,
                ))
                _state = hf_hub_download_vision_weights(
                    repo_id=_PRIMARY_HF_NAME,
                    prefix_strip=("vision_encoder.", "vision_tower.",
                                  "image_encoder.", "vit.",
                                  "encoder.", "backbone.",
                                  "trunk.", "module.",
                                  "model.", "state_dict."),
                )
                _msg = _skel.load_state_dict(_state, strict=False)
                warnings.warn(f"[panderm] loaded via hf_hub_download: {_msg}")
                self.backbone = HuggingFaceViTWrapper(_skel)
        else:
            raise RuntimeError(
                "PanDermEncoder does not support pretrained=False. "
                "This encoder requires pretrained weights from PanDerm "
                "(https://github.com/SiyuanYan1/PanDerm). "
                "Pass pretrained=True or provide a local checkpoint via pretrained_path."
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
