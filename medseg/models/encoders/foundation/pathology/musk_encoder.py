"""MUSK pathology foundation-model encoder.

Reference: Xiang et al. (Bo Wang lab), "A vision-language foundation model for
computational pathology", Nature Methods, 2024.

MUSK is a BEiT-3 style multimodal (image + text) foundation model trained on
large-scale pathology image-text pairs. Its vision tower is a ViT-Large with
``patch_size=16`` and a native pretraining spatial resolution of ``384x384``.

The official weights are hosted at ``xiangjx/musk`` on HuggingFace Hub.
``pretrained=True`` auto-downloads from HF Hub.
``pretrained=False`` raises ``RuntimeError``.

Registered as ``"musk"`` in ``ENCODER_REGISTRY``.
"""
# Source: https://huggingface.co/xiangjx/musk

from __future__ import annotations

import warnings
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.registry import ENCODER_REGISTRY
from medseg.models.encoders.foundation._base import DPTHead, BaseFoundationEncoder, load_hf_vit, hf_hub_download_vision_weights, HuggingFaceViTWrapper


# MUSK official release on the Hugging Face hub.
_PRIMARY_HF_NAME = "xiangjx/musk"
PRIMARY_BACKBONE_NAME = _PRIMARY_HF_NAME


@ENCODER_REGISTRY.register("musk")
class MUSKEncoder(BaseFoundationEncoder):
    """MUSK (pathology BEiT-3 ViT-L/16) encoder with DPT-style multi-block pyramid.

    The vision tower is a BEiT-3 ViT-Large/16 (``embed_dim=1024``,
    ``patch_size=16``) pretrained at ``384x384`` on a large pathology
    image-text corpus.
    """

    native_img_size: int = 384
    PATCH_SIZE = 16
    EMBED_DIM = 1024

    def __init__(self, in_channels: int = 3, img_size: Optional[int] = None,
                 pretrained: bool = True, pretrained_path: Optional[str] = None,
                 freeze: bool = True, unfreeze_last_n: int = 0,
                 inference_only: bool = False, **kwargs):
        resolved_img_size = int(img_size) if img_size is not None \
            else int(self.native_img_size)

        super().__init__(in_channels=in_channels, img_size=resolved_img_size,
                         pretrained=pretrained,
                         pretrained_path=pretrained_path,
                         freeze=freeze, unfreeze_last_n=unfreeze_last_n,
                         inference_only=inference_only, **kwargs)

        # ------------------------------------------------------------------
        # Backbone (MUSK) — loaded via transformers AutoModel
        # ------------------------------------------------------------------
        if pretrained:
            try:
                self.backbone = load_hf_vit(
                    hf_name=_PRIMARY_HF_NAME,
                    pretrained_path=pretrained_path,
                    trust_remote_code=True,
                    model_cls_name="AutoModel",
                )
            except Exception:
                # Fallback: download vision weights from the full MUSK model
                # and load into a fresh HF ViT-L/16 skeleton.
                import transformers
                _skel = transformers.ViTModel(transformers.ViTConfig(
                    hidden_size=1024, num_hidden_layers=24,
                    num_attention_heads=16, intermediate_size=4096,
                    patch_size=16, image_size=224,
                ))
                _state = hf_hub_download_vision_weights(
                    repo_id=_PRIMARY_HF_NAME,
                    prefix_strip=("beit3.vision_encoder.", "beit3.",
                                  "vision_encoder.", "visual.",
                                  "image_encoder.", "vit.",
                                  "encoder.", "backbone.",
                                  "trunk.", "module."),
                )
                _msg = _skel.load_state_dict(_state, strict=False)
                warnings.warn(f"[musk] loaded via hf_hub_download: {_msg}")
                self.backbone = HuggingFaceViTWrapper(_skel)
        else:
            raise RuntimeError(
                "MUSKEncoder does not support pretrained=False. "
                "This encoder requires pretrained weights from 'xiangjx/musk'. "
                "Pass pretrained=True to auto-download, or provide a local "
                "checkpoint via pretrained_path."
            )
        self._backbone_name = PRIMARY_BACKBONE_NAME

        # ------------------------------------------------------------------
        # Geometry / dims
        # ------------------------------------------------------------------
        dim = int(self.backbone.embed_dim)
        self.embed_dim = dim
        self.patch_size = int(self.backbone.patch_embed.patch_size)
        self.num_prefix_tokens = int(self.backbone.num_prefix_tokens)

        # ------------------------------------------------------------------
        # DPT-style multi-block projector
        # ------------------------------------------------------------------
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
