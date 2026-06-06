"""Qwen2.5-VL vision-tower encoder (foundation-model encoder).

Qwen2.5-VL (Alibaba, 2024) is a multimodal LLM whose image side is a custom
ViT with M-RoPE (Multimodal Rotary Position Embedding) supporting dynamic
resolution. The 7B configuration ships a ~675M-parameter vision tower.

``pretrained=True`` auto-downloads from ``Qwen/Qwen2.5-VL-7B-Instruct``
via ``transformers.AutoModel`` and extracts the vision tower.
``pretrained=False`` raises RuntimeError.

Output:
    forward(x) -> List[Tensor] of length 4 with channels [dim/8, dim/4, dim/2, dim]
    (deepest LAST).
"""
# Source: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct

from __future__ import annotations

import math
import warnings
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.registry import ENCODER_REGISTRY
from medseg.models.encoders.foundation._base import DPTHead, BaseFoundationEncoder, HuggingFaceViTWrapper


_PRIMARY_HF_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
PRIMARY_BACKBONE_NAME = _PRIMARY_HF_NAME
_EMBED_DIM = 1024
_PATCH_SIZE = 14


@ENCODER_REGISTRY.register("qwen25_vl_vision")
class Qwen25VLVisionEncoder(BaseFoundationEncoder):
    """Qwen2.5-VL vision tower with DPT-style multi-block multi-scale output."""

    native_img_size: int = 384

    def __init__(self, in_channels: int = 3, img_size: Optional[int] = None,
                 pretrained: bool = True, pretrained_path: Optional[str] = None,
                 freeze: bool = True, unfreeze_last_n: int = 0,
                 inference_only: bool = False, **kwargs):
        if img_size is None:
            img_size = self.native_img_size
        super().__init__(in_channels=in_channels, img_size=img_size,
                         pretrained=pretrained, pretrained_path=pretrained_path,
                         freeze=freeze, unfreeze_last_n=unfreeze_last_n,
                         inference_only=inference_only, **kwargs)

        self.embed_dim = _EMBED_DIM
        self.patch_size = _PATCH_SIZE

        if pretrained:
            _src = pretrained_path or _PRIMARY_HF_NAME
            try:
                import transformers
                _full_model = transformers.AutoModel.from_pretrained(
                    _src, trust_remote_code=True,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Qwen2.5-VL auto-download from '{_PRIMARY_HF_NAME}' failed: "
                    f"{type(e).__name__}: {e}. Provide a local checkpoint via "
                    f"pretrained_path."
                ) from e
            # Extract the vision tower: Qwen2.5-VL stores it under .visual or .model.visual.
            _visual = getattr(_full_model, "visual", None)
            if _visual is None:
                _model_inner = getattr(_full_model, "model", None)
                if _model_inner is not None:
                    _visual = getattr(_model_inner, "visual", None)
            if _visual is None:
                for attr in ("vision_model", "vision_tower", "image_encoder"):
                    _visual = getattr(_full_model, attr, None)
                    if _visual is not None:
                        break
            if _visual is None:
                raise RuntimeError(
                    f"Could not find vision tower in Qwen2.5-VL model from '{_src}'."
                )
            _vit = getattr(_visual, "trunk", None) or getattr(_visual, "backbone", _visual)
            if not hasattr(_vit, 'embed_dim'):
                _vit.embed_dim = _EMBED_DIM
            if not hasattr(_vit, 'num_features'):
                _vit.num_features = _EMBED_DIM
            if not hasattr(_vit, 'num_prefix_tokens'):
                _vit.num_prefix_tokens = 1
            if not hasattr(_vit, 'patch_embed'):
                class _PE:
                    pass
                _vit.patch_embed = _PE()
                _vit.patch_embed.patch_size = _PATCH_SIZE
            self.backbone = HuggingFaceViTWrapper(_vit)
            del _full_model
        else:
            raise RuntimeError(
                "Qwen25VLVisionEncoder does not support pretrained=False. "
                "This encoder requires Qwen2.5-VL vision tower weights from "
                f"'{_PRIMARY_HF_NAME}'. Pass pretrained=True to auto-download, "
                "or provide a local checkpoint via pretrained_path."
            )
        self._backbone_name = PRIMARY_BACKBONE_NAME
        self.patch_size = int(self.backbone.patch_embed.patch_size)
        self.embed_dim = int(self.backbone.embed_dim)
        self.num_prefix_tokens = int(self.backbone.num_prefix_tokens)

        if in_channels != 3:
            self.input_adapter = nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)
        else:
            self.input_adapter = nn.Identity()

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

    def _pad_to_multiple(self, x):
        H, W = x.shape[-2], x.shape[-1]
        unit = self.patch_size * 2
        new_h = int(math.ceil(H / unit) * unit) if H % unit else H
        new_w = int(math.ceil(W / unit) * unit) if W % unit else W
        if new_h != H or new_w != W:
            x = F.pad(x, (0, new_w - W, 0, new_h - H))
        return x, (H, W)

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
