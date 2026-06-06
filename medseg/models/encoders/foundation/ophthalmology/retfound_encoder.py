"""RETFound retinal foundation-model encoder (ophthalmology).

Reference:
    Zhou et al., "RETFound: A Foundation Model for Retinal Images",
    Nature Communications, 2024 (arXiv:2311.16164).

RETFound is a MAE-style self-supervised ViT-Large/16 (``embed_dim=1024``,
``patch_size=16``) pre-trained on ~1.6 million retinal images.  The
official weights are hosted at ``open-eye/RETFound_MAE`` on HuggingFace
Hub (a transformers ``ViTModel`` checkpoint).

``pretrained=True`` auto-downloads from HF Hub.
``pretrained=False`` raises ``RuntimeError``.

The ViT token grid is reshaped to ``(B, dim, h, w)`` and projected into
a four-stage DPT-style multi-block pyramid (deepest LAST), matching the
``BaseFoundationEncoder`` contract used by the rest of the foundation
family.

Registered as ``"retfound"`` in ``ENCODER_REGISTRY``.
"""
# Source: https://huggingface.co/open-eye/RETFound_MAE

from __future__ import annotations

import warnings
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.registry import ENCODER_REGISTRY
from medseg.models.encoders.foundation._base import DPTHead, BaseFoundationEncoder, load_hf_vit


_PRIMARY_HF_NAME = "open-eye/RETFound_MAE"
PRIMARY_BACKBONE_NAME = _PRIMARY_HF_NAME
_EMBED_DIM = 1024
_PATCH_SIZE = 16


@ENCODER_REGISTRY.register("retfound")
class RETFoundEncoder(BaseFoundationEncoder):
    """RETFound (retinal MAE ViT-L/16) encoder with DPT-style multi-block pyramid.

    The backbone is a MAE-pretrained ViT-Large/16 (``embed_dim=1024``,
    ``patch_size=16``).  Its final-layer patch tokens are reshaped to
    ``(B, 1024, h, w)`` and projected by four DPT-style multi-block stages into
    a multi-scale pyramid (deepest LAST):

    * stage 0 — 4x up   -> 1x1 conv to ``dim/8`` channels  (~ H/4)
    * stage 1 — 2x up   -> 1x1 conv to ``dim/4`` channels  (~ H/8)
    * stage 2 — identity -> 1x1 conv to ``dim/2`` channels (~ H/16)
    * stage 3 — 2x down -> 1x1 conv to ``dim``   channels  (~ H/32)

    ``out_channels = [dim/8, dim/4, dim/2, dim]`` (deepest LAST).

    Args:
        in_channels: Number of input image channels (default 3).
        img_size: Reference spatial size (default 224).
        pretrained: If True, auto-download from HF Hub.
        pretrained_path: Optional local checkpoint path.
        freeze / unfreeze_last_n / inference_only: Standard freeze controls.
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

        # Channel adapter for non-RGB inputs.
        if in_channels != 3:
            self.input_adapter: nn.Module = nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)
        else:
            self.input_adapter = nn.Identity()

        # Backbone — ViTModel via transformers.
        if pretrained:
            self.backbone = load_hf_vit(
                hf_name=_PRIMARY_HF_NAME,
                pretrained_path=pretrained_path,
                model_cls_name="ViTModel",
            )
        else:
            raise RuntimeError(
                "RETFoundEncoder does not support pretrained=False. "
                "This encoder requires pretrained weights from "
                "'open-eye/RETFound_MAE'. Pass pretrained=True to "
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
