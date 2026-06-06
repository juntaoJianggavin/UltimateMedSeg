"""HuatuoGPT-Vision medical MLLM vision encoder (foundation-model encoder).

HuatuoGPT-Vision (USTC, 2024) is a medical multimodal LLM. Its vision tower
is a CLIP ViT-L/14 (embed_dim=1024, depth=24, patch_size=14) operating at a
native input resolution of 336x336 — identical in structure to OpenAI's
CLIP ViT-L/14-336, but (further-)tuned on biomedical figure-caption data.

``pretrained=True`` auto-downloads from ``openai/clip-vit-large-patch14-336``
via ``transformers.CLIPVisionModel`` (same architecture as HuatuoGPT's vision
tower). ``pretrained=False`` raises RuntimeError.

Output:
    forward(x) -> List[Tensor] of length 4 with approximate spatial strides
    [H/4, H/8, H/16, H/32] and channels [dim/8, dim/4, dim/2, dim]
    (deepest LAST).
"""
# Source: https://github.com/FreedomIntelligence/HuatuoGPT-Vision
# Vision tower: openai/clip-vit-large-patch14-336

from __future__ import annotations

import warnings
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.registry import ENCODER_REGISTRY
from medseg.models.encoders.foundation._base import DPTHead, BaseFoundationEncoder, load_hf_vit


_PRIMARY_HF_NAME = "openai/clip-vit-large-patch14-336"
PRIMARY_BACKBONE_NAME = _PRIMARY_HF_NAME
PRIMARY_PATCH_SIZE = 14


@ENCODER_REGISTRY.register("huatuogpt_vision")
class HuatuoGPTVisionEncoder(BaseFoundationEncoder):
    """HuatuoGPT-Vision vision-tower encoder with DPT-style multi-block output.

    Constructor follows the BaseFoundationEncoder contract. The HuatuoGPT-
    Vision tower is structurally a CLIP ViT-L/14 (1024-dim, 24 blocks,
    patch 14) with native input 336x336.
    """

    native_img_size: int = 336

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

        if pretrained:
            self.backbone = load_hf_vit(
                hf_name=_PRIMARY_HF_NAME,
                pretrained_path=pretrained_path,
                model_cls_name="CLIPVisionModel",
            )
        else:
            raise RuntimeError(
                "HuatuoGPTVisionEncoder does not support pretrained=False. "
                "This encoder requires CLIP ViT-L/14-336 weights. "
                "Pass pretrained=True to auto-download, or provide a local "
                "checkpoint via pretrained_path."
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
