"""MoNET (Kim et al., Nature Medicine 2024) — Medical cONcept rETriever for dermatology.

MoNET is a CLIP-style vision-language foundation model trained on ~105K
dermatology image-text pairs from medical literature. Vision tower: CLIP
ViT-L/14 (hidden_size=1024, num_layers=24, patch_size=14).

Verified artifact: ``suinleelab/monet`` (HuggingFace, public, ungated). The
HF repo stores a ``transformers.CLIPModel`` checkpoint (config.json +
model.safetensors with the standard HF CLIP layout) — NOT a timm-native
checkpoint — so we load it via ``transformers.CLIPModel.from_pretrained``
and keep only the vision tower.
"""
# Source: https://huggingface.co/suinleelab/monet
# Source: https://github.com/suinleelab/MONET
# Citation: Kim, C., Gadgil, S.U., DeGrave, A.J. et al. Transparent medical
#   image AI via an image-text foundation model grounded in medical literature.
#   Nature Medicine 30, 1154-1165 (2024).

from __future__ import annotations
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.models.encoders.foundation._base import DPTHead, BaseFoundationEncoder, load_with_ssl_fallback, HuggingFaceViTWrapper
from medseg.registry import ENCODER_REGISTRY


MONET_HF_NAME = "suinleelab/monet"  # transformers CLIPModel artifact


def _build_clip_vision(pretrained: bool, hf_name: str):
    """Load the MoNET CLIP vision tower via transformers (no timm)."""
    try:
        from transformers import CLIPModel, CLIPVisionConfig, CLIPVisionModel
    except ImportError as e:
        raise RuntimeError(
            "MoNET encoder requires the `transformers` library. "
            "Install with: pip install transformers"
        ) from e

    if pretrained:
        full_clip = load_with_ssl_fallback(
            CLIPModel.from_pretrained, hf_name,
        )
        vision = full_clip.vision_model  # drops text tower
        del full_clip
        return vision

    raise RuntimeError(
        "MoNETDermEncoder does not support pretrained=False. "
        "This encoder requires pretrained weights from 'suinleelab/monet'. "
        "Pass pretrained=True to auto-download, or provide a local "
        "checkpoint via pretrained_path."
    )


@ENCODER_REGISTRY.register("monet_derm")
class MoNETDermEncoder(BaseFoundationEncoder):
    """MoNET CLIP ViT-L/14 dermatology vision encoder (transformers-loaded)."""

    native_img_size: int = 224

    def __init__(self, in_channels=3, img_size=None, pretrained=True,
                 pretrained_path=None, freeze=True, unfreeze_last_n=0,
                 inference_only=False, **kwargs):
        super().__init__(in_channels=in_channels,
                         img_size=img_size or self.native_img_size,
                         pretrained=pretrained, pretrained_path=pretrained_path,
                         freeze=freeze, unfreeze_last_n=unfreeze_last_n,
                         inference_only=inference_only, **kwargs)

        self.input_adapter = (nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)
                              if in_channels != 3 else nn.Identity())

        # Resolve where to load from (local path > explicit hf-hub override > MONET default)
        hf_name = pretrained_path if pretrained_path else MONET_HF_NAME

        _vision = _build_clip_vision(pretrained, hf_name)
        self.backbone = HuggingFaceViTWrapper(_vision)

        self._dim = int(self.backbone.embed_dim)
        self._patch_size = int(self.backbone.patch_embed.patch_size)

        # DPT head: 从不同深度 block 构建真正多尺度金字塔
        # DPT head: genuine multi-scale pyramid from different-depth blocks
        self.dpt = DPTHead(
            embed_dim=self._dim,
            num_prefix_tokens=int(self.backbone.num_prefix_tokens),
        )
        self.out_channels = self.dpt.out_channels
        self._block_indices = DPTHead.default_block_indices(len(self.backbone.blocks))

        self._maybe_inject_adapters()
        self._apply_freeze_policy()
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.input_adapter(x)
        B, _, H, W = x.shape
        p = self._patch_size

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
