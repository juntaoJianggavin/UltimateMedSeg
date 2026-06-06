"""BiomedCLIP image encoder (foundation-model encoder).

BiomedCLIP (Microsoft, 2023) is a ViT-B/16 image tower paired with a
PubMedBERT text tower, contrastively pre-trained on ~15M biomedical
image-text pairs from PMC. We only use the vision tower as a backbone
here.

The official weights live under
``microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`` on the
HuggingFace Hub. This is an open_clip-format checkpoint; the vision tower
is loaded natively via ``open_clip.create_model``. ``pretrained=True``
auto-downloads from HF Hub; ``pretrained_path`` accepts a local checkpoint.
``pretrained=False`` is not supported (raises ``RuntimeError``).

Output:
    forward(x) -> List[Tensor] of length 4. Approximate spatial strides
    [H/4, H/8, H/16, H/32]; channels [dim/8, dim/4, dim/2, dim] with
    dim=768 (ViT-B/16).
"""
# Source: https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224

from __future__ import annotations

import warnings
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.registry import ENCODER_REGISTRY
from medseg.models.encoders.foundation._base import DPTHead, BaseFoundationEncoder, load_with_ssl_fallback, hf_hub_download_vision_weights


# Canonical embed dim / patch for BiomedCLIP (ViT-B/16).
_EMBED_DIM = 768
_PATCH_SIZE = 16

# BiomedCLIP vision tower is ViT-B/16 from open_clip.
# Official HF artifact: microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224.
PRIMARY_BACKBONE_NAME = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


@ENCODER_REGISTRY.register("biomedclip")
class BiomedCLIPEncoder(BaseFoundationEncoder):
    """BiomedCLIP ViT-B/16 image encoder with DPT-style multi-block multi-scale output.

    Constructor follows the BaseFoundationEncoder contract. Auto-downloads
    BiomedCLIP weights from HuggingFace Hub.
    """

    def __init__(self, in_channels: int = 3, img_size: int = 224,
                 pretrained: bool = True, pretrained_path: Optional[str] = None,
                 freeze: bool = True, unfreeze_last_n: int = 0,
                 inference_only: bool = False, **kwargs):
        super().__init__(in_channels=in_channels, img_size=img_size,
                         pretrained=pretrained, pretrained_path=pretrained_path,
                         freeze=freeze, unfreeze_last_n=unfreeze_last_n,
                         inference_only=inference_only, **kwargs)

        self.embed_dim = _EMBED_DIM
        self.patch_size = _PATCH_SIZE

        # Load the BiomedCLIP vision tower via open_clip (native loading).
        # No silent fallback: raises RuntimeError if weights cannot be obtained.
        if pretrained:
            try:
                import open_clip
            except ImportError:
                raise RuntimeError(
                    "BiomedCLIPEncoder requires open_clip. "
                    "Install with: pip install open_clip_torch"
                )
            _hf_name = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
            try:
                _clip_model = open_clip.create_model(
                    _hf_name, pretrained=not pretrained_path,
                )
            except Exception as e:
                raise RuntimeError(
                    "BiomedCLIP auto-download from "
                    f"'{_hf_name}' failed: {type(e).__name__}: {e}. "
                    "Provide a local checkpoint via pretrained_path."
                ) from e
            self.backbone = _clip_model.visual
            del _clip_model
        else:
            raise RuntimeError(
                "BiomedCLIPEncoder does not support pretrained=False. "
                "This encoder requires pretrained weights from "
                "'microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'. "
                "Pass pretrained=True to auto-download, or provide a local "
                "checkpoint via pretrained_path."
            )
        self._backbone_name = PRIMARY_BACKBONE_NAME
        self.embed_dim = int(getattr(self.backbone, "embed_dim", _EMBED_DIM))
        _ps = getattr(getattr(self.backbone, "patch_embed", None), "patch_size", _PATCH_SIZE)
        if isinstance(_ps, (tuple, list)):
            self.patch_size = int(_ps[0])
        else:
            self.patch_size = int(_ps)
        self.num_prefix_tokens = int(getattr(self.backbone, "num_prefix_tokens", 1))

        # Adapter for non-RGB inputs (BiomedCLIP expects 3 channels).
        if in_channels != 3:
            self.input_adapter = nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)
        else:
            self.input_adapter = nn.Identity()

        # DPT-style multi-block: project the token grid (B, dim, h, w) to four
        # pyramid stages with channels [dim/8, dim/4, dim/2, dim].
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
