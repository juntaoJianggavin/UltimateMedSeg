"""PLIP foundation-model encoder (pathology vision-language).

Reference: Huang et al., "A visual-language foundation model for pathology
image analysis using medical Twitter", Nature Medicine, 2023.

PLIP is a CLIP-style ViT-B/32 dual encoder fine-tuned on ~200k pathology
image-text pairs scraped from medical Twitter (OpenPath). Its vision tower
shares the architecture of OpenAI CLIP ViT-B/32 (``embed_dim=768``,
``patch_size=32``). The community release is hosted on Hugging Face at
``vinid/plip``.

``pretrained=True`` auto-downloads the vision tower via
``transformers.CLIPVisionModel``. ``pretrained=False`` raises
``RuntimeError``.

Registered as ``"plip"`` in ``ENCODER_REGISTRY``.
"""
# Source: https://huggingface.co/vinid/plip

from __future__ import annotations

import math
import warnings
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.registry import ENCODER_REGISTRY
from medseg.models.encoders.foundation._base import DPTHead, BaseFoundationEncoder, load_hf_vit


# Architecture constants for CLIP ViT-B/32.
_PLIP_EMBED_DIM = 768
_PLIP_PATCH_SIZE = 32
_PLIP_NATIVE_IMG_SIZE = 224

# Community PLIP release on the Hugging Face hub.
_PRIMARY_HF_NAME = "vinid/plip"
PRIMARY_BACKBONE_NAME = _PRIMARY_HF_NAME


@ENCODER_REGISTRY.register("plip")
class PLIPEncoder(BaseFoundationEncoder):
    """PLIP (pathology CLIP ViT-B/32) encoder with DPT-style multi-block output.

    The backbone is a ViT-Base/32 (``embed_dim=768``, ``patch_size=32``).
    Its final-layer patch tokens are reshaped to ``(B, 768, h, w)`` and
    projected by four independent ``Conv2d`` ops to a 4-stage pyramid:

    * stage 0 - ``ConvTranspose2d`` 8x up  -> 1x1 conv to ``dim/8``  channels
    * stage 1 - ``ConvTranspose2d`` 4x up  -> 1x1 conv to ``dim/4``  channels
    * stage 2 - ``Identity``               -> 1x1 conv to ``dim/2``  channels
    * stage 3 - ``MaxPool2d`` 2x down      -> 1x1 conv to ``dim``    channels

    ``out_channels = [dim/8, dim/4, dim/2, dim]`` (deepest LAST).
    """

    PATCH_SIZE = _PLIP_PATCH_SIZE
    EMBED_DIM = _PLIP_EMBED_DIM
    native_img_size: int = _PLIP_NATIVE_IMG_SIZE

    def __init__(self, in_channels: int = 3, img_size: Optional[int] = None,
                 pretrained: bool = True, pretrained_path: Optional[str] = None,
                 freeze: bool = True, unfreeze_last_n: int = 0,
                 inference_only: bool = False, **kwargs):
        if img_size is None:
            img_size = self.native_img_size

        super().__init__(
            in_channels=in_channels, img_size=img_size,
            pretrained=pretrained, pretrained_path=pretrained_path,
            freeze=freeze, unfreeze_last_n=unfreeze_last_n,
            inference_only=inference_only, **kwargs,
        )

        # ---- channel adapter for non-RGB inputs (PLIP is RGB-only) -------
        if in_channels != 3:
            self.input_adapter: nn.Module = nn.Conv2d(in_channels, 3,
                                                     kernel_size=1, bias=False)
        else:
            self.input_adapter = nn.Identity()

        # ---- backbone — loaded via transformers CLIPVisionModel -----------
        ref_size = self._pad_to_multiple_size(int(img_size))

        if pretrained:
            self.backbone = load_hf_vit(
                hf_name=_PRIMARY_HF_NAME,
                pretrained_path=pretrained_path,
                model_cls_name="CLIPVisionModel",
            )
        else:
            raise RuntimeError(
                "PLIPEncoder does not support pretrained=False. "
                "This encoder requires pretrained weights from 'vinid/plip'. "
                "Pass pretrained=True to auto-download, or provide a local "
                "checkpoint via pretrained_path."
            )
        self._backbone_name = PRIMARY_BACKBONE_NAME

        # ---- backbone introspection -------------------------------------
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

    @classmethod
    def _pad_to_multiple_size(cls, size: int) -> int:
        m = cls.PATCH_SIZE * 8  # 32 * 8 = 256
        return ((int(size) + m - 1) // m) * m

    def _pad_input(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        _, _, H, W = x.shape
        ps = self.patch_size
        H_pad = int(math.ceil(H / ps) * ps)
        W_pad = int(math.ceil(W / ps) * ps)
        if H_pad != H or W_pad != W:
            x = F.pad(x, (0, W_pad - W, 0, H_pad - H))
        return x, H_pad, W_pad
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
