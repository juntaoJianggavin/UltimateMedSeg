"""RETFound-DINOv2 retinal foundation-model encoder (ophthalmology).

Reference:
    Zhou et al., "RETFound: A Foundation Model for Retinal Images",
    Nature Communications, 2024 (arXiv:2311.16164).  DINOv2 variant.

RETFound-DINOv2 is a DINOv2-style ViT-Large/14 (``embed_dim=1024``,
``patch_size=14``) pretrained on ~1.6 million retinal images using the
DINOv2 self-distillation framework.  This is the newer, stronger variant
of the original RETFound-MAE model.

The official weights are hosted at
``YukunZhou/RETFound_dinov2_shanghai`` on HuggingFace Hub
(a transformers ``Dinov2Model`` checkpoint).

``pretrained=True`` auto-downloads from HF Hub.
``pretrained=False`` raises ``RuntimeError``.

Registered as ``"retfound_dinov2"`` in ``ENCODER_REGISTRY``.
"""
# Source: https://huggingface.co/YukunZhou/RETFound_dinov2_shanghai
# Paper:  https://www.nature.com/articles/s41467-024-50803-3

from __future__ import annotations

import warnings
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.registry import ENCODER_REGISTRY
from medseg.models.encoders.foundation._base import BaseFoundationEncoder, load_hf_vit


_PRIMARY_HF_NAME = "YukunZhou/RETFound_dinov2_shanghai"
PRIMARY_BACKBONE_NAME = _PRIMARY_HF_NAME
_EMBED_DIM = 1024
_PATCH_SIZE = 14


@ENCODER_REGISTRY.register("retfound_dinov2")
class RETFoundDINOv2Encoder(BaseFoundationEncoder):
    """RETFound-DINOv2 (retinal DINOv2 ViT-L/14) encoder with FPN-from-tokens pyramid.

    The backbone is a DINOv2-pretrained ViT-Large/14 (``embed_dim=1024``,
    ``patch_size=14``).  ``out_channels = [dim/8, dim/4, dim/2, dim]``
    (deepest LAST).
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

        # Backbone — Dinov2Model via transformers.
        if pretrained:
            self.backbone = load_hf_vit(
                hf_name=_PRIMARY_HF_NAME,
                pretrained_path=pretrained_path,
                model_cls_name="Dinov2Model",
            )
        else:
            raise RuntimeError(
                "RETFoundDINOv2Encoder does not support pretrained=False. "
                "This encoder requires pretrained weights from "
                "'YukunZhou/RETFound_dinov2_shanghai'. Pass pretrained=True "
                "to auto-download, or provide a local checkpoint via pretrained_path."
            )
        self._backbone_name = PRIMARY_BACKBONE_NAME

        # Introspection.
        self.patch_size = int(self.backbone.patch_embed.patch_size)
        self.embed_dim = int(self.backbone.embed_dim)
        self.num_prefix_tokens = int(self.backbone.num_prefix_tokens)

        # FPN-from-tokens projector.
        dim = self.embed_dim
        c0, c1, c2, c3 = max(dim // 8, 1), max(dim // 4, 1), max(dim // 2, 1), dim
        self.out_channels: List[int] = [c0, c1, c2, c3]

        self.scale0 = nn.Sequential(
            nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
            nn.GELU(),
            nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
        )
        self.proj0 = nn.Conv2d(dim // 4, c0, kernel_size=1)

        self.scale1 = nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2)
        self.proj1 = nn.Conv2d(dim // 2, c1, kernel_size=1)

        self.scale2 = nn.Identity()
        self.proj2 = nn.Conv2d(dim, c2, kernel_size=1)

        self.scale3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.proj3 = nn.Conv2d(dim, c3, kernel_size=1)

        self._maybe_inject_adapters()
        self._apply_freeze_policy()

    # ------------------------------------------------------------------
    def _tokens_to_grid(self, tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
        if tokens.dim() == 4:
            if tokens.shape[1] == self.embed_dim:
                return tokens.contiguous()
            return tokens.permute(0, 3, 1, 2).contiguous()
        B, N, C = tokens.shape
        target = h * w
        if N == target + self.num_prefix_tokens:
            tokens = tokens[:, self.num_prefix_tokens:, :]
        elif N == target + 1:
            tokens = tokens[:, 1:, :]
        elif N != target:
            extra = N - target
            if extra > 0:
                tokens = tokens[:, extra:, :]
            else:
                raise RuntimeError(
                    f"RETFoundDINOv2Encoder: token count {N} incompatible with grid {h}x{w}.")
        return tokens.transpose(1, 2).contiguous().view(B, C, h, w)

    @staticmethod
    def _crop_or_resize(t: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        H, W = t.shape[-2:]
        if H == target_h and W == target_w:
            return t
        if H >= target_h and W >= target_w:
            return t[..., :target_h, :target_w].contiguous()
        return F.interpolate(t, size=(target_h, target_w), mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.input_adapter(x)
        B, _, H, W = x.shape
        p = self.patch_size
        pad_h = (p - H % p) % p
        pad_w = (p - W % p) % p
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        Hp, Wp = x.shape[-2], x.shape[-1]
        h, w = Hp // p, Wp // p

        tokens = self.backbone.forward_features(x)
        if isinstance(tokens, (list, tuple)):
            tokens = tokens[0]
        grid = self._tokens_to_grid(tokens, h, w)

        s0 = self.proj0(self.scale0(grid))
        s1 = self.proj1(self.scale1(grid))
        s2 = self.proj2(self.scale2(grid))
        s3 = self.proj3(self.scale3(grid))

        sizes = [
            (max(H // 4, 1), max(W // 4, 1)),
            (max(H // 8, 1), max(W // 8, 1)),
            (max(H // 16, 1), max(W // 16, 1)),
            (max(H // 32, 1), max(W // 32, 1)),
        ]
        return [
            self._crop_or_resize(s0, *sizes[0]),
            self._crop_or_resize(s1, *sizes[1]),
            self._crop_or_resize(s2, *sizes[2]),
            self._crop_or_resize(s3, *sizes[3]),
        ]
