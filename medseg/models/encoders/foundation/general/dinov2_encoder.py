"""DINOv2 foundation-model encoder (DPT-style multi-block pyramid).

DINOv2 基础模型编码器，使用 DPT 风格的多 block 特征金字塔。

Reference:
    Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision", 2024.
    Ranftl et al., "Vision Transformers for Dense Prediction", ICCV 2021 (DPT head).

从 ViT 的 4 个均匀间隔 block 提取 token，构建真正多层次语义金字塔。
Extracts tokens from 4 evenly-spaced ViT blocks to build a genuine
multi-level semantic pyramid (shallow=texture, deep=semantics).
"""

from __future__ import annotations
import warnings
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.registry import ENCODER_REGISTRY
from medseg.models.encoders.foundation._base import BaseFoundationEncoder, DPTHead, load_with_ssl_fallback


_VARIANT_TO_NAME = {
    "small": "vit_small_patch14_dinov2",
    "base":  "vit_base_patch14_dinov2",
    "large": "vit_large_patch14_dinov2",
    "giant": "vit_giant_patch14_dinov2",
}


@ENCODER_REGISTRY.register("dinov2")
class DINOv2Encoder(BaseFoundationEncoder):
    """DINOv2 ViT encoder + DPT head 多尺度金字塔。
    DINOv2 ViT encoder with DPT-style multi-scale pyramid.

    与之前的 FPN-from-tokens 方案不同，DPT head 从 ViT 的不同深度 block
    提取 token，每一级的语义抽象层次真正不同（浅层=纹理，深层=语义）。
    Unlike the previous FPN-from-tokens approach, the DPT head extracts
    tokens from different-depth blocks so each level has genuinely
    different semantic abstraction.

    Args:
        variant: "small" / "base" / "large" / "giant"
    """

    PATCH_SIZE = 14
    PAD_MULTIPLE = 14 * 4  # 需要被 patch_size * 4 整除以保证金字塔尺寸
                            # Must be divisible by patch_size * 4 for pyramid sizes

    def __init__(self, in_channels: int = 3, img_size: int = 224,
                 pretrained: bool = True, pretrained_path: Optional[str] = None,
                 freeze: bool = True, unfreeze_last_n: int = 0,
                 inference_only: bool = False,
                 variant: str = "small", **kwargs):
        super().__init__(
            in_channels=in_channels, img_size=img_size,
            pretrained=pretrained, pretrained_path=pretrained_path,
            freeze=freeze, unfreeze_last_n=unfreeze_last_n,
            inference_only=inference_only, **kwargs,
        )

        if variant not in _VARIANT_TO_NAME:
            raise ValueError(
                f"Unknown DINOv2 variant '{variant}'. "
                f"Expected one of {sorted(_VARIANT_TO_NAME)}."
            )
        self.variant = variant

        import timm
        ref_size = self._pad_to_multiple(int(img_size))
        self.backbone = load_with_ssl_fallback(
            timm.create_model, _VARIANT_TO_NAME[variant],
            pretrained=pretrained,
            img_size=ref_size,
            in_chans=in_channels,
            dynamic_img_size=True,
        )

        if pretrained_path:
            try:
                sd = torch.load(pretrained_path, map_location="cpu")
                if isinstance(sd, dict) and "model" in sd:
                    sd = sd["model"]
                msg = self.backbone.load_state_dict(sd, strict=False)
                warnings.warn(f"DINOv2 loaded local weights from {pretrained_path}: {msg}")
            except Exception as e:
                warnings.warn(f"Failed to load DINOv2 checkpoint {pretrained_path}: {e}")

        dim = int(self.backbone.embed_dim)
        self._embed_dim = dim
        self._num_prefix_tokens = int(getattr(self.backbone, "num_prefix_tokens", 1))
        self._num_blocks = len(self.backbone.blocks)

        # DPT head: 从 4 个不同深度的 block 构建真正多尺度金字塔
        # DPT head: build genuine multi-scale pyramid from 4 different-depth blocks
        self.dpt = DPTHead(
            embed_dim=dim,
            num_prefix_tokens=self._num_prefix_tokens,
        )
        self.out_channels = self.dpt.out_channels

        self._block_indices = DPTHead.default_block_indices(self._num_blocks)

        self._maybe_inject_adapters()
        self._apply_freeze_policy()

    @classmethod
    def _pad_to_multiple(cls, size: int) -> int:
        m = cls.PAD_MULTIPLE
        return ((int(size) + m - 1) // m) * m

    def _pad_input(self, x: torch.Tensor):
        _, _, H, W = x.shape
        Hp, Wp = self._pad_to_multiple(H), self._pad_to_multiple(W)
        pad_h, pad_w = Hp - H, Wp - W
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        return x, (H, W), (Hp, Wp)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x_pad, (H, W), (Hp, Wp) = self._pad_input(x)

        # 从不同深度 block 提取 token（DPT 核心）
        # Extract tokens from different-depth blocks (DPT core)
        multi_tokens = self.backbone.get_intermediate_layers(
            x_pad, n=self._block_indices
        )

        h_patches = Hp // self.PATCH_SIZE
        w_patches = Wp // self.PATCH_SIZE

        return self.dpt(list(multi_tokens), h_patches, w_patches, H, W)
