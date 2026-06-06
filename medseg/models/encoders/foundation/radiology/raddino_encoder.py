"""Rad-DINO foundation-model encoder (DPT-style multi-block pyramid).

Rad-DINO 放射科基础模型编码器，使用 DPT 风格的多 block 特征金字塔。

Reference:
    Perez-Garcia et al., "RAD-DINO: Exploring Scalable Medical Image
    Encoders Beyond Text Supervision", Microsoft Research, 2024.
    Ranftl et al., "Vision Transformers for Dense Prediction", ICCV 2021 (DPT head).

从 ViT 的 4 个均匀间隔 block 提取 token，构建真正多层次语义金字塔。
Extracts tokens from 4 evenly-spaced ViT blocks to build a genuine
multi-level semantic pyramid (shallow=texture, deep=semantics).

Registered under the key ``"raddino"`` in ``ENCODER_REGISTRY``.
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.registry import ENCODER_REGISTRY
from medseg.models.encoders.foundation._base import DPTHead, BaseFoundationEncoder, load_hf_vit


# Primary weight source: HuggingFace Hub (microsoft/rad-dino, Dinov2Model).
PRIMARY_BACKBONE_NAME = "microsoft/rad-dino"


@ENCODER_REGISTRY.register("raddino")
class RadDINOEncoder(BaseFoundationEncoder):
    """Rad-DINO chest-X-ray encoder + DPT head 多尺度金字塔。
    Rad-DINO chest-X-ray encoder with DPT-style multi-scale pyramid.

    与之前的 FPN-from-tokens 方案不同，DPT head 从 ViT 的不同深度 block
    提取 token，每一级的语义抽象层次真正不同（浅层=纹理，深层=语义）。
    Unlike the previous FPN-from-tokens approach, the DPT head extracts
    tokens from different-depth blocks so each level has genuinely
    different semantic abstraction.

    Parameters
    ----------
    in_channels : int
        Number of input image channels.
    img_size : int
        Reference spatial size used to instantiate the backbone's positional
        embedding grid.
    pretrained : bool
        Load Rad-DINO weights from HuggingFace Hub via
        ``transformers.Dinov2Model``.
    pretrained_path : Optional[str]
        Optional path to a local Rad-DINO checkpoint.
    freeze / unfreeze_last_n / inference_only :
        Standard freeze controls inherited via :class:`FreezeMixin`.
    """

    PATCH_SIZE = 14
    PAD_MULTIPLE = 14 * 8

    def __init__(self, in_channels: int = 3, img_size: int = 224,
                 pretrained: bool = True, pretrained_path: Optional[str] = None,
                 freeze: bool = True, unfreeze_last_n: int = 0,
                 inference_only: bool = False, **kwargs):
        super().__init__(in_channels=in_channels, img_size=img_size,
                         pretrained=pretrained, pretrained_path=pretrained_path,
                         freeze=freeze, unfreeze_last_n=unfreeze_last_n,
                         inference_only=inference_only, **kwargs)

        # ------------------------------------------------------------------
        # Backbone -- DINOv2 ViT-B/14 via transformers.
        # ------------------------------------------------------------------
        self._backbone_name = PRIMARY_BACKBONE_NAME
        if pretrained:
            self.backbone = load_hf_vit(
                hf_name=PRIMARY_BACKBONE_NAME,
                pretrained_path=pretrained_path,
                model_cls_name="Dinov2Model",
            )
        else:
            raise RuntimeError(
                "RadDINOEncoder does not support pretrained=False. "
                "This encoder requires pretrained weights from 'microsoft/rad-dino'. "
                "Pass pretrained=True to auto-download, or provide a local "
                "checkpoint via pretrained_path."
            )

        # ------------------------------------------------------------------
        # Backbone introspection.
        # ------------------------------------------------------------------
        self.patch_size = int(self.backbone.patch_embed.patch_size)
        self._pad_multiple = self.patch_size * 8
        self._embed_dim = int(self.backbone.embed_dim)
        self._num_prefix_tokens = int(self.backbone.num_prefix_tokens)

        dim = self._embed_dim

        # ------------------------------------------------------------------
        # DPT head: 从不同深度 block 构建真正多尺度金字塔
        # DPT head: genuine multi-scale pyramid from different-depth blocks
        # ------------------------------------------------------------------
        self.dpt = DPTHead(
            embed_dim=dim,
            num_prefix_tokens=self._num_prefix_tokens,
        )
        self.out_channels = self.dpt.out_channels
        self._block_indices = DPTHead.default_block_indices(len(self.backbone.blocks))

        self._maybe_inject_adapters()
        self._apply_freeze_policy()

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------
    @classmethod
    def _pad_to_multiple(cls, size: int) -> int:
        m = cls.PAD_MULTIPLE
        return ((int(size) + m - 1) // m) * m

    def _pad_input(self, x: torch.Tensor):
        _, _, H, W = x.shape
        m = getattr(self, "_pad_multiple", self.PAD_MULTIPLE)
        Hp = ((H + m - 1) // m) * m
        Wp = ((W + m - 1) // m) * m
        pad_h, pad_w = Hp - H, Wp - W
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        return x, (H, W), (Hp, Wp)

    # ----------------------------------------------------------------------
    # Forward
    # ----------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x_pad, (H, W), (Hp, Wp) = self._pad_input(x)

        # 从不同深度 block 提取 token（DPT 核心）
        # Extract tokens from different-depth blocks (DPT core)
        multi_tokens = self.backbone.get_intermediate_layers(
            x_pad, n=self._block_indices,
        )

        h_patches = Hp // self.patch_size
        w_patches = Wp // self.patch_size

        return self.dpt(list(multi_tokens), h_patches, w_patches, H, W)
