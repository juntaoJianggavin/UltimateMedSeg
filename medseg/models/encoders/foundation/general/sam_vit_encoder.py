"""SAM image encoder backed by a timm ViT (B/L/H) with an DPT-style multi-block head.

This module exposes :class:`SAMViTEncoder` (registered key ``"sam_vit"``).
It loads a plain ViT backbone via :mod:`timm`, drops the prefix (CLS /
register) tokens, reshapes the patch-token sequence to a ``(B, dim, h, w)``
grid at scale ``H / patch_size`` and then synthesises four multi-scale
features at approximate scales ``[H/4, H/8, H/16, H/32]`` following the
ViTDet/SAM DPT-style multi-block recipe:

    * scale[0] (H/4)  -> ConvTranspose2d 4x upsample + 1x1 reduce
    * scale[1] (H/8)  -> ConvTranspose2d 2x upsample + 1x1 reduce
    * scale[2] (H/16) -> Identity + 1x1 reduce
    * scale[3] (H/32) -> MaxPool2d stride 2 + 1x1 reduce  (deepest LAST)

``out_channels = [dim/8, dim/4, dim/2, dim]``.
"""

from __future__ import annotations

import math
import warnings
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.registry import ENCODER_REGISTRY
from medseg.models.encoders.foundation._base import DPTHead, BaseFoundationEncoder, load_with_ssl_fallback


# Primary backbone name per variant.
_VARIANT_TO_NAME = {
    "base":  "vit_base_patch16_224",
    "large": "vit_large_patch16_224",
    "huge":  "vit_huge_patch14_224",
}


def _build_timm_vit(variant: str, img_size: int, in_chans: int,
                    pretrained: bool):
    """Create a timm ViT backbone for the requested variant."""
    import timm

    PRIMARY_BACKBONE_NAME = _VARIANT_TO_NAME[variant]
    model = load_with_ssl_fallback(
        timm.create_model, PRIMARY_BACKBONE_NAME,
        pretrained=pretrained,
        num_classes=0,
        img_size=img_size,
        in_chans=in_chans,
    )
    return model, PRIMARY_BACKBONE_NAME


@ENCODER_REGISTRY.register("sam_vit")
class SAMViTEncoder(BaseFoundationEncoder):
    native_img_size: int = 1024   # SAM ViT was pretrained at 1024x1024

    """SAM-style ViT encoder with an DPT-style multi-block multi-scale head.

    Parameters
    ----------
    in_channels:
        Number of input image channels. Passed straight to the ViT patch
        embedding so non-RGB inputs are also supported (pretrained weights are
        only meaningful for ``in_channels == 3``).
    img_size:
        Input spatial size the backbone is *built* for. Inputs at runtime are
        right/bottom padded to a multiple of ``patch_size * 2`` so the /32
        branch always has at least one pixel; outputs are cropped back to
        ``ceil(H / scale)`` to match the original spatial region.
    pretrained:
        Whether to download timm pretrained weights for the backbone.
    pretrained_path:
        Optional path to a local SAM/ViT state-dict loaded into the backbone
        with ``strict=False``. Takes precedence over ``pretrained=True``.
    freeze / unfreeze_last_n / inference_only:
        Standard freezing controls from :class:`BaseFoundationEncoder`.
    vit_variant:
        One of ``"base"`` (default), ``"large"``, ``"huge"``.
    """

    def __init__(
        self,
        in_channels: int = 3,
        img_size: int = 224,
        pretrained: bool = True,
        pretrained_path: Optional[str] = None,
        freeze: bool = True,
        unfreeze_last_n: int = 0,
        inference_only: bool = False,
        vit_variant: str = "base",
        **kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            img_size=img_size,
            pretrained=pretrained,
            pretrained_path=pretrained_path,
            freeze=freeze,
            unfreeze_last_n=unfreeze_last_n,
            inference_only=inference_only,
            **kwargs,
        )

        if vit_variant not in _VARIANT_TO_NAME:
            raise ValueError(
                f"[sam_vit] unknown vit_variant='{vit_variant}'. "
                f"Expected one of {sorted(_VARIANT_TO_NAME)}."
            )
        self.vit_variant = vit_variant

        # ---- Backbone --------------------------------------------------
        # When a custom checkpoint path is given we skip the timm pretrained
        # download (the local file will replace those weights anyway).
        self.backbone, self._backbone_name = _build_timm_vit(
            variant=vit_variant,
            img_size=img_size,
            in_chans=in_channels,
            pretrained=pretrained and pretrained_path is None,
        )

        if pretrained_path is not None:
            try:
                state = torch.load(pretrained_path, map_location="cpu")
                if isinstance(state, dict) and "model" in state:
                    state = state["model"]
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                msg = self.backbone.load_state_dict(state, strict=False)
                print(
                    f"[sam_vit] loaded weights from {pretrained_path}: {msg}"
                )
            except Exception as e:  # pragma: no cover
                warnings.warn(
                    f"[sam_vit] failed to load '{pretrained_path}': {e}"
                )

        # ---- Geometry --------------------------------------------------
        self.embed_dim = int(getattr(self.backbone, "embed_dim", 768))
        ps = self.backbone.patch_embed.patch_size
        if isinstance(ps, (tuple, list)):
            self.patch_size = int(ps[0])
        else:
            self.patch_size = int(ps)
        self.num_prefix_tokens = int(
            getattr(self.backbone, "num_prefix_tokens", 1)
        )

        # ---- Output channels (deepest LAST) ----------------------------
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

    def _pad_to_multiple(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Right/bottom pad ``x`` so H,W are multiples of ``patch_size * 2``.

        The factor of 2 ensures the /32 (MaxPool2d) branch sees at least one
        even-sided input. Returns the padded tensor and the original ``(H, W)``.
        """
        H, W = x.shape[-2], x.shape[-1]
        unit = self.patch_size * 2
        new_h = int(math.ceil(H / unit) * unit) if H % unit else H
        new_w = int(math.ceil(W / unit) * unit) if W % unit else W
        if new_h != H or new_w != W:
            x = F.pad(x, (0, new_w - W, 0, new_h - H))
        return x, (H, W)
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
