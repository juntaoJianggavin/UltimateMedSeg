"""DINO ViT foundation encoder (Caron et al., 2021).

Wraps a timm ViT pretrained with the original self-DIstillation with NO labels
(DINO) recipe. We support two backbone variants:

    variant="small" -> timm 'vit_small_patch16_224.dino'   (embed_dim=384)
    variant="base"  -> timm 'vit_base_patch16_224.dino'    (embed_dim=768)

Single-resolution ViT tokens are reshaped to a 2D feature map and projected
into a four-stage DPT-style multi-block pyramid expected by the decoder family.
The deepest (lowest-resolution) feature is returned LAST, per the
``BaseFoundationEncoder`` contract.

Padding: inputs are reflect-padded so H, W are multiples of ``patch_size`` and
each pyramid feature is cropped back to its natural fraction of the original
input on the way out.
"""

from __future__ import annotations

import warnings
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.registry import ENCODER_REGISTRY
from medseg.models.encoders.foundation._base import DPTHead, BaseFoundationEncoder, load_with_ssl_fallback


_VARIANT_TO_NAME = {
    "small": "vit_small_patch16_224.dino",
    "base": "vit_base_patch16_224.dino",
}


def _build_timm_vit(variant: str, pretrained: bool, img_size: int,
                    in_chans: int):
    """Build the timm DINO ViT for the requested variant."""
    import timm

    primary = _VARIANT_TO_NAME[variant]
    return load_with_ssl_fallback(
        timm.create_model, primary,
        pretrained=pretrained, num_classes=0,
        in_chans=in_chans, img_size=img_size,
        dynamic_img_size=True,
    )


@ENCODER_REGISTRY.register("dino")
class DINOEncoder(BaseFoundationEncoder):
    """DINO ViT encoder with DPT-style multi-block multi-scale projection.

    Parameters
    ----------
    in_channels : int
        Input image channels.  Forwarded to timm's ``in_chans``.
    img_size : int
        Nominal input size used to size positional embeddings.  Actual
        forward inputs may be larger/smaller (the backbone is built with
        ``dynamic_img_size=True``).
    pretrained : bool
        Load DINO-pretrained weights from timm if True.
    pretrained_path : Optional[str]
        Optional path to a local checkpoint loaded after construction.
    freeze / unfreeze_last_n / inference_only :
        Standard freeze controls inherited via FreezeMixin.
    variant : str
        One of {"small", "base"} (default "small").
    """

    def __init__(self, in_channels: int = 3, img_size: int = 224,
                 pretrained: bool = True, pretrained_path: Optional[str] = None,
                 freeze: bool = True, unfreeze_last_n: int = 0,
                 inference_only: bool = False, variant: str = "small",
                 **kwargs):
        super().__init__(in_channels=in_channels, img_size=img_size,
                         pretrained=pretrained,
                         pretrained_path=pretrained_path,
                         freeze=freeze, unfreeze_last_n=unfreeze_last_n,
                         inference_only=inference_only, **kwargs)

        if variant not in _VARIANT_TO_NAME:
            raise ValueError(
                f"DINOEncoder: unknown variant '{variant}'. "
                f"Expected one of {sorted(_VARIANT_TO_NAME)}.")
        self.variant = variant

        # ------------------------------------------------------------------
        # Backbone
        # ------------------------------------------------------------------
        self.backbone = _build_timm_vit(
            variant=variant, pretrained=pretrained, img_size=img_size,
            in_chans=in_channels)

        # Optional local checkpoint override.
        if pretrained_path is not None:
            try:
                state = torch.load(pretrained_path, map_location="cpu")
                if isinstance(state, dict) and "model" in state:
                    state = state["model"]
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                msg = self.backbone.load_state_dict(state, strict=False)
                # Surface mismatches but do not raise.
                missing = getattr(msg, "missing_keys", [])
                unexpected = getattr(msg, "unexpected_keys", [])
                if missing or unexpected:
                    warnings.warn(
                        f"DINOEncoder: pretrained_path load: "
                        f"{len(missing)} missing, {len(unexpected)} unexpected.")
            except Exception as e:  # noqa: BLE001
                warnings.warn(
                    f"DINOEncoder: failed to load pretrained_path "
                    f"'{pretrained_path}': {e}")

        # ------------------------------------------------------------------
        # DPT-style multi-block projector
        # ------------------------------------------------------------------
        dim = int(self.backbone.embed_dim)
        patch = self.backbone.patch_embed.patch_size
        self.patch_size = int(patch[0]) if isinstance(patch, (tuple, list)) else int(patch)
        self.embed_dim = dim

        # DPT head: 从不同深度 block 构建真正多尺度金字塔
        # DPT head: genuine multi-scale pyramid from different-depth blocks
        self.dpt = DPTHead(
            embed_dim=self.embed_dim,
            num_prefix_tokens=int(self.backbone.num_prefix_tokens),
        )
        self.out_channels = self.dpt.out_channels
        self._block_indices = DPTHead.default_block_indices(len(self.backbone.blocks))

        self._maybe_inject_adapters()
        self._apply_freeze_policy()
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
