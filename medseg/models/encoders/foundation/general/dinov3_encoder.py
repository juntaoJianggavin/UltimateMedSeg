"""DINOv3 foundation-model encoder (Meta 2025) with DPT-style multi-block multi-scale.

Wraps a timm ViT pretrained with DINOv3 self-supervision and converts its
single-scale patch-token grid into a 4-stage feature pyramid by applying
separate Conv2d operators (transposed-conv up-samples, identity, and a
max-pool down-sample) directly on the reshaped tokens.

Registered as ``"dinov3"`` in ``ENCODER_REGISTRY``.
"""
# Source: Meta AI DINOv3 (arXiv:2508.10104, Aug 2025).
# Requires timm >= 1.0.20 (DINOv3 backbone support added 2025-09-17).
# Official variants: ViT-S/16, ViT-S+/16, ViT-B/16, ViT-L/16, ViT-H+/16, ViT-7B/16
# (H+ is ViT-H-Plus, NOT standard ViT-H; no 'giant' variant exists).

from __future__ import annotations

import math
import warnings
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.registry import ENCODER_REGISTRY
from medseg.models.encoders.foundation._base import DPTHead, BaseFoundationEncoder, load_with_ssl_fallback


# Preferred DINOv3 timm names per "variant" kwarg.
# Official DINOv3 variants (Meta 2025):
#   ViT-S/16  (21M), ViT-S+/16 (29M), ViT-B/16 (86M),
#   ViT-L/16 (300M), ViT-H+/16 (840M), ViT-7B/16 (6716M).
# timm resolves these to .lvd1689m (web-pretrained) checkpoints automatically.
_DINOV3_NAMES = {
    "small":      "vit_small_patch16_dinov3",
    "base":       "vit_base_patch16_dinov3",
    "large":      "vit_large_patch16_dinov3",
    "huge_plus":  "vit_huge_plus_patch16_dinov3",
    "7b":         "vit_7b_patch16_dinov3",
}


@ENCODER_REGISTRY.register("dinov3")
class DINOv3Encoder(BaseFoundationEncoder):
    """DINOv3 ViT encoder with an DPT-style multi-block multi-scale projector.

    The backbone is a timm ViT (`vit_<variant>_patch16_dinov3`); the patch
    tokens of its last layer are reshaped to ``(B, D, h, w)`` and projected
    by four independent ``Conv2d`` ops to produce a 4-stage pyramid:

    * stage 0 — ``ConvTranspose2d`` 8x up   ->  1x1 conv to ``D/8`` channels
    * stage 1 — ``ConvTranspose2d`` 4x up   ->  1x1 conv to ``D/4`` channels
    * stage 2 — ``Identity``                ->  1x1 conv to ``D/2`` channels
    * stage 3 — ``MaxPool2d``    2x down    ->  1x1 conv to ``D``   channels

    ``out_channels`` is ``[D/8, D/4, D/2, D]`` (deepest LAST). Inputs whose
    spatial size is not a multiple of the backbone's patch size are zero-
    padded; the output features are cropped back to the natural multiples
    of the original input spatial size.

    Args:
        in_channels: Number of input image channels (default 3). A 1x1 conv
            adapter is inserted when ``!= 3`` (DINOv3 is RGB-only).
        img_size: Reference spatial size used to instantiate the backbone.
        pretrained: If True, attempt to download DINOv3 self-supervised
            weights via timm (falls back gracefully on network failure).
        pretrained_path: Optional local ``state_dict`` to load instead.
        freeze: If True, freeze the backbone's parameters (FPN stays
            trainable).
        unfreeze_last_n: If > 0, unfreeze the last ``N`` transformer blocks
            and the final norm.
        inference_only: If True, ``eval()`` and freeze every parameter.
        variant: One of ``{"small","base","large","huge_plus","7b"}``.
    """

    def __init__(self, in_channels: int = 3, img_size: int = 224,
                 pretrained: bool = True, pretrained_path: Optional[str] = None,
                 freeze: bool = True, unfreeze_last_n: int = 0,
                 inference_only: bool = False, variant: str = "small",
                 **kwargs):
        super().__init__(in_channels=in_channels, img_size=img_size,
                         pretrained=pretrained, pretrained_path=pretrained_path,
                         freeze=freeze, unfreeze_last_n=unfreeze_last_n,
                         inference_only=inference_only, **kwargs)

        if variant not in _DINOV3_NAMES:
            raise ValueError(
                f"Unknown DINOv3 variant '{variant}'. "
                f"Expected one of {sorted(_DINOV3_NAMES)}."
            )
        self.variant = variant

        import timm
        PRIMARY_BACKBONE_NAME = _DINOV3_NAMES[variant]
        self._backbone_name = PRIMARY_BACKBONE_NAME

        # ---- backbone -------------------------------------------------
        if pretrained_path is None:
            self.backbone = load_with_ssl_fallback(
                timm.create_model, PRIMARY_BACKBONE_NAME,
                pretrained=pretrained,
                num_classes=0,
                img_size=img_size,
                dynamic_img_size=True,
            )
        else:
            # User-provided checkpoint takes precedence: build without
            # downloading timm weights, then load the local state dict.
            self.backbone = timm.create_model(
                PRIMARY_BACKBONE_NAME,
                pretrained=False,
                num_classes=0,
                img_size=img_size,
                dynamic_img_size=True,
            )
            state = torch.load(pretrained_path, map_location="cpu")
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            msg = self.backbone.load_state_dict(state, strict=False)
            warnings.warn(f"DINOv3 local checkpoint load: {msg}")

        # ---- backbone introspection -----------------------------------
        ps = self.backbone.patch_embed.patch_size
        self.patch_size = int(ps[0]) if isinstance(ps, (tuple, list)) else int(ps)
        self.embed_dim = int(getattr(self.backbone, "embed_dim",
                                     getattr(self.backbone, "num_features", 384)))
        self.num_prefix_tokens = int(getattr(self.backbone, "num_prefix_tokens", 1))

        # ---- optional channel adapter (DINOv3 is RGB-only) ------------
        if in_channels != 3:
            self.input_adapter: nn.Module = nn.Conv2d(in_channels, 3,
                                                     kernel_size=1, bias=False)
        else:
            self.input_adapter = nn.Identity()

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

    def _pad_to_multiple(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """Zero-pad ``x`` so its spatial dims are multiples of ``patch_size``."""
        _, _, H, W = x.shape
        ps = self.patch_size
        H_pad = int(math.ceil(H / ps) * ps)
        W_pad = int(math.ceil(W / ps) * ps)
        if H_pad != H or W_pad != W:
            x = F.pad(x, (0, W_pad - W, 0, H_pad - H))
        return x, H_pad, W_pad

    def _get_intermediate_layers(self, x: torch.Tensor,
                                  indices: List[int]) -> List[torch.Tensor]:
        """Extract intermediate layer outputs from a timm ViT backbone.

        Handles backbones with or without ``get_intermediate_layers``.
        For timm VisionTransformer / Eva, manually runs the forward pass
        through patch_embed → pos_embed → blocks, collecting at specified indices.
        """
        bb = self.backbone
        # Prefer native method when available (some newer timm versions add it)
        if hasattr(bb, 'get_intermediate_layers'):
            return list(bb.get_intermediate_layers(x, n=indices))

        # Manual extraction for timm VisionTransformer / Eva
        x = bb.patch_embed(x)

        # patch_embed may return 3D [B,N,C] or 4D depending on timm version
        if x.dim() == 4:
            # timm ViT patch_embed returns NHWC [B, h, w, C]
            x = x.flatten(1, 2)  # [B, h, w, C] -> [B, N, C]

        # Position embedding
        pos_embed = bb.pos_embed
        if hasattr(bb, 'pos_embed_token'):
            # Some Eva variants use separate pos tokens
            x = x + bb.pos_embed_token
        elif pos_embed is not None:
            if pos_embed.ndim == 3:
                # Standard timm: pos_embed is (1, N_patches+prefix, D)
                x = x + pos_embed
            else:
                # Reshape if needed
                x = x + pos_embed.flatten(0, 1).unsqueeze(0)

        # Prepend prefix tokens (cls_token, register tokens, etc.)
        if hasattr(bb, 'cls_token') and bb.cls_token is not None:
            cls_tokens = bb.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        if hasattr(bb, 'reg_token') and bb.reg_token is not None:
            reg_tokens = bb.reg_token.expand(x.shape[0], -1, -1)
            x = torch.cat((reg_tokens, x), dim=1)

        # Dropout
        if hasattr(bb, 'pos_drop'):
            x = bb.pos_drop(x)
        elif hasattr(bb, 'drop'):
            x = bb.drop(x)

        # Run through blocks, collecting at specified indices
        target_set = set(indices)
        intermediates = {}
        for i, block in enumerate(bb.blocks):
            x = block(x)
            if i in target_set:
                intermediates[i] = x

        # Apply final norm to each collected output
        if hasattr(bb, 'norm') and bb.norm is not None:
            for idx in intermediates:
                intermediates[idx] = bb.norm(intermediates[idx])

        return [intermediates[i] for i in indices]

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
        multi_tokens = self._get_intermediate_layers(
            x, self._block_indices,
        )

        h_patches = Hp // p
        w_patches = Wp // p

        return self.dpt(list(multi_tokens), h_patches, w_patches, H, W)
