"""KEEP foundation-model encoder (knowledge-enhanced pathology, 2024/2026).

Reference:
    Zhou et al., "Knowledge-enhanced pretraining for vision-language
    pathology foundation model on cancer diagnosis", Cancer Cell, 2026
    (arXiv:2412.13126).

# Source: https://github.com/MAGIC-AI4Med/KEEP
# HF artifact: https://huggingface.co/Astaxanthin/KEEP   (verified, ~1.66 GB)

KEEP is a CLIP-style vision-language pathology model. Its vision tower is
**ViT-Large/16** (``embed_dim=1024``, ``patch_size=16``, depth=24, heads=16,
``init_values=1e-5`` LayerScale). The HF repo ships a custom ``KEEPModel``
(text = BertModel, vision = ViT-L/16) loaded via
``transformers.AutoModel.from_pretrained("Astaxanthin/KEEP",
trust_remote_code=True)``.

``pretrained=True`` is required — the full KEEP model is auto-downloaded
from HF Hub and the vision tower is extracted. ``pretrained=False`` raises
``RuntimeError``.

This wrapper produces a 4-stage feature pyramid (deepest LAST) suitable for
U-Net-style decoders by reshaping the ViT patch tokens into a spatial grid
and applying four parallel up/down-sampling + 1x1 projection branches.

Registered as ``"keep"`` in ``ENCODER_REGISTRY``.
"""
# Source: https://github.com/MAGIC-AI4Med/KEEP  (HF: Astaxanthin/KEEP)

from __future__ import annotations

import math
import warnings
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.registry import ENCODER_REGISTRY
from medseg.models.encoders.foundation._base import DPTHead, BaseFoundationEncoder, HuggingFaceViTWrapper


# Architecture constants for KEEP's ViT-Large/16 vision tower.
_KEEP_EMBED_DIM = 1024
_KEEP_PATCH_SIZE = 16
_KEEP_HF_NAME = "Astaxanthin/KEEP"

# Primary weight source: HuggingFace Hub (Astaxanthin/KEEP, custom transformers model).
PRIMARY_BACKBONE_NAME = _KEEP_HF_NAME


@ENCODER_REGISTRY.register("keep")
class KEEPEncoder(BaseFoundationEncoder):
    """KEEP (knowledge-enhanced pathology, ViT-Large/16) encoder.

    The vision tower is a ViT-Large/16 (``embed_dim=1024``, ``patch_size=16``,
    depth=24, heads=16, ``init_values=1e-5`` LayerScale), matching the
    ``Astaxanthin/KEEP`` HuggingFace artifact.
    Its final-layer patch tokens are reshaped to ``(B, 1024, h, w)`` and
    projected by four independent ``Conv2d`` ops to a 4-stage pyramid:

    * stage 0 - ``ConvTranspose2d`` 4x up  -> 1x1 conv to ``dim/8``  channels
    * stage 1 - ``ConvTranspose2d`` 2x up  -> 1x1 conv to ``dim/4``  channels
    * stage 2 - ``Identity``               -> 1x1 conv to ``dim/2``  channels
    * stage 3 - ``MaxPool2d`` 2x down      -> 1x1 conv to ``dim``    channels

    ``out_channels = [dim/8, dim/4, dim/2, dim]`` (deepest LAST). Inputs
    whose spatial size is not a multiple of the patch size are zero-padded
    on the bottom/right; outputs are cropped/resampled back to the canonical
    pyramid sizes of the ORIGINAL un-padded input.

    Args:
        in_channels: Number of input image channels (default 3). A 1x1 conv
            adapter is inserted when ``!= 3`` (KEEP is RGB-only).
        img_size: Reference spatial size used to instantiate the backbone.
            ``None`` -> use ``native_img_size`` (224). ``dynamic_img_size=
            True`` is set, so other sizes also work at runtime.
        pretrained: If True, download CLIP ViT-B/16 weights. Raises a
            clear ``RuntimeError`` if the load fails (no silent downgrade).
        pretrained_path: Optional local ``state_dict`` to load instead.
        freeze: If True, freeze the backbone (FPN stays trainable).
        unfreeze_last_n: If > 0, unfreeze the last ``N`` transformer blocks.
        inference_only: If True, ``eval()`` and freeze every parameter.
    """

    native_img_size: int = 224
    PATCH_SIZE = _KEEP_PATCH_SIZE
    EMBED_DIM = _KEEP_EMBED_DIM

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

        # ---- channel adapter for non-RGB inputs (KEEP is RGB-only) -------
        if in_channels != 3:
            self.input_adapter: nn.Module = nn.Conv2d(
                in_channels, 3, kernel_size=1, bias=False,
            )
            backbone_in_chans = 3
        else:
            self.input_adapter = nn.Identity()
            backbone_in_chans = 3

        # ---- backbone ---------------------------------------------------
        # Load the KEEP model via transformers and extract the vision tower.
        ref_size = self._pad_to_multiple_size(int(img_size))

        if pretrained:
            try:
                import transformers
            except ImportError:
                raise RuntimeError(
                    "transformers is required for KEEP. Install with: pip install transformers"
                )
            _src = pretrained_path or _KEEP_HF_NAME
            try:
                _keep_model = transformers.AutoModel.from_pretrained(
                    _src, trust_remote_code=True,
                )
            except Exception as e:
                raise RuntimeError(
                    f"KEEP auto-download from '{_KEEP_HF_NAME}' failed: "
                    f"{type(e).__name__}: {e}. Provide a local checkpoint via "
                    f"pretrained_path."
                ) from e
            # Extract the vision tower from the KEEP model.
            _visual = getattr(_keep_model, "visual", None)
            if _visual is None:
                _visual = getattr(_keep_model, "vision_model", None)
            if _visual is None:
                _visual = getattr(_keep_model, "vision_tower", None)
            if _visual is None:
                raise RuntimeError(
                    f"Could not find vision tower in KEEP model from '{_src}'. "
                    f"Expected attributes: 'visual', 'vision_model', or 'vision_tower'."
                )
            _vit = getattr(_visual, "trunk", None) or getattr(_visual, "backbone", _visual)
            # Add compatibility attributes for HuggingFaceViTWrapper.
            if not hasattr(_vit, 'embed_dim'):
                _vit.embed_dim = _KEEP_EMBED_DIM
            if not hasattr(_vit, 'num_features'):
                _vit.num_features = _KEEP_EMBED_DIM
            if not hasattr(_vit, 'num_prefix_tokens'):
                _vit.num_prefix_tokens = 1
            if not hasattr(_vit, 'patch_embed'):
                class _PE:
                    pass
                _vit.patch_embed = _PE()
                _vit.patch_embed.patch_size = _KEEP_PATCH_SIZE
            self.backbone = HuggingFaceViTWrapper(_vit)
            del _keep_model
        else:
            raise RuntimeError(
                "KEEPEncoder does not support pretrained=False. "
                "This encoder requires pretrained weights from 'Astaxanthin/KEEP'. "
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
        """Round ``size`` up to a multiple of ``patch_size * 8`` so that all
        four pyramid stages have integer spatial dims."""
        m = cls.PATCH_SIZE * 8  # 16 * 8 = 128
        return ((int(size) + m - 1) // m) * m

    def _pad_input(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """Zero-pad ``x`` so its spatial dims are multiples of ``patch_size``."""
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
