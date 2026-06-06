"""UltraDINO fetal-ultrasound foundation-model encoder.

Reference:
    Ambsdorf et al., "General Methods Make Great Domain-specific Foundation
    Models: A Case-study on Fetal Ultrasound", MICCAI 2025
    (arXiv:2506.19552).

UltraDINO is a DINOv2/iBOT-pretrained ViT-B/16 (``embed_dim=768``,
``patch_size=16``) trained from scratch on 2 million fetal ultrasound
images (FUS2M, Danish national fetal ultrasound database).

Weights are released on GitHub at ``jakobamb/UltraDINO`` (DINOv2 format).
``pretrained=True`` attempts to auto-download; if that fails, provide a
local checkpoint via ``pretrained_path``.
``pretrained=False`` raises ``RuntimeError``.

The ViT token grid is projected into a 4-stage DPT-style multi-block pyramid
(deepest LAST), matching the ``BaseFoundationEncoder`` contract.

Registered as ``"ultradino"`` in ``ENCODER_REGISTRY``.
"""
# Source: https://github.com/jakobamb/UltraDINO

from __future__ import annotations

import warnings
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.registry import ENCODER_REGISTRY
from medseg.models.encoders.foundation._base import DPTHead, BaseFoundationEncoder, HuggingFaceViTWrapper


# Auto-download URL (GitHub raw).
_DOWNLOAD_URL = (
    "https://github.com/jakobamb/UltraDINO/raw/main/"
    "pretrained/ultradino_vitb16.pth"
)

_EMBED_DIM = 768
_PATCH_SIZE = 16
_NUM_LAYERS = 12
_NUM_HEADS = 12
_INTERMEDIATE_SIZE = 3072  # 4 * embed_dim

# DINOv2 / iBOT checkpoint key prefixes to strip.
_PREFIX_STRIP = ("module.", "backbone.", "student.", "teacher.", "model.")


def _load_dinov2_vit_b16(pretrained_path: Optional[str] = None) -> nn.Module:
    """Build a HF ViTModel skeleton (ViT-B/16) and load UltraDINO weights.

    Returns a ``HuggingFaceViTWrapper``-compatible module.
    """
    from transformers import ViTConfig, ViTModel

    cfg = ViTConfig(
        hidden_size=_EMBED_DIM,
        num_hidden_layers=_NUM_LAYERS,
        num_attention_heads=_NUM_HEADS,
        intermediate_size=_INTERMEDIATE_SIZE,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
        image_size=224,
        patch_size=_PATCH_SIZE,
        num_channels=3,
        qkv_bias=True,
    )
    vit = ViTModel(cfg)

    # Try to load pretrained weights.
    state = None
    if pretrained_path:
        ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            for key in ("state_dict", "model", "model_state_dict"):
                if key in ckpt and isinstance(ckpt[key], dict):
                    ckpt = ckpt[key]
                    break
        state = ckpt if isinstance(ckpt, dict) else None

    if state is None:
        try:
            ckpt = torch.hub.load_state_dict_from_url(
                _DOWNLOAD_URL, map_location="cpu", check_hash=False
            )
            if isinstance(ckpt, dict):
                for key in ("state_dict", "model", "model_state_dict"):
                    if key in ckpt and isinstance(ckpt[key], dict):
                        ckpt = ckpt[key]
                        break
            state = ckpt if isinstance(ckpt, dict) else None
        except Exception as e:
            warnings.warn(
                f"UltraDINO auto-download failed: {type(e).__name__}: {e}. "
                "Provide a local checkpoint via pretrained_path. "
                "Download from: https://github.com/jakobamb/UltraDINO"
            )
            raise RuntimeError(
                "UltraDINO pretrained weights could not be loaded. "
                "Download the checkpoint from https://github.com/jakobamb/UltraDINO "
                "and provide the local path via pretrained_path."
            ) from e

    # Strip known prefixes.
    if isinstance(state, dict):
        cleaned = {}
        for k, v in state.items():
            nk = k
            for pref in _PREFIX_STRIP:
                if nk.startswith(pref):
                    nk = nk[len(pref):]
                    break
            cleaned[nk] = v
        state = cleaned
        # Remove teacher / projection head keys.
        state = {k: v for k, v in state.items()
                 if not any(skip in k for skip in ("head", "prototypes", "teacher_head"))}

    if state is not None:
        missing, unexpected = vit.load_state_dict(state, strict=False)
        if missing:
            warnings.warn(f"UltraDINO: {len(missing)} missing keys.")
        if unexpected:
            warnings.warn(f"UltraDINO: {len(unexpected)} unexpected keys.")

    return HuggingFaceViTWrapper(vit)


@ENCODER_REGISTRY.register("ultradino")
class UltraDINOEncoder(BaseFoundationEncoder):
    """UltraDINO fetal-ultrasound encoder (DINOv2 ViT-B/16, ``embed_dim=768``).

    Parameters
    ----------
    in_channels : int
        Number of input image channels.
    img_size : int
        Reference spatial size (default 224).
    pretrained : bool
        Attempt to load UltraDINO pretrained weights.
    pretrained_path : Optional[str]
        Path to a local UltraDINO ``.pth`` checkpoint.
    freeze / unfreeze_last_n / inference_only :
        Standard freeze controls inherited via :class:`FreezeMixin`.
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

        # Channel adapter for non-RGB inputs.
        if in_channels != 3:
            self.input_adapter: nn.Module = nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)
        else:
            self.input_adapter = nn.Identity()

        if pretrained:
            self.backbone = _load_dinov2_vit_b16(pretrained_path=pretrained_path)
        else:
            raise RuntimeError(
                "UltraDINOEncoder does not support pretrained=False. "
                "This encoder requires pretrained weights from UltraDINO. "
                "Download the checkpoint from https://github.com/jakobamb/UltraDINO "
                "and provide the local path via pretrained_path."
            )

        # Introspection.
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
