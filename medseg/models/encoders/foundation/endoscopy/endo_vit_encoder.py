"""EndoViT foundation-model encoder (endoscopy).

Reference: Batic et al., "EndoViT: pretraining vision transformers on a
large collection of endoscopic images" (Int J CARS, 2024).

EndoViT is a Masked-Autoencoder-pretrained ViT-Base/16 (``embed_dim=768``,
``patch_size=16``) trained on Endo700k. Weights are hosted at
``egeozsoy/EndoViT`` on HuggingFace Hub. ``pretrained=True`` auto-downloads.
``pretrained=False`` raises ``RuntimeError``.

Registered as ``"endo_vit"`` in ``ENCODER_REGISTRY``.
"""
# Source: https://huggingface.co/egeozsoy/EndoViT

from __future__ import annotations

import math
import warnings
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.registry import ENCODER_REGISTRY
from medseg.models.encoders.foundation._base import (BaseFoundationEncoder, hf_hub_download_vision_weights,
                     HuggingFaceViTWrapper, DPTHead)


# Architecture constants.
_ENDOVIT_EMBED_DIM = 768
_ENDOVIT_PATCH_SIZE = 16

_PRIMARY_HF_NAME = "egeozsoy/EndoViT"
PRIMARY_BACKBONE_NAME = _PRIMARY_HF_NAME


@ENCODER_REGISTRY.register("endo_vit")
class EndoViTEncoder(BaseFoundationEncoder):
    """EndoViT (endoscopy MAE ViT-B/16) encoder with DPT-style multi-block output.

    The backbone is a ViT-Base/16 (``embed_dim=768``, ``patch_size=16``).
    ``out_channels = [dim/8, dim/4, dim/2, dim]`` (deepest LAST).
    """

    native_img_size: int = 224
    PATCH_SIZE = _ENDOVIT_PATCH_SIZE
    EMBED_DIM = _ENDOVIT_EMBED_DIM

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

        # ---- channel adapter for non-RGB inputs (EndoViT is RGB-only) ----
        if in_channels != 3:
            self.input_adapter: nn.Module = nn.Conv2d(in_channels, 3,
                                                     kernel_size=1, bias=False)
        else:
            self.input_adapter = nn.Identity()

        # ---- backbone — HF ViTModel + weight download --------------------
        if pretrained:
            import transformers
            _cfg = transformers.ViTConfig(
                hidden_size=_ENDOVIT_EMBED_DIM, num_hidden_layers=12,
                num_attention_heads=12, intermediate_size=3072,
                patch_size=_ENDOVIT_PATCH_SIZE, image_size=224,
            )
            _vit = transformers.ViTModel(_cfg)
            if not pretrained_path:
                try:
                    state = hf_hub_download_vision_weights(
                        repo_id=_PRIMARY_HF_NAME,
                        filename="pytorch_model.bin",
                        prefix_strip=("module.", "backbone."),
                    )
                    msg = _vit.load_state_dict(state, strict=False)
                    warnings.warn(
                        f"[endo_vit] auto-downloaded weights from "
                        f"{_PRIMARY_HF_NAME}: {msg}"
                    )
                except Exception as e:
                    raise RuntimeError(
                        f"EndoViT auto-download from '{_PRIMARY_HF_NAME}' "
                        f"failed: {type(e).__name__}: {e}. Provide a local "
                        f"checkpoint via pretrained_path."
                    ) from e
            self.backbone = HuggingFaceViTWrapper(_vit)
        else:
            raise RuntimeError(
                "EndoViTEncoder does not support pretrained=False. "
                "This encoder requires pretrained weights from "
                "'egeozsoy/EndoViT'. Pass pretrained=True to auto-download, "
                "or provide a local checkpoint via pretrained_path."
            )
        self._backbone_name = PRIMARY_BACKBONE_NAME

        # ---- optional local checkpoint -----------------------------------
        if pretrained_path:
            try:
                state = torch.load(pretrained_path, map_location="cpu")
                if isinstance(state, dict):
                    for key in ("state_dict", "model", "teacher", "student"):
                        if key in state and isinstance(state[key], dict):
                            state = state[key]
                            break
                if isinstance(state, dict):
                    cleaned = {}
                    for k, v in state.items():
                        nk = k
                        for pref in ("module.", "backbone."):
                            if nk.startswith(pref):
                                nk = nk[len(pref):]
                        cleaned[nk] = v
                    msg = self.backbone.model.load_state_dict(cleaned, strict=False)
                    warnings.warn(
                        f"[endo_vit] loaded local weights from "
                        f"'{pretrained_path}': {msg}"
                    )
            except Exception as e:
                warnings.warn(
                    f"[endo_vit] failed to load checkpoint "
                    f"'{pretrained_path}': {e}"
                )

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

    # ------------------------------------------------------------------
    @classmethod
    def _pad_to_multiple_size(cls, size: int) -> int:
        m = cls.PATCH_SIZE * 8
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
