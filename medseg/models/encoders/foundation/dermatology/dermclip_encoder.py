"""DermCLIP — CLIP variant trained on dermatology image-text pairs.

This wrapper targets the public **DermLIP-ViT-B/16** checkpoint released with
the Derm1M paper (ICCV'25 Highlight, Yan et al.), which is the closest public
artifact to the generic "DermCLIP" name.

Architecture: CLIP ViT-B/16, embed_dim=768, patch_size=16 (open_clip).

Source:
  # Source: https://github.com/SiyuanYan1/Derm1M
  # HF artifact: https://huggingface.co/redlessone/DermLIP_ViT-B-16
  # Paper: Yan et al., "Derm1M: A Million-Scale Vision-Language Dataset Aligned
  #        with Clinical Ontology Knowledge for Dermatology", arXiv:2503.14911

The HF artifact is an open_clip checkpoint. Loaded natively via
``open_clip.create_model``. ``pretrained=True`` auto-downloads from HF Hub;
``pretrained_path`` accepts a local checkpoint.
``pretrained=False`` is not supported (raises ``RuntimeError``).
"""
# Source: https://github.com/SiyuanYan1/Derm1M  (HF: redlessone/DermLIP_ViT-B-16)

from __future__ import annotations
import warnings
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.models.encoders.foundation._base import BaseFoundationEncoder
from medseg.registry import ENCODER_REGISTRY


# Verified open_clip artifact: https://huggingface.co/redlessone/DermLIP_ViT-B-16
PRIMARY_BACKBONE_NAME = "redlessone/DermLIP_ViT-B-16"
_EMBED_DIM = 768
_PATCH_SIZE = 16


@ENCODER_REGISTRY.register("dermclip")
class DermCLIPEncoder(BaseFoundationEncoder):
    """Dermatology CLIP vision encoder (open_clip native loading)."""

    native_img_size: int = 224

    def __init__(self, in_channels=3, img_size=None, pretrained=True,
                 pretrained_path=None, freeze=True, unfreeze_last_n=0,
                 inference_only=False, **kwargs):
        super().__init__(in_channels=in_channels,
                         img_size=img_size or self.native_img_size,
                         pretrained=pretrained, pretrained_path=pretrained_path,
                         freeze=freeze, unfreeze_last_n=unfreeze_last_n,
                         inference_only=inference_only, **kwargs)

        self.input_adapter = (nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)
                              if in_channels != 3 else nn.Identity())

        self.embed_dim = _EMBED_DIM
        self.patch_size = _PATCH_SIZE

        # Load the DermCLIP vision tower via open_clip (native loading).
        if pretrained:
            try:
                import open_clip
            except ImportError:
                raise RuntimeError(
                    "DermCLIPEncoder requires open_clip. "
                    "Install with: pip install open_clip_torch"
                )
            _hf_name = f"hf-hub:{PRIMARY_BACKBONE_NAME}"
            try:
                _clip_model = open_clip.create_model(
                    _hf_name, pretrained=not pretrained_path,
                )
            except Exception as e:
                raise RuntimeError(
                    "DermCLIP auto-download from "
                    f"'{_hf_name}' failed: {type(e).__name__}: {e}. "
                    "Provide a local checkpoint via pretrained_path."
                ) from e
            self.backbone = _clip_model.visual
            del _clip_model
        else:
            raise RuntimeError(
                "DermCLIPEncoder does not support pretrained=False. "
                "This encoder requires pretrained weights from "
                f"'{PRIMARY_BACKBONE_NAME}'. "
                "Pass pretrained=True to auto-download, or provide a local "
                "checkpoint via pretrained_path."
            )
        self._backbone_name = PRIMARY_BACKBONE_NAME
        self.embed_dim = int(getattr(self.backbone, "embed_dim", _EMBED_DIM))
        _ps = getattr(getattr(self.backbone, "patch_embed", None), "patch_size", _PATCH_SIZE)
        if isinstance(_ps, (tuple, list)):
            self.patch_size = int(_ps[0])
        else:
            self.patch_size = int(_ps)
        self.num_prefix_tokens = int(getattr(self.backbone, "num_prefix_tokens", 1))

        d = self.embed_dim
        chs = [d // 8, d // 4, d // 2, d]
        self.out_channels = chs
        self.proj0 = nn.Conv2d(d, chs[0], kernel_size=1, bias=False)
        self.proj1 = nn.Conv2d(d, chs[1], kernel_size=1, bias=False)
        self.proj2 = nn.Conv2d(d, chs[2], kernel_size=1, bias=False)
        self.proj3 = nn.Conv2d(d, chs[3], kernel_size=1, bias=False)

        # Optionally load a local pretrained checkpoint.
        if pretrained_path:
            try:
                state = torch.load(pretrained_path, map_location="cpu")
                if isinstance(state, dict):
                    for key in ("state_dict", "model"):
                        if key in state and isinstance(state[key], dict):
                            state = state[key]
                            break
                if isinstance(state, dict):
                    cleaned = {}
                    for k, v in state.items():
                        nk = k
                        for prefix in ("visual.trunk.", "visual.", "vision_model.",
                                       "image_encoder."):
                            if nk.startswith(prefix):
                                nk = nk[len(prefix):]
                                break
                        cleaned[nk] = v
                    state = cleaned
                msg = self.backbone.load_state_dict(state, strict=False)
                warnings.warn(f"[dermclip] loaded pretrained_path: {msg}")
            except Exception as e:
                warnings.warn(f"[dermclip] failed to load pretrained_path: {e}")

        self._maybe_inject_adapters()
        self._apply_freeze_policy()

    def forward(self, x: torch.Tensor):
        x = self.input_adapter(x)
        B, _, H, W = x.shape
        ps = self.patch_size
        pad_h = (-H) % ps
        pad_w = (-W) % ps
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        H_p, W_p = x.shape[-2:]

        tokens = self.backbone.forward_features(x)
        h, w = H_p // ps, W_p // ps
        n_spatial = h * w
        if tokens.dim() == 3 and tokens.shape[1] != n_spatial:
            tokens = tokens[:, -n_spatial:, :]
        elif tokens.dim() == 4:
            tokens = tokens.flatten(2).transpose(1, 2)
        grid = tokens.transpose(1, 2).reshape(B, self.embed_dim, h, w)

        sizes = [(H // 4, W // 4), (H // 8, W // 8), (H // 16, W // 16), (H // 32, W // 32)]
        projs = [self.proj0, self.proj1, self.proj2, self.proj3]
        return [F.interpolate(proj(grid), size=sz, mode='bilinear', align_corners=False)
                for proj, sz in zip(projs, sizes)]
