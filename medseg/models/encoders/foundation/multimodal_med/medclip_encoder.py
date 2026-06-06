"""MedCLIP ViT image encoder (foundation-model encoder).

MedCLIP (Wang et al., 2022) is a contrastive vision-language model pretrained
on medical image-text pairs. The official release uses a ViT-B/16 backbone
(``embed_dim=768``, ``patch_size=16``) loaded via ``transformers.ViTModel``.
Weights are auto-downloaded from Google Cloud Storage.

# Source: https://github.com/RyanWangZf/MedCLIP
# Artifact (MedCLIP-ViT): https://storage.googleapis.com/pytrial/medclip-vit-pretrained.zip

Output:
    forward(x) -> List[Tensor] of length 4. Approximate spatial strides
    [H/4, H/8, H/16, H/32]; channels [dim/8, dim/4, dim/2, dim] with
    dim=768 (ViT-B).
"""

from __future__ import annotations

import warnings
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.registry import ENCODER_REGISTRY
from medseg.models.encoders.foundation._base import DPTHead, BaseFoundationEncoder, HuggingFaceViTWrapper, convert_timm_vit_state_to_hf


# MedCLIP's vision tower is ViT-B/16; embed_dim=768, patch_size=16.
_MEDCLIP_EMBED_DIM = 768
_MEDCLIP_PATCH_SIZE = 16
_MEDCLIP_GCS_URL = "https://storage.googleapis.com/pytrial/medclip-vit-pretrained.zip"

# Primary weight source: Google Cloud Storage.
PRIMARY_BACKBONE_NAME = "medclip-vit-gcs"


@ENCODER_REGISTRY.register("medclip")
class MedCLIPEncoder(BaseFoundationEncoder):
    """MedCLIP ViT-B/16 image encoder with DPT-style multi-block multi-scale output.

    Constructor follows the BaseFoundationEncoder contract. MedCLIP weights
    are auto-downloaded from GCS or loaded via ``pretrained_path``.
    """

    def __init__(self, in_channels: int = 3, img_size: int = 224,
                 pretrained: bool = True, pretrained_path: Optional[str] = None,
                 freeze: bool = True, unfreeze_last_n: int = 0,
                 inference_only: bool = False, **kwargs):
        super().__init__(in_channels=in_channels, img_size=img_size,
                         pretrained=pretrained, pretrained_path=pretrained_path,
                         freeze=freeze, unfreeze_last_n=unfreeze_last_n,
                         inference_only=inference_only, **kwargs)

        self.embed_dim = _MEDCLIP_EMBED_DIM
        self.patch_size = _MEDCLIP_PATCH_SIZE

        # Build the ViT-B/16 backbone via transformers.
        if pretrained:
            try:
                from transformers import ViTModel, ViTConfig
            except ImportError:
                raise RuntimeError(
                    "transformers is required for MedCLIP. Install with: pip install transformers"
                )

            if pretrained_path:
                # Load from local checkpoint.
                _cfg = ViTConfig(
                    hidden_size=768, num_hidden_layers=12,
                    num_attention_heads=12, intermediate_size=3072,
                    hidden_act="gelu", layer_norm_eps=1e-12,
                    image_size=224, patch_size=16,
                )
                _vit = ViTModel(_cfg)
                try:
                    _state = torch.load(pretrained_path, map_location="cpu",
                                        weights_only=False)
                    if isinstance(_state, dict):
                        for _key in ("state_dict", "model"):
                            if _key in _state and isinstance(_state[_key], dict):
                                _state = _state[_key]
                                break
                    # Strip common MedCLIP prefixes.
                    if isinstance(_state, dict):
                        _cleaned = {}
                        for _k, _v in _state.items():
                            _nk = _k
                            for _pref in ("visual.trunk.", "visual.", "vision_model.",
                                          "image_encoder.", "module."):
                                if _nk.startswith(_pref):
                                    _nk = _nk[len(_pref):]
                                    break
                            _cleaned[_nk] = _v
                        _state = _cleaned
                    # Convert timm-format keys to HF format if needed.
                    _is_timm = any("blocks." in k for k in _state)
                    if _is_timm:
                        _state = convert_timm_vit_state_to_hf(_state)
                    _msg = _vit.load_state_dict(_state, strict=False)
                    warnings.warn(f"[medclip] loaded pretrained_path: {_msg}")
                except Exception as e:
                    warnings.warn(f"[medclip] failed to load pretrained_path: {e}")
                self.backbone = HuggingFaceViTWrapper(_vit)
            else:
                # Auto-download from GCS.
                import os
                import zipfile
                _cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "medseg")
                os.makedirs(_cache_dir, exist_ok=True)
                _zip_path = os.path.join(_cache_dir, "medclip-vit-pretrained.zip")
                if not os.path.isfile(_zip_path):
                    try:
                        from torch.hub import download_url_to_file
                        download_url_to_file(_MEDCLIP_GCS_URL, _zip_path)
                    except Exception as _dl_err:
                        raise RuntimeError(
                            f"MedCLIP auto-download failed: {_dl_err}. "
                            f"Download manually from {_MEDCLIP_GCS_URL} and pass "
                            f"pretrained_path=<path>/pytorch_model.bin."
                        ) from _dl_err
                _ckpt_path = None
                try:
                    with zipfile.ZipFile(_zip_path, 'r') as _zf:
                        for _name in _zf.namelist():
                            if _name.endswith((".bin", ".pt", ".pth")):
                                _ckpt_path = _zf.extract(_name, _cache_dir)
                                break
                except Exception as _zip_err:
                    raise RuntimeError(f"Failed to extract MedCLIP zip: {_zip_err}") from _zip_err
                if _ckpt_path is None:
                    raise RuntimeError("No .bin/.pt/.pth file found in MedCLIP zip.")

                _cfg = ViTConfig(
                    hidden_size=768, num_hidden_layers=12,
                    num_attention_heads=12, intermediate_size=3072,
                    hidden_act="gelu", layer_norm_eps=1e-12,
                    image_size=224, patch_size=16,
                )
                _vit = ViTModel(_cfg)
                try:
                    _state = torch.load(_ckpt_path, map_location="cpu",
                                        weights_only=False)
                    if isinstance(_state, dict):
                        for _key in ("state_dict", "model"):
                            if _key in _state and isinstance(_state[_key], dict):
                                _state = _state[_key]
                                break
                    if isinstance(_state, dict):
                        _cleaned = {}
                        for _k, _v in _state.items():
                            _nk = _k
                            for _pref in ("visual.trunk.", "visual.", "vision_model.",
                                          "image_encoder.", "module."):
                                if _nk.startswith(_pref):
                                    _nk = _nk[len(_pref):]
                                    break
                            _cleaned[_nk] = _v
                        _state = _cleaned
                    _is_timm = any("blocks." in k for k in _state)
                    if _is_timm:
                        _state = convert_timm_vit_state_to_hf(_state)
                    _msg = _vit.load_state_dict(_state, strict=False)
                    warnings.warn(f"[medclip] auto-downloaded MedCLIP weights: {_msg}")
                except Exception as _load_err:
                    raise RuntimeError(
                        f"[medclip] failed to load auto-downloaded weights: {_load_err}."
                    ) from _load_err
                self.backbone = HuggingFaceViTWrapper(_vit)
        else:
            raise RuntimeError(
                "MedCLIPEncoder does not support pretrained=False. "
                "This encoder requires pretrained MedCLIP weights. "
                "Pass pretrained=True to auto-download from GCS, or provide "
                "a local checkpoint via pretrained_path."
            )
        self._backbone_name = PRIMARY_BACKBONE_NAME
        self.num_prefix_tokens = int(self.backbone.num_prefix_tokens)

        # Adapter for non-RGB inputs (MedCLIP/ViT-B expects 3 channels).
        if in_channels != 3:
            self.input_adapter = nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)
        else:
            self.input_adapter = nn.Identity()

        # DPT-style multi-block: project the token grid (B, dim, h, w) to four
        # pyramid stages with channels [dim/8, dim/4, dim/2, dim].
        # Spatial scales (deepest LAST):
        #   stage 0: ConvTranspose 4x  -> ~H/4,  dim/8
        #   stage 1: ConvTranspose 2x  -> ~H/8,  dim/4
        #   stage 2: Identity          -> ~H/16, dim/2
        #   stage 3: MaxPool 2x        -> ~H/32, dim
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
