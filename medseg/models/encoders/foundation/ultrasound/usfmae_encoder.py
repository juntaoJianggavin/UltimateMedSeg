"""USF-MAE ultrasound foundation-model encoder.

Reference:
    Megahed et al., "USF-MAE: Ultrasound Self-Supervised Foundation Model
    with Masked Autoencoding", Biomedical Signal Processing and Control,
    2026 (arXiv:2510.22990).

USF-MAE is a masked-autoencoder (MAE) ViT-Base/16 (``embed_dim=768``,
``patch_size=16``) pre-trained on 370K ultrasound images from 46 open-source
datasets (OpenUS-46).  Weights are released on GitHub at
``Yusufii9/USF-MAE`` (Facebook MAE checkpoint format).

``pretrained=True`` attempts to auto-download the checkpoint via
``torch.hub.load_state_dict_from_url``.  If the download fails (e.g. no
network), provide a local checkpoint path via ``pretrained_path``.
``pretrained=False`` raises ``RuntimeError``.

The ViT encoder token grid is projected into a 4-stage DPT-style multi-block
pyramid (deepest LAST), matching the ``BaseFoundationEncoder`` contract.

Registered as ``"usfmae"`` in ``ENCODER_REGISTRY``.
"""
# Source: https://github.com/Yusufii9/USF-MAE

from __future__ import annotations

import warnings
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.registry import ENCODER_REGISTRY
from medseg.models.encoders.foundation._base import DPTHead, BaseFoundationEncoder, load_hf_vit


# Auto-download URL (GitHub release asset).
_DOWNLOAD_URL = (
    "https://github.com/Yusufii9/USF-MAE/raw/main/"
    "checkpoints/usf_mae_vitb16_100ep.pth"
)

_EMBED_DIM = 768
_PATCH_SIZE = 16
_NUM_LAYERS = 12
_NUM_HEADS = 12
_INTERMEDIATE_SIZE = 3072  # 4 * embed_dim

# MAE checkpoint key prefixes to strip.
_PREFIX_STRIP = ("module.", "encoder.", "model.", "mae.")


def _load_mae_vit_b16(pretrained_path: Optional[str] = None) -> nn.Module:
    """Build a HF ViTModel skeleton (ViT-B/16) and load USF-MAE weights.

    Returns a ``HuggingFaceViTWrapper``-compatible module with
    ``forward_features``, ``embed_dim``, ``patch_embed.patch_size``, etc.
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
        # Auto-download from GitHub.
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
                f"USF-MAE auto-download failed: {type(e).__name__}: {e}. "
                "Provide a local checkpoint via pretrained_path. "
                "Download from: https://github.com/Yusufii9/USF-MAE"
            )
            raise RuntimeError(
                "USF-MAE pretrained weights could not be loaded. "
                "Download the checkpoint from https://github.com/Yusufii9/USF-MAE "
                "and provide the local path via pretrained_path."
            ) from e

    # Strip known MAE prefixes.
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

        # Remove decoder keys (MAE has encoder + decoder).
        state = {k: v for k, v in state.items()
                 if not any(skip in k for skip in ("decoder", "mask_token"))}

    if state is not None:
        missing, unexpected = vit.load_state_dict(state, strict=False)
        if missing:
            n_miss = len(missing)
            warnings.warn(f"USF-MAE: {n_miss} missing keys (expected for CLS head).")
        if unexpected:
            n_unexp = len(unexpected)
            if n_unexp > 10:
                warnings.warn(f"USF-MAE: {n_unexp} unexpected keys (decoder / MAE head).")

    from medseg.models.encoders.foundation._base import HuggingFaceViTWrapper
    return HuggingFaceViTWrapper(vit)


@ENCODER_REGISTRY.register("usfmae")
class USFMAEEncoder(BaseFoundationEncoder):
    """USF-MAE ultrasound encoder (MAE ViT-B/16, ``embed_dim=768``).

    Parameters
    ----------
    in_channels : int
        Number of input image channels.
    img_size : int
        Reference spatial size (default 224).
    pretrained : bool
        Attempt to load USF-MAE pretrained weights.
    pretrained_path : Optional[str]
        Path to a local USF-MAE ``.pth`` checkpoint (Facebook MAE format).
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
            self.backbone = _load_mae_vit_b16(pretrained_path=pretrained_path)
        else:
            raise RuntimeError(
                "USFMAEEncoder does not support pretrained=False. "
                "This encoder requires pretrained weights from USF-MAE. "
                "Download the checkpoint from https://github.com/Yusufii9/USF-MAE "
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
