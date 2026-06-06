"""FLAIR foundation-model encoder (ophthalmology).

Reference:
    Silva-Rodriguez et al., "FLAIR: A Foundation LAnguage-Image model of the
    Retina for fundus image understanding", Medical Image Analysis, 2025
    (arXiv:2308.07898).

FLAIR is a CLIP-style vision-language model trained on fundus images. Its
vision tower is a **ResNet-50** backbone, loaded natively via ``torchvision``.
A 5-stage feature pyramid at strides [2, 4, 8, 16, 32] is exposed for
U-Net-style segmentation decoders.

``pretrained=True`` loads torchvision ImageNet-pretrained ResNet-50 weights.
If a FLAIR-specific checkpoint is desired, pass ``pretrained_path`` pointing
to the local ``model.safetensors`` from ``jusiro2/FLAIR``.
``pretrained=False`` raises RuntimeError.

Source: https://github.com/jusiro/FLAIR (HF: jusiro2/FLAIR)
"""

from __future__ import annotations

import math
import warnings
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.registry import ENCODER_REGISTRY
from medseg.models.encoders.foundation._base import BaseFoundationEncoder


_PRIMARY_BACKBONE = "torchvision.models.resnet50"
PRIMARY_BACKBONE_NAME = _PRIMARY_BACKBONE

_FLAIR_DEFAULT_CHANNELS: List[int] = [64, 256, 512, 1024, 2048]
_FLAIR_STRIDE = 32


@ENCODER_REGISTRY.register("flair")
class FLAIREncoder(BaseFoundationEncoder):
    """FLAIR (retinal vision-language ResNet-50) encoder.

    Wraps a torchvision ResNet-50 backbone and exposes 5 native feature maps
    at strides 2/4/8/16/32 with channels ``[64, 256, 512, 1024, 2048]``
    (deepest LAST).
    """

    native_img_size: int = 224
    _STAGE_ATTRS: Tuple[str, ...] = ("layer1", "layer2", "layer3", "layer4")

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

        if pretrained:
            try:
                import torchvision
                weights = torchvision.models.ResNet50_Weights.DEFAULT
                self.backbone = torchvision.models.resnet50(weights=weights)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load torchvision ResNet-50 pretrained weights: "
                    f"{type(e).__name__}: {e}. Provide a local checkpoint via "
                    f"pretrained_path."
                ) from e
            # Replace the first conv if in_channels != 3.
            if in_channels != 3:
                self.backbone.conv1 = nn.Conv2d(
                    in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False,
                )
        else:
            raise RuntimeError(
                "FLAIREncoder does not support pretrained=False. "
                "This encoder requires ResNet-50 weights. "
                "Pass pretrained=True to load torchvision ImageNet weights, "
                "or provide a local FLAIR checkpoint via pretrained_path."
            )
        self._backbone_name = PRIMARY_BACKBONE_NAME

        # Remove FC head — we only need feature extraction.
        self.backbone.fc = nn.Identity()
        self.backbone.avgpool = nn.Identity()

        # Load optional local FLAIR checkpoint.
        if pretrained_path is not None:
            self._load_local_checkpoint(pretrained_path)

        self.out_channels: List[int] = list(_FLAIR_DEFAULT_CHANNELS)

        self._maybe_inject_adapters()
        self._apply_freeze_policy()

    def _load_local_checkpoint(self, pretrained_path: str) -> None:
        try:
            state = torch.load(pretrained_path, map_location="cpu")
        except Exception as e:
            warnings.warn(
                f"[flair] failed to read pretrained_path "
                f"'{pretrained_path}': {e}."
            )
            return

        if isinstance(state, dict):
            for key in ("model", "state_dict", "visual", "vision_model"):
                if key in state and isinstance(state[key], dict):
                    state = state[key]
                    break

        if not isinstance(state, dict):
            warnings.warn(
                f"[flair] pretrained_path '{pretrained_path}' did not yield "
                "a state_dict; ignored."
            )
            return

        cleaned = {}
        for k, v in state.items():
            nk = k
            for pref in ("module.", "backbone.", "visual."):
                if nk.startswith(pref):
                    nk = nk[len(pref):]
            cleaned[nk] = v

        try:
            msg = self.backbone.load_state_dict(cleaned, strict=False)
            missing = list(getattr(msg, "missing_keys", []))
            unexpected = list(getattr(msg, "unexpected_keys", []))
            warnings.warn(
                f"[flair] loaded local weights from '{pretrained_path}': "
                f"{len(missing)} missing, {len(unexpected)} unexpected."
            )
        except Exception as e:
            warnings.warn(
                f"[flair] failed to apply pretrained_path "
                f"'{pretrained_path}': {e}"
            )

    def unfreeze_last_n_blocks(self, n: int) -> None:
        if n <= 0 or not hasattr(self, "backbone"):
            return
        stages = [getattr(self.backbone, a) for a in self._STAGE_ATTRS
                  if hasattr(self.backbone, a)]
        if not stages:
            super().unfreeze_last_n_blocks(n)
            return
        for stage in stages[-n:]:
            for p in stage.parameters():
                p.requires_grad = True
        if n >= len(stages):
            for attr in ("bn1", "act1"):
                if hasattr(self.backbone, attr):
                    m = getattr(self.backbone, attr)
                    if isinstance(m, nn.Module):
                        for p in m.parameters():
                            p.requires_grad = True

    def _pad_input(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        _, _, H, W = x.shape
        m = _FLAIR_STRIDE
        H_pad = int(math.ceil(H / m) * m)
        W_pad = int(math.ceil(W / m) * m)
        if H_pad != H or W_pad != W:
            x = F.pad(x, (0, W_pad - W, 0, H_pad - H))
        return x, H_pad, W_pad

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        _, _, H_in, W_in = x.shape
        x_pad, _, _ = self._pad_input(x)

        bb = self.backbone
        # Stem: stride 2
        stem = bb.relu(bb.bn1(bb.conv1(x_pad)))  # (B, 64, H/2, W/2)
        # Stride 4: maxpool + layer1
        pool = bb.maxpool(stem)  # (B, 64, H/4, W/4)
        s1 = bb.layer1(pool)     # (B, 256, H/4, W/4)
        s2 = bb.layer2(s1)      # (B, 512, H/8, W/8)
        s3 = bb.layer3(s2)      # (B, 1024, H/16, W/16)
        s4 = bb.layer4(s3)      # (B, 2048, H/32, W/32)

        feats = [stem, s1, s2, s3, s4]

        out: List[torch.Tensor] = []
        for i, f in enumerate(feats):
            stride = 2 ** (i + 1)
            th = max(H_in // stride, 1)
            tw = max(W_in // stride, 1)
            if f.shape[-2:] != (th, tw):
                fh, fw = f.shape[-2:]
                if fh >= th and fw >= tw:
                    f = f[..., :th, :tw]
                else:
                    f = F.interpolate(
                        f, size=(th, tw), mode="bilinear", align_corners=False,
                    )
            out.append(f)
        return out
