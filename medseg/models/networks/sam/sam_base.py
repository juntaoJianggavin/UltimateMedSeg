"""SAMBase — common base for all SAM-family medical seg models."""
# Source: INTERNAL — framework adaptation (this repo).

from __future__ import annotations

import os as _os
# Default to HF mirror (https://hf-mirror.com) so weight downloads work in
# environments without direct access to huggingface.co. Users can override
# by setting HF_ENDPOINT before importing this module.
_os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import warnings
from typing import Optional
import torch
import torch.nn as nn


def load_with_ssl_fallback(load_fn, *args, **kwargs):
    """Try the loader; on SSL failure retry once with unverified context.

    Does NOT silently fall back to random init or a different model. If the
    pretrained weights cannot be loaded, raises RuntimeError with a clear
    message.

    Args:
        load_fn: callable that performs the load (e.g. timm.create_model).
        *args, **kwargs: forwarded to load_fn. A non-False 'pretrained' kwarg
            indicates pretrained weights are desired.
    """
    import ssl
    # If the caller already disabled pretrained, just call through — no retry needed.
    if kwargs.get("pretrained", True) is False:
        return load_fn(*args, **kwargs)

    try:
        return load_fn(*args, **kwargs)
    except Exception as e1:
        # Retry with unverified SSL context (corporate / local cert miss).
        prev = ssl._create_default_https_context
        try:
            ssl._create_default_https_context = ssl._create_unverified_context
            return load_fn(*args, **kwargs)
        except Exception as e2:
            # No silent fallback — raise a clear error.
            model_name = ""
            if args:
                model_name = str(args[0])
            elif "model_name" in kwargs:
                model_name = str(kwargs["model_name"])
            raise RuntimeError(
                f"Failed to load pretrained weights for '{model_name}'. "
                f"Initial error: {type(e1).__name__}: {e1}. "
                f"SSL-bypass retry error: {type(e2).__name__}: {e2}. "
                f"Either: (a) provide a local checkpoint via 'pretrained_path', "
                f"(b) ensure network access to download the official weights, "
                f"or (c) explicitly pass pretrained=False to construct with random init."
            ) from e2
        finally:
            ssl._create_default_https_context = prev


class SAMBase(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, img_size=1024,
                 freeze_image_encoder=True, freeze_prompt_encoder=True,
                 freeze_mask_decoder=False, unfreeze_last_n_blocks=0,
                 pretrained=True, pretrained_path=None, inference_only=False, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.img_size = img_size
        self._pretrained = pretrained
        self._pretrained_path = pretrained_path
        self._freeze_cfg = {
            "image_encoder": freeze_image_encoder,
            "prompt_encoder": freeze_prompt_encoder,
            "mask_decoder": freeze_mask_decoder,
            "unfreeze_last_n_blocks": int(unfreeze_last_n_blocks),
            "inference_only": bool(inference_only),
        }

    def apply_freeze(self):
        cfg = self._freeze_cfg
        if cfg["image_encoder"] and getattr(self, "image_encoder", None) is not None:
            for p in self.image_encoder.parameters(): p.requires_grad = False
        if cfg["prompt_encoder"] and getattr(self, "prompt_encoder", None) is not None:
            for p in self.prompt_encoder.parameters(): p.requires_grad = False
        if cfg["mask_decoder"] and getattr(self, "mask_decoder", None) is not None:
            for p in self.mask_decoder.parameters(): p.requires_grad = False
        n = cfg["unfreeze_last_n_blocks"]
        if n > 0 and getattr(self, "image_encoder", None) is not None:
            blocks = None
            for attr in ("blocks", "layers", "transformer_blocks"):
                if hasattr(self.image_encoder, attr):
                    blocks = list(getattr(self.image_encoder, attr))
                    break
            if blocks:
                for blk in blocks[-n:]:
                    for p in blk.parameters(): p.requires_grad = True
                for attr in ("norm", "norm_post", "ln_post"):
                    if hasattr(self.image_encoder, attr):
                        m = getattr(self.image_encoder, attr)
                        if isinstance(m, nn.Module):
                            for p in m.parameters(): p.requires_grad = True
        if cfg["inference_only"]:
            self.eval()
            for p in self.parameters(): p.requires_grad = False

    def trainable_param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def frozen_param_count(self):
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)
