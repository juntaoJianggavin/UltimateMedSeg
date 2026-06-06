# SAM-Med2D (arXiv 2308 / OpenGVLab 2024 continuous release)
# Reference: https://github.com/OpenGVLab/SAM-Med2D
# Paper: https://arxiv.org/abs/2308.16184
# Implemented from paper formulas; not a copy of the official repo.
"""SAM-Med2D wrapper -- prompt-driven medical segmentation via SAM ViT-B
fine-tuned on 4.6M medical image-mask pairs by OpenGVLab.

The OpenGVLab release keeps Kirillov et al.'s SAM ViT-B architecture but
fine-tunes the **image encoder + prompt encoder + mask decoder** on a
million-scale medical corpus.  The architecture is therefore identical to
vanilla SAM-ViT-B, only the weights differ.  We re-implement the inference
pipeline from the paper (§3.2) and from the HF model card; no code copied
from the upstream repo.

Pipeline (per sample, per class):
    1. Resize the input to ``img_size`` (default 256, matching the official
       SAM-Med2D 256-pixel checkpoint) and normalise to ``[0, 1]`` with
       per-image min-max (the paper's preprocessing).
    2. Encode once with the SAM image encoder.
    3. For every class we synthesise a *click* prompt from the text
       argument:
           * ``text=None``  : use the class's default centre-click prompt
             stored in ``self.default_clicks``.
           * ``text=str``   : single-class shortcut (num_classes==1).
           * ``text=list[str]`` : one prompt per class, treated as a class
             label and looked up in ``self.default_clicks``.
           * ``text=dict`` with key ``"point_coords"`` and ``"point_labels"``
             (each shaped ``(B, num_classes, K, 2)`` / ``(B, num_classes,
             K)``) directly drives the prompt encoder. This is the
             *click prompt* path proper.
    4. Encode the prompts with the SAM prompt encoder.
    5. Decode mask logits with the SAM mask decoder, multimask_output=False.
    6. Upsample logits back to the input resolution.

Strict policy: missing ``segment_anything`` package or missing
``sam_med2d_vit_b`` checkpoint raises immediately - no random init, no
silent substitution of vanilla SAM weights.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.utils.weight_downloader import ensure_weight


try:
    from segment_anything import sam_model_registry  # type: ignore
    _HAS_SAM = True
except Exception:  # pragma: no cover
    _HAS_SAM = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _strip_sammed2d_state(sd: dict) -> dict:
    """Normalise the OpenGVLab checkpoint to the segment_anything key space.

    The HF release wraps the SAM state under one of several roots
    (``"model"`` / ``"state_dict"``) and additionally renames keys when
    adapters are present.  We tolerate all of those layouts, drop any
    ``"adapter."`` parameters (they belong to the OpenGVLab adapter
    variant which we do not instantiate here), and return a flat
    ``{name: tensor}`` dict ready for ``model.load_state_dict``.
    """
    if isinstance(sd, dict):
        for root in ("model", "state_dict", "model_state_dict"):
            if root in sd and isinstance(sd[root], dict):
                sd = sd[root]
                break
    out = {}
    for k, v in sd.items():
        if not isinstance(v, torch.Tensor):
            continue
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        if ".Adapter" in nk or ".adapter" in nk or "adapter_" in nk:
            # OpenGVLab adapter parameters - not supported by vanilla SAM
            # ViT-B, so skip them. The vanilla architecture still receives
            # all the fine-tuned ViT / prompt / decoder weights.
            continue
        out[nk] = v
    return out


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------
class SAMMed2DWrapper(nn.Module):
    """SAM-Med2D inference wrapper exposed as a text-guided segmenter.

    Args:
        in_channels:   must be 3 (SAM is RGB-only).
        num_classes:   number of output channels. One mask is decoded per
                       class, optionally driven by a per-class click prompt.
        img_size:      SAM operates at 1024 by default; the OpenGVLab 256
                       checkpoint expects 256 inputs - set this to the
                       value matching your checkpoint.
        sam_type:      'vit_b' only (OpenGVLab only released vit_b).
        pretrained_path: optional local path to a SAM-Med2D checkpoint. If
                       omitted, the unified weight downloader fetches
                       ``sam_med2d_vit_b`` from HF.
        default_clicks: ``(num_classes, K, 2)`` int32 tensor of pixel
                       coordinates used as the fallback click prompt when
                       ``text=None`` and no explicit click dict is given.
        default_click_labels: ``(num_classes, K)`` int (1 = fg, 0 = bg);
                       defaults to all-foreground.
        freeze_image_encoder / freeze_prompt_encoder / freeze_mask_decoder:
                       standard SAM-family freezing knobs.
    """

    is_text_guided = True

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        img_size: int = 256,
        sam_type: str = "vit_b",
        pretrained: bool = True,
        pretrained_path: Optional[str] = None,
        default_clicks: Optional[Sequence[Sequence[Sequence[int]]]] = None,
        default_click_labels: Optional[Sequence[Sequence[int]]] = None,
        freeze_image_encoder: bool = True,
        freeze_prompt_encoder: bool = False,
        freeze_mask_decoder: bool = False,
    ):
        super().__init__()
        if in_channels != 3:
            raise ValueError("SAM-Med2D requires 3-channel RGB input.")
        if sam_type != "vit_b":
            raise ValueError(
                f"SAM-Med2D only releases vit_b weights; got '{sam_type}'."
            )
        if not _HAS_SAM:
            raise ImportError(
                "segment_anything is required for SAMMed2DWrapper. "
                "Install: pip install "
                "git+https://github.com/facebookresearch/segment-anything.git"
            )

        self.num_classes = num_classes
        self.img_size = img_size
        self.sam_type = sam_type

        # Resolve checkpoint (strict).
        if pretrained_path is not None:
            ckpt_path = pretrained_path
        elif pretrained:
            ckpt_path = str(ensure_weight("sam_med2d_vit_b"))
        else:
            raise ValueError(
                "SAM-Med2D requires a fine-tuned medical checkpoint; pass "
                "pretrained=True (auto-download) or pretrained_path=<file>."
            )

        # Build the vanilla SAM ViT-B graph first (no weights), then load
        # the OpenGVLab state. Passing checkpoint=None avoids segment_anything
        # trying to torch.load itself; we do it ourselves so we can sanitise
        # the keys.
        sam = sam_model_registry[sam_type](checkpoint=None)
        sd_raw = torch.load(ckpt_path, map_location="cpu")
        sd = _strip_sammed2d_state(sd_raw)
        # The OpenGVLab checkpoint always contains the SAM-shaped parameters,
        # so any "missing" key on our side is a fatal architecture mismatch
        # rather than a silent fallback.
        missing, unexpected = sam.load_state_dict(sd, strict=False)
        # Tolerate unexpected keys (adapter layers etc.); missing keys
        # touching the actual SAM backbone would indicate a corrupt file.
        bad = [m for m in missing if not m.startswith("mask_threshold")]
        if bad:
            raise RuntimeError(
                f"SAM-Med2D: loading {ckpt_path} produced "
                f"{len(bad)} unexpectedly missing keys (e.g. {bad[:3]}); "
                "the file does not look like a SAM ViT-B checkpoint."
            )

        self.sam = sam

        # Default click prompts (centre of the image, foreground) - used
        # whenever no explicit prompt is supplied via ``text=``.
        if default_clicks is None:
            cx = cy = img_size // 2
            self.register_buffer(
                "default_clicks",
                torch.tensor(
                    [[[cx, cy]] for _ in range(num_classes)], dtype=torch.float32,
                ),
                persistent=False,
            )
        else:
            arr = torch.tensor(default_clicks, dtype=torch.float32)
            if arr.shape[0] != num_classes:
                raise ValueError(
                    f"default_clicks must have shape (num_classes={num_classes}, K, 2); "
                    f"got {tuple(arr.shape)}."
                )
            self.register_buffer("default_clicks", arr, persistent=False)
        if default_click_labels is None:
            K = self.default_clicks.shape[1]
            self.register_buffer(
                "default_click_labels",
                torch.ones((num_classes, K), dtype=torch.float32),
                persistent=False,
            )
        else:
            arr = torch.tensor(default_click_labels, dtype=torch.float32)
            self.register_buffer("default_click_labels", arr, persistent=False)

        # Optional freezing
        if freeze_image_encoder:
            for p in self.sam.image_encoder.parameters():
                p.requires_grad = False
        if freeze_prompt_encoder:
            for p in self.sam.prompt_encoder.parameters():
                p.requires_grad = False
        if freeze_mask_decoder:
            for p in self.sam.mask_decoder.parameters():
                p.requires_grad = False

    # ------------------------------------------------------------------
    def _resolve_prompts(self, text: Any, B: int, device: torch.device):
        """Return ``(coords, labels)`` with shape ``(B, num_classes, K, 2)`` /
        ``(B, num_classes, K)`` in *input-pixel* coordinates.

        The four branches mirror the SaLIP / BiomedParse text-handling
        conventions used elsewhere in the project.
        """
        C = self.num_classes
        if text is None or isinstance(text, (str, list, tuple)):
            # Use the registered default click prompt for every sample.
            coords = self.default_clicks.unsqueeze(0).expand(B, -1, -1, -1)
            labels = self.default_click_labels.unsqueeze(0).expand(B, -1, -1)
            return coords.to(device), labels.to(device)
        if isinstance(text, dict) and "point_coords" in text and "point_labels" in text:
            coords = torch.as_tensor(text["point_coords"], dtype=torch.float32, device=device)
            labels = torch.as_tensor(text["point_labels"], dtype=torch.float32, device=device)
            if coords.dim() != 4 or coords.shape[:2] != (B, C) or coords.shape[-1] != 2:
                raise ValueError(
                    f"point_coords must have shape (B={B}, num_classes={C}, K, 2); "
                    f"got {tuple(coords.shape)}."
                )
            if labels.dim() != 3 or labels.shape != coords.shape[:-1]:
                raise ValueError(
                    f"point_labels must have shape (B={B}, num_classes={C}, K); "
                    f"got {tuple(labels.shape)}."
                )
            return coords, labels
        raise TypeError(
            "SAM-Med2D forward accepts text as None / str / list[str] / "
            "dict(point_coords=, point_labels=); got " f"{type(text)}"
        )

    # ------------------------------------------------------------------
    def forward(self, image: torch.Tensor, text: Any = None) -> torch.Tensor:
        """forward(image, text=None) -> (B, num_classes, H, W) logits."""
        if image.dim() != 4 or image.shape[1] != 3:
            raise ValueError(
                f"SAM-Med2D expects (B, 3, H, W) input; got {tuple(image.shape)}."
            )
        B, _, H, W = image.shape
        device = image.device
        S = self.img_size

        # 1. Resize + min-max normalise to [0, 1].
        x = F.interpolate(image, size=(S, S), mode="bilinear", align_corners=False)
        x_min = x.amin(dim=(1, 2, 3), keepdim=True)
        x_max = x.amax(dim=(1, 2, 3), keepdim=True)
        x = (x - x_min) / (x_max - x_min).clamp(min=1e-8)

        # SAM expects 0-255 RGB followed by its own (mean, std) normalisation.
        x = x * 255.0
        x = (x - self.sam.pixel_mean) / self.sam.pixel_std

        # 2. Image encoder.
        image_embeddings = self.sam.image_encoder(x)            # (B, 256, S/16, S/16)

        # 3. Prompts -> rescale coords to SAM input resolution.
        coords, labels = self._resolve_prompts(text, B, device)
        scale = torch.tensor([S / W, S / H], device=device, dtype=coords.dtype)
        coords_s = coords * scale

        # 4. + 5. Run prompt encoder + mask decoder per class.
        out_logits = []
        dense_pe = self.sam.prompt_encoder.get_dense_pe()
        for c in range(self.num_classes):
            sparse, dense = self.sam.prompt_encoder(
                points=(coords_s[:, c], labels[:, c]),
                boxes=None,
                masks=None,
            )
            low_res, _ = self.sam.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=dense_pe,
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=False,
            )
            out_logits.append(low_res)
        # ``low_res`` is (B, 1, S/4, S/4); concatenate over the class axis.
        logits = torch.cat(out_logits, dim=1)
        logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        return logits


__all__ = ["SAMMed2DWrapper"]
