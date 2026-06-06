# MedCLIP-SAM (MICCAI 2024)
# Reference: https://github.com/HealthX-Lab/MedCLIP-SAM
# Paper: https://arxiv.org/abs/2403.20253
# Implemented from paper formulas; not a copy of the official repo.
"""MedCLIP-SAM: a bridge between MedCLIP and SAM for zero-shot text-driven
medical image segmentation (MICCAI 2024).

Algorithm (faithful to the paper, NOT copied from the repo):

    1. Compute a *saliency map* ``S(p) = cos(F_p, t)`` for every spatial
       location ``p`` of the (Med)CLIP image encoder, where ``F_p`` is the
       per-patch vision feature and ``t`` is the CLIP text embedding of the
       organ / pathology prompt.  This is the gScoreCAM/MaskCLIP-style
       saliency the paper uses to drive box prompts (Sec. 3.2 of the paper).
    2. Threshold the saliency at its 75-th percentile and take the tight
       bounding box of the resulting binary mask.  When the mask is empty
       we use a centre-crop fallback box equal to 50 % of the image.
    3. Feed that bounding box as the SAM **box prompt** and run a single
       SAM forward pass.  The returned mask becomes the prediction for that
       (sample, class) pair.

Implementation policy (strict, no fallback):
    * Both ``segment_anything`` and ``transformers`` are required imports.
      Missing either at *forward-time* raises an explicit ImportError.
    * The SAM checkpoint is auto-downloaded via
      :func:`medseg.utils.weight_downloader.ensure_weight` (which itself
      raises a :class:`~medseg.utils.weight_downloader.WeightDownloadError`
      with a manual download URL when every source fails).
    * CLIP loading goes through :func:`hf_from_pretrained` — failures
      surface the manual HuggingFace URL.

The module is *training-free*; ``train_text_guided.py`` should invoke it
with ``--eval-only`` exactly as for SaLIP.
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.utils.weight_downloader import hf_from_pretrained


# ---------------------------------------------------------------------------
# optional deps
# ---------------------------------------------------------------------------
try:
    from segment_anything import sam_model_registry, SamPredictor  # type: ignore
    _HAS_SAM = True
except Exception:  # pragma: no cover
    _HAS_SAM = False

try:
    from transformers import CLIPModel, CLIPProcessor  # type: ignore
    _HAS_HF = True
except Exception:  # pragma: no cover
    _HAS_HF = False


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def _build_sam(sam_type: str, sam_ckpt: Optional[str], device: torch.device):
    if not _HAS_SAM:
        raise ImportError(
            "segment_anything is required for MedCLIPSAM. "
            "Install: pip install git+https://github.com/facebookresearch/segment-anything.git"
        )
    if sam_type not in sam_model_registry:
        raise KeyError(
            f"Unknown SAM type '{sam_type}'. Available: {list(sam_model_registry.keys())}"
        )
    if not sam_ckpt:
        from medseg.utils.weight_downloader import ensure_weight
        key_map = {"vit_b": "sam_vit_b", "vit_l": "sam_vit_l", "vit_h": "sam_vit_h"}
        if sam_type not in key_map:
            raise KeyError(
                f"No registered auto-download for SAM type '{sam_type}'. "
                f"Pass an explicit sam_ckpt path."
            )
        sam_ckpt = str(ensure_weight(key_map[sam_type]))
    sam = sam_model_registry[sam_type](checkpoint=sam_ckpt)
    sam.to(device)
    sam.eval()
    return sam


def _build_clip(hf_name: str, device: torch.device):
    if not _HAS_HF:
        raise ImportError(
            "transformers is required for MedCLIPSAM. Install: pip install transformers"
        )
    clip = hf_from_pretrained(CLIPModel, hf_name).to(device).eval()
    proc = hf_from_pretrained(CLIPProcessor, hf_name)
    return clip, proc


# ---------------------------------------------------------------------------
# Saliency helper (gScoreCAM-style; per-patch cosine on the last hidden state)
# ---------------------------------------------------------------------------
def _clip_patch_saliency(
    clip,
    proc,
    image_np,
    text_prompts: Sequence[str],
    device: torch.device,
) -> torch.Tensor:
    """Compute a (C, H, W) saliency map per text prompt for one image.

    Steps mirror Eq. (1) of the paper:
      1. Forward CLIP vision encoder with ``output_hidden_states=True``.
      2. Drop the [CLS] token and reshape the remaining patch features into
         a (h, w, D) grid.
      3. L2-normalise patch features and the text embedding, take cosine
         similarity → per-patch heat map.
    """
    from PIL import Image
    img_pil = Image.fromarray(image_np.astype("uint8"))

    with torch.no_grad():
        text_inputs = proc(
            text=list(text_prompts), return_tensors="pt", padding=True, truncation=True,
        )
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        txt_emb = clip.get_text_features(**text_inputs)             # (C, D)
        txt_emb = F.normalize(txt_emb, dim=-1)

        img_inputs = proc(images=img_pil, return_tensors="pt")
        img_inputs = {k: v.to(device) for k, v in img_inputs.items()}
        vision = clip.vision_model(
            pixel_values=img_inputs["pixel_values"], output_hidden_states=False,
        )
        last = vision.last_hidden_state                              # (1, 1+N, D_v)
        patch = last[:, 1:, :]                                       # drop [CLS]
        # Project to the joint embed dim
        patch_proj = patch @ clip.visual_projection.weight.t()       # (1, N, D)
        patch_proj = F.normalize(patch_proj, dim=-1)
        # cosine sim per patch per class -> (C, N)
        sim = txt_emb @ patch_proj[0].t()
        N = sim.shape[-1]
        side = int(round(math.sqrt(N)))
        if side * side != N:
            raise ValueError(
                f"CLIP vision encoder produced {N} patch tokens; expected a square grid."
            )
        sim = sim.view(-1, side, side)                               # (C, side, side)

    return sim


def _box_from_saliency(
    saliency: torch.Tensor,
    H: int,
    W: int,
    percentile: float = 0.75,
) -> Tuple[int, int, int, int]:
    """Return a tight bbox from a saliency map using a percentile threshold."""
    sal = F.interpolate(
        saliency[None, None], size=(H, W), mode="bilinear", align_corners=False
    )[0, 0]
    flat = sal.flatten()
    if flat.numel() == 0:
        return (W // 4, H // 4, 3 * W // 4, 3 * H // 4)
    thr = torch.quantile(flat, percentile)
    mask = sal > thr
    if not mask.any():
        return (W // 4, H // 4, 3 * W // 4, 3 * H // 4)
    ys, xs = torch.where(mask)
    x0, x1 = int(xs.min().item()), int(xs.max().item()) + 1
    y0, y1 = int(ys.min().item()), int(ys.max().item()) + 1
    # Guard against degenerate boxes
    x0 = max(0, min(W - 2, x0))
    y0 = max(0, min(H - 2, y0))
    x1 = max(x0 + 1, min(W, x1))
    y1 = max(y0 + 1, min(H, y1))
    return (x0, y0, x1, y1)


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------
class MedCLIPSAM(nn.Module):
    """MedCLIP-SAM zero-shot text-to-mask wrapper (CLIP saliency -> SAM box prompt)."""

    is_text_guided = True

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        img_size: int = 1024,
        sam_type: str = "vit_b",
        sam_ckpt: Optional[str] = None,
        clip_hf_name: str = "openai/clip-vit-base-patch32",
        text_prompts: Optional[Sequence[str]] = None,
        saliency_percentile: float = 0.75,
        multimask_output: bool = False,
    ):
        super().__init__()
        if in_channels != 3:
            raise ValueError("MedCLIPSAM requires 3-channel RGB input.")
        if not 0.0 <= saliency_percentile < 1.0:
            raise ValueError("saliency_percentile must be in [0, 1)")
        self.num_classes = num_classes
        self.img_size = img_size
        self.sam_type = sam_type
        self.sam_ckpt = sam_ckpt
        self.clip_hf_name = clip_hf_name
        self.saliency_percentile = saliency_percentile
        self.multimask_output = multimask_output
        self.text_prompts = (
            list(text_prompts) if text_prompts is not None
            else [f"a medical image of class {i}" for i in range(num_classes)]
        )
        if len(self.text_prompts) != num_classes:
            raise ValueError(
                f"text_prompts has {len(self.text_prompts)} entries but num_classes={num_classes}"
            )
        self._sam = None
        self._predictor = None
        self._clip = None
        self._clip_proc = None
        # anchor buffer keeps Module.device tracking working
        self.register_buffer("_anchor", torch.zeros(1))

    # ------------------------------------------------------------------
    def _lazy_build(self, device: torch.device):
        if self._sam is None:
            self._sam = _build_sam(self.sam_type, self.sam_ckpt, device)
            self._predictor = SamPredictor(self._sam)
        if self._clip is None:
            self._clip, self._clip_proc = _build_clip(self.clip_hf_name, device)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward(self, image: torch.Tensor, text: Any = None) -> torch.Tensor:
        """forward(image, text=None) -> (B, num_classes, H, W) logits.

        For each sample, for each class:
            1. Compute CLIP saliency for the class prompt.
            2. Convert saliency to a bounding box.
            3. Run SAM with that bbox as a prompt.
            4. Write the SAM mask into the output volume.
        """
        # 如果 text=None 且 yaml 里配了 text_prompts，自动使用
        # If text=None and text_prompts configured in yaml, use them automatically
        if text is None and hasattr(self, '_default_text_prompts') and self._default_text_prompts:
            text = self._default_text_prompts
        device = image.device
        self._lazy_build(device)
        B, _, H, W = image.shape

        if text is None:
            text_list = self.text_prompts
        elif isinstance(text, str):
            if self.num_classes != 1:
                raise ValueError("single string prompt only valid for num_classes=1")
            text_list = [text]
        elif isinstance(text, (list, tuple)):
            text_list = [str(t) for t in text]
            if len(text_list) != self.num_classes:
                raise ValueError(
                    f"text must list {self.num_classes} prompts; got {len(text_list)}"
                )
        else:
            raise TypeError(
                "MedCLIPSAM forward accepts text as None / str / list-of-str only; "
                f"got {type(text)}"
            )

        import numpy as np
        out = image.new_zeros((B, self.num_classes, H, W))
        img_np = (
            image.detach().cpu().clamp(0, 1).mul(255).permute(0, 2, 3, 1).numpy().astype("uint8")
        )

        for b in range(B):
            saliency = _clip_patch_saliency(
                self._clip, self._clip_proc, img_np[b], text_list, device
            )  # (C, h, w)
            self._predictor.set_image(img_np[b])
            for c in range(self.num_classes):
                bbox = _box_from_saliency(
                    saliency[c], H, W, percentile=self.saliency_percentile
                )
                box_arr = torch.tensor(bbox, dtype=torch.float32).cpu().numpy()
                masks, scores, _ = self._predictor.predict(
                    box=box_arr,
                    multimask_output=self.multimask_output,
                )
                # pick highest-IoU mask if multimask_output
                if self.multimask_output:
                    idx = int(scores.argmax())
                    m = masks[idx]
                else:
                    m = masks[0]
                out[b, c] = torch.from_numpy(m.astype("float32")).to(device)

        # Convert binary mask to logits-friendly range so downstream sigmoid/argmax behaves
        return out * 10.0 - 5.0


__all__ = ["MedCLIPSAM"]
