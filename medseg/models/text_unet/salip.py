# SaLIP (BMVC 2024)
# Reference: https://github.com/aleemsidra/SaLIP
# Paper: https://arxiv.org/abs/2404.06362
# Implemented from paper formulas; not a copy of the official repo.
"""SaLIP: SAM + Linguistic Instance Prompts for zero-shot medical segmentation.

The SaLIP pipeline is *inference-only* and contains three stages:

    1. Run SAM's automatic mask generator on the image, yielding a set of
       class-agnostic proposals {M_k}.
    2. For each proposal, crop the image with the proposal's bounding box
       (paper additionally tries to mask-out the background; we follow the
       paper's "crop-and-blacken" variant).
    3. Score every crop against the textual prompt with CLIP image-text
       similarity: ``s_k = cos(CLIP_img(crop_k), CLIP_txt(prompt))``.
       The proposal with the highest similarity becomes the predicted mask.

We re-implement this from the paper description without copying upstream
code.  Because SaLIP is training-free, ``forward(image, text=None)`` runs
the pipeline end-to-end in ``torch.no_grad()`` and returns a logit tensor
constructed by writing the chosen binary mask into a (B, C, H, W) volume.

Strict policy (no fallback):
    * If ``segment_anything`` is not installed, the constructor raises.
    * If ``transformers`` (for CLIP) is missing, the constructor raises.
    * If the SAM checkpoint is missing and ``pretrained=True``, we raise.

We use a *grid-of-points* automatic mask generator written in pure torch
that calls ``SamPredictor`` (or a SAM model directly) at a coarse grid.
This keeps the file self-contained and avoids depending on the upstream
``SamAutomaticMaskGenerator``'s extra post-processing dependencies.
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
# Helper: lazy import + strict checkpoint load
# ---------------------------------------------------------------------------
def _build_sam(sam_type: str, sam_ckpt: Optional[str], device: torch.device):
    if not _HAS_SAM:
        raise ImportError(
            "segment_anything is required for SaLIP. "
            "Install: pip install git+https://github.com/facebookresearch/segment-anything.git"
        )
    if sam_type not in sam_model_registry:
        raise KeyError(f"Unknown SAM type '{sam_type}'. Available: {list(sam_model_registry.keys())}")

    if not sam_ckpt:
        # Auto-download via the unified weight downloader; raises
        # WeightDownloadError (with manual URL) on failure.
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
            "transformers is required for SaLIP's CLIP scorer. "
            "Install: pip install transformers"
        )
    clip = hf_from_pretrained(CLIPModel, hf_name).to(device).eval()
    proc = hf_from_pretrained(CLIPProcessor, hf_name)
    return clip, proc


# ---------------------------------------------------------------------------
# Point-grid mask generator (lightweight stand-in for SamAutomaticMaskGenerator)
# ---------------------------------------------------------------------------
class _GridPointGenerator(nn.Module):
    def __init__(self, points_per_side: int = 16, pred_iou_thresh: float = 0.7, min_mask_area: int = 50):
        super().__init__()
        self.points_per_side = points_per_side
        self.pred_iou_thresh = pred_iou_thresh
        self.min_mask_area = min_mask_area

    @staticmethod
    def _grid_points(H: int, W: int, n: int) -> torch.Tensor:
        ys = torch.linspace(0, H - 1, n)
        xs = torch.linspace(0, W - 1, n)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([gx.flatten(), gy.flatten()], dim=-1)  # (n*n, 2) in (x, y)

    def __call__(self, predictor, image_np):
        H, W = image_np.shape[:2]
        pts = self._grid_points(H, W, self.points_per_side).numpy()
        masks, scores, boxes = [], [], []
        for p in pts:
            mask_logits, iou_preds, _ = predictor.predict(
                point_coords=p[None, :], point_labels=[1], multimask_output=True
            )
            for m, s in zip(mask_logits, iou_preds):
                if s < self.pred_iou_thresh:
                    continue
                area = int(m.sum())
                if area < self.min_mask_area:
                    continue
                ys, xs = m.nonzero()
                if ys.size == 0:
                    continue
                bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
                masks.append(m)
                scores.append(float(s))
                boxes.append(bbox)
        return masks, scores, boxes


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------
class SaLIP(nn.Module):
    """SaLIP — training-free SAM+CLIP referring segmentation.

    Args:
        num_classes:    number of output channels (one selected mask per class).
        img_size:       expected input H=W. (paper uses 1024 for SAM.)
        sam_type:       'vit_b' | 'vit_l' | 'vit_h'.
        sam_ckpt:       path to the SAM checkpoint (.pth). Required when
                        ``pretrained=True``.
        clip_hf_name:   HF identifier for the CLIP backbone used for scoring.
        text_prompts:   default text query per class (used when ``text=None``).
        points_per_side: density of the grid used by the proposal generator.
        pred_iou_thresh: minimum SAM-predicted IoU to keep a proposal.
        pretrained:     when True, builds the SAM checkpoint at construction
                        time; otherwise SAM and CLIP are built lazily and
                        every forward call MUST succeed in loading them.
    """

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
        points_per_side: int = 16,
        pred_iou_thresh: float = 0.7,
        min_mask_area: int = 50,
        crop_strategy: str = "blacken_bg",
    ):
        super().__init__()
        if in_channels != 3:
            raise ValueError("SaLIP requires 3-channel RGB input.")
        if crop_strategy not in ("crop", "blacken_bg"):
            raise ValueError("crop_strategy must be 'crop' or 'blacken_bg'")
        self.num_classes = num_classes
        self.img_size = img_size
        self.sam_type = sam_type
        self.sam_ckpt = sam_ckpt
        self.clip_hf_name = clip_hf_name
        self.text_prompts = (
            list(text_prompts) if text_prompts is not None
            else [f"a medical image of class {i}" for i in range(num_classes)]
        )
        if len(self.text_prompts) != num_classes:
            raise ValueError(
                f"text_prompts has {len(self.text_prompts)} entries but num_classes={num_classes}"
            )
        self.crop_strategy = crop_strategy
        self.mask_gen = _GridPointGenerator(points_per_side, pred_iou_thresh, min_mask_area)

        # placeholders — lazily filled on first forward
        self._sam = None
        self._predictor = None
        self._clip = None
        self._clip_proc = None
        # we keep one buffer so the module has a parameter / device anchor
        self.register_buffer("_anchor", torch.zeros(1))

    # ------------------------------------------------------------------
    def _lazy_build(self, device: torch.device):
        if self._sam is None:
            self._sam = _build_sam(self.sam_type, self.sam_ckpt, device)
            self._predictor = SamPredictor(self._sam)
        if self._clip is None:
            self._clip, self._clip_proc = _build_clip(self.clip_hf_name, device)

    # ------------------------------------------------------------------
    def _score_proposals_with_clip(
        self,
        image_np,
        masks: List,
        boxes: List[Tuple[int, int, int, int]],
        text: List[str],
        device: torch.device,
    ) -> torch.Tensor:
        """Score every (proposal, text) pair with CLIP.

        Returns a (num_classes, K) cosine-similarity matrix.
        """
        import numpy as np
        from PIL import Image
        K = len(masks)
        C = len(text)
        if K == 0:
            return torch.full((C, 0), -1.0, device=device)

        crops = []
        for m, b in zip(masks, boxes):
            x0, y0, x1, y1 = b
            patch = image_np[y0:y1, x0:x1].copy()
            if self.crop_strategy == "blacken_bg":
                local_mask = m[y0:y1, x0:x1]
                patch = patch * local_mask[..., None]
            crops.append(Image.fromarray(patch.astype("uint8")))

        with torch.no_grad():
            inputs = self._clip_proc(
                text=text, images=crops, return_tensors="pt", padding=True, truncation=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            out = self._clip(**inputs)
            img_emb = out.image_embeds                       # (K, D)
            txt_emb = out.text_embeds                         # (C, D)
            img_emb = F.normalize(img_emb, dim=-1)
            txt_emb = F.normalize(txt_emb, dim=-1)
            sim = txt_emb @ img_emb.t()                       # (C, K)
        return sim

    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward(self, image: torch.Tensor, text: Any = None) -> torch.Tensor:
        # 如果 text=None 且 yaml 里配了 text_prompts，自动使用
        # If text=None and text_prompts configured in yaml, use them automatically
        if text is None and hasattr(self, '_default_text_prompts') and self._default_text_prompts:
            text = self._default_text_prompts
        """forward(image, text=None) -> (B, num_classes, H, W) logits.

        For each (sample, class) we write the SAM proposal whose CLIP
        similarity to the class's text prompt is highest.  The "logit" is
        simply the (binary) mask ∈ {0, 1} cast to float; this keeps the
        return signature compatible with the rest of the framework.
        """
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
                "SaLIP forward accepts text as None / str / list-of-str only; "
                f"got {type(text)}"
            )

        # Per-sample loop (SAM operates on numpy arrays)
        import numpy as np
        out = image.new_zeros((B, self.num_classes, H, W))
        img_np = image.detach().cpu().clamp(0, 1).mul(255).permute(0, 2, 3, 1).numpy().astype("uint8")
        for b in range(B):
            self._predictor.set_image(img_np[b])
            masks, _scores, boxes = self.mask_gen(self._predictor, img_np[b])
            if not masks:
                continue
            sim = self._score_proposals_with_clip(img_np[b], masks, boxes, text_list, device)
            best = sim.argmax(dim=-1)  # (num_classes,)
            for c in range(self.num_classes):
                m = masks[int(best[c].item())].astype("float32")
                out[b, c] = torch.from_numpy(m).to(device)
        # convert to logits-ish range so downstream sigmoid/threshold still works
        return out * 10.0 - 5.0


__all__ = ["SaLIP"]
