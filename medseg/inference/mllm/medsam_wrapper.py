"""MedSAM mask generator wrapper.

# Reference: https://github.com/bowang-lab/MedSAM
# Reference: https://huggingface.co/wanglab/medsam-vit-base
# Paper: https://www.nature.com/articles/s41467-024-44824-z
# Paper: https://arxiv.org/abs/2304.12306  (preprint of MedSAM)

MedSAM (Ma et al., Nature Communications 2024, "Segment Anything in
Medical Images") is a SAM-ViT-B model fine-tuned on > 1.5M medical
image-mask pairs. It accepts a single bounding-box prompt per call and
returns one binary mask.

This wrapper exposes the same ``predict_from_boxes(image, boxes) ->
(N, H, W) uint8`` interface as :class:`SAM2MaskGenerator`, so the
unified pipeline can swap SAM2 ↔ MedSAM via the YAML
``mask_generator.type`` field.

Official inference recipe (from ``MedSAM_Inference.ipynb`` /
``utils/demo.py``)::

    # 1. Resize 1024×1024 (no aspect-ratio preservation).
    img_1024 = skimage.transform.resize(
        img_rgb, (1024, 1024), order=3, preserve_range=True,
        anti_aliasing=True,
    ).astype(np.uint8)
    # 2. Per-image min-max normalisation (NOT /255).
    img_1024 = (img_1024 - img_1024.min()) / np.clip(
        img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
    )
    img_t = torch.tensor(img_1024).float().permute(2,0,1)[None].to(device)
    # 3. Encode image once, then per box:
    image_embedding = medsam.image_encoder(img_t)
    box_1024 = box_xyxy_pixel / np.array([W, H, W, H]) * 1024
    sparse, dense = medsam.prompt_encoder(
        points=None, boxes=box_torch, masks=None,
    )
    low_res, _ = medsam.mask_decoder(
        image_embeddings=image_embedding,
        image_pe=medsam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=False,
    )
    prob = torch.sigmoid(low_res)        # then upsample to (H, W) and threshold > 0.5

Strict policy: missing ``segment_anything`` or missing checkpoint -> raise
(no auto mock). The ``_mock_predict`` method is retained only for
explicit opt-in pipeline-assembly tests.

External requirements (user supplies):
  - ``pip install git+https://github.com/facebookresearch/segment-anything.git``
  - MedSAM checkpoint ``medsam_vit_b.pth`` (set ``$MEDSAM_CHECKPOINT``
    or pass ``checkpoint=...``). Download URL is documented in the
    official MedSAM README.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import numpy as np

from medseg.inference.mllm.base import BBox

logger = logging.getLogger(__name__)


class MedSAMMaskGenerator:
    """MedSAM mask generator (box prompt -> binary mask).

    Parameters
    ----------
    model_type : str
        SAM ViT variant. MedSAM officially releases only ``vit_b``.
    checkpoint : str | None
        Path to ``medsam_vit_b.pth``. If ``None`` we try the
        environment variable ``MEDSAM_CHECKPOINT``.
    device : str
        ``cuda`` / ``cpu``.
    image_size : int
        Resize side used by MedSAM (1024 by default, matching the
        official demo notebook).
    """

    DEFAULT_MODEL = "vit_b"
    DEFAULT_CHECKPOINT_ENV = "MEDSAM_CHECKPOINT"
    OFFICIAL_HF_PATH = "wanglab/medsam-vit-base"

    def __init__(
        self,
        model_id: str = "vit_b",
        checkpoint: Optional[str] = None,
        device: str = "cuda",
        image_size: int = 1024,
        multimask: bool = False,                          # kept for parity with SAM2
        **kwargs,
    ):
        # ``model_id`` is named like SAM2 wrapper for unified YAML schema.
        self.model_type = model_id
        self.checkpoint = checkpoint or os.environ.get(self.DEFAULT_CHECKPOINT_ENV)
        self.device = device
        self.image_size = image_size
        self.multimask = multimask           # MedSAM only supports single mask, ignored
        self.model = None
        self.mock_mode = False
        self._load_model()

    # ------------------------------------------------------------
    def _load_model(self) -> None:
        import torch
        from segment_anything import sam_model_registry  # type: ignore
        from medseg.utils.weight_downloader import ensure_weight

        # Resolution order:
        #   1. explicit ctor `checkpoint=...`
        #   2. $MEDSAM_CHECKPOINT env var
        #   3. auto-download via the unified weight downloader
        # ensure_weight raises WeightDownloadError with the manual URL
        # (Google Drive / Zenodo) on failure — no silent fallback.
        if self.checkpoint and os.path.isfile(self.checkpoint):
            ckpt_path = self.checkpoint
        else:
            ckpt_path = str(ensure_weight("medsam_vit_b"))
            self.checkpoint = ckpt_path

        self.model = sam_model_registry[self.model_type](
            checkpoint=ckpt_path,
        )
        self.model = self.model.to(self.device).eval()
        logger.info(
            f"MedSAM loaded: {self.model_type} from {ckpt_path} on {self.device}"
        )

    # ------------------------------------------------------------
    def _mock_predict(
        self,
        image: np.ndarray,
        boxes: List[BBox],
    ) -> np.ndarray:
        """Mock: fill each bbox interior with foreground (same shape as SAM2 mock)."""
        h, w = image.shape[:2]
        if len(boxes) == 0:
            return np.zeros((0, h, w), dtype=np.uint8)
        masks = np.zeros((len(boxes), h, w), dtype=np.uint8)
        for i, b in enumerate(boxes):
            x1, y1, x2, y2 = b.to_pixel(w, h)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            masks[i, y1:y2, x1:x2] = 1
        return masks

    # ------------------------------------------------------------
    @staticmethod
    def _resize_image(image: np.ndarray, size: int):
        """Resize HxWx3 image to (size, size) and apply MedSAM's official
        **per-image min-max** normalisation to ``[0, 1]``.

        NOTE: the previous version divided by 255 — this is *not* what
        MedSAM does and can degrade the mask decoder accuracy. The
        official notebook (``MedSAM_Inference.ipynb``) uses min-max
        normalisation; we follow it exactly.
        """
        import cv2  # type: ignore

        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        img_resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)
        img_resized = img_resized.astype(np.float32)
        lo, hi = img_resized.min(), img_resized.max()
        denom = max(hi - lo, 1e-8)
        return (img_resized - lo) / denom

    @staticmethod
    def _resize_mask_back(mask_low: np.ndarray, h: int, w: int) -> np.ndarray:
        import cv2  # type: ignore

        return cv2.resize(
            mask_low.astype(np.uint8),
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        )

    # ------------------------------------------------------------
    def predict_from_boxes(
        self,
        image: np.ndarray,
        boxes: List[BBox],
    ) -> np.ndarray:
        """Generate masks for one image with multiple boxes.

        Parameters
        ----------
        image : (H, W, 3) uint8 RGB
        boxes : list of normalised :class:`BBox`

        Returns
        -------
        masks : (N, H, W) uint8 binary masks (one per box)
        """
        # Explicit empty-input contract: no boxes -> no masks.
        h, w = image.shape[:2]
        if len(boxes) == 0:
            return np.zeros((0, h, w), dtype=np.uint8)
        if self.mock_mode:
            return self._mock_predict(image, boxes)
        if self.model is None:
            raise RuntimeError(
                "MedSAM model is None but mock_mode is False; load failed silently."
            )

        # Strict: no inference-time mock fallback; let exceptions propagate.
        import torch

        S = self.image_size

        # 1. Image preprocess (resize → min-max → CHW tensor); embed once.
        img_S = self._resize_image(image, S)
        img_t = (
            torch.from_numpy(img_S)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device)
        )
        with torch.inference_mode():
            image_embedding = self.model.image_encoder(img_t)

            masks_out = []
            for b in boxes:
                # 2. BBox: normalised [0,1] -> pixel @ original (W,H) -> rescale to S.
                x1p = b.x1 * w
                y1p = b.y1 * h
                x2p = b.x2 * w
                y2p = b.y2 * h
                sx = S / float(w)
                sy = S / float(h)
                box_S = np.array(
                    [[x1p * sx, y1p * sy, x2p * sx, y2p * sy]],
                    dtype=np.float32,
                )
                box_torch = torch.as_tensor(box_S, device=self.device)[None]
                # 3. Prompt encoder with single box.
                sparse, dense = self.model.prompt_encoder(
                    points=None, boxes=box_torch, masks=None,
                )
                # 4. Mask decoder.
                low_res, _ = self.model.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=self.model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse,
                    dense_prompt_embeddings=dense,
                    multimask_output=False,
                )
                # 5. Upsample low-res logits to S×S, then sigmoid → threshold →
                #    resize back to original (H, W). (Matches the official
                #    notebook which uses F.interpolate to image_size first.)
                import torch.nn.functional as F  # local import keeps top clean
                low_res_full = F.interpolate(
                    low_res, size=(S, S), mode="bilinear", align_corners=False,
                )
                prob = torch.sigmoid(low_res_full).cpu().numpy()[0, 0]
                mask_S = (prob > 0.5).astype(np.uint8)
                masks_out.append(self._resize_mask_back(mask_S, h, w))

        return np.stack(masks_out, axis=0).astype(np.uint8)
