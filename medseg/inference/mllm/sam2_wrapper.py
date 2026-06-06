"""SAM2 mask generator wrapper.

# Reference: https://github.com/facebookresearch/sam2
# Reference: https://huggingface.co/facebook/sam2-hiera-large
# Paper: https://arxiv.org/abs/2408.00714  (SAM 2, Meta AI 2024)

SAM2 produces high-quality masks given box / point / mask prompts. This
wrapper converts the project-wide normalised :class:`BBox` to the pixel
xyxy boxes the official ``SAM2ImagePredictor`` expects, and returns
``(num_boxes, H, W) uint8`` masks.

Official inference recipe (per the SAM2 README / image_predictor demo)::

    predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
    predictor.set_image(image_rgb)                  # HWC uint8
    masks, scores, logits = predictor.predict(
        point_coords=None, point_labels=None,
        box=boxes_xyxy_pixels,                       # (N, 4) float
        multimask_output=False,
    )

Strict policy: missing ``sam2`` package or load failure -> raise. Empty
input boxes produce an empty ``(0, H, W)`` array (no mock). The
``_mock_predict`` method is retained ONLY for explicit opt-in tests
(``self.mock_mode = True``).
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from medseg.inference.mllm.base import BBox

logger = logging.getLogger(__name__)


class SAM2MaskGenerator:
    """SAM2 mask 生成器（box prompt → mask）。"""

    DEFAULT_MODEL = "facebook/sam2-hiera-large"

    def __init__(
        self,
        model_id: str = "facebook/sam2-hiera-large",
        device: str = "cuda",
        multimask: bool = False,
    ):
        self.model_id = model_id
        self.device = device
        self.multimask = multimask
        self.predictor = None
        self.mock_mode = False
        self._load_model()

    # ------------------------------------------------------------
    def _load_model(self) -> None:
        try:
            from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore
            self.predictor = SAM2ImagePredictor.from_pretrained(self.model_id)
            try:
                # 部分 SAM2 版本 predictor.model 支持 .to(device)
                if hasattr(self.predictor, "model"):
                    self.predictor.model.to(self.device)
            except Exception:
                pass
            logger.info(f"SAM2 loaded: {self.model_id} on {self.device}")
        except Exception:
            # Strict: no mock fallback on load failure.
            raise

    # ------------------------------------------------------------
    def _mock_predict(
        self,
        image: np.ndarray,
        boxes: List[BBox],
    ) -> np.ndarray:
        """Mock：把每个 bbox 内部填充为前景。"""
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
    def predict_from_boxes(
        self,
        image: np.ndarray,
        boxes: List[BBox],
    ) -> np.ndarray:
        """对一张图、若干 bbox 生成 mask。

        Args:
            image: (H, W, 3) uint8 RGB
            boxes: 归一化 BBox 列表

        Returns:
            masks: (N, H, W) uint8 二值 mask
        """
        h, w = image.shape[:2]
        if len(boxes) == 0:
            return np.zeros((0, h, w), dtype=np.uint8)
        if self.mock_mode:
            return self._mock_predict(image, boxes)
        if self.predictor is None:
            raise RuntimeError(
                "SAM2 predictor is None but mock_mode is False; load failed silently."
            )

        # Strict: no inference-time mock fallback.
        import torch
        box_array = np.stack(
            [
                [b.x1 * w, b.y1 * h, b.x2 * w, b.y2 * h]
                for b in boxes
            ],
            axis=0,
        ).astype(np.float32)

        self.predictor.set_image(image)
        with torch.inference_mode():
            masks, scores, _ = self.predictor.predict(
                box=box_array,
                multimask_output=self.multimask,
            )
        # 统一成 (N, H, W) uint8
        if masks.ndim == 4:
            # multimask: (N, M, H, W)，取 score 最高的
            best = scores.argmax(axis=1)
            masks = np.stack(
                [masks[i, best[i]] for i in range(masks.shape[0])], axis=0
            )
        if masks.ndim == 2:
            masks = masks[None]
        return masks.astype(np.uint8)
