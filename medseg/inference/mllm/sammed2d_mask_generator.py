# SAM-Med2D MaskGenerator wrapper (pipeline segmenter 端)
# SAM-Med2D MaskGenerator wrapper (pipeline segmenter side)
# Reference: https://github.com/OpenGVLab/SAM-Med2D
# Paper: https://arxiv.org/abs/2308.16184
"""SAM-Med2D mask generator for the MLLM grounding pipeline.

与 SAM2MaskGenerator / MedSAMMaskGenerator 实现相同的 predict_from_boxes 接口，
可作为 pipeline.py 中 detector → segmenter 的 segmenter 端使用。
Implements the same predict_from_boxes interface as SAM2MaskGenerator /
MedSAMMaskGenerator, usable as the segmenter in pipeline.py.

用法 / Usage (yaml):
    mask_generator:
      type: sammed2d
      checkpoint: ./weights/sam-med2d_b.pth   # 或自动下载 / or auto-download
      device: cuda
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class SAMMed2DMaskGenerator:
    """SAM-Med2D box-prompt mask generator。
    SAM-Med2D box-prompt mask generator.

    接口与 SAM2MaskGenerator / MedSAMMaskGenerator 一致：
    Interface matches SAM2MaskGenerator / MedSAMMaskGenerator:
        predict_from_boxes(image, boxes) -> (N, H, W) uint8 masks
    """

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        model_type: str = "vit_b",
        device: str = "cuda",
        image_size: int = 256,
        **kwargs,
    ):
        self.device = device
        self.image_size = image_size
        self.model = None
        self.mock_mode = False

        # 加载 SAM-Med2D 权重 / Load SAM-Med2D weights
        from segment_anything import sam_model_registry, SamPredictor

        if not checkpoint:
            from medseg.utils.weight_downloader import ensure_weight
            checkpoint = str(ensure_weight("sam_med2d_vit_b"))

        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(
                f"SAM-Med2D checkpoint not found: {checkpoint}. "
                f"Download from https://github.com/OpenGVLab/SAM-Med2D"
            )

        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam = sam.to(device).eval()
        self.predictor = SamPredictor(sam)
        logger.info(f"SAM-Med2D loaded: {model_type} from {checkpoint} on {device}")

    def predict_from_boxes(
        self,
        image: np.ndarray,
        boxes: List,
    ) -> np.ndarray:
        """对一张图 + 若干 bbox 生成 mask。
        Generate masks for an image given bounding boxes.

        Args:
            image: (H, W, 3) uint8 RGB
            boxes: BBox 列表（归一化坐标）/ List of BBox (normalised coords)

        Returns:
            masks: (N, H, W) uint8 二值 mask / binary masks
        """
        h, w = image.shape[:2]
        if len(boxes) == 0:
            return np.zeros((0, h, w), dtype=np.uint8)

        if self.mock_mode or self.predictor is None:
            return self._mock_predict(image, boxes)

        self.predictor.set_image(image)

        masks_out = []
        for b in boxes:
            x1, y1, x2, y2 = b.to_pixel(w, h)
            box_np = np.array([[x1, y1, x2, y2]])
            mask_logits, _, _ = self.predictor.predict(
                box=box_np,
                multimask_output=False,
            )
            mask = (mask_logits[0] > 0).astype(np.uint8)
            masks_out.append(mask)

        return np.stack(masks_out, axis=0) if masks_out else np.zeros((0, h, w), dtype=np.uint8)

    def _mock_predict(self, image, boxes):
        h, w = image.shape[:2]
        masks = np.zeros((len(boxes), h, w), dtype=np.uint8)
        for i, b in enumerate(boxes):
            x1, y1, x2, y2 = b.to_pixel(w, h)
            masks[i, max(0,y1):min(h,y2), max(0,x1):min(w,x2)] = 1
        return masks
