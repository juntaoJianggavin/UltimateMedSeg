# LiteMedSAM MaskGenerator wrapper (pipeline segmenter 端)
# LiteMedSAM MaskGenerator wrapper (pipeline segmenter side)
# Reference: https://github.com/bowang-lab/MedSAM/tree/LiteMedSAM
# Paper: https://arxiv.org/abs/2403.20329
"""LiteMedSAM mask generator for the MLLM grounding pipeline.

轻量级 MedSAM (TinyViT-5M encoder)，可作为 pipeline 的 segmenter 端。
Lightweight MedSAM (TinyViT-5M encoder), usable as pipeline segmenter.

用法 / Usage (yaml):
    mask_generator:
      type: lite_medsam
      checkpoint: ./weights/lite_medsam.pth   # 可选 / optional
      device: cuda
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class LiteMedSAMMaskGenerator:
    """LiteMedSAM box-prompt mask generator。
    LiteMedSAM box-prompt mask generator.

    接口与 SAM2MaskGenerator / MedSAMMaskGenerator 一致。
    Interface matches SAM2MaskGenerator / MedSAMMaskGenerator.
    """

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        device: str = "cuda",
        image_size: int = 256,
        **kwargs,
    ):
        self.device = device
        self.image_size = image_size
        self.mock_mode = False

        # 加载 LiteMedSAM 模型 / Load LiteMedSAM model
        from medseg.models.networks.sam.lite_medsam import LiteMedSAM
        self.model = LiteMedSAM(
            in_channels=3,
            num_classes=1,
            img_size=image_size,
            pretrained=True,
            checkpoint=checkpoint,
        )
        self.model = self.model.to(device).eval()
        logger.info(f"LiteMedSAM loaded on {device}")

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
        import torch.nn.functional as F

        h, w = image.shape[:2]
        if len(boxes) == 0:
            return np.zeros((0, h, w), dtype=np.uint8)

        if self.mock_mode:
            return self._mock_predict(image, boxes)

        masks_out = []
        for b in boxes:
            x1, y1, x2, y2 = b.to_pixel(w, h)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                masks_out.append(np.zeros((h, w), dtype=np.uint8))
                continue

            # 裁剪 ROI 并 resize / Crop ROI and resize
            roi = image[y1:y2, x1:x2, :]
            roi_t = torch.from_numpy(roi).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            roi_t = F.interpolate(roi_t, size=(self.image_size, self.image_size),
                                  mode='bilinear', align_corners=False)
            roi_t = roi_t.to(self.device)

            # Box prompt (整个 ROI = [0,0,1,1])
            box_prompt = torch.tensor([[0.0, 0.0, 1.0, 1.0]], device=self.device)

            with torch.no_grad():
                logits = self.model(roi_t, text={"boxes": box_prompt})
                mask_roi = (logits[0, 0] > 0).cpu().numpy().astype(np.uint8)

            # Resize mask 回原始 ROI 尺寸 / Resize mask back to original ROI size
            import cv2
            mask_full = np.zeros((h, w), dtype=np.uint8)
            mask_resized = cv2.resize(mask_roi, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
            mask_full[y1:y2, x1:x2] = mask_resized
            masks_out.append(mask_full)

        return np.stack(masks_out, axis=0) if masks_out else np.zeros((0, h, w), dtype=np.uint8)

    def _mock_predict(self, image, boxes):
        h, w = image.shape[:2]
        masks = np.zeros((len(boxes), h, w), dtype=np.uint8)
        for i, b in enumerate(boxes):
            x1, y1, x2, y2 = b.to_pixel(w, h)
            masks[i, max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = 1
        return masks
