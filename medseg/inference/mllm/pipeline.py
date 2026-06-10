"""MLLM × SAM2 / MedSAM grounding-then-segmentation pipeline.

# Reference: https://github.com/facebookresearch/sam2
# Reference: https://github.com/bowang-lab/MedSAM
# Reference: https://github.com/IDEA-Research/GroundingDINO
# Paper: https://arxiv.org/abs/2408.00714  (SAM 2)
# Paper: https://www.nature.com/articles/s41467-024-44824-z  (MedSAM)

Three-stage Detect-then-Segment paradigm:
  Step 1. MLLM (Qwen-VL / InternVL / Grounding-DINO / ...) 文本 → 每类归一化 bbox
  Step 2. SAM2 或 MedSAM 用 bbox 作 prompt → 每个 bbox 的高质量二值 mask
  Step 3. (可选) 现有 medseg 分割模型在 ROI 内 fine-grain refinement

Output:  multi-class label map ``(H, W)``，类别 id 与 ``class_names`` 一一对应。

Strict policy: refinement-config load failures raise (no silent skip).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import numpy as np

from medseg.inference.mllm.base import MLLMGrounder, BBox, DetectionResult
from medseg.inference.mllm.sam2_wrapper import SAM2MaskGenerator

logger = logging.getLogger(__name__)


@dataclass
class PipelineOutput:
    """Pipeline 输出。"""
    label_map: np.ndarray                              # (H, W) int, 0=bg
    per_class_masks: Dict[str, np.ndarray]             # class_name -> (H, W) uint8
    detection: DetectionResult
    refined: bool = False
    raw_logits: Optional[np.ndarray] = None            # (C, H, W) 若 refinement 启用


class MLLMGroundingSegPipeline:
    """MLLM Grounding × SAM2 (× Refinement) 推理 pipeline。"""

    def __init__(
        self,
        grounder: MLLMGrounder,
        mask_generator: SAM2MaskGenerator,
        class_names: List[str],
        refinement_model: Optional[Any] = None,         # torch.nn.Module
        refinement_img_size: int = 224,
        roi_padding: float = 0.1,
    ):
        self.grounder = grounder
        self.mask_generator = mask_generator
        self.class_names = list(class_names)
        self.refinement_model = refinement_model
        self.refinement_img_size = refinement_img_size
        self.roi_padding = roi_padding

    # ------------------------------------------------------------
    def _compose_label_map(
        self,
        per_class_masks: Dict[str, np.ndarray],
        h: int,
        w: int,
    ) -> np.ndarray:
        """把按类别的二值 mask 合成 label map (0 = bg, k = class k)。"""
        label_map = np.zeros((h, w), dtype=np.int32)
        for cid, name in enumerate(self.class_names, start=1):
            m = per_class_masks.get(name)
            if m is None or m.sum() == 0:
                continue
            # 后写入覆盖先写入（同 instance 重叠时取后类）
            label_map[m > 0] = cid
        return label_map

    # ------------------------------------------------------------
    def _refine_roi(
        self,
        image: np.ndarray,
        bbox: BBox,
        class_id: int,
    ) -> Optional[np.ndarray]:
        """对一个 ROI 做精细分割（占位实现，可按需扩展）。"""
        if self.refinement_model is None:
            return None
        try:
            import torch
            import torch.nn.functional as F
            h, w = image.shape[:2]
            # 外扩
            pad = self.roi_padding
            x1 = max(0.0, bbox.x1 - pad)
            y1 = max(0.0, bbox.y1 - pad)
            x2 = min(1.0, bbox.x2 + pad)
            y2 = min(1.0, bbox.y2 + pad)
            px1, py1, px2, py2 = (
                int(x1 * w),
                int(y1 * h),
                int(x2 * w),
                int(y2 * h),
            )
            if px2 - px1 < 4 or py2 - py1 < 4:
                return None
            roi = image[py1:py2, px1:px2]

            # 简单 resize → forward
            roi_t = torch.from_numpy(roi).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            roi_t = F.interpolate(
                roi_t,
                size=(self.refinement_img_size, self.refinement_img_size),
                mode="bilinear",
                align_corners=False,
            )
            with torch.no_grad():
                logits = self.refinement_model(roi_t)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            prob = logits.softmax(1)[0, class_id]  # (h', w')
            mask_small = (prob > 0.5).cpu().numpy().astype(np.uint8)
            # resize 回 ROI 尺寸
            import cv2  # type: ignore
            mask_roi = cv2.resize(
                mask_small,
                (px2 - px1, py2 - py1),
                interpolation=cv2.INTER_NEAREST,
            )
            full = np.zeros((h, w), dtype=np.uint8)
            full[py1:py2, px1:px2] = mask_roi
            return full
        except Exception as e:
            logger.error(f"Refinement failed: {e}")
            return None

    # ------------------------------------------------------------
    def __call__(self, image: np.ndarray) -> PipelineOutput:
        return self.run(image)

    def run(self, image: np.ndarray) -> PipelineOutput:
        """单张图像端到端推理。

        Args:
            image: (H, W, 3) RGB

        Returns:
            PipelineOutput
        """
        h, w = image.shape[:2]
        # Step 1: MLLM grounding
        detection = self.grounder.detect(image, self.class_names)

        # Step 2: SAM2 box → mask
        per_class_masks: Dict[str, np.ndarray] = {}
        for cid, name in enumerate(self.class_names, start=1):
            boxes = detection.boxes_by_class.get(name, [])
            if len(boxes) == 0:
                per_class_masks[name] = np.zeros((h, w), dtype=np.uint8)
                continue

            masks = self.mask_generator.predict_from_boxes(image, boxes)
            # 合并这个类下所有 instance（按 OR）
            union = np.any(masks > 0, axis=0).astype(np.uint8) if masks.shape[0] > 0 \
                else np.zeros((h, w), dtype=np.uint8)

            # Step 3 (optional): refinement
            if self.refinement_model is not None:
                refined = self._refine_roi(image, boxes[0], cid)
                if refined is not None:
                    union = np.maximum(union, refined)
            per_class_masks[name] = union

        label_map = self._compose_label_map(per_class_masks, h, w)
        return PipelineOutput(
            label_map=label_map,
            per_class_masks=per_class_masks,
            detection=detection,
            refined=self.refinement_model is not None,
        )


# ================================================================
# Factory
# ================================================================
def build_pipeline_from_config(cfg: Dict[str, Any]) -> MLLMGroundingSegPipeline:
    """根据 yaml cfg 构建 pipeline。

    期望 cfg 结构（见 configs/training_paradigms/text_guided/synapse_qwen2vl_sam2.yaml）：
      mllm: {type, model_id, device, dtype, prompt_template, class_names, ...}
      mask_generator: {type, model_id, device, multimask, ...}
      refinement: {enabled, config_path, roi_padding, ...}  # optional
    """
    from medseg.inference.mllm import MLLM_REGISTRY, MASK_GENERATOR_REGISTRY

    mllm_cfg = cfg.get("mllm", {})

    # 统一格式 / Unified format:
    #   mllm:
    #     class_names: [...]
    #     grounder: {type, model_id, device, ...}
    #     mask_generator: {type, device, ...}
    #     refinement: {enabled, ...}  # 可选 / optional
    if "grounder" not in mllm_cfg:
        raise ValueError(
            "Pipeline yaml 格式错误：mllm 下必须有 grounder 和 mask_generator 字段。\n"
            "Pipeline yaml format error: mllm must have grounder and mask_generator fields.\n"
            "正确格式 / Correct format:\n"
            "  mllm:\n"
            "    class_names: [...]\n"
            "    grounder: {type: grounding_dino, ...}\n"
            "    mask_generator: {type: sam2, ...}"
        )

    grounder_cfg = mllm_cfg["grounder"]
    mg_cfg = mllm_cfg["mask_generator"]
    refine_cfg = mllm_cfg.get("refinement", {}) or {}
    mllm_type = grounder_cfg["type"]
    grounder_kwargs = {k: v for k, v in grounder_cfg.items() if k != "type"}

    # ---- MLLM Grounder ----
    if mllm_type not in MLLM_REGISTRY:
        raise ValueError(
            f"Unknown mllm type: {mllm_type}. "
            f"Available: {list(MLLM_REGISTRY.keys())}"
        )
    grounder_cls = MLLM_REGISTRY[mllm_type]
    grounder = grounder_cls(**grounder_kwargs)

    # ---- Mask Generator (sam2 / medsam) ----
    mg_type = mg_cfg.get("type", "sam2")
    if mg_type not in MASK_GENERATOR_REGISTRY:
        raise ValueError(
            f"Unknown mask_generator type: {mg_type}. "
            f"Available: {list(MASK_GENERATOR_REGISTRY.keys())}"
        )
    mg_cls = MASK_GENERATOR_REGISTRY[mg_type]
    mg_kwargs = {k: v for k, v in mg_cfg.items() if k != "type"}
    mask_generator = mg_cls(**mg_kwargs)

    # ---- Refinement (optional) ----
    # Strict: if the YAML enables refinement, missing config / build
    # failures raise rather than silently disabling the third stage.
    refinement_model = None
    if refine_cfg.get("enabled", False):
        import yaml as _yaml
        from medseg.model_builder import build_model
        with open(refine_cfg["config_path"]) as _fh:
            seg_cfg = _yaml.safe_load(_fh)
        refinement_model = build_model(seg_cfg)
        refinement_model.eval()

    return MLLMGroundingSegPipeline(
        grounder=grounder,
        mask_generator=mask_generator,
        class_names=mllm_cfg["class_names"],
        refinement_model=refinement_model,
        refinement_img_size=refine_cfg.get("img_size", 224),
        roi_padding=refine_cfg.get("roi_padding", 0.1),
    )
