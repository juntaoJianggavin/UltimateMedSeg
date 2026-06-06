"""Grounding DINO grounder wrapper (MLLMGrounder 接口适配).

# Reference: https://github.com/IDEA-Research/GroundingDINO
# Paper: https://arxiv.org/abs/2303.05499  (Grounding DINO, ECCV 2024)

把已有的 ``medseg.grounding_dino_wrapper.GroundingDINODetector`` 适配到
MLLMGrounder 抽象，可与 SAM2 / MedSAM 组合复用 MLLMGroundingSegPipeline。
GroundingDINO 不是 MLLM，但 input→output 形态（自然语言 → bbox）与 MLLM
grounding 一致；统一抽象便于调用方按 YAML 切换。

Coordinate convention: the official ``predict()`` returns boxes in
**normalised [0, 1]** xyxy space, identical to the project-wide
:class:`BBox` convention — no rescaling is needed.

Strict policy: GroundingDINO is treated as a hard dependency; load
failure raises (no auto mock).
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np

from medseg.inference.mllm.base import MLLMGrounder, BBox, DetectionResult

logger = logging.getLogger(__name__)


class GroundingDINOGrounder(MLLMGrounder):
    """Grounding DINO 包装为 MLLMGrounder。"""

    DEFAULT_PROMPT = "a medical CT image of {class_name}"

    def __init__(
        self,
        model_id: str = "tiny",                    # 'tiny' (SwinT) | 'base' (SwinB)
        device: str = "cuda",
        dtype: str = "float32",                    # GroundingDINO 默认 fp32
        prompt_template: str | None = None,
        weights_path: str | None = None,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        **kwargs,
    ):
        self.weights_path = weights_path
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        # 不走 LLM generate，所以 max_new_tokens 留默认
        super().__init__(
            model_id=model_id,
            device=device,
            dtype=dtype,
            prompt_template=prompt_template or self.DEFAULT_PROMPT,
            max_new_tokens=0,
            **kwargs,
        )

    # ------------------------------------------------------------
    def _load_model(self) -> None:
        # Strict: re-raise any load failure (the inner detector now also
        # raises rather than auto-mocking).
        from medseg.grounding_dino_wrapper import GroundingDINODetector
        self._detector = GroundingDINODetector(
            model_type=self.model_id,
            device=self.device,
            model_path=self.weights_path,
        )
        if self._detector.model is None:
            raise RuntimeError(
                "GroundingDINODetector loaded but .model is None — "
                "checkpoint missing or load step silently failed."
            )
        self.model = self._detector.model
        self.processor = None
        logger.info(
            f"GroundingDINO grounder loaded (model_type={self.model_id})"
        )

    # ------------------------------------------------------------
    def _build_text_prompt(self, class_names: List[str]) -> str:
        """Grounding DINO 用 ' . ' 分隔多类别短语。"""
        parts = [self.prompt_template.format(class_name=n) for n in class_names]
        return " . ".join(parts) + " ."

    # ------------------------------------------------------------
    def _detect_single_class(self, image: np.ndarray, class_name: str) -> List[BBox]:
        """Single-class entry point (the multi-class ``detect()`` is preferred)."""
        if self.mock_mode:
            return self._mock_detect_single_class(image, class_name)
        if self._detector is None:
            raise RuntimeError("GroundingDINO detector is None.")
        text_prompt = self.prompt_template.format(class_name=class_name) + " ."
        boxes, scores, phrases = self._detector.detect(
            image=image,
            text_prompt=text_prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
        )
        return self._to_bboxes(boxes, scores, phrases, class_name)

    # ------------------------------------------------------------
    def detect(
        self,
        image: np.ndarray,
        class_names: List[str],
    ) -> DetectionResult:
        """Multi-class fast path: GroundingDINO accepts ``"a . b . c ."``
        as a single caption and detects all phrases in one forward.
        """
        h, w = image.shape[:2]
        result = DetectionResult(image_shape=(h, w))

        if self.mock_mode:
            for name in class_names:
                result.boxes_by_class[name] = self._mock_detect_single_class(
                    image, name
                )
            return result
        if self._detector is None:
            raise RuntimeError("GroundingDINO detector is None.")

        # Strict: no inference-time mock fallback; let errors propagate.
        text_prompt = self._build_text_prompt(class_names)
        boxes, scores, phrases = self._detector.detect(
            image=image,
            text_prompt=text_prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
        )
        grouped: dict = {name: [] for name in class_names}
        for box, score, phrase in zip(boxes, scores, phrases):
            for name in class_names:
                if name.lower() in str(phrase).lower():
                    grouped[name].append(
                        BBox(
                            x1=float(box[0]),
                            y1=float(box[1]),
                            x2=float(box[2]),
                            y2=float(box[3]),
                            score=float(score),
                            label=name,
                        )
                    )
                    break
        result.boxes_by_class = grouped
        result.raw_response = f"GroundingDINO: {len(boxes)} raw boxes"
        return result

    # ------------------------------------------------------------
    @staticmethod
    def _to_bboxes(boxes, scores, phrases, class_name: str) -> List[BBox]:
        out: List[BBox] = []
        for box, score in zip(boxes, scores):
            out.append(
                BBox(
                    x1=float(box[0]),
                    y1=float(box[1]),
                    x2=float(box[2]),
                    y2=float(box[3]),
                    score=float(score),
                    label=class_name,
                )
            )
        return out
