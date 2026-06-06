"""GenericVLGrounder: shared base for general-purpose Vision-Language models
that do **not** emit native grounding tokens (unlike Qwen2-VL / Qwen3-VL).

# Reference: https://github.com/haotian-liu/LLaVA
# Reference: https://github.com/microsoft/Phi-3CookBook
# Reference: https://github.com/OpenBMB/MiniCPM-V
# Reference: https://github.com/THUDM/CogVLM2
# Paper: https://arxiv.org/abs/2310.03744  (LLaVA-1.5)

Strategy:
    1. Prompt the model to output a JSON dict of normalised bbox coords.
    2. Parse the response with two regexes (JSON-style + bare coords),
       both robust to surrounding text.
    3. Strict policy: an empty / unparseable response returns an empty
       bbox list; any model-side exception is re-raised. We never
       silently substitute a mock bbox at inference time.

Subclasses must implement:
    _load_model()           : bring up self.model / self.processor
    _generate_response(image, prompt) -> str
                              : run the model and return the raw text reply.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional

import numpy as np

from medseg.inference.mllm.base import MLLMGrounder, BBox

logger = logging.getLogger(__name__)


# ---------- response parsers -------------------------------------------------
# 1) JSON-style: {"bbox": [x1, y1, x2, y2]} or {"box": [x1,y1,x2,y2]}
_JSON_BBOX_RE = re.compile(
    r'"(?:bbox|box|bounding_box)"\s*:\s*\[\s*'
    r'([-+]?\d*\.?\d+)\s*,\s*'
    r'([-+]?\d*\.?\d+)\s*,\s*'
    r'([-+]?\d*\.?\d+)\s*,\s*'
    r'([-+]?\d*\.?\d+)\s*\]'
)
# 2) Bare list-style: [x1, y1, x2, y2] or (x1,y1,x2,y2)
_PLAIN_BBOX_RE = re.compile(
    r'[\[\(]\s*'
    r'([-+]?\d*\.?\d+)\s*,\s*'
    r'([-+]?\d*\.?\d+)\s*,\s*'
    r'([-+]?\d*\.?\d+)\s*,\s*'
    r'([-+]?\d*\.?\d+)\s*[\]\)]'
)


def _parse_bbox_response(
    response: str,
    class_name: str,
    coord_scale: float = 1.0,
) -> List[BBox]:
    """Extract bboxes from a free-form VL reply.

    Args:
        response: model output text.
        class_name: label to attach.
        coord_scale: divide raw coords by this; e.g. 1000.0 if the model
            outputs [0, 1000] integer space, 1.0 if [0, 1] normalised,
            or pass image_w/h for pixel space coords (handled by caller).
    """
    boxes: List[BBox] = []

    def _push(x1, y1, x2, y2):
        x1, y1, x2, y2 = (float(v) / coord_scale for v in (x1, y1, x2, y2))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        # Skip degenerate boxes
        if x2 - x1 < 1e-4 or y2 - y1 < 1e-4:
            return
        boxes.append(
            BBox(
                x1=max(0.0, min(1.0, x1)),
                y1=max(0.0, min(1.0, y1)),
                x2=max(0.0, min(1.0, x2)),
                y2=max(0.0, min(1.0, y2)),
                score=1.0,
                label=class_name,
            )
        )

    # Try JSON pattern first (more reliable when the model follows the prompt)
    for m in _JSON_BBOX_RE.finditer(response):
        _push(*m.groups())
    if boxes:
        return boxes

    # Fallback: any [x1,y1,x2,y2] / (x1,y1,x2,y2) tuple
    for m in _PLAIN_BBOX_RE.finditer(response):
        _push(*m.groups())
    return boxes


class GenericVLGrounder(MLLMGrounder):
    """Base grounder for general VL models without native box tokens."""

    DEFAULT_PROMPT = (
        "Locate the {class_name} in this medical image and return ONLY a "
        'JSON object: {{"bbox": [x1, y1, x2, y2]}} where each value is a '
        "float in [0, 1] (normalised coordinates relative to the image "
        "width/height). If multiple instances exist, return the most "
        "salient one. Do not output any other text."
    )

    # Coordinate scale of the model output. Override in subclass if the
    # model emits coords in [0, 1000] or pixel space.
    COORD_SCALE: float = 1.0

    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        prompt_template: Optional[str] = None,
        max_new_tokens: int = 128,
        **kwargs,
    ):
        super().__init__(
            model_id=model_id,
            device=device,
            dtype=dtype,
            prompt_template=prompt_template or self.DEFAULT_PROMPT,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Subclass must implement
    # ------------------------------------------------------------------
    def _load_model(self) -> None:    # pragma: no cover - abstract
        raise NotImplementedError

    def _generate_response(self, image: np.ndarray, prompt: str) -> str:    # pragma: no cover
        raise NotImplementedError

    # ------------------------------------------------------------------
    def _detect_single_class(self, image: np.ndarray, class_name: str) -> List[BBox]:
        # Explicit opt-in mock path (set by caller for assembly tests).
        if self.mock_mode:
            return self._mock_detect_single_class(image, class_name)
        if self.model is None:
            # _load_model() should have raised; surface the inconsistency
            # rather than silently producing a fake bbox.
            raise RuntimeError(
                f"[{type(self).__name__}] model is None but mock_mode is False; "
                "the load step did not raise and the instance is unusable."
            )
        # Strict: no inference-time fallback. Parsing failures return an
        # empty list (downstream pipeline treats it as 'no detection'),
        # but any model-side exception is re-raised.
        prompt = self.prompt_template.format(class_name=class_name)
        response = self._generate_response(image, prompt)
        boxes = _parse_bbox_response(response, class_name, self.COORD_SCALE)
        if not boxes:
            logger.debug(
                f"[{type(self).__name__}] could not parse bbox from "
                f"response: {response[:200]!r}"
            )
        return boxes
