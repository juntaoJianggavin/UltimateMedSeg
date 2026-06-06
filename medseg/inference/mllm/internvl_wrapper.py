"""InternVL grounding wrapper.

# Reference: https://github.com/OpenGVLab/InternVL
# Reference: https://huggingface.co/OpenGVLab/InternVL2_5-8B
# Paper: https://arxiv.org/abs/2312.14238  (InternVL, CVPR 2024)
# Paper: https://arxiv.org/abs/2412.05271  (InternVL 2.5)

InternVL 2 / 2.5 (OpenGVLab) supports natural-language visual grounding.
The native output markup is::

    <ref>{class_name}</ref><box>[[x1, y1, x2, y2]]</box>

Coordinates are absolute integers in ``[0, 1000]``; we divide by 1000
to map into the project-wide normalised :class:`BBox` space.
"""

from __future__ import annotations

import logging
import re
from typing import List

import numpy as np

from medseg.inference.mllm.base import MLLMGrounder, BBox

logger = logging.getLogger(__name__)


_INTERNVL_BOX_RE = re.compile(
    r"<box>\s*\[?\[?\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]?\]?\s*</box>"
)


class InternVLGrounder(MLLMGrounder):
    """InternVL 系列 grounding wrapper（推理-only）。"""

    DEFAULT_PROMPT = (
        "<image>\nPlease provide the bounding box coordinates of the {class_name} "
        "in this medical image, in the format <box>[[x1, y1, x2, y2]]</box>."
    )

    def __init__(
        self,
        model_id: str = "OpenGVLab/InternVL2_5-8B",
        device: str = "cuda",
        dtype: str = "bfloat16",
        prompt_template: str | None = None,
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

    # ------------------------------------------------------------
    def _load_model(self) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            self.model = (
                AutoModel.from_pretrained(
                    self.model_id,
                    torch_dtype=dtype_map.get(self.dtype, torch.bfloat16),
                    trust_remote_code=True,
                )
                .eval()
                .to(self.device)
            )
            self.processor = AutoTokenizer.from_pretrained(
                self.model_id, trust_remote_code=True, use_fast=False
            )
            logger.info(f"InternVL loaded: {self.model_id} on {self.device}")
        except Exception:
            # Strict: no mock fallback on load failure.
            raise

    # ------------------------------------------------------------
    def _parse_response(
        self,
        response: str,
        class_name: str,
    ) -> List[BBox]:
        boxes: List[BBox] = []
        for m in _INTERNVL_BOX_RE.finditer(response):
            x1, y1, x2, y2 = (int(v) for v in m.groups())
            boxes.append(
                BBox(
                    x1=max(0.0, x1 / 1000.0),
                    y1=max(0.0, y1 / 1000.0),
                    x2=min(1.0, x2 / 1000.0),
                    y2=min(1.0, y2 / 1000.0),
                    score=1.0,
                    label=class_name,
                )
            )
        return boxes

    # ------------------------------------------------------------
    def _preprocess_image(self, image: np.ndarray):
        """转 PIL → 调用 InternVL 内置 image transform。

        PIL is a hard dependency for the wrapper; if missing we raise
        (strict policy — no mock substitution).
        """
        from PIL import Image  # noqa: F401  - hard dep, let ImportError propagate
        if image.dtype != np.uint8:
            image = (image * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(image).convert("RGB")

    # ------------------------------------------------------------
    def _detect_single_class(self, image: np.ndarray, class_name: str) -> List[BBox]:
        if self.mock_mode:
            return self._mock_detect_single_class(image, class_name)
        if self.model is None:
            raise RuntimeError(
                "InternVL model is None but mock_mode is False; "
                "_load_model() did not raise — call site is inconsistent."
            )

        pil_img = self._preprocess_image(image)
        # InternVL exposes a custom .chat() method (trust_remote_code).
        question = self.prompt_template.format(class_name=class_name)
        generation_config = dict(
            max_new_tokens=self.max_new_tokens, do_sample=False
        )
        response = self.model.chat(
            self.processor,
            pil_img,
            question,
            generation_config,
        )
        return self._parse_response(response, class_name)
