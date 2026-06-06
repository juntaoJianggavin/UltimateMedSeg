"""Qwen2-VL grounding wrapper.

# Reference: https://github.com/QwenLM/Qwen2-VL
# Reference: https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct
# Paper: https://arxiv.org/abs/2409.12191  (Qwen2-VL technical report)

Qwen2-VL (Alibaba, 2024) natively emits visual-grounding tokens. The exact
output markup is:

    <|object_ref_start|>{class_name}<|object_ref_end|>
    <|box_start|>(x1,y1),(x2,y2)<|box_end|>

Coordinates are integers in the absolute domain ``[0, 1000]`` (regardless
of the input image size — the model internally normalises). We divide by
``1000`` to convert to the project-wide ``[0, 1]`` :class:`BBox` space.
"""

from __future__ import annotations

import logging
import re
from typing import List

import numpy as np

from medseg.inference.mllm.base import MLLMGrounder, BBox

logger = logging.getLogger(__name__)


# Qwen2-VL grounding 输出抓取正则
_QWEN_BOX_RE = re.compile(
    r"<\|box_start\|>\(?\s*(\d+)\s*,\s*(\d+)\s*\)?\s*,?\s*\(?\s*(\d+)\s*,\s*(\d+)\s*\)?\s*<\|box_end\|>"
)


class Qwen2VLGrounder(MLLMGrounder):
    """Qwen2-VL 系列 grounding wrapper（推理-only，无训练）。"""

    DEFAULT_PROMPT = (
        "Locate the {class_name} in this medical image. "
        "Output the bounding box in the format "
        "<|box_start|>(x1,y1),(x2,y2)<|box_end|>."
    )

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2-VL-7B-Instruct",
        device: str = "cuda",
        dtype: str = "bfloat16",
        prompt_template: str | None = None,
        max_new_tokens: int = 128,
        use_flash_attn: bool = True,
        **kwargs,
    ):
        self.use_flash_attn = use_flash_attn
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
        # Strict: no mock fallback on load failure (per project policy:
        # silently switching to a mock detector hides real errors and gives
        # bogus boxes). Caller must explicitly construct with mock_mode=True
        # if a deterministic mock is wanted for unit testing.
        import torch
        from transformers import (
            Qwen2VLForConditionalGeneration,
            AutoProcessor,
        )
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        kw = {}
        if self.use_flash_attn:
            kw["attn_implementation"] = "flash_attention_2"
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=dtype_map.get(self.dtype, torch.bfloat16),
            device_map=self.device,
            **kw,
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        logger.info(f"Qwen2-VL loaded: {self.model_id} on {self.device}")

    # ------------------------------------------------------------
    def _build_messages(self, image: np.ndarray, class_name: str):
        """构造 Qwen2-VL chat messages。"""
        text = self.prompt_template.format(class_name=class_name)
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text},
                ],
            }
        ]

    # ------------------------------------------------------------
    def _parse_response(
        self,
        response: str,
        img_h: int,
        img_w: int,
        class_name: str,
    ) -> List[BBox]:
        """从 Qwen2-VL 输出文本中抓取 bbox。"""
        boxes: List[BBox] = []
        for m in _QWEN_BOX_RE.finditer(response):
            x1, y1, x2, y2 = (int(v) for v in m.groups())
            # Qwen2-VL 坐标域 [0, 1000] → 归一化到 [0, 1]
            boxes.append(
                BBox(
                    x1=x1 / 1000.0,
                    y1=y1 / 1000.0,
                    x2=x2 / 1000.0,
                    y2=y2 / 1000.0,
                    score=1.0,
                    label=class_name,
                )
            )
        return boxes

    # ------------------------------------------------------------
    def _detect_single_class(self, image: np.ndarray, class_name: str) -> List[BBox]:
        if self.mock_mode:
            return self._mock_detect_single_class(image, class_name)
        if self.model is None:
            raise RuntimeError(
                "Qwen2-VL model is None but mock_mode is False; "
                "_load_model() did not raise — call site is inconsistent."
            )

        try:
            from qwen_vl_utils import process_vision_info  # type: ignore
        except ImportError:
            process_vision_info = None  # type: ignore

        # No except-and-fallback-to-mock here: a runtime failure should
        # surface, not be masked by a deterministic dummy detector.
        import torch
        messages = self._build_messages(image, class_name)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if process_vision_info is not None:
            image_inputs, video_inputs = process_vision_info(messages)
        else:
            image_inputs, video_inputs = [image], None

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            generated = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens
            )
        generated_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated)
        ]
        response = self.processor.batch_decode(
            generated_trimmed, skip_special_tokens=False
        )[0]

        h, w = image.shape[:2]
        return self._parse_response(response, h, w, class_name)
