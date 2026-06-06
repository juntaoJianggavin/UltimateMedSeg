"""LLaVA / LLaVA-NeXT grounding wrapper.

# Reference: https://github.com/haotian-liu/LLaVA
# Reference: https://github.com/LLaVA-VL/LLaVA-NeXT
# Reference: https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf
# Paper: https://arxiv.org/abs/2310.03744  (LLaVA-1.5, CVPR 2024)
# Paper: https://arxiv.org/abs/2407.07895  (LLaVA-NeXT)

LLaVA-1.5 / LLaVA-NeXT (a.k.a. LLaVA-1.6) is a general-purpose VL chat
model **without** native grounding tokens, so we:

    1. Prompt it to output ``{"bbox": [x1,y1,x2,y2]}`` with normalised
       (``[0, 1]``) coordinates.
    2. Parse the response with :func:`_parse_bbox_response` inherited
       from :class:`GenericVLGrounder`.

Strict policy: a parse failure returns an empty bbox list; any model
exception is re-raised (no mock substitution).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from medseg.inference.mllm.generic_vl_base import GenericVLGrounder

logger = logging.getLogger(__name__)


class LLaVAGrounder(GenericVLGrounder):
    """LLaVA / LLaVA-NeXT grounding wrapper (inference-only)."""

    # LLaVA emits free-form text; the prompt asks for normalised [0, 1] coords.
    COORD_SCALE = 1.0

    def __init__(
        self,
        model_id: str = "llava-hf/llava-v1.6-mistral-7b-hf",
        device: str = "cuda",
        dtype: str = "bfloat16",
        prompt_template: Optional[str] = None,
        max_new_tokens: int = 128,
        use_next: bool = True,
        **kwargs,
    ):
        self.use_next = use_next
        super().__init__(
            model_id=model_id,
            device=device,
            dtype=dtype,
            prompt_template=prompt_template,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )

    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        try:
            import torch
            from transformers import AutoProcessor

            if self.use_next:
                from transformers import (  # type: ignore
                    LlavaNextForConditionalGeneration as _LlavaModel,
                )
            else:
                from transformers import (  # type: ignore
                    LlavaForConditionalGeneration as _LlavaModel,
                )
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            self.model = _LlavaModel.from_pretrained(
                self.model_id,
                torch_dtype=dtype_map.get(self.dtype, torch.bfloat16),
                device_map=self.device,
            )
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            logger.info(
                f"LLaVA{'-NeXT' if self.use_next else ''} loaded: "
                f"{self.model_id} on {self.device}"
            )
        except Exception:
            # Strict: no mock fallback on load failure. Re-raise the
            # original error so the caller sees what actually broke.
            raise

    # ------------------------------------------------------------------
    def _generate_response(self, image: np.ndarray, prompt: str) -> str:
        import torch
        from PIL import Image

        # Cast to PIL RGB
        if image.dtype != np.uint8:
            image = (image * 255).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(image).convert("RGB")

        # LLaVA-NeXT chat template
        conv = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        chat = self.processor.apply_chat_template(conv, add_generation_prompt=True)
        inputs = self.processor(
            images=pil_img, text=chat, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        # Strip the input prompt tokens
        prompt_len = inputs["input_ids"].shape[1]
        gen_ids = out_ids[0, prompt_len:]
        return self.processor.tokenizer.decode(gen_ids, skip_special_tokens=True)
