"""Phi-3.5-Vision grounding wrapper.

# Reference: https://github.com/microsoft/Phi-3CookBook
# Reference: https://huggingface.co/microsoft/Phi-3.5-vision-instruct
# Paper: https://arxiv.org/abs/2404.14219  (Phi-3 technical report)

Phi-3.5-Vision (Microsoft, 2024) is a small (~4B) instruction-tuned VL
model without native grounding tokens, so we use the generic JSON-bbox
prompt + parser. The processor expects a ``<|image_1|>`` placeholder in
the chat message; image is supplied as a list to the processor call.

Strict policy: parse failure -> empty bbox list; model exception -> raise.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from medseg.inference.mllm.generic_vl_base import GenericVLGrounder

logger = logging.getLogger(__name__)


class Phi3VGrounder(GenericVLGrounder):
    """Phi-3.5-Vision grounding wrapper (inference-only)."""

    COORD_SCALE = 1.0

    def __init__(
        self,
        model_id: str = "microsoft/Phi-3.5-vision-instruct",
        device: str = "cuda",
        dtype: str = "bfloat16",
        prompt_template: Optional[str] = None,
        max_new_tokens: int = 128,
        num_crops: int = 4,
        **kwargs,
    ):
        self.num_crops = num_crops
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
            from transformers import AutoModelForCausalLM, AutoProcessor

            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=dtype_map.get(self.dtype, torch.bfloat16),
                device_map=self.device,
                trust_remote_code=True,
                _attn_implementation="eager",
            ).eval()
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                num_crops=self.num_crops,
            )
            logger.info(f"Phi-3.5-Vision loaded: {self.model_id} on {self.device}")
        except Exception:
            # Strict: no mock fallback on load failure. Re-raise the
            # original error so the caller sees what actually broke.
            raise

    # ------------------------------------------------------------------
    def _generate_response(self, image: np.ndarray, prompt: str) -> str:
        import torch
        from PIL import Image

        if image.dtype != np.uint8:
            image = (image * 255).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(image).convert("RGB")

        # Phi-3.5-Vision uses <|image_1|> placeholder
        messages = [{"role": "user", "content": f"<|image_1|>\n{prompt}"}]
        chat = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(chat, [pil_img], return_tensors="pt").to(self.device)

        with torch.no_grad():
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )
        gen_ids = out_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(
            gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
