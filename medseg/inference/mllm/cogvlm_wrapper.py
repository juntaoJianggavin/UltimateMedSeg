"""CogVLM2 grounding wrapper.

# Reference: https://github.com/THUDM/CogVLM2
# Reference: https://huggingface.co/THUDM/cogvlm2-llama3-chat-19B
# Paper: https://arxiv.org/abs/2311.03079  (CogVLM)
# Paper: https://arxiv.org/abs/2408.16500  (CogVLM2 / GLM-4V)

CogVLM2 (THUDM, 2024) is a strong open VL model with optional grounding
support. The model is loaded with ``trust_remote_code=True`` and exposes
a custom ``build_conversation_input_ids`` API (per the official
modeling file).

Coordinate convention:
  - default chat ckpt → free-form JSON-bbox in normalised ``[0, 1]``
  - grounding ckpt    → native ``[[x1,y1,x2,y2]]`` in ``[0, 1000]``
                        (pass ``coord_scale=1000.0`` at construction).
Strict policy: parse failure -> empty bbox list; model exception -> raise.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from medseg.inference.mllm.generic_vl_base import GenericVLGrounder

logger = logging.getLogger(__name__)


class CogVLMGrounder(GenericVLGrounder):
    """CogVLM2 grounding wrapper (inference-only)."""

    # Default normalised; users with grounding-tuned ckpt should pass coord_scale=1000.
    COORD_SCALE = 1.0

    def __init__(
        self,
        model_id: str = "THUDM/cogvlm2-llama3-chat-19B",
        device: str = "cuda",
        dtype: str = "bfloat16",
        prompt_template: Optional[str] = None,
        max_new_tokens: int = 128,
        coord_scale: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(
            model_id=model_id,
            device=device,
            dtype=dtype,
            prompt_template=prompt_template,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )
        if coord_scale is not None:
            # Allow per-instance override (e.g. 1000.0 for grounding ckpts)
            self.COORD_SCALE = float(coord_scale)

    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            self.processor = AutoTokenizer.from_pretrained(
                self.model_id, trust_remote_code=True
            )
            self.model = (
                AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    torch_dtype=dtype_map.get(self.dtype, torch.bfloat16),
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
                .eval()
                .to(self.device)
            )
            logger.info(f"CogVLM2 loaded: {self.model_id} on {self.device}")
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

        # CogVLM2 custom API - returned by trust_remote_code modeling file
        inputs = self.model.build_conversation_input_ids(
            self.processor,
            query=prompt,
            history=[],
            images=[pil_img],
            template_version="chat",
        )
        # Move tensors to device
        device = self.device
        model_inputs = {
            "input_ids": inputs["input_ids"].unsqueeze(0).to(device),
            "token_type_ids": inputs["token_type_ids"].unsqueeze(0).to(device),
            "attention_mask": inputs["attention_mask"].unsqueeze(0).to(device),
            "images": [[img.to(device).to(self.model.dtype) for img in inputs["images"]]],
        }
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
        }
        with torch.no_grad():
            output_ids = self.model.generate(**model_inputs, **gen_kwargs)
            output_ids = output_ids[:, model_inputs["input_ids"].shape[1]:]
            response = self.processor.decode(output_ids[0], skip_special_tokens=True)
        return str(response).strip()
