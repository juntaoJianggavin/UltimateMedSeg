"""MiniCPM-V grounding wrapper.

# Reference: https://github.com/OpenBMB/MiniCPM-V
# Reference: https://huggingface.co/openbmb/MiniCPM-V-2_6
# Paper: https://arxiv.org/abs/2408.01800  (MiniCPM-V technical report)

MiniCPM-V 2.5 / 2.6 (OpenBMB) supports grounding when explicitly
prompted. The model exposes a custom ``.chat()`` API via
``trust_remote_code=True``. We use the generic JSON-bbox prompt and the
parser inherited from :class:`GenericVLGrounder`.

Strict policy: parse failure -> empty bbox list; model exception -> raise.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from medseg.inference.mllm.generic_vl_base import GenericVLGrounder

logger = logging.getLogger(__name__)


class MiniCPMVGrounder(GenericVLGrounder):
    """MiniCPM-V grounding wrapper (inference-only)."""

    COORD_SCALE = 1.0

    def __init__(
        self,
        model_id: str = "openbmb/MiniCPM-V-2_6",
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
            prompt_template=prompt_template,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )

    # ------------------------------------------------------------------
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
                self.model_id, trust_remote_code=True
            )
            logger.info(f"MiniCPM-V loaded: {self.model_id} on {self.device}")
        except Exception:
            # Strict: no mock fallback on load failure. Re-raise the
            # original error so the caller sees what actually broke.
            raise

    # ------------------------------------------------------------------
    def _generate_response(self, image: np.ndarray, prompt: str) -> str:
        from PIL import Image

        if image.dtype != np.uint8:
            image = (image * 255).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(image).convert("RGB")

        msgs = [{"role": "user", "content": [pil_img, prompt]}]
        # MiniCPM-V exposes a .chat method
        response = self.model.chat(
            image=None,
            msgs=msgs,
            tokenizer=self.processor,
            sampling=False,
            max_new_tokens=self.max_new_tokens,
        )
        if isinstance(response, tuple):
            response = response[0]
        return str(response)
