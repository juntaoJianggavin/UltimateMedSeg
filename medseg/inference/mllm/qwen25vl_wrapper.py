"""Qwen2.5-VL grounding wrapper.

# Reference: https://github.com/QwenLM/Qwen2.5-VL
# Reference: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
# Paper: https://arxiv.org/abs/2502.13923  (Qwen2.5-VL technical report)

Qwen2.5-VL (Alibaba, 2025) keeps the same native grounding output format
as Qwen2-VL::

    <|object_ref_start|>{class_name}<|object_ref_end|>
    <|box_start|>(x1,y1),(x2,y2)<|box_end|>

Coordinates remain in the absolute integer domain ``[0, 1000]``, so we
reuse the parser / prompt / mock logic from :class:`Qwen2VLGrounder` and
only swap the HuggingFace model class.
"""

from __future__ import annotations

import logging

from medseg.inference.mllm.qwen2vl_wrapper import Qwen2VLGrounder

logger = logging.getLogger(__name__)


class Qwen25VLGrounder(Qwen2VLGrounder):
    """Qwen2.5-VL grounding wrapper (inference-only)."""

    DEFAULT_PROMPT = (
        "Locate the {class_name} in this medical image. "
        "Output the bounding box in the format "
        "<|box_start|>(x1,y1),(x2,y2)<|box_end|>."
    )

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "cuda",
        dtype: str = "bfloat16",
        prompt_template: str | None = None,
        max_new_tokens: int = 128,
        use_flash_attn: bool = True,
        **kwargs,
    ):
        super().__init__(
            model_id=model_id,
            device=device,
            dtype=dtype,
            prompt_template=prompt_template,
            max_new_tokens=max_new_tokens,
            use_flash_attn=use_flash_attn,
            **kwargs,
        )

    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        """Load Qwen2.5-VL via transformers. Raises if the model class is
        missing — we deliberately do **not** substitute ``AutoModelForVision2Seq``,
        because the auto wrapper would silently drop Qwen2.5-VL-specific
        forward-pass logic (M-RoPE, windowed attention) and quietly degrade
        grounding accuracy. The user must upgrade ``transformers>=4.49``.
        """
        import torch
        from transformers import (  # type: ignore
            AutoProcessor,
            Qwen2_5_VLForConditionalGeneration as _Qwen25VLModel,
        )

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        kw = {}
        if self.use_flash_attn:
            kw["attn_implementation"] = "flash_attention_2"
        self.model = _Qwen25VLModel.from_pretrained(
            self.model_id,
            torch_dtype=dtype_map.get(self.dtype, torch.bfloat16),
            device_map=self.device,
            **kw,
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        logger.info(f"Qwen2.5-VL loaded: {self.model_id} on {self.device}")
