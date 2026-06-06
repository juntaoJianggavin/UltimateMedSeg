"""Qwen3-VL grounding wrapper.

# Reference: https://github.com/QwenLM/Qwen3-VL
# Reference: https://huggingface.co/Qwen/Qwen3-VL-7B-Instruct
# Paper: https://arxiv.org/abs/2409.12191  (Qwen-VL family; Qwen3-VL extends)

Qwen3-VL (Alibaba, 2025) is the successor of Qwen2-VL with the same
native visual grounding output format::

    <|object_ref_start|>{class_name}<|object_ref_end|>
    <|box_start|>(x1,y1),(x2,y2)<|box_end|>

Coordinates remain absolute integers in ``[0, 1000]``, so we reuse the
parser by inheriting :class:`Qwen2VLGrounder` and only swap the model
class to ``Qwen3VLForConditionalGeneration``.
"""

from __future__ import annotations

import logging

from medseg.inference.mllm.qwen2vl_wrapper import Qwen2VLGrounder

logger = logging.getLogger(__name__)


class Qwen3VLGrounder(Qwen2VLGrounder):
    """Qwen3-VL grounding wrapper.

    Inherits all parsing / prompt / mock behaviour from
    :class:`Qwen2VLGrounder`; only the HuggingFace model class is
    swapped to ``Qwen3VLForConditionalGeneration``.
    """

    DEFAULT_PROMPT = (
        "Locate the {class_name} in this medical image. "
        "Output the bounding box in the format "
        "<|box_start|>(x1,y1),(x2,y2)<|box_end|>."
    )

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-7B-Instruct",
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

    # ------------------------------------------------------------
    def _load_model(self) -> None:
        """Load Qwen3-VL via transformers. Raises on any failure.

        We deliberately do **not** substitute ``AutoModelForVision2Seq`` on
        ImportError — the generic wrapper drops Qwen3-VL-specific forward
        logic (M-RoPE-2D, dynamic image resolution) and would silently emit
        wrong grounding boxes. The user must upgrade ``transformers`` to a
        version exposing ``Qwen3VLForConditionalGeneration``.
        """
        import torch
        from transformers import (  # type: ignore
            AutoProcessor,
            Qwen3VLForConditionalGeneration as _Qwen3VLModel,
        )

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        kw = {}
        if self.use_flash_attn:
            kw["attn_implementation"] = "flash_attention_2"
        self.model = _Qwen3VLModel.from_pretrained(
            self.model_id,
            torch_dtype=dtype_map.get(self.dtype, torch.bfloat16),
            device_map=self.device,
            **kw,
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        logger.info(f"Qwen3-VL loaded: {self.model_id} on {self.device}")
