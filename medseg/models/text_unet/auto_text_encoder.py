"""自动文本编码器：用户传字符串，内部自动 tokenize。
Auto text encoder: users pass strings, tokenization is handled internally.

用法 / Usage:
    在模型 __init__ 中:
        self.text_enc = AutoTextEncoder("openai/clip-vit-base-patch32", max_length=77)

    在模型 forward 中:
        # text 可以是字符串、字符串列表、或已 tokenize 的 dict
        # text can be a string, list of strings, or pre-tokenized dict
        encoded = self.text_enc(text, device=image.device)
        # encoded = {"input_ids": (B, L), "attention_mask": (B, L)}

yaml 中用户只需写:
    arch_params:
      text_prompts:
        - "a lung lesion"
        - "normal lung tissue"
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class AutoTextEncoder(nn.Module):
    """自动 tokenizer + 编码器。接受字符串，输出 token dict。
    Auto tokenizer + encoder. Accepts strings, outputs token dicts.

    支持三种输入 / Supports three input types:
        1. str 或 List[str] → 自动 tokenize / auto tokenize
        2. dict {input_ids, attention_mask} → 直接透传 / pass through
        3. Tensor (B, L, D) → 直接透传 / pass through (pre-computed embeddings)
    """

    def __init__(
        self,
        tokenizer_name: str,
        max_length: int = 77,
        tokenizer_type: str = "auto",
    ):
        super().__init__()
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length
        self._tokenizer = None
        self._tokenizer_type = tokenizer_type

    def _get_tokenizer(self):
        """懒加载 tokenizer / Lazy-load tokenizer."""
        if self._tokenizer is not None:
            return self._tokenizer

        from medseg.utils.weight_downloader import hf_from_pretrained

        if self._tokenizer_type == "clip":
            from transformers import CLIPTokenizer
            self._tokenizer = hf_from_pretrained(CLIPTokenizer, self.tokenizer_name)
        elif self._tokenizer_type == "clip_processor":
            from transformers import CLIPProcessor
            self._tokenizer = hf_from_pretrained(CLIPProcessor, self.tokenizer_name)
        else:
            from transformers import AutoTokenizer
            self._tokenizer = hf_from_pretrained(AutoTokenizer, self.tokenizer_name)

        return self._tokenizer

    def forward(
        self,
        text: Union[str, List[str], Dict[str, torch.Tensor], torch.Tensor, None],
        device: torch.device = torch.device("cpu"),
        batch_size: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """将任意形式的 text 转为 {input_ids, attention_mask} dict。
        Convert any text format to {input_ids, attention_mask} dict.

        Args:
            text: 字符串 / 字符串列表 / 已 tokenize 的 dict / 预计算 embedding
                  string / list of strings / pre-tokenized dict / pre-computed embedding
            device: 输出张量的设备 / device for output tensors
            batch_size: 当 text 是单个字符串时，复制到 batch_size 份
                        when text is a single string, replicate to batch_size copies

        Returns:
            dict with "input_ids" (B, L) and "attention_mask" (B, L)
        """
        # 已经是 dict → 直接透传
        if isinstance(text, dict):
            return {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in text.items()}

        # 已经是 Tensor → 视为预计算 embedding，直接返回
        if isinstance(text, torch.Tensor):
            return {"embeddings": text.to(device)}

        # None → 返回空（让模型用默认 prompt）
        if text is None:
            return None

        # 字符串 → tokenize
        tokenizer = self._get_tokenizer()

        if isinstance(text, str):
            text = [text] * batch_size

        encoded = tokenizer(
            text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"].to(device),
            "attention_mask": encoded["attention_mask"].to(device),
        }


# ======================================================================
# 预置 tokenizer 工厂 / Pre-configured tokenizer factories
# ======================================================================

def clip_text_encoder(clip_name: str = "openai/clip-vit-base-patch32", max_length: int = 77):
    """CLIP tokenizer (用于 CRIS / CausalCLIPSeg / CXR-CLIP-Seg / TP-DRSeg)"""
    return AutoTextEncoder(clip_name, max_length=max_length, tokenizer_type="clip")

def bert_text_encoder(bert_name: str = "bert-base-uncased", max_length: int = 24):
    """BERT/CXR-BERT tokenizer (用于 LanGuideMedSeg / LViT)"""
    return AutoTextEncoder(bert_name, max_length=max_length, tokenizer_type="auto")

def pubmedbert_text_encoder(max_length: int = 64):
    """PubMedBERT tokenizer (用于 BiomedParse / TPRO)"""
    return AutoTextEncoder(
        "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        max_length=max_length, tokenizer_type="auto",
    )
