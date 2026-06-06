"""MediSee (ACM Multimedia 2025) — Reasoning-based Pixel-level Perception in Medical Images.

上游来源: https://github.com/Edisonhimself/MediSee  (commit @ main, vendor 时间 2026-04)

* ``_vendor/`` —— 上游 ``model/`` 与 ``utils/`` 整树, **1:1 保留, 不修改任何一行**
  以便对照 paper 复现; vendor 内的绝对导入 (``from utils.utils import ...``,
  ``from model.MediSee import MediSeeForCausalLM``) 通过
  :mod:`medseg.inference.mllm.medisee._path_setup` 注入 ``sys.path`` 解析。
* :mod:`weights_loader` —— 4 套 HF 权重的自动下载 (``snapshot_download``)。
* :mod:`wrapper` —— medseg 框架 glue, 暴露 :class:`MediSeeWrapper`。

依赖 (pip):
* ``transformers >= 4.37``
* ``huggingface_hub >= 0.20``
* ``peft`` (训练时 LoRA, 推理可选)
* ``deepspeed`` (训练时, 推理可选)
* ``bitsandbytes`` (4/8bit 加载, 可选)
* ``sentencepiece`` (Mistral tokenizer)
"""

from .wrapper import MediSeeWrapper

__all__ = ["MediSeeWrapper"]
