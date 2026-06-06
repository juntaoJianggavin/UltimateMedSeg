"""MLLM-based Grounding × Segmentation pipeline.

模块结构：
- base.py                : MLLMGrounder 抽象基类（统一 detect 接口）
- generic_vl_base.py     : GenericVLGrounder（无原生 box token 的 VL 模型共享基类）
- qwen2vl_wrapper.py     : Qwen2-VL grounding wrapper（原生 [0,1000] box token）
- qwen25vl_wrapper.py    : Qwen2.5-VL grounding wrapper
- qwen3vl_wrapper.py     : Qwen3-VL grounding wrapper
- internvl_wrapper.py    : InternVL grounding wrapper
- llava_wrapper.py       : LLaVA / LLaVA-NeXT grounding wrapper
- minicpm_v_wrapper.py   : MiniCPM-V grounding wrapper
- phi3v_wrapper.py       : Phi-3.5-Vision grounding wrapper
- cogvlm_wrapper.py      : CogVLM2 grounding wrapper
- grounding_dino_grounder.py : Grounding DINO（专用开放词检测器）
- sam2_wrapper.py        : SAM2 mask decoder wrapper (box→mask)
- medsam_wrapper.py      : MedSAM mask decoder wrapper
- pipeline.py            : MLLMGroundingSegPipeline 三阶段 pipeline

设计 Paradigm: Detect-then-Segment
  Step 1. MLLM 自然语言 → 归一化 bbox
  Step 2. SAM2/MedSAM 用 bbox 作为 prompt → 高质量 mask
  Step 3. (可选) 现有分割模型在 ROI 内做 fine-grain 精细分割

Strict no-fallback policy (project-wide):
  缺依赖 / 缺权重 / 推理报错 一律 raise，**不会**自动切到 mock 输出。
  仅当调用方显式设置 ``instance.mock_mode = True`` 才会走 mock 路径
  （仅用于 YAML/pipeline 装配的单元测试）。

依赖（可选，按需安装）：
  Qwen2-VL / Qwen2.5-VL : pip install transformers>=4.45 accelerate qwen-vl-utils
  Qwen3-VL  : pip install transformers>=4.50 accelerate
  InternVL  : pip install transformers>=4.40 timm einops
  LLaVA-NeXT: pip install transformers>=4.40 accelerate
  MiniCPM-V : pip install transformers timm sentencepiece
  Phi-3.5-V : pip install transformers>=4.43 accelerate
  CogVLM2   : pip install transformers accelerate sentencepiece einops
  SAM2      : pip install git+https://github.com/facebookresearch/sam2.git
"""

from medseg.inference.mllm.base import MLLMGrounder, BBox, DetectionResult
from medseg.inference.mllm.generic_vl_base import GenericVLGrounder
from medseg.inference.mllm.qwen2vl_wrapper import Qwen2VLGrounder
from medseg.inference.mllm.qwen25vl_wrapper import Qwen25VLGrounder
from medseg.inference.mllm.qwen3vl_wrapper import Qwen3VLGrounder
from medseg.inference.mllm.internvl_wrapper import InternVLGrounder
from medseg.inference.mllm.llava_wrapper import LLaVAGrounder
from medseg.inference.mllm.minicpm_v_wrapper import MiniCPMVGrounder
from medseg.inference.mllm.phi3v_wrapper import Phi3VGrounder
from medseg.inference.mllm.cogvlm_wrapper import CogVLMGrounder
from medseg.inference.mllm.grounding_dino_grounder import GroundingDINOGrounder
from medseg.inference.mllm.sam2_wrapper import SAM2MaskGenerator
from medseg.inference.mllm.medsam_wrapper import MedSAMMaskGenerator
from medseg.inference.mllm.pipeline import MLLMGroundingSegPipeline, build_pipeline_from_config

MLLM_REGISTRY = {
    # Qwen family (native grounding tokens)
    "qwen2vl": Qwen2VLGrounder,
    "qwen25vl": Qwen25VLGrounder,
    "qwen2_5vl": Qwen25VLGrounder,         # alias
    "qwen3vl": Qwen3VLGrounder,
    # InternVL family
    "internvl": InternVLGrounder,
    # General VL models (JSON-bbox prompted)
    "llava": LLaVAGrounder,
    "llava_next": LLaVAGrounder,            # alias (set use_next=True in cfg)
    "minicpm_v": MiniCPMVGrounder,
    "phi3v": Phi3VGrounder,
    "phi35v": Phi3VGrounder,                # alias
    "cogvlm": CogVLMGrounder,
    "cogvlm2": CogVLMGrounder,              # alias
    # Specialised open-vocab detector
    "grounding_dino": GroundingDINOGrounder,
}

from medseg.inference.mllm.sammed2d_mask_generator import SAMMed2DMaskGenerator
from medseg.inference.mllm.lite_medsam_mask_generator import LiteMedSAMMaskGenerator

# MediSee (LLM reasoning segmenter，原 mllm_seg，已合并)
from medseg.inference.mllm.medisee import MediSeeWrapper

MASK_GENERATOR_REGISTRY = {
    "sam2": SAM2MaskGenerator,
    "medsam": MedSAMMaskGenerator,
    "sammed2d": SAMMed2DMaskGenerator,      # SAM-Med2D (点/框 prompt)
    "lite_medsam": LiteMedSAMMaskGenerator,  # LiteMedSAM (轻量级)
}

__all__ = [
    "MLLMGrounder",
    "GenericVLGrounder",
    "BBox",
    "DetectionResult",
    "Qwen2VLGrounder",
    "Qwen25VLGrounder",
    "Qwen3VLGrounder",
    "InternVLGrounder",
    "LLaVAGrounder",
    "MiniCPMVGrounder",
    "Phi3VGrounder",
    "CogVLMGrounder",
    "GroundingDINOGrounder",
    "SAM2MaskGenerator",
    "MedSAMMaskGenerator",
    "MLLMGroundingSegPipeline",
    "build_pipeline_from_config",
    "MLLM_REGISTRY",
    "MASK_GENERATOR_REGISTRY",
]
