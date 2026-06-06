# 文本引导分割

[English](text_guided.md)

本框架支持两种文本引导范式：可训练模型 (end-to-end) 和推理 pipeline (detector + segmenter)。

---

## 可训练模型 (12)

位于 `medseg/models/text_unet/`，均为 2D 端到端文本-视觉分割模型。

| Key | 模型 | 论文 | 发表 | GitHub |
|-----|------|------|------|--------|
| `tganet` | TGANet | Tomar et al. | MICCAI 2022 | [nikhilroxtomar/TGANet](https://github.com/nikhilroxtomar/TGANet) |
| `lvit` | LViT | Li et al. | TMI 2023 | [HUANGLIZI/LViT](https://github.com/HUANGLIZI/LViT) |
| `languide` | LanGuideMedSeg | Zhong et al. | MICCAI 2023 | [Junelin2333/LanGuideMedSeg-MICCAI2023](https://github.com/Junelin2333/LanGuideMedSeg-MICCAI2023) |
| `clip_universal` | CLIP-Driven Universal Model | Liu et al. | ICCV 2023 | [ljwztc/CLIP-Driven-Universal-Model](https://github.com/ljwztc/CLIP-Driven-Universal-Model) |
| `cris` | CRIS | Wang et al. | CVPR 2022 | [DerrickWang005/CRIS.pytorch](https://github.com/DerrickWang005/CRIS.pytorch) |
| `biomedparse` | BiomedParse | Zhao et al. | Nature Methods 2024 | [microsoft/BiomedParse](https://github.com/microsoft/BiomedParse) |
| `tpro` | TPRO | Zhang et al. | MICCAI 2023 | [shijun18/TPRO](https://github.com/shijun18/TPRO) |
| `salip` | SaLIP | Aleem et al. | BMVC 2024 | [aleemsidra/SaLIP](https://github.com/aleemsidra/SaLIP) |
| `causal_clipseg` | CausalCLIPSeg | Chen et al. | MICCAI 2024 | [WUTCM-Lab/CausalCLIPSeg](https://github.com/WUTCM-Lab/CausalCLIPSeg) |
| `medclip_sam` | MedCLIP-SAM | Koleilat et al. | MICCAI 2024 | [HealthX-Lab/MedCLIP-SAM](https://github.com/HealthX-Lab/MedCLIP-SAM) |
| `tp_drseg` | TPDRSeg | - | - | - |
| `cxrclipseg` | CXRCLIPSeg | - | - | - |

### 文本输入格式

文本通过配置中的 `class_names` 提供：

```yaml
model:
  text_guided:
    model_type: TextPromptUNet
    prompt_mode: clip              # clip | bert | word2vec
    embed_dim: 512
    use_external_encoder: true
    class_names:                   # 自然语言描述
      - background region
      - spleen organ
      - right kidney organ
      - liver organ
```

### 可训练模型配置

**基于 CLIP (TextPromptUNet)：**

```yaml
model:
  text_guided:
    model_type: TextPromptUNet
    prompt_mode: clip
    embed_dim: 512
    use_external_encoder: true
    class_names:
      - background region
      - spleen organ
      - right kidney organ
  encoder:
    name: timm_vit_clip_base
    pretrained: true
    in_channels: 3
    img_size: 256

data:
  type: synapse
  img_size: 256
  train_dir: ./data/Synapse/train_npz
  val_dir: ./data/Synapse/test_vol_h5

training:
  epochs: 200
  batch_size: 8
  optimizer:
    name: adamw
    lr: 1e-4
  scheduler:
    name: cosine
    min_lr: 1e-6
  loss:
    name: compound
    params:
      ce_weight: 1.0
      dice_weight: 1.0
```

**LViT（逐图文本）：**

```yaml
model:
  num_classes: 1
  img_size: 224
  architecture: lvit
  arch_params:
    base_channel: 64
    text_len: 10
    text_embed_dim: 768

data:
  type: mosmed_plus
  img_size: 224
  data_root: ./data/MosMedDataPlus
  tokenizer_name: bert-base-uncased
  text_max_length: 10
  text_source: dataset       # 从数据集 Excel 读取逐图文本
```

### 训练

```bash
python train_text_guided.py --config configs/training_paradigms/text_guided/synapse_clip.yaml
```

---

## 推理 Pipeline

检测-再分割：检测器定位 + 分割器生成掩码。

### 检测器 (5)

| 检测器 | 类型 | 来源 |
|--------|------|------|
| Grounding DINO | 开放词汇检测器 | [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) |
| Qwen2-VL | MLLM 定位 | [QwenLM/Qwen2-VL](https://github.com/QwenLM/Qwen2-VL) |
| Qwen2.5-VL | MLLM 定位 | [QwenLM/Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) |
| Qwen3-VL | MLLM 定位 | [QwenLM/Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) |
| InternVL | MLLM 定位 | [OpenGVLab/InternVL](https://github.com/OpenGVLab/InternVL) |

### 分割器 (4)

| 分割器 | 论文 | 来源 |
|--------|------|------|
| SAM2 | Meta, 2024 | [facebookresearch/sam2](https://github.com/facebookresearch/sam2) |
| MedSAM | Ma et al., NatComm 2024 | [bowang-lab/MedSAM](https://github.com/bowang-lab/MedSAM) |
| SAM-Med2D | - | - |
| LiteMedSAM | - | - |

### Pipeline 配置

```yaml
mllm:
  class_names:
    - spleen
    - right kidney
    - left kidney
    - liver
    - stomach
  grounder:
    type: grounding_dino     # grounding_dino | qwen2_vl | qwen3_vl | internvl
    model_id: tiny
    device: cuda
    box_threshold: 0.35
    text_threshold: 0.25
    prompt_template: "a medical CT image of {class_name}"
  mask_generator:
    type: sam2               # sam2 | medsam | sammed2d | litemedsam
    model_id: facebook/sam2-hiera-large
    device: cuda
    multimask: false
  refinement:
    enabled: false

data:
  type: synapse
  img_size: 1024
  test_dir: ./data/Synapse/test_vol_h5
```

### 使用方法

```python
import yaml
from medseg.mllm import build_pipeline_from_config

cfg = yaml.safe_load(open('configs/training_paradigms/text_guided/synapse_grounding_dino_sam2.yaml'))
pipe = build_pipeline_from_config(cfg)

out = pipe(image_rgb_uint8)
label_map = out.label_map            # (H, W) int
per_class = out.per_class_masks      # {'spleen': mask, ...}
```

配置位于 `configs/training_paradigms/text_guided/`。
