# 编码器

[English](encoders.md)

本项目提供 169 个注册编码器，分为两种使用模式。

## 两种 Encoder 模式

### 1. 注册表 Encoder

直接使用注册名称（如 `timm_resnet50`、`dinov2`、`biomedclip`）。

### 2. 动态 timm Encoder

以 `timm_` 前缀 + 任意 timm 模型名，自动创建 encoder。即使未在注册表中预注册，也可使用 timm 库中的 1000+ 模型。

```yaml
model:
  encoder:
    name: timm_efficientnet_b7  # 未预注册但可用
    pretrained: true
```

---

## Foundation 模型编码器

Foundation 模型编码器使用 **DPT head**（Dense Prediction Transformer）从 ViT 不同 block 提取多尺度特征（浅层=纹理，深层=语义），构建真正的多层次语义金字塔，可直接接入任意 decoder。

### 通用

| 名称 | 论文 | 年份 | HF Repo | YAML |
|---|---|---|---|---|
| `dinov2` | DINOv2: Learning Robust Visual Features without Supervision | 2024 | `facebook/dinov2-*` | [dinov2_emcad.yaml](../../configs/architectures/combinations/general/dinov2_emcad.yaml), [dinov2_cascade_full.yaml](../../configs/architectures/combinations/general/dinov2_cascade_full.yaml), [dinov2_cfm.yaml](../../configs/architectures/combinations/general/dinov2_cfm.yaml) |
| `dino` | DINO: Self-Distillation with No Labels | ICCV 2021 | `facebook/dino-*` | [dino_cascade_full.yaml](../../configs/architectures/combinations/general/dino_cascade_full.yaml), [dino_emcad.yaml](../../configs/architectures/combinations/general/dino_emcad.yaml) |
| `clip_vit` | CLIP: Learning Transferable Visual Models | ICML 2021 | `openai/clip-*` | [clip_vit_cascade_full.yaml](../../configs/architectures/combinations/general/clip_vit_cascade_full.yaml), [clip_vit_emcad.yaml](../../configs/architectures/combinations/general/clip_vit_emcad.yaml) |
| `sam_vit` | Segment Anything (ViT encoder) | ICCV 2023 | `facebook/sam-*` | [sam_vit_cascade_full.yaml](../../configs/architectures/combinations/general/sam_vit_cascade_full.yaml), [sam_vit_cfm.yaml](../../configs/architectures/combinations/general/sam_vit_cfm.yaml), [sam_vit_emcad.yaml](../../configs/architectures/combinations/general/sam_vit_emcad.yaml) |
| `dinov3` | DINOv3 | 2025 | `facebook/dinov3-*` | [dinov3_cascade_full.yaml](../../configs/architectures/combinations/general/dinov3_cascade_full.yaml), [dinov3_cfm.yaml](../../configs/architectures/combinations/general/dinov3_cfm.yaml), [dinov3_emcad.yaml](../../configs/architectures/combinations/general/dinov3_emcad.yaml) |

### 病理

| 名称 | 论文 | 年份 | HF Repo | YAML |
|---|---|---|---|---|
| `phikon` | Scaling Self-Supervised Learning for Histopathology (Phikon) | 2024 | `owkin/phikon` | [phikon.yaml](../../configs/architectures/foundation/pathology/phikon.yaml) |
| `uni` | Towards a General-Purpose Foundation Model for Computational Pathology | Nature Med 2024 | `MahmoodLab/UNI` (gated) | [uni.yaml](../../configs/architectures/foundation/pathology/uni.yaml) |
| `plip` | PLIP: A Visual-Language Foundation Model for Pathology | Nature Med 2023 | `vinid/plip` | [plip.yaml](../../configs/architectures/foundation/pathology/plip.yaml) |
| `musk` | MUSK: Multi-task Self-supervised Pathology | 2024 | - | [musk.yaml](../../configs/architectures/foundation/pathology/musk.yaml) |
| `phikon_v2` | Phikon-v2 | 2024 | `owkin/phikon-v2` | [phikon_v2_cascade_full.yaml](../../configs/architectures/combinations/general/phikon_v2_cascade_full.yaml) |

### 放射科

| 名称 | 论文 | 年份 | HF Repo | YAML |
|---|---|---|---|---|
| `raddino` | RAD-DINO: Scalable Medical Image Encoders Beyond Text Supervision | 2024 | `microsoft/rad-dino` | [raddino_cascade_full.yaml](../../configs/architectures/combinations/general/raddino_cascade_full.yaml) |
| `omnirad` | OmniRad | 2024 | - | [omnirad_cascade_full.yaml](../../configs/architectures/combinations/general/omnirad_cascade_full.yaml) |
| `medsiglip` | MedSigLIP | 2024 | - | 未提供 |

### 眼科

| 名称 | 论文 | 年份 | HF Repo | YAML |
|---|---|---|---|---|
| `retfound` | RETFound | Nature 2023 | - | 未提供 |
| `retfound_dinov2` | RETFound-DINOv2 | 2024 | - | [retfound_dinov2_cascade_full.yaml](../../configs/architectures/combinations/general/retfound_dinov2_cascade_full.yaml) |
| `flair` | FLAIR: Fine-grained Language-informed Retinal Analysis | 2024 | - | [flair.yaml](../../configs/architectures/foundation/ophthalmology/flair.yaml) |
| `ophmae` | OphMAE | 2024 | - | [ophmae_cascade_full.yaml](../../configs/architectures/combinations/general/ophmae_cascade_full.yaml) |

### 皮肤科

| 名称 | 论文 | 年份 | HF Repo | YAML |
|---|---|---|---|---|
| `panderm` | PanDerm | 2024 | - | [panderm_cascade_full.yaml](../../configs/architectures/combinations/general/panderm_cascade_full.yaml) |
| `dermclip` | DermCLIP | 2024 | - | 未提供 |
| `monet_derm` | Monet-Derm | 2024 | - | 未提供 |

### 内镜

| 名称 | 论文 | 年份 | HF Repo | YAML |
|---|---|---|---|---|
| `endo_vit` | EndoViT | 2024 | - | [endo_vit.yaml](../../configs/architectures/foundation/endoscopy/endo_vit.yaml) |

### 多模态医学

| 名称 | 论文 | 年份 | HF Repo | YAML |
|---|---|---|---|---|
| `biomedclip` | BiomedCLIP | NeurIPS 2023 | `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` | [biomedclip_cascade_full.yaml](../../configs/architectures/combinations/general/biomedclip_cascade_full.yaml), [biomedclip_emcad.yaml](../../configs/architectures/combinations/general/biomedclip_emcad.yaml) |
| `medclip` | MedCLIP | EMNLP 2022 | - | [medclip.yaml](../../configs/architectures/foundation/multimodal_med/medclip.yaml) |
| `keep` | KEEP | 2024 | - | [keep.yaml](../../configs/architectures/foundation/multimodal_med/keep.yaml) |

### 超声

| 名称 | 论文 | 年份 | HF Repo | YAML |
|---|---|---|---|---|
| `usfmae` | USF-MAE | 2024 | - | [usfmae.yaml](../../configs/architectures/foundation/ultrasound/usfmae.yaml) |
| `ultradino` | UltraDINO | 2024 | - | [ultradino_cascade_full.yaml](../../configs/architectures/combinations/general/ultradino_cascade_full.yaml) |
| `ultrafedfm` | UltraFedFM | 2024 | - | [ultrafedfm_cascade_full.yaml](../../configs/architectures/combinations/general/ultrafedfm_cascade_full.yaml) |

### 多模态大语言模型视觉编码器

从多模态大语言模型中提取视觉编码器，用于分割任务。

| 名称 | 论文 | 年份 | YAML |
|---|---|---|---|
| `qwen3_vl_vision` | Qwen3-VL | 2025 | [qwen3_vl_vision.yaml](../../configs/architectures/foundation/mllm_vision/qwen3_vl_vision.yaml) |
| `qwen25_vl_vision` | Qwen2.5-VL | 2025 | [qwen25_vl_vision.yaml](../../configs/architectures/foundation/mllm_vision/qwen25_vl_vision.yaml) |
| `llava_med_vision` | LLaVA-Med | NeurIPS 2023 | [llava_med_vision.yaml](../../configs/architectures/foundation/mllm_vision/llava_med_vision.yaml) |
| `medgemma_vision` | MedGemma | 2025 | [medgemma_vision.yaml](../../configs/architectures/foundation/mllm_vision/medgemma_vision.yaml) |
| `healthgpt_vision` | HealthGPT | 2025 | [healthgpt_vision.yaml](../../configs/architectures/foundation/mllm_vision/healthgpt_vision.yaml) |
| `huatuogpt_vision` | HuatuoGPT-Vision | 2024 | [huatuogpt_vision.yaml](../../configs/architectures/foundation/mllm_vision/huatuogpt_vision.yaml) |
| `hulumed_vision` | HuluMed-Vision | 2024 | [hulumed_vision.yaml](../../configs/architectures/foundation/mllm_vision/hulumed_vision.yaml) |
| `lingshu_vision` | LingShu-Vision | 2024 | [lingshu_vision.yaml](../../configs/architectures/foundation/mllm_vision/lingshu_vision.yaml) |

---

## 预注册 timm Encoder

以下为项目预注册并测试过的 timm encoder（部分列表）：

| 系列 | 编码器名称 | YAML 示例 |
|---|---|---|
| ResNet | `timm_resnet18`, `timm_resnet34`, `timm_resnet50`, `timm_resnet101`, `timm_resnet152` | [unet_resnet50.yaml](../../configs/architectures/combinations/general/unet_resnet50.yaml) |
| ResNeXt | `timm_resnext50_32x4d`, `timm_resnext101_32x8d` | - |
| Wide ResNet | `timm_wide_resnet50_2`, `timm_wide_resnet101_2` | - |
| EfficientNet | `timm_efficientnet_b0` ~ `b5`, `timm_efficientnetv2_s`, `timm_efficientnetv2_m` | [efficientnetv2_cascade_full.yaml](../../configs/architectures/combinations/general/efficientnetv2_cascade_full.yaml) |
| ConvNeXt | `timm_convnext_tiny/small/base/large`, `timm_convnextv2_tiny/base` | [convnext_cascade_full.yaml](../../configs/architectures/combinations/general/convnext_cascade_full.yaml) |
| DenseNet | `timm_densenet121/161/169/201` | [unet_densenet121.yaml](../../configs/architectures/combinations/general/unet_densenet121.yaml) |
| VGG | `timm_vgg16`, `timm_vgg16_bn`, `timm_vgg19`, `timm_vgg19_bn` | - |
| Swin Transformer | `timm_swin_tiny/small/base_patch4_window7_224`, `timm_swinv2_tiny_window8_256` | [unet_swin_tiny.yaml](../../configs/architectures/combinations/general/unet_swin_tiny.yaml) |
| PVTv2 | `timm_pvt_v2_b0` ~ `b4` | [pvtv2_emcad.yaml](../../configs/architectures/combinations/general/pvtv2_emcad.yaml) |
| SegFormer MiT | `timm_mit_b0` ~ `b5` | - |
| MaxViT | `timm_maxvit_tiny/small_tf_224` | [maxvit_cascade_full.yaml](../../configs/architectures/combinations/general/maxvit_cascade_full.yaml) |
| ViT (CLIP) | `timm_vit_clip_base/large/huge` | [unet_vit_clip_base.yaml](../../configs/architectures/combinations/general/unet_vit_clip_base.yaml) |
| ViT (DINOv2) | `timm_vit_dinov2_base/large/giant` | [unet_vit_dinov2_base.yaml](../../configs/architectures/combinations/general/unet_vit_dinov2_base.yaml) |
| ViT (DINOv3) | `timm_vit_dinov3_small/base/large/huge_plus/7b` | [unet_vit_dinov3_base.yaml](../../configs/architectures/combinations/general/unet_vit_dinov3_base.yaml) |
| ViT (MAE) | `timm_vit_mae_base/large` | [unet_vit_mae_base.yaml](../../configs/architectures/combinations/general/unet_vit_mae_base.yaml) |
| ViT (SAM) | `timm_vit_sam_base/large/huge` | [unet_vit_sam_base.yaml](../../configs/architectures/combinations/general/unet_vit_sam_base.yaml) |
| MobileNet | `timm_mobilenetv2_100`, `timm_mobilenetv3_large/small_100` | [unet_mobilenetv3.yaml](../../configs/architectures/combinations/general/unet_mobilenetv3.yaml) |
| Others | `timm_inception_v3`, `timm_ghostnet_100`, `timm_mobilevit_s`, `timm_poolformer_s12/s24`, `timm_fastvit_t8`, `timm_coatnet_0_224` | [unet_fastvit_t8.yaml](../../configs/architectures/combinations/general/unet_fastvit_t8.yaml) |

---

## YAML 使用示例

### Foundation Encoder + 自由组合

```yaml
model:
  num_classes: 9
  img_size: 224
  encoder:
    name: dinov2
    pretrained: true
    in_channels: 3
    params:
      variant: base       # small / base / large / giant
  decoder:
    name: emcad
    params: {}
  skip_connection:
    name: concat
  bottleneck:
    name: none

data:
  type: synapse
  img_size: 224
  train_dir: ./data/Synapse/train_npz
  test_dir: ./data/Synapse/test_vol_h5
  train_list: ./data/Synapse/lists/lists_Synapse/train.txt
  test_list: ./data/Synapse/lists/lists_Synapse/test_vol.txt

training:
  epochs: 200
  batch_size: 16
  num_workers: 4
  loss:
    name: compound
    params:
      losses:
        - name: ce
          weight: 0.4
        - name: dice
          weight: 0.6
  optimizer:
    name: adamw
    lr: 0.0001
    weight_decay: 0.0001
  scheduler:
    name: cosine
    min_lr: 0.000001
```

### 动态 timm Encoder

```yaml
model:
  num_classes: 2
  img_size: 224
  encoder:
    name: timm_efficientnet_b7   # 任意 timm 模型
    pretrained: true
    in_channels: 3
  decoder:
    name: unet
  skip_connection:
    name: concat
  bottleneck:
    name: none

data:
  type: binary
  img_size: 224
  train_dir: ./data/your_dataset/train
  test_dir: ./data/your_dataset/test

training:
  epochs: 100
  batch_size: 16
  num_workers: 4
  loss:
    name: compound
    params:
      losses:
        - name: ce
          weight: 0.4
        - name: dice
          weight: 0.6
  optimizer:
    name: adamw
    lr: 0.0001
    weight_decay: 0.0001
  scheduler:
    name: cosine
    min_lr: 0.000001
```
