# 编码器

[English](encoders.md)

本项目提供 172 个注册编码器，分为两种使用模式。

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

| 名称 | 论文 | 年份 | HF Repo |
|---|---|---|---|
| `dinov2` | DINOv2: Learning Robust Visual Features without Supervision | 2024 | `facebook/dinov2-*` |
| `dino` | DINO: Self-Distillation with No Labels | ICCV 2021 | `facebook/dino-*` |
| `clip_vit` | CLIP: Learning Transferable Visual Models | ICML 2021 | `openai/clip-*` |
| `sam_vit` | Segment Anything (ViT encoder) | ICCV 2023 | `facebook/sam-*` |
| `dinov3` | DINOv3 | 2025 | `facebook/dinov3-*` |

### 病理

| 名称 | 论文 | 年份 | HF Repo |
|---|---|---|---|
| `phikon` | Scaling Self-Supervised Learning for Histopathology (Phikon) | 2024 | `owkin/phikon` |
| `uni` | Towards a General-Purpose Foundation Model for Computational Pathology | Nature Med 2024 | `MahmoodLab/UNI` (gated) |
| `plip` | PLIP: A Visual-Language Foundation Model for Pathology | Nature Med 2023 | `vinid/plip` |
| `musk` | MUSK: Multi-task Self-supervised Pathology | 2024 | - |
| `phikon_v2` | Phikon-v2 | 2024 | `owkin/phikon-v2` |
| `path_foundation` | PathFoundation | 2024 | - |

### 放射科

| 名称 | 论文 | 年份 | HF Repo |
|---|---|---|---|
| `raddino` | RAD-DINO: Scalable Medical Image Encoders Beyond Text Supervision | 2024 | `microsoft/rad-dino` |
| `cxr_foundation` | CXR-Foundation | 2024 | - |
| `omnirad` | OmniRad | 2024 | - |
| `medsiglip` | MedSigLIP | 2024 | - |

### 眼科

| 名称 | 论文 | 年份 | HF Repo |
|---|---|---|---|
| `retfound` | RETFound | Nature 2023 | - |
| `retfound_dinov2` | RETFound-DINOv2 | 2024 | - |
| `flair` | FLAIR: Fine-grained Language-informed Retinal Analysis | 2024 | - |
| `ophmae` | OphMAE | 2024 | - |

### 皮肤科

| 名称 | 论文 | 年份 | HF Repo |
|---|---|---|---|
| `derm_foundation` | DermFoundation | 2024 | - |
| `panderm` | PanDerm | 2024 | - |
| `dermclip` | DermCLIP | 2024 | - |
| `monet_derm` | Monet-Derm | 2024 | - |

### 内镜

| 名称 | 论文 | 年份 | HF Repo |
|---|---|---|---|
| `endo_vit` | EndoViT | 2024 | - |

### 多模态医学

| 名称 | 论文 | 年份 | HF Repo |
|---|---|---|---|
| `biomedclip` | BiomedCLIP | NeurIPS 2023 | `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` |
| `medclip` | MedCLIP | EMNLP 2022 | - |
| `keep` | KEEP | 2024 | - |

### 超声

| 名称 | 论文 | 年份 | HF Repo |
|---|---|---|---|
| `usfmae` | USF-MAE | 2024 | - |
| `ultradino` | UltraDINO | 2024 | - |
| `ultrafedfm` | UltraFedFM | 2024 | - |

### 多模态大语言模型视觉编码器

从多模态大语言模型中提取视觉编码器，用于分割任务。

| 名称 | 论文 | 年份 |
|---|---|---|
| `qwen3_vl_vision` | Qwen3-VL | 2025 |
| `qwen25_vl_vision` | Qwen2.5-VL | 2025 |
| `llava_med_vision` | LLaVA-Med | NeurIPS 2023 |
| `medgemma_vision` | MedGemma | 2025 |
| `healthgpt_vision` | HealthGPT | 2025 |
| `huatuogpt_vision` | HuatuoGPT-Vision | 2024 |
| `hulumed_vision` | HuluMed-Vision | 2024 |
| `lingshu_vision` | LingShu-Vision | 2024 |

---

## 预注册 timm Encoder

以下为项目预注册并测试过的 timm encoder（部分列表）：

| 系列 | 编码器名称 |
|---|---|
| ResNet | `timm_resnet18`, `timm_resnet34`, `timm_resnet50`, `timm_resnet101`, `timm_resnet152` |
| ResNeXt | `timm_resnext50_32x4d`, `timm_resnext101_32x8d` |
| Wide ResNet | `timm_wide_resnet50_2`, `timm_wide_resnet101_2` |
| EfficientNet | `timm_efficientnet_b0` ~ `b5`, `timm_efficientnetv2_s`, `timm_efficientnetv2_m` |
| ConvNeXt | `timm_convnext_tiny/small/base/large`, `timm_convnextv2_tiny/base` |
| DenseNet | `timm_densenet121/161/169/201` |
| VGG | `timm_vgg16`, `timm_vgg16_bn`, `timm_vgg19`, `timm_vgg19_bn` |
| Swin Transformer | `timm_swin_tiny/small/base_patch4_window7_224`, `timm_swinv2_tiny_window8_256` |
| PVTv2 | `timm_pvt_v2_b0` ~ `b4` |
| SegFormer MiT | `timm_mit_b0` ~ `b5` |
| MaxViT | `timm_maxvit_tiny/small_tf_224` |
| ViT (CLIP) | `timm_vit_clip_base/large/huge` |
| ViT (DINOv2) | `timm_vit_dinov2_base/large/giant` |
| ViT (DINOv3) | `timm_vit_dinov3_small/base/large/huge_plus/7b` |
| ViT (MAE) | `timm_vit_mae_base/large` |
| ViT (SAM) | `timm_vit_sam_base/large/huge` |
| MobileNet | `timm_mobilenetv2_100`, `timm_mobilenetv3_large/small_100` |
| Others | `timm_inception_v3`, `timm_ghostnet_100`, `timm_mobilevit_s`, `timm_poolformer_s12/s24`, `timm_fastvit_t8`, `timm_coatnet_0_224` |

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
