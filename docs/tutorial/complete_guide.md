# APRIL-MedSeg — Complete User Tutorial (From Zero to Hero)

> A hands-on guide covering installation, dataset preparation, training, evaluation, advanced paradigms, deployment, and custom extensions.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Installation](#2-installation)
3. [Project Structure Overview](#3-project-structure-overview)
4. [Preparing Your Dataset](#4-preparing-your-dataset)
5. [Understanding the YAML Config System](#5-understanding-the-yaml-config-system)
6. [Training Your First Model (Supervised)](#6-training-your-first-model-supervised)
7. [Monitoring Training Progress](#7-monitoring-training-progress)
8. [Evaluating and Testing Your Model](#8-evaluating-and-testing-your-model)
9. [Advanced Inference: Ensemble and TTA](#9-advanced-inference-ensemble-and-tta)
10. [Exploring Architectures: Modular Combination](#10-exploring-architectures-modular-combination)
11. [Using Foundation Models (Transfer Learning)](#11-using-foundation-models-transfer-learning)
12. [Semi-Supervised Training](#12-semi-supervised-training)
13. [Domain Adaptation Training](#13-domain-adaptation-training)
14. [Knowledge Distillation](#14-knowledge-distillation)
15. [Weakly Supervised Training](#15-weakly-supervised-training)
16. [Text-Guided Segmentation](#16-text-guided-segmentation)
17. [MLLM Inference Pipeline (Grounding DINO + SAM)](#17-mllm-inference-pipeline)
18. [ONNX Export and Deployment](#18-onnx-export-and-deployment)
19. [Model Profiling (FLOPs / Params / FPS)](#19-model-profiling)
20. [Custom Extensions](#20-custom-extensions)

---

## 1. Introduction

**APRIL-MedSeg** is a modular 2D medical image segmentation toolbox built on PyTorch. It provides:

- **130** complete network architectures (CNN, Transformer, Mamba, RWKV, KAN, SAM family, etc.)
- **177** encoders (including 39 foundation models across 9 medical modalities + dynamic timm wrapper)
- **45** decoders (cascade, attention, transformer, MLP, etc.)
- **81** loss functions (supervised, distillation, domain adaptation, weakly supervised)
- **25** skip connection types
- **6** training paradigms (supervised, semi-supervised, domain adaptation, distillation, weakly supervised, text-guided)
- **917** ready-to-use YAML configs
- **24** augmentation methods configurable via YAML

**Why use this framework?**

- **One-line architecture swap**: Change encoder, decoder, skip connection, or bottleneck in YAML — no code changes.
- **Reproducible**: Built-in seed control, deterministic mode, and config inheritance.
- **Production-ready**: AMP, DDP, ONNX export, TTA, ensemble, and model profiling out of the box.
- **Research-friendly**: 6 training paradigms, MLLM inference pipeline, and easy custom extensions.

---

## 2. Installation

### 2.1 Prerequisites

- Python >= 3.8
- PyTorch >= 2.0
- CUDA (recommended) / CPU / Apple Silicon (MPS)

### 2.2 Clone and Install

```bash
git clone https://github.com/juntaoJianggavin/APRIL-MedSeg.git
cd APRIL-MedSeg

# Install core dependencies
pip install -r requirements.txt

# Install in editable (dev) mode
pip install -e .
```

### 2.3 Optional Dependencies

Install only what you need:

```bash
# Foundation models (DINOv2, CLIP, SAM, etc.)
pip install timm transformers huggingface_hub safetensors

# Albumentations augmentation library
pip install albumentations

# Training visualization (TensorBoard / WandB)
pip install tensorboard wandb

# MLLM inference pipeline (GroundingDINO + SAM)
pip install groundingdino-py
pip install git+https://github.com/facebookresearch/segment-anything.git

# ONNX export and verification
pip install onnx onnxruntime

# Lion optimizer
pip install lion-pytorch
```

### 2.4 Verify Installation

```python
from medseg.model_builder import build_model
from medseg.registry import ENCODER_REGISTRY, DECODER_REGISTRY
import torch

# Quick smoke test
cfg = {
    "model": {
        "num_classes": 2,
        "img_size": 224,
        "encoder": {"name": "timm_resnet34", "pretrained": False, "in_channels": 3},
        "decoder": {"name": "unet"},
        "bottleneck": {"name": "none"},
    }
}
model = build_model(cfg)
x = torch.randn(1, 3, 224, 224)
out = model(x)
print(f"Output shape: {out.shape}")  # → (1, 2, 224, 224)
```

### 2.5 Automatic Weight Download

Some encoders (foundation models) require pretrained weights. The framework provides a utility:

```bash
# List all auto-downloadable weights
python -m medseg.utils.weight_downloader list

# Download specific weights (e.g., MedSAM ViT-B)
python -m medseg.utils.weight_downloader download medsam_vit_b

# Check cache status
python -m medseg.utils.weight_downloader check
```

> **Note**: `timm` encoder weights are downloaded automatically on first use — no manual management needed.

---

## 3. Project Structure Overview

```
APRIL-MedSeg/
├── medseg/                    # Core framework (pip installable)
│   ├── models/
│   │   ├── encoders/          # 177 encoders (CNN, Transformer, Mamba, RWKV, timm, foundation)
│   │   ├── decoders/          # 45 decoders
│   │   ├── skip_connections/  # 25 skip connection types
│   │   ├── bottlenecks/       # 17 bottlenecks
│   │   ├── networks/          # 130 complete pre-assembled architectures
│   │   └── text_unet/         # 12 text-guided models
│   ├── training/              # Training paradigm implementations
│   │   ├── semi/              # 20 semi-supervised methods
│   │   ├── domain_adaptation/ # 18 domain adaptation methods
│   │   ├── distillation/      # 27 distillation methods
│   │   └── weakly_supervised/ # 20 weakly supervised methods
│   ├── inference/             # Inference utilities
│   │   ├── ensemble.py        # Multi-model ensemble
│   │   ├── tta.py             # Test-Time Augmentation
│   │   └── mllm/              # MLLM grounding+segmentation pipeline
│   ├── losses/                # 81 loss functions
│   ├── datasets/              # Dataset classes and augmentations
│   ├── utils/                 # Config, AMP/DDP, logger, warmup, metrics, etc.
│   ├── model_builder.py       # YAML → model assembler
│   └── registry.py            # Component registries
├── configs/                   # 917 YAML configs
├── scripts/                   # Utility scripts (ONNX export, visualization, etc.)
├── train.py                   # Supervised training entry point
├── semi_train.py              # Semi-supervised training
├── train_domain_adaptation.py # Domain adaptation training
├── train_distillation.py      # Knowledge distillation training
├── train_weakly_supervised.py # Weakly supervised training
├── train_text_guided.py       # Text-guided training
├── test.py                    # Inference and evaluation
├── profile_model.py           # Model profiling (FLOPs/params/FPS)
└── data/                      # Your datasets go here
```

### Key Entry Points

| Script | Purpose |
|--------|---------|
| `train.py` | Standard supervised training (AMP + DDP + Logger) |
| `semi_train.py` | Semi-supervised training (20 methods) |
| `train_domain_adaptation.py` | Domain adaptation (18 methods) |
| `train_distillation.py` | Knowledge distillation (27 methods) |
| `train_weakly_supervised.py` | Weakly supervised (20 methods) |
| `train_text_guided.py` | Text-guided segmentation |
| `test.py` | Evaluation / inference (single, ensemble, TTA) |
| `profile_model.py` | FLOPs, params, and FPS profiling |

---

## 4. Preparing Your Dataset

### 4.1 Supported Dataset Types

| Type | Config Value | Description |
|------|-------------|-------------|
| Synapse | `synapse` | Multi-organ CT (TransUNet format: `.npz`/`.h5`) |
| ACDC | `acdc` | Cardiac MRI (TransUNet format) |
| Generic | `generic` / `binary` / `image_mask` | Images + masks directories (PNG/JPG/TIFF) |
| QaTa-COV19 | `qata_covid19` | Chest X-ray + per-image text (LViT format) |
| MosMedData+ | `mosmed_plus` | COVID CT + per-image text (LViT format) |

### 4.2 Generic Dataset (Most Common)

For most custom datasets, use the `generic` (or `binary`) type. Prepare your data as:

```
data/YourDataset/
├── images/
│   ├── 001.png
│   ├── 002.png
│   └── ...
├── masks/
│   ├── 001.png    # pixel values = class indices (0=background, 1=class1, ...)
│   ├── 002.png
│   └── ...
```

Or separate into explicit train/val/test directories:

```
data/YourDataset/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

> **Important**: Mask pixel values are used directly as class indices. For binary segmentation: 0 = background, 1 = foreground. For multi-class: 0, 1, 2, 3, etc.

### 4.3 Data Split Methods

The framework supports three ways to split data:

```yaml
# Method 1: Explicit train/val/test directories
data:
  type: generic
  train_dir: ./data/YourDataset/train
  val_dir: ./data/YourDataset/val
  test_dir: ./data/YourDataset/test

# Method 2: Automatic ratio-based split from a single directory
data:
  type: generic
  root_dir: ./data/YourDataset
  train_ratio: 0.7
  val_ratio: 0.15
  random_state: 42

# Method 3: N-fold cross-validation
data:
  type: generic
  root_dir: ./data/YourDataset
  n_splits: 5
  fold_idx: 0          # Change at runtime: --override data.fold_idx=1
  random_state: 42
```

### 4.4 Data Augmentation

Three augmentation modes are available:

**Mode 1: Basic** (default — built-in flips, rotations, scaling)
```yaml
training:
  augmentation: basic   # or omit for default
```

**Mode 2: Albumentations** (requires `pip install albumentations`)
```yaml
training:
  augmentation: albumentations
  aug_params:
    p_flip: 0.5
    p_rotate: 0.3
    p_color: 0.3
    p_elastic: 0.2
```

**Mode 3: YAML Pipeline** (recommended — 24 methods, fully configurable)
```yaml
training:
  augmentation: pipeline
  aug_pipeline:
    - name: horizontal_flip
      params: { p: 0.5 }
    - name: vertical_flip
      params: { p: 0.5 }
    - name: random_rotate90
      params: { p: 0.5 }
    - name: random_rotate
      params: { p: 0.3, degrees_range: [-30, 30] }
    - name: elastic_deform
      params: { p: 0.3, alpha_range: [20, 80], sigma_range: [3, 7] }
    - name: copy_paste
      params: { p: 0.3, max_objects: 2, scale_range: [0.5, 1.5] }
    - name: mosaic
      params: { p: 0.3, offset_range: [0.0, 0.2] }
    - name: clahe
      params: { p: 0.3, clip_limit_range: [1.0, 5.0] }
    - name: gaussian_noise
      params: { p: 0.2, std_range: [0.01, 0.08] }
```

**Available augmentation methods (24):**

| Category | Methods |
|----------|---------|
| Geometric | `horizontal_flip`, `vertical_flip`, `random_rotate90`, `random_rotate`, `random_affine`, `random_perspective`, `random_scale`, `elastic_deform`, `grid_mask` |
| Pixel-level | `photometric_distortion`, `color_jitter`, `brightness_contrast`, `gamma_correction`, `clahe`, `gaussian_blur`, `gaussian_noise`, `sharpness`, `posterize`, `random_solarize`, `channel_dropout` |
| Masking | `random_erasing`, `coarse_dropout`, `grid_mask` |
| Sample-level | `copy_paste`, `mosaic` |

---

## 5. Understanding the YAML Config System

Every experiment is driven by a single YAML file. The config has three main sections: `model`, `data`, and `training`.

### 5.1 Two Model Config Modes

**Mode 1: Modular combination** — Mix and match encoder + decoder + skip + bottleneck:

```yaml
model:
  num_classes: 2
  img_size: 256
  encoder:
    name: timm_resnet50       # Any registered encoder or timm_xxx
    pretrained: true
    in_channels: 3
  decoder:
    name: unet                # Any registered decoder
    params: {}
  skip_connection:
    name: concat              # Any registered skip connection
  bottleneck:
    name: aspp                # Any registered bottleneck
```

**Mode 2: Complete architecture** — Use a pre-assembled network:

```yaml
model:
  num_classes: 9
  img_size: 224
  architecture: transunet     # Any registered architecture name
  arch_params: {}
```

### 5.2 Config Inheritance

Avoid duplication with the `_base_` field:

```yaml
# base_config.yaml — shared settings
model:
  encoder:
    name: timm_resnet50
    pretrained: true
  decoder:
    name: unet
data:
  type: generic
  img_size: 256

# child_config.yaml — only overrides
_base_: ../base_config.yaml
model:
  num_classes: 4              # Override
training:
  epochs: 300                 # Override
```

The framework auto-merges `_base_` configs via `medseg.utils.config.load_config()`.

### 5.3 Complete Config Example

```yaml
model:
  num_classes: 2
  img_size: 256
  encoder:
    name: timm_resnet50
    pretrained: true
    in_channels: 3
  decoder:
    name: unet
    params: {}
  bottleneck:
    name: none

data:
  type: generic
  img_size: 256
  root_dir: ./data/SkinLesion
  train_ratio: 0.7
  val_ratio: 0.15
  random_state: 42

training:
  random_state: 42
  deterministic: true
  amp: true                        # Mixed precision
  parallel: auto                   # Auto-select DDP/DP/single
  logger: tensorboard              # tensorboard / wandb / both / none
  augmentation: pipeline
  aug_pipeline:
    - name: horizontal_flip
      params: { p: 0.5 }
    - name: random_rotate90
      params: { p: 0.3 }
  epochs: 200
  batch_size: 16
  num_workers: 4
  val_interval: 10
  save_interval: 50
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
    name: warmup_cosine
    warmup_epochs: 10
    warmup_lr: 0.000001
    min_lr: 0.000001
```

### 5.4 CLI Overrides

Override any config value at runtime without editing the file:

```bash
python train.py --config configs/my_config.yaml \
    --override training.epochs=300 training.batch_size=32 model.num_classes=4
```

---

## 6. Training Your First Model (Supervised)

### 6.1 Basic Training

```bash
# Train with a pre-made config
python train.py --config configs/architectures/networks/general/transunet.yaml \
    --output_dir output/transunet

# Train a modular combination
python train.py --config configs/default.yaml \
    --output_dir output/resnet50_unet
```

### 6.2 With Mixed Precision (AMP)

AMP reduces memory usage by ~40% and speeds up training by ~30% on modern GPUs:

```bash
python train.py --config configs/default.yaml \
    --output_dir output/resnet50_unet --amp
```

Or set in YAML: `training.amp: true`

### 6.3 Multi-GPU DDP Training

```bash
# 4 GPUs
torchrun --nproc_per_node=4 train.py \
    --config configs/default.yaml \
    --output_dir output/resnet50_unet --amp
```

The framework auto-detects DDP and wraps the model with `DistributedDataParallel`.

### 6.4 DataParallel (Single Process, Multi-GPU)

```yaml
training:
  parallel: dp    # Force DataParallel (single process, multiple GPUs)
```

### 6.5 Training with Warmup Scheduler

Warmup gradually increases LR from a small value to the target LR over the first N epochs, stabilizing early training:

```yaml
training:
  scheduler:
    name: warmup_cosine
    warmup_epochs: 10
    warmup_lr: 0.000001    # Start LR
    min_lr: 0.000001       # Final LR after cosine decay
```

### 6.6 Reproducibility

```yaml
training:
  random_state: 42        # Global random seed
  deterministic: true     # cuDNN deterministic mode (5-10% slower but identical results)
```

### 6.7 Resuming Training

```bash
python train.py --config configs/default.yaml \
    --output_dir output/resnet50_unet \
    --resume output/resnet50_unet/checkpoint_epoch100.pth
```

### 6.8 What Gets Saved

- `best_model.pth` — Saved whenever validation Dice improves
- `checkpoint_epoch{N}.pth` — Saved every `save_interval` epochs

Each checkpoint contains:
```python
{
    'epoch': int,
    'model_state_dict': state_dict,
    'optimizer_state_dict': optimizer_state,
    'best_dice': float,
}
```

### 6.9 5-Fold Cross-Validation Example

```bash
for i in 0 1 2 3 4; do
    python train.py --config configs/architectures/networks/general/resnet50_unet_5fold.yaml \
        --override data.fold_idx=$i \
        --output_dir output/5fold/fold_$i
done
```

Then report the mean ± std of the 5 best validation Dice scores.

---

## 7. Monitoring Training Progress

### 7.1 TensorBoard

```yaml
training:
  logger: tensorboard
```

```bash
# Launch TensorBoard
tensorboard --logdir output/transunet/tb_logs
```

Logged metrics: `train/loss`, `lr/group0`, and validation Dice.

### 7.2 Weights & Biases (WandB)

```yaml
training:
  logger: wandb
  wandb_project: my_medseg_experiment
  wandb_entity: null          # Optional: your WandB team/entity
```

```bash
# WandB auto-initializes during training
python train.py --config configs/default.yaml --output_dir output/exp1
```

### 7.3 Both TensorBoard and WandB

```yaml
training:
  logger: both
  wandb_project: my_experiment
```

### 7.4 Console Logging

All training scripts print epoch-level progress to stdout:

```
Epoch [50/200] Loss: 0.3421 LR: 0.000085 Time: 12.3s Val_Dice: 0.8234
```

---

## 8. Evaluating and Testing Your Model

### 8.1 Single Model Evaluation

```bash
python test.py --config configs/architectures/networks/general/transunet.yaml \
    --checkpoint output/transunet/best_model.pth
```

This prints per-class Dice, IoU, and HD95 metrics:

```
DICE:
  Class 1: 0.8923
  Class 2: 0.8156
  ...
  Mean: 0.8540

IOU:
  Class 1: 0.8067
  ...
```

### 8.2 Saving Prediction Results

```bash
python test.py --config configs/architectures/networks/general/transunet.yaml \
    --checkpoint output/transunet/best_model.pth \
    --save_pred --output_dir test_output/
```

Predictions are saved as `.npy` files in `test_output/predictions/`.

### 8.3 Visualizing Predictions

```bash
python scripts/visualize.py \
    --config configs/architectures/networks/general/transunet.yaml \
    --checkpoint output/transunet/best_model.pth \
    --input ./data/test/images/ \
    --output vis_output/
```

For each input image, three files are generated:
- `xxx_input.png` — Original image
- `xxx_pred.png` — Colorized prediction mask
- `xxx_overlay.png` — Image + translucent mask overlay

### 8.4 Evaluation Metrics

The framework computes three standard metrics:

| Metric | Description | Range |
|--------|-------------|-------|
| **Dice** | F1 score between prediction and ground truth | 0.0 – 1.0 (higher is better) |
| **IoU** | Jaccard index (intersection over union) | 0.0 – 1.0 (higher is better) |
| **HD95** | 95th percentile Hausdorff distance | 0.0 – ∞ (lower is better, requires `medpy`) |

---

## 9. Advanced Inference: Ensemble and TTA

### 9.1 Multi-Checkpoint Ensemble

Average predictions from multiple checkpoints to improve robustness:

```bash
# Equal-weight logit averaging
python test.py --config configs/default.yaml \
    --checkpoint ckpt_fold0.pth ckpt_fold1.pth ckpt_fold2.pth \
    --ensemble-average logit

# Weighted logit averaging
python test.py --config configs/default.yaml \
    --checkpoint ckpt_a.pth ckpt_b.pth ckpt_c.pth \
    --ensemble-weights 0.5 0.3 0.2 \
    --ensemble-average logit
```

**Three averaging modes:**

| Mode | Description |
|------|-------------|
| `logit` | Average raw logits before argmax (recommended) |
| `softmax` | Average softmax probabilities |
| `sigmoid` | Average sigmoid probabilities |

### 9.2 Test-Time Augmentation (TTA)

Apply augmentations at test time, predict on each augmented version, then merge:

```bash
python test.py --config configs/default.yaml \
    --checkpoint output/best_model.pth \
    --tta \
    --tta-augs identity rot90 rot180 rot270 hflip vflip \
    --tta-merge mean
```

**Available TTA augmentations:**
`identity`, `rot90`, `rot180`, `rot270`, `hflip`, `vflip`, `brightness_up`, `brightness_down`, `contrast_up`, `contrast_down`, `gamma_up`, `gamma_down`

**Merge strategies:**

| Strategy | Description |
|----------|-------------|
| `mean` | Arithmetic mean of predictions |
| `gmean` | Geometric mean |
| `max` | Element-wise maximum |
| `median` | Element-wise median |

### 9.3 Combining Ensemble + TTA

```bash
python test.py --config configs/default.yaml \
    --checkpoint ckpt_a.pth ckpt_b.pth \
    --ensemble-average logit \
    --tta --tta-merge mean
```

The ensemble runs first, then TTA wraps the ensemble output.

---

## 10. Exploring Architectures: Modular Combination

The framework's key innovation: freely combine any encoder + decoder + skip connection + bottleneck.

### 10.1 Using a Pre-assembled Architecture

```yaml
model:
  num_classes: 2
  img_size: 224
  architecture: transunet    # or swinunet, medsam, vmunet, etc.
```

130 complete architectures are available. Some popular ones:

| Category | Examples |
|----------|----------|
| CNN | UNet++, UNet3+, Attention-UNet, nnU-Net, MedNeXt |
| Transformer | TransUNet, Swin-UNet, DAEFormer, MISSFormer, HiFormer |
| Mamba/SSM | VM-UNet, U-Mamba, Swin-UMamba, LKM-UNet |
| SAM family | MedSAM, SAM-Med2D, SAM2, SAMUS |
| KAN/MLP | U-KAN, Rolling-UNet, UNeXt |

### 10.2 Mixing Encoder + Decoder

```yaml
model:
  num_classes: 2
  img_size: 256
  encoder:
    name: timm_convnext_tiny
    pretrained: true
  decoder:
    name: cascade             # CASCADE decoder
  skip_connection:
    name: cab                 # Channel Attention Block
  bottleneck:
    name: aspp
```

### 10.3 Using Any timm Model as Encoder

Prefix any `timm` model name with `timm_`:

```yaml
encoder:
  name: timm_efficientnet_b7
  pretrained: true
```

```yaml
encoder:
  name: timm_swin_base_patch4_window12_384
  pretrained: true
```

Over 1000 timm models are available. Check: `python -c "import timm; print(len(timm.list_models()))"`

### 10.4 Exploring Decoders

| Category | Decoders |
|----------|----------|
| Basic | `unet`, `bilinear`, `deconv`, `dw_sep` |
| Dense | `unetpp`, `unet3p` |
| Cascade | `cascade`, `emcad`, `g_cascade`, `cfm`, `merit` |
| Attention | `attention`, `ham`, `lawin` |
| Transformer | `daeformer`, `mtunet`, `swinunet`, `nnformer`, `uctransnet` |
| MLP | `segformer_mlp`, `mlp_decoder` |
| Pyramid | `upernet` |

### 10.5 Exploring Skip Connections

| Category | Options |
|----------|---------|
| Basic | `concat`, `dense` |
| Attention | `ag`, `cab`, `sab`, `scse`, `cbam`, `gating`, `gru`, `gab` |
| Transformer | `cross_attn`, `trans_fusion`, `agg_attn`, `missformer`, `uctrans` |
| Fusion | `bi_fusion`, `deformable`, `multi_scale`, `feature_refine`, `ccm`, `sdi` |

### 10.6 Exploring Bottlenecks

`none`, `basic`, `aspp`, `dense_aspp`, `ppm`, `transformer`, `mamba`, `rwkv`, `se`, `dual_attention`, `cbam`, `acmix`

### 10.7 Ablation Study Configs

The project includes pre-built ablation configs:

```
configs/architectures/
├── decoder_study/      # 3 encoders × 44 decoders + 1 augmented = 133 configs
├── skip_study/         # 3 encoders × 25 skips = 75 configs
├── bottleneck_study/   # 3 encoders × 17 bottlenecks = 51 configs
└── combinations/       # Free encoder+decoder combos = 169 configs
```

Run experiment scripts for batch ablation:

```bash
bash scripts/experiments/run_decoder_study.sh
bash scripts/experiments/run_skip_study.sh
bash scripts/experiments/run_bottleneck_study.sh
```

---

## 11. Using Foundation Models (Transfer Learning)

The framework includes 39 foundation model encoders covering 9 medical modalities.

### 11.1 Available Foundation Models

| Modality | Models |
|----------|--------|
| General | DINOv2, DINOv3, DINO, CLIP-ViT, SAM-ViT |
| Pathology | Phikon, Phikon-v2, UNI, PLIP, MUSK, KEEP |
| Radiology | Rad-DINO, OmniRad, BioViL, CheXZero |
| Ophthalmology | RETFound-DINOv2, RETFound, FLAIR, OphMAE |
| Dermatology | DermCLIP, MoNet, PanDerm |
| General Medical | BiomedCLIP, MedCLIP, MedSigLIP |
| MLLM Vision | Qwen2.5-VL, Qwen3-VL, MedGemma, LLaVA-Med, HuatuoGPT, HealthGPT, HuLuMed, LingShu |
| Ultrasound | UltraFedFM, US-FMAE, SAMUS |
| Endoscopy | Endo-ViT, Endo-FM, Surgical-SAM |

### 11.2 Using a Foundation Model

```yaml
model:
  num_classes: 4
  img_size: 224
  encoder:
    name: dinov2_vit_b         # Foundation model encoder
    pretrained: true
    freeze_cfg:
      freeze: true              # Freeze encoder weights
      unfreeze_last_n: 2        # Unfreeze last 2 transformer blocks
  decoder:
    name: unet
  bottleneck:
    name: none
```

### 11.3 Fine-Tuning Strategies

| Strategy | Config | Description |
|----------|--------|-------------|
| Full fine-tuning | `freeze: false` | Train all encoder parameters |
| Frozen encoder | `freeze: true, unfreeze_last_n: 0` | Only train decoder |
| Partial fine-tuning | `freeze: true, unfreeze_last_n: 2` | Unfreeze last N blocks |
| Inference only | `inference_only: true` | Encoder in eval mode, no grad |

### 11.4 Foundation Model Configs

Pre-built configs for 54 foundation model combinations:

```
configs/architectures/foundation/
├── general/          # DINOv2, CLIP, etc.
├── pathology/        # Phikon, UNI, PLIP, MUSK
├── radiology/        # Rad-DINO, OmniRad
├── ophthalmology/    # RETFound, FLAIR
├── dermatology/      # DermCLIP, PanDerm
├── multimodal_med/   # BiomedCLIP, MedCLIP
├── mllm_vision/      # Qwen-VL, MedGemma
├── endoscopy/        # EndoViT
└── ultrasound/       # UltraFedFM
```

---

## 12. Semi-Supervised Training

When you have abundant unlabeled data but limited annotations, semi-supervised training leverages both.

### 12.1 Available Methods (20)

Mean Teacher, CPS, CCT, UniMatch, FixMatch, FlexMatch, FreeMatch, SoftMatch, UA-MT, URPC, Deep Co-Training, Pi-Model, Temporal Ensembling, Pseudo-Label, ICT, R-Drop, Cross-Teaching, CorrMatch, AllSpark, DiffRect

### 12.2 Dataset Setup

```
data/
├── labeled/           # Images with pixel-level masks
│   ├── images/
│   └── masks/
├── unlabeled/         # Images without masks
│   └── images/
├── val/               # Validation set (with masks)
│   ├── images/
│   └── masks/
└── test/              # Test set (with masks)
    ├── images/
    └── masks/
```

### 12.3 Config Example (Mean Teacher)

```yaml
model:
  num_classes: 4
  img_size: 224
  encoder:
    name: timm_resnet34
    pretrained: true
  decoder:
    name: bilinear
  bottleneck:
    name: none

data:
  img_size: 224
  labeled_dir: ./data/labeled
  unlabeled_dir: ./data/unlabeled
  val_dir: ./data/val
  test_dir: ./data/test

semi:
  method: mean_teacher
  params:
    ema_decay: 0.999            # EMA decay for teacher model
    consistency_weight: 1.0     # Weight on consistency loss
    rampup_epochs: 40           # Epochs to ramp up consistency weight

training:
  epochs: 200
  labeled_batch_size: 8
  unlabeled_batch_size: 8
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
```

### 12.4 Running

```bash
# Mean Teacher
python semi_train.py --config configs/training_paradigms/semi_supervision/mean_teacher.yaml \
    --output_dir output/semi_mt

# CPS (Cross Pseudo Supervision)
python semi_train.py --config configs/training_paradigms/semi_supervision/cps.yaml \
    --output_dir output/semi_cps

# UniMatch
python semi_train.py --config configs/training_paradigms/semi_supervision/unimatch.yaml \
    --output_dir output/semi_unimatch
```

---

## 13. Domain Adaptation Training

When training and deployment data come from different distributions (e.g., different scanners, hospitals).

### 13.1 Available Methods (18)

Source Only, AdvEnt, DANN, TENT, DPL, CBMT, FDA, CRST, PixMatch, MIC, DAFormer, HRDA, PiPa, DDB, SePiCo, DiGA, MICDrop, SemiVL

### 13.2 Dataset Setup

```
data/
├── source/             # Source domain (labeled)
│   ├── images/
│   └── masks/
├── target/             # Target domain (unlabeled)
│   └── images/
└── target_val/         # Target validation (labeled)
    ├── images/
    └── masks/
```

### 13.3 Source+Target Adaptation (e.g., AdvEnt)

```yaml
model:
  num_classes: 4
  img_size: 224
  encoder:
    name: timm_resnet34
    pretrained: true
  decoder:
    name: bilinear
  bottleneck:
    name: none

data:
  source:
    image_dir: ./data/source/images
    mask_dir: ./data/source/masks
  target:
    root: ./data/target/images
  val:
    image_dir: ./data/target_val/images
    mask_dir: ./data/target_val/masks

domain_adaptation:
  method: advent
  params: {}

training:
  epochs: 100
  batch_size: 8
  optimizer:
    name: adamw
    lr: 0.0001
```

```bash
python train_domain_adaptation.py \
    --config configs/training_paradigms/domain_adaptation/advent.yaml \
    --output_dir output/da_advent
```

### 13.4 Source-Free Adaptation (e.g., TENT)

Source-free methods adapt using only target data (with a pretrained source model):

```yaml
data:
  target:
    root: ./data/target/images
  val:
    image_dir: ./data/target_val/images
    mask_dir: ./data/target_val/masks
  pretrained_model: ./output/source_trained/best_model.pth   # Required!

domain_adaptation:
  method: tent
  params: {}
```

```bash
python train_domain_adaptation.py \
    --config configs/training_paradigms/domain_adaptation/tent.yaml \
    --output_dir output/da_tent
```

> **Important**: Source-free methods require `data.pretrained_model` — they refuse to adapt a randomly initialized model.

---

## 14. Knowledge Distillation

Transfer knowledge from a large teacher model to a smaller student model.

### 14.1 Available Methods (27)

Vanilla KD, FitNets, AT, FSP, NST, RKD, VID, DKD, MGD, DIST, CIRKD, CWD, ReviewKD, SimKD, NORM, SDD, AICSD, LSKD, TTM, CTKD, MLKD + 4 medical-specific methods

### 14.2 Workflow

1. **Train the teacher** first:
```bash
python train.py --config configs/training_paradigms/distillation/teacher_large.yaml \
    --output_dir output/teacher
```

2. **Distill to student**:
```bash
python train_distillation.py \
    --teacher_config configs/training_paradigms/distillation/teacher_large.yaml \
    --student_config configs/training_paradigms/distillation/student_small.yaml \
    --teacher_ckpt output/teacher/best_model.pth \
    --distillation_type logit \
    --temperature 4.0 \
    --alpha 0.5 \
    --output_dir output/kd_logit
```

### 14.3 Distillation Types

| Type | CLI Flag | Description |
|------|----------|-------------|
| Logit | `--distillation_type logit` | Match softened output distributions |
| Feature | `--distillation_type feature` | Match intermediate feature maps |
| Attention | `--distillation_type attention` | Match attention maps |
| Multi-scale | `--distillation_type multi_scale` | Match features at multiple scales |
| Hint | `--distillation_type hint` | FitNets-style hint learning |

### 14.4 Key Parameters

- `--temperature` (default 4.0): Higher = softer distributions = more knowledge transfer
- `--alpha` (default 0.5): Weight of KD loss vs. supervised loss

---

## 15. Weakly Supervised Training

When only weak annotations are available (bounding boxes, image-level labels, scribbles).

### 15.1 Available Methods (20)

Box, CAM, Point, Scribble, MIL, TreeEnergy, SEAM, PuzzleCAM, AdvCAM, MCTformer, EPS, BoxInst, ReCAM, ToCo, LPCAM, MARS, DuPL, MoRe, PSDPM, SemPLeS

### 15.2 Box-Supervised Training

```bash
python train_weakly_supervised.py \
    --config configs/training_paradigms/weak_supervision/box_supervised.yaml \
    --supervision_type box \
    --output_dir output/weak_box
```

### 15.3 CAM-Based Training

```bash
python train_weakly_supervised.py \
    --config configs/training_paradigms/weak_supervision/cam.yaml \
    --supervision_type cam \
    --output_dir output/weak_cam
```

### 15.4 Supervision Types

| Type | `--supervision_type` | Annotation Required |
|------|---------------------|---------------------|
| Bounding box | `box` | Bounding boxes per object |
| CAM | `cam` | Image-level class labels |
| MIL | `mil` | Image-level class labels |
| EM pseudo-label | `em` | Weak annotations + EM refinement |
| Image-level | `image_label` | Multi-label per image |

---

## 16. Text-Guided Segmentation

Use natural language descriptions to guide segmentation.

### 16.1 Available Models (12)

CRIS, BiomedParse, LanGuideMedSeg, LViT, TGANet, TPRO, CausalCLIPSeg, CLIP-Universal, CXR-CLIP-Seg, TP-DRSeg, MedCLIP-SAM, SaLIP

### 16.2 Training

```bash
python train_text_guided.py \
    --config configs/training_paradigms/text_guided/synapse_clip.yaml \
    --output_dir output/text_cris
```

### 16.3 Config Example

```yaml
model:
  num_classes: 9
  img_size: 224
  text_guided:
    model_type: TextPromptUNet
    class_names: [background, spleen, kidney_R, kidney_L, gallbladder,
                  liver, stomach, aorta, pancreas]
    prompt_mode: learnable
    embed_dim: 512
    encoder_channels: [64, 128, 256, 512]

data:
  type: synapse
  img_size: 224
  train_dir: ./data/Synapse/train_npz
  val_dir: ./data/Synapse/test_vol_h5

training:
  epochs: 200
  batch_size: 8
  optimizer:
    name: adamw
    lr: 0.0001
```

### 16.4 Freezing the Encoder

```bash
python train_text_guided.py \
    --config configs/training_paradigms/text_guided/synapse_clip.yaml \
    --freeze_encoder \
    --output_dir output/text_cris_frozen
```

---

## 17. MLLM Inference Pipeline

The framework provides a **Detect-then-Segment** pipeline using Multimodal Large Language Models.

### 17.1 Architecture

```
Text Prompt → MLLM Detector → Bounding Box → SAM Segmenter → Mask
```

### 17.2 Available Components

**Detectors (5):**
- GroundingDINO (specialized open-vocabulary detector)
- Qwen2-VL / Qwen2.5-VL / Qwen3-VL (native grounding tokens)
- InternVL

**Segmenters (4):**
- SAM2 (Meta AI's Segment Anything 2)
- MedSAM (medical-domain SAM)
- SAM-Med2D (medical SAM variant)
- LiteMedSAM (lightweight medical SAM)

**5 detectors × 4 segmenters = 20 possible combinations**

### 17.3 Python API Usage

```python
from medseg.inference.mllm import (
    MLLMGroundingSegPipeline,
    MLLM_REGISTRY,
    MASK_GENERATOR_REGISTRY,
)

# Build detector
detector = MLLM_REGISTRY["grounding_dino"](
    model_config="GroundingDINO_SwinT_OGC",
    box_threshold=0.3,
    text_threshold=0.25,
)

# Build segmenter
segmenter = MASK_GENERATOR_REGISTRY["sam2"](
    checkpoint="sam2_hiera_large.pt",
)

# Build pipeline
pipeline = MLLMGroundingSegPipeline(
    grounder=detector,
    mask_generator=segmenter,
)

# Run inference
result = pipeline.predict(
    image="path/to/image.png",
    text_prompt="liver tumor",
)
# result.mask is the predicted segmentation mask
```

### 17.4 Running the Smoke Test

```bash
python scripts/smoke_mllm_inference.py
```

---

## 18. ONNX Export and Deployment

### 18.1 Basic Export

```bash
python scripts/export_onnx.py \
    --config configs/architectures/networks/general/transunet.yaml \
    --checkpoint output/transunet/best_model.pth \
    --output transunet.onnx
```

### 18.2 With Verification

```bash
python scripts/export_onnx.py \
    --config configs/architectures/networks/general/transunet.yaml \
    --checkpoint output/transunet/best_model.pth \
    --output transunet.onnx \
    --verify
```

This exports the model and verifies it with ONNX Runtime, comparing outputs with PyTorch.

### 18.3 Dynamic Input Shapes

```bash
python scripts/export_onnx.py \
    --config configs/architectures/networks/general/transunet.yaml \
    --checkpoint output/transunet/best_model.pth \
    --output transunet.onnx \
    --dynamic
```

### 18.4 Custom Image Size and Opset

```bash
python scripts/export_onnx.py \
    --config configs/architectures/networks/general/transunet.yaml \
    --checkpoint output/transunet/best_model.pth \
    --output transunet.onnx \
    --img_size 512 \
    --opset 17
```

### 18.5 Deploying with ONNX Runtime

```python
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("transunet.onnx")
input_name = sess.get_inputs()[0].name

# Prepare input (1, 3, 224, 224) normalized float32
image = np.random.randn(1, 3, 224, 224).astype(np.float32)

# Run inference
output = sess.run(None, {input_name: image})[0]

# Post-process
pred = output.argmax(axis=1)  # (1, 224, 224) class indices
```

---

## 19. Model Profiling

### 19.1 FLOPs and Parameters

```bash
python profile_model.py --config configs/architectures/networks/general/transunet.yaml
```

Output:
```
======================================================================
  Model: transunet.yaml
======================================================================
  FLOPs:            11.32 GFLOPs
  Active Params:    27.15 M params
======================================================================
```

### 19.2 FPS Benchmark

```bash
python profile_model.py \
    --config configs/architectures/networks/general/transunet.yaml \
    --fps
```

### 19.3 Custom FPS Settings

```bash
python profile_model.py \
    --config configs/architectures/networks/general/transunet.yaml \
    --fps --warmup 50 --runs 200 --batch_size 4 --device cuda
```

### 19.4 Per-Module Breakdown

```bash
python profile_model.py \
    --config configs/architectures/networks/general/transunet.yaml \
    --detail
```

### 19.5 Batch Profile All Configs in a Directory

```bash
python profile_model.py --config_dir configs/architectures/networks/general/
python profile_model.py --config_dir configs/architectures/networks/general/ --fps
```

---

## 20. Custom Extensions

### 20.1 Adding a New Encoder

Create `medseg/models/encoders/cnn/my_encoder.py`:

```python
import torch
import torch.nn as nn
from medseg.registry import ENCODER_REGISTRY

@ENCODER_REGISTRY.register("my_encoder")
class MyEncoder(nn.Module):
    def __init__(self, pretrained=False, in_channels=3, img_size=224, **kwargs):
        super().__init__()
        # Define your encoder layers
        self.layer1 = nn.Conv2d(in_channels, 64, 3, stride=2, padding=1)
        self.layer2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.layer3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.layer4 = nn.Conv2d(256, 512, 3, stride=2, padding=1)

        # MUST define out_channels: list of channel counts per stage
        self.out_channels = [64, 128, 256, 512]

    def forward(self, x):
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return [f1, f2, f3, f4]  # Multi-scale features (shallow → deep)
```

Then import it in `medseg/models/encoders/cnn/__init__.py`:

```python
from . import my_encoder
```

Use in YAML:

```yaml
model:
  encoder:
    name: my_encoder
```

### 20.2 Adding a New Decoder

```python
from medseg.registry import DECODER_REGISTRY

@DECODER_REGISTRY.register("my_decoder")
class MyDecoder(nn.Module):
    has_internal_skip = False  # Set True if decoder handles skip internally

    def __init__(self, encoder_channels, bottleneck_channels,
                 skip_connection=None, img_size=224, **kwargs):
        super().__init__()
        # Build decoder layers...
        self.out_channels = encoder_channels[0]  # Output channels

    def forward(self, bottleneck_feat, skip_features):
        # Decode using bottleneck output and skip features
        return decoded_feature
```

### 20.3 Adding a New Loss Function

```python
from medseg.registry import LOSS_REGISTRY

@LOSS_REGISTRY.register("my_loss")
class MyLoss(nn.Module):
    def __init__(self, weight=1.0, **kwargs):
        super().__init__()
        self.weight = weight

    def forward(self, pred, target):
        # Compute loss
        return loss_value * self.weight
```

Use in YAML:

```yaml
training:
  loss:
    name: my_loss
    params:
      weight: 2.0
```

### 20.4 Adding a New Augmentation

Create in `medseg/datasets/advanced_aug.py`:

```python
from medseg.registry import AUGMENTATION_REGISTRY

@AUGMENTATION_REGISTRY.register("my_augmentation")
class MyAugmentation:
    def __init__(self, p=0.5, **kwargs):
        self.p = p

    def set_dataset(self, dataset):
        """Optional: access dataset for sample-level augmentations."""
        self.dataset = dataset

    def __call__(self, sample: dict) -> dict:
        import random
        if random.random() > self.p:
            return sample
        image, label = sample['image'], sample['label']
        # ... implement augmentation logic ...
        return {'image': image, 'label': label}
```

Use in YAML:

```yaml
training:
  augmentation: pipeline
  aug_pipeline:
    - name: my_augmentation
      params: { p: 0.5, custom_param: 42 }
```

### 20.5 Adding a New Skip Connection

```python
from medseg.registry import SKIP_REGISTRY

@SKIP_REGISTRY.register("my_skip")
class MySkip(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, encoder_feat, decoder_feat):
        # Fuse encoder and decoder features at the same resolution
        return fused_feature
```

### 20.6 Adding a New Bottleneck

```python
from medseg.registry import BOTTLENECK_REGISTRY

@BOTTLENECK_REGISTRY.register("my_bottleneck")
class MyBottleneck(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.out_channels = in_channels  # Or different if channels change

    def forward(self, x):
        # Process the deepest encoder feature
        return processed_feature
```

---

## Quick Reference: All Training Scripts

| Script | Paradigm | Key Arguments |
|--------|----------|---------------|
| `train.py` | Supervised | `--config`, `--output_dir`, `--amp`, `--override` |
| `semi_train.py` | Semi-supervised | `--config`, `--output_dir`, `--resume` |
| `train_domain_adaptation.py` | Domain adaptation | `--config`, `--output_dir` |
| `train_distillation.py` | Distillation | `--teacher_config`, `--student_config`, `--teacher_ckpt`, `--temperature`, `--alpha` |
| `train_weakly_supervised.py` | Weakly supervised | `--config`, `--supervision_type`, `--output_dir` |
| `train_text_guided.py` | Text-guided | `--config`, `--output_dir`, `--freeze_encoder` |
| `test.py` | Evaluation | `--config`, `--checkpoint`, `--tta`, `--ensemble-average` |
| `profile_model.py` | Profiling | `--config`, `--fps`, `--detail`, `--config_dir` |

---

## Quick Reference: All YAML Config Fields

```yaml
# ===== MODEL =====
model:
  num_classes: 2                    # Number of output classes
  img_size: 224                     # Input image size
  architecture: transunet           # OR modular: encoder+decoder+skip+bottleneck
  encoder:
    name: timm_resnet50             # Encoder name (timm_xxx or registered name)
    pretrained: true                # Use pretrained weights
    in_channels: 3                  # Input channels (3=RGB, 1=grayscale)
    freeze_cfg:                     # Foundation model freezing
      freeze: true
      unfreeze_last_n: 2
  decoder:
    name: unet
    params: {}
  skip_connection:
    name: concat
  bottleneck:
    name: none

# ===== DATA =====
data:
  type: generic                     # synapse / acdc / generic / binary / qata_covid19 / mosmed_plus
  img_size: 224
  # Option A: Explicit directories
  train_dir: ./data/train
  val_dir: ./data/val
  test_dir: ./data/test
  # Option B: Auto-split
  root_dir: ./data/all
  train_ratio: 0.7
  val_ratio: 0.15
  # Option C: N-fold
  n_splits: 5
  fold_idx: 0

# ===== TRAINING =====
training:
  random_state: 42
  deterministic: true
  amp: true                         # Mixed precision
  parallel: auto                    # auto / ddp / dp / none
  logger: tensorboard               # tensorboard / wandb / both / none
  augmentation: pipeline            # basic / albumentations / pipeline / none
  epochs: 200
  batch_size: 16
  num_workers: 4
  val_interval: 10
  save_interval: 50
  loss:
    name: compound
    params:
      losses:
        - name: ce
          weight: 0.4
        - name: dice
          weight: 0.6
  optimizer:
    name: adamw                     # adamw / adam / sgd / lion
    lr: 0.0001
    weight_decay: 0.0001
  scheduler:
    name: warmup_cosine             # cosine / step / poly / warmup_cosine / warmup_poly
    warmup_epochs: 10
    warmup_lr: 0.000001
    min_lr: 0.000001
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `KeyError: 'xxx' not found in encoders` | Check encoder name spelling. For timm models, use `timm_` prefix. |
| Out of memory | Reduce `batch_size`, enable `amp: true`, or use a smaller encoder. |
| DDP hangs | Ensure `torchrun` is used (not `python`). Check firewall/security settings. |
| HD95 returns NaN | Install `medpy`: `pip install medpy` |
| ONNX export fails | Try different `--opset` version (13-17). Some ops need newer opset. |
| Foundation model download fails | Check internet. Use `python -m medseg.utils.weight_downloader download <name>` manually. |
| `Decoder requires encoder` error | Some decoders only work with specific encoders. Use compatible combinations. |
| Validation Dice = 0 | Check that mask pixel values match class indices (0, 1, 2, ...). |

---

*This tutorial covers the complete APRIL-MedSeg workflow from installation to deployment. For detailed documentation on specific components, see the [docs/](../docs/) directory and the [tutorial series](README.md).*
