# Research Guide

[中文文档](research_guide_CN.md)

This document provides systematic research suggestions for 5 directions, including recommended baselines, datasets, comparisons, and experiment scripts.

---

## 1. General SOTA Architecture Benchmark

### Goal
Fair comparison of architectures across multiple medical segmentation benchmarks.

### Datasets
| Dataset | Modality | Classes | Description |
|---------|----------|---------|-------------|
| Synapse | Abdominal CT | 9 | Multi-organ, TransUNet standard benchmark |
| ACDC | Cardiac MRI | 4 | Cardiac structure segmentation |
| BUSI | Breast ultrasound | 2 | Tumor segmentation, 5-fold CV |
| CVC-ClinicDB | Colonoscopy | 2 | Polyp segmentation |
| GlaS | Pathology H&E | 2 | Gland segmentation |
| Kvasir-SEG | Gastroenteroscopy | 2 | Polyp segmentation |
| ISIC 2018 | Dermoscopy | 2 | Skin lesion segmentation |

### Recommended Baselines

**Must-run (covers all architecture types)**:

| Architecture | Type | Paper | Config key |
|--------------|------|-------|------------|
| TransUNet | Transformer | Chen et al., 2021 | `transunet` |
| Swin-UNet | Transformer | Cao et al., 2022 | `swinunet` |
| VM-UNet | Mamba | Chen et al., 2024 | `vm_unet` |
| RWKV-UNet | RWKV | — | `rwkv_unet` |
| RIR-Zigzag | RWKV | TMI 2025 | `rir_zigzag` |
| Rolling-UNet | MLP | AAAI 2024 | `rolling_unet` |
| U-KAN | KAN | AAAI 2025 | `ukan` |
| Mobile-U-ViT | Lightweight Transformer | — | `mobile_u_vit` |
| UNet (basic) | CNN baseline | Ronneberger 2015 | encoder `basic` + decoder `unet` |
| UNet++ | CNN Dense | Zhou 2018 | `unetpp` |
| Attention UNet | CNN Attention | Oktay 2018 | `attention_unet` |

**Additionally recommended on BUSI/CVC-ClinicDB/GlaS/Kvasir-SEG**:
- PolypPVT, CASCADE, HSNet, SSFormer (polyp/gland-specific methods)

### Experiment Script

```bash
# Run all baselines
bash scripts/experiments/run_sota_benchmark.sh

# Single model × single dataset
python train.py --config configs/architectures/networks/general/transunet.yaml \
    --output_dir output/sota/transunet_synapse --amp
```

---

## 2. Decoder Ablation Study

### Goal
Fix encoder, compare decoders to find the best encoder-decoder pairing.

### Experiment Design

**3 representative encoders** × **all 40 decoders**:

| Encoder | Type | Rationale |
|---------|------|-----------|
| `basic` | Original UNet Conv | Simplest baseline, eliminates encoder influence |
| `timm_resnet50` | CNN (ImageNet pretrained) | Classic CNN backbone |
| `timm_pvt_v2_b2` | Transformer (ImageNet) | Transformer representative |

**Datasets**: Synapse + ISIC 2018 (one multi-organ, one binary)

### YAML Example

```yaml
# configs/architectures/decoder_study/general/resnet50_unet.yaml
model:
  num_classes: 2
  img_size: 256
  encoder:
    name: timm_resnet50
    pretrained: true
    in_channels: 3
  decoder:
    name: unet              # replace this line to switch decoder
    params: {}
  skip_connection:
    name: concat
  bottleneck:
    name: none
```

### Experiment Script

```bash
# All decoders × 3 encoders
bash scripts/experiments/run_decoder_study.sh

# Single combination
python train.py --config configs/architectures/decoder_study/general/resnet50_emcad.yaml \
    --output_dir output/decoder_study/resnet50_emcad --amp
```

### Available YAMLs

`configs/architectures/decoder_study/general/` contains 120 YAMLs (3 encoders × 40 decoders).

---

## 3. Bottleneck Ablation Study

### Experiment Design

**3 encoders × 17 bottlenecks**, decoder fixed to `bilinear`, skip fixed to `concat`.

```yaml
model:
  encoder: { name: timm_resnet50, pretrained: true }
  decoder: { name: bilinear }
  bottleneck: { name: aspp }      # replace this line
  skip_connection: { name: concat }
```

### Experiment Script

```bash
bash scripts/experiments/run_bottleneck_study.sh
```

Available YAMLs: `configs/architectures/bottleneck_study/general/` (51 files).

---

## 4. Skip Connection Ablation Study

### Experiment Design

**3 encoders × 25 skips**, decoder fixed to `unet`, bottleneck fixed to `none`.

**Focus on new methods**:
- `skvmpp` (SK-VM++, BSPC 2025) — Mamba-assisted skip
- `ta_mosc` (UTANet, AAAI 2025) — Task-adaptive mixture skip
- `uctrans` (UCTransNet, AAAI 2022) — Channel-wise Transformer skip
- `sdi` (U-Net V2, ISBI 2025) — Scale-Diverse Integration

```bash
bash scripts/experiments/run_skip_study.sh
```

Available YAMLs: `configs/architectures/skip_study/general/` (75 files).

---

## 5. Foundation Model Encoder Study

### Goal
Compare general vs domain-specific foundation models as encoders.

### Recommended Comparisons

#### General vs Specialized (Skin/Pathology/Radiology/Ophthalmology)

| General Encoder | Specialized Encoder | Dataset |
|---|---|---|
| `dinov2` (base) | `panderm` | ISIC 2017/2018, PH2 |
| `dinov2` (base) | `phikon` / `uni` / `plip` | GlaS, PanNuke, MoNuSeg |
| `dinov2` (base) | `raddino` | Montgomery+Shenzhen CXR |
| `dinov2` (base) | `retfound_dinov2` / `flair` | DRIVE, CHASE_DB1, REFUGE |
| `clip_vit` (base) | `biomedclip` / `medclip` | Cross-modal comparison |

#### MLLM Vision Tower vs Traditional Foundation

| MLLM Vision | Traditional Foundation | Dataset |
|---|---|---|
| `qwen3_vl_vision` | `dinov2` (large) | Synapse, ACDC |
| `medgemma_vision` | `biomedclip` | Multimodal benchmark |

### Notes

- All foundation encoders use **DPT head** (multi-scale features from different depth blocks)
- When encoder is frozen, only decoder is trained; parameter counts should distinguish trainable vs frozen
- Recommended: use `unet` decoder, `concat` skip uniformly

### YAML Example

```yaml
model:
  num_classes: 2
  img_size: native            # recommended native size for foundation models
  encoder:
    name: dinov2              # replace with phikon / panderm / raddino etc.
    pretrained: true
    params:
      variant: base
    freeze_cfg:
      freeze: true
      unfreeze_last_n: 4      # fine-tune last 4 blocks
  decoder:
    name: unet
  bottleneck:
    name: none
```

Available YAMLs: `configs/architectures/foundation/` (57 files).

---

## 6. Lightweight Skin Cancer Segmentation

### Goal
Compare lightweight networks on skin lesion segmentation, evaluate params-accuracy tradeoff.

### Datasets
- **ISIC 2017** — training set, official train/val/test split
- **ISIC 2018** — training set, official train/val/test split
- **PH2** — external validation (200 images, 5-fold CV)

### Baselines

| Network | Params | Paper | Config key |
|---------|--------|-------|------------|
| EGE-UNet | ~50K | MICCAI 2023 W, [GitHub](https://github.com/JCruan519/EGE-UNet) | `ege_unet` |
| Lite-UNet | ~60K | 2023 | `lite_unet` |
| U-Lite | ~60K | 2023 | `u_lite` |
| MALUNet | ~170K | BIBM 2022, [GitHub](https://github.com/JCruan519/MALUNet) | `malunet` |
| LV-UNet | ~400K | BIBM 2024, [GitHub](https://github.com/juntaoJianggavin/LV-UNet) | `lv_unet` |
| UltraLight-VM-UNet | ~50K | 2024, [GitHub](https://github.com/wurenkai/UltraLight-VM-UNet) | `ultralight_vmunet` |
| UltraLBM-UNet | ~50K | 2024 | `ultralbm_unet` |
| MK-UNet | ~200K | ICCV 2025, [GitHub](https://github.com/SLDGroup/MK-UNet) | `mk_unet` |

### Experiment Script

```bash
bash scripts/experiments/run_lightweight_skin.sh
```

---

## 7. Training Paradigm Study

### Semi-Supervised

**Dataset**: BUSI (5-fold, 10%/20%/50% labeled ratio)
**Backbone**: UNet (basic encoder + unet decoder) + RWKV-UNet

| Method | Config |
|--------|--------|
| Mean Teacher | `configs/training_paradigms/semi_supervision/mean_teacher.yaml` |
| CPS | `configs/training_paradigms/semi_supervision/cps.yaml` |
| UniMatch | `configs/training_paradigms/semi_supervision/unimatch.yaml` |
| FixMatch | `configs/training_paradigms/semi_supervision/fixmatch.yaml` |
| AugSeg (CVPR 2023) | — (config not yet available) |
| CorrMatch (CVPR 2024) | `configs/training_paradigms/semi_supervision/corrmatch.yaml` |

### Domain Adaptation

**Scenario**: Synapse → ACDC cross-modality (CT→MRI)
**Methods**: AdvEnt / DANN / FDA / MIC / HRDA

### Knowledge Distillation

**Teacher**: TransUNet (large model) → **Student**: U-Lite (lightweight)
**Methods**: Vanilla KD / DKD / CWD / MGD / DIST

### Weakly Supervised

**Dataset**: BUSI (box annotation) / Kvasir-SEG (point annotation)
**Methods**: BoxSup / PointSup / CAM / GatedCRF / TreeEnergy

```bash
bash scripts/experiments/run_semi_study.sh
bash scripts/experiments/run_da_study.sh
bash scripts/experiments/run_kd_study.sh
bash scripts/experiments/run_weak_study.sh
```

---

## 8. Text-Guided Segmentation

> **(To be continued)**

### Trainable Model Comparison

Compare LanGuideMedSeg / LViT / MediSee etc. on QaTa-COV19 and MosMedData+ (methods requiring per-image text).

### Inference Pipeline Comparison

Compare different detector × segmenter combinations for zero-shot segmentation on Synapse.

---

## How to Register a New Model

### 1. Register New Encoder

```python
# medseg/models/encoders/cnn/my_encoder.py
from medseg.registry import ENCODER_REGISTRY

@ENCODER_REGISTRY.register("my_encoder")
class MyEncoder(nn.Module):
    def __init__(self, pretrained=False, in_channels=3, img_size=224, **kwargs):
        super().__init__()
        self.out_channels = [64, 128, 256, 512]
    def forward(self, x):
        return [f1, f2, f3, f4]
```

Add `from . import my_encoder` in `medseg/models/encoders/cnn/__init__.py`.

### 2. Register New Decoder

```python
# medseg/models/decoders/basic/my_decoder.py
from medseg.registry import DECODER_REGISTRY

@DECODER_REGISTRY.register("my_decoder")
class MyDecoder(nn.Module):
    has_internal_skip = False
    def __init__(self, encoder_channels, bottleneck_channels, skip_connection=None, **kwargs):
        super().__init__()
        self.out_channels = encoder_channels[0]
    def forward(self, bottleneck_feat, skip_features):
        return decoded
```

### 3. Register New Loss

```python
# medseg/losses/my_loss.py
from medseg.registry import LOSS_REGISTRY

@LOSS_REGISTRY.register("my_loss")
class MyLoss(nn.Module):
    def forward(self, pred, target):
        return loss_value
```

### 4. Register New Skip Connection

```python
# medseg/models/skip_connections/attention/my_skip.py
from medseg.registry import SKIP_REGISTRY

@SKIP_REGISTRY.register("my_skip")
class MySkip(nn.Module):
    def get_out_channels(self, dec_ch, skip_ch):
        return dec_ch + skip_ch
    def forward(self, decoder_feat, skip_feat):
        return torch.cat([decoder_feat, skip_feat], dim=1)
```

### 5. Register New Bottleneck

```python
from medseg.registry import BOTTLENECK_REGISTRY

@BOTTLENECK_REGISTRY.register("my_bottleneck")
class MyBottleneck(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.out_channels = in_channels
    def forward(self, x):
        return refined_x
```

### 6. Register New Semi-Supervised Method

Inherit `BaseSemiMethod`, implement `build()` / `train_step()` / `update()`, add to `_SEMI_METHODS` dict in `medseg/training/semi/__init__.py`.

### After Registration

1. Import in corresponding `__init__.py`
2. Create YAML config (reference with `name: my_xxx`)
3. Add entry to corresponding `docs/` document
4. Add to comparison table in this document
5. Run `python scripts/test_all_configs.py` to verify

---

## 9. Domain-Specific Model Benchmarks

The following experiments are grouped by medical modality, each including
general baselines + domain-specific models. All include 4 general baselines:
UNet / Attention-UNet / UNet++ / ResNet50+UNet. SAM-family models are excluded
(they have their own prompt-based evaluation paradigm).

### 9.1 Polyp Segmentation

**Datasets**: CVC-ClinicDB (5-fold) + Kvasir-SEG (5-fold)

| Model | Architecture Innovation | Key | Source |
|-------|------------------------|-----|--------|
| **Baseline** | | | |
| UNet | Standard CNN baseline | `encoder: basic` + `decoder: unet` | Ronneberger 2015 |
| Attention-UNet | Attention gate | `attention_unet` | Oktay 2018 |
| UNet++ | Dense nested skip | `unetpp` | Zhou 2018 |
| ResNet50+UNet | ImageNet pretrained | `encoder: timm_resnet50` + `decoder: unet` | — |
| **Domain-Specific** | | | |
| SEPNet | MAP(RFB) + CRC progressive refinement | `sepnet` | — |
| CTNet | SMIM multi-scale + CIM cross-layer fusion | `ctnet` | — |
| Polyper | Swin-T dual-branch (region+boundary) + BGM | `polyper` | — |
| PolypPVT | PVTv2 + CFM + cascade attention | `polyp_pvt` | AAAI 2023 |
| CASCADE | Cascaded attention decoder | `cascade` | MICCAI 2023 |
| HSNet | PVTv2 + cascaded CSA | `hsnet` | 2023 |
| SSFormer | MiT-B2 + PLD decoder | `ssformer` | 2023 |
| LDNet | Lesion-aware dynamic kernel | `ldnet` | 2022 |
| ESFPNet | Efficient sparse FPN | `esfpnet` | 2023 |
| MIST | Multi-task seg transformer | `mist` | 2023 |
| FCBFormer | FCN + Transformer fusion | `fcbformer` | 2022 |
| TransNetR | Transformer + residual | `transnetr` | 2022 |

```bash
bash scripts/experiments/run_polyp_benchmark.sh
```

### 9.2 Skin Segmentation

**Training**: ISIC 2017, ISIC 2018 | **External validation**: PH2

| Model | Architecture Innovation | Key | Params |
|-------|------------------------|-----|--------|
| **Baseline** | (same as above) | | |
| **Lightweight Domain-Specific** | | | |
| EGE-UNet | Group Enhanced + GHPA + deep supervision | `ege_unet` | ~50K |
| Lite-UNet | Lightweight Conv encoder | `lite_unet` | ~60K |
| U-Lite | Axial depthwise conv | `u_lite` | ~60K |
| MALUNet | Multi-axis large-kernel + DGA | `malunet` | ~170K |
| LV-UNet | MobileNetV3 + VanillaNet decoder | `lv_unet` | ~400K |
| UltraLight-VM-UNet | Ultra-lightweight Mamba | `ultralight_vmunet` | ~50K |
| UltraLBM-UNet | Ultra-lightweight bidirectional Mamba | `ultralbm_unet` | ~50K |
| MK-UNet | Multi-kernel IRB + CBAM | `mk_unet` | ~200K |
| **Mamba-Specific** | | | |
| MUCM-Net | UCMBlock (Mamba + shifted MLP) | `mucm_net` | — |
| AC-MambaSeg | Adaptive conv + Mamba bottleneck + CBAM skip | `ac_mambaseg` | — |
| SkinMamba | Cross-scale Mamba + FFT boundary | `skin_mamba` | — |
| DermoMamba | Cross-scale Mamba + PCA + 3-direction SweepMamba | `dermomamba` | — |

```bash
bash scripts/experiments/run_skin_benchmark.sh
```

### 9.3 Retinal Vessel Segmentation

**Datasets**: DRIVE (train20/test20), STARE (5-fold), CHASE_DB1

| Model | Architecture Innovation | Key |
|-------|------------------------|-----|
| **Baseline** | (same as above) | |
| FR-UNet | Full-Resolution multi-branch vessel seg | `fr_unet` |
| SerpMamba | Serpentine scan 4-direction SS2D | `serp_mamba` |
| MambaVesselNet++ | CNN-Mamba hybrid + 3-direction scan | `mamba_vesselnet_pp` |

```bash
bash scripts/experiments/run_retinal_benchmark.sh
```

### 9.4 Ultrasound Segmentation

**Dataset**: BUSI (5-fold)

| Model | Architecture Innovation | Key |
|-------|------------------------|-----|
| **Baseline** | (same as above) | |
| AAU-Net | Adaptive Attention (BUSI-specific) | `aau_net` |
| DCM-Net | Dual encoder CNN+Mamba + CBFM cross-branch fusion | `dcm_net` |
| UU-Mamba | U-Mamba + uncertainty-aware output | `uu_mamba` |
| ViM-UNet | Vision Mamba encoder + UNet decoder | `vim_unet` |

```bash
bash scripts/experiments/run_ultrasound_benchmark.sh
```

### 9.5 Pathology Segmentation

**Dataset**: GlaS (train80%/test)

| Model | Architecture Innovation | Key |
|-------|------------------------|-----|
| **Baseline** | (same as above) | |
| U-VixLSTM | Vision-xLSTM (mLSTM) encoder + skip decoder | `u_vixlstm` |
| TransNuSeg | Multi-task decoder + shared QKV attention (MICCAI 2023) | `transnuseg` |
| HoverNetLite | NP+HV dual-branch nuclei seg (lightweight HoVerNet) | `hovernet_lite` |
| NuLite | Lightweight nuclei seg | `nulite` |

```bash
bash scripts/experiments/run_pathology_benchmark.sh
```

---

## Experiment Scripts Overview

| Script | Purpose | Models |
|--------|---------|--------|
| `run_sota_benchmark.sh` | General SOTA architecture comparison | 11 |
| `run_decoder_study.sh` | Decoder ablation (3 enc × 15 dec) | 45 |
| `run_bottleneck_study.sh` | Bottleneck ablation (3 enc × 9 bn) | 27 |
| `run_skip_study.sh` | Skip ablation (3 enc × 12 skip) | 36 |
| `run_lightweight_skin.sh` | Lightweight skin segmentation | 8 |
| `run_polyp_benchmark.sh` | Polyp domain-specific models | 16 |
| `run_skin_benchmark.sh` | Skin domain-specific models | 16 |
| `run_retinal_benchmark.sh` | Retinal domain-specific models | 7 |
| `run_ultrasound_benchmark.sh` | Ultrasound domain-specific models | 8 |
| `run_pathology_benchmark.sh` | Pathology domain-specific models | 8 |
| `run_semi_study.sh` | Semi-supervised paradigms | 6 |
| `run_da_study.sh` | Domain adaptation paradigms | 8 |
| `run_kd_study.sh` | Knowledge distillation | 7 |
| `run_weak_study.sh` | Weakly supervised paradigms | 6 |

All scripts are in `scripts/experiments/`, first argument can specify fold or dataset.
