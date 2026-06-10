<div align="center">
  <img src="figs/logo.png" alt="UltimateMedSeg Logo" width="500"/>
  <p>
    <strong>Juntao Jiang</strong>,
    <strong>Jinsheng Bai</strong>,
    <strong>Linxuan Fan</strong>,
    <strong>Jiangning Zhang</strong>,
    <strong>Yong Liu</strong>
  </p>

  <p>
    <a href="README_CN.md">中文文档</a>
  </p>
</div>

> **128** networks · **169** encoders · **40** decoders · **88** losses · **25** skip connections · **17** bottlenecks · **6** training paradigms · **24** augmentations · **876** YAML configs · switch anything with one line of YAML

---

## 📑 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Tutorial](#tutorial)
- [Project Structure](#project-structure)
- [Model Components](#model-components)
- [Training Paradigms](#training-paradigms)
- [Deployment & Efficiency](#deployment--efficiency)
- [Datasets](#datasets)
- [Config System](#config-system)
- [Custom Extensions](#custom-extensions)
- [Citation & License](#citation--license)

---

## 📦 Installation

### Requirements

- Python >= 3.8
- PyTorch >= 2.0
- CUDA (recommended) / CPU / Apple Silicon (MPS)

### Basic Installation

```bash
git clone <repo_url>
cd segmentation_tool

# Install dependencies
pip install -r requirements.txt

# Install in dev mode
pip install -e .
```

### Optional Dependencies

```bash
# Foundation models
pip install timm transformers huggingface_hub safetensors

# Data augmentation
pip install albumentations

# Training visualization
pip install tensorboard wandb

# MLLM inference pipeline
pip install groundingdino-py
pip install git+https://github.com/facebookresearch/segment-anything.git

# ONNX export & verification
pip install onnx onnxruntime

# Lion optimizer
pip install lion-pytorch
```

### Automatic Weight Download

```bash
# List all auto-downloadable weights
python -m medseg.utils.weight_downloader list

# Download specific weights
python -m medseg.utils.weight_downloader download medsam_vit_b

# Check cache status
python -m medseg.utils.weight_downloader check
```

timm encoder weights are downloaded automatically, no manual management needed.

---

## 🚀 Quick Start

### 1. Standard Supervised Training

```bash
# ResNet50 + UNet decoder
python train.py --config configs/architectures/networks/general/aau_net.yaml \
    --output_dir output/aau_net

# With AMP mixed precision
python train.py --config configs/architectures/networks/general/transunet.yaml \
    --output_dir output/transunet --amp

# Multi-GPU DDP training
torchrun --nproc_per_node=4 train.py \
    --config configs/architectures/networks/general/swinunet.yaml \
    --output_dir output/swinunet --amp
```

### 2. Semi-Supervised Training

```bash
python semi_train.py --config configs/training_paradigms/semi_supervision/mean_teacher.yaml \
    --output_dir output/semi_mt
```

### 3. ONNX Export

```bash
python scripts/export_onnx.py \
    --config configs/architectures/networks/general/transunet.yaml \
    --checkpoint output/best_model.pth \
    --output model.onnx --verify
```

### 4. Prediction Visualization

```bash
python scripts/visualize.py \
    --config configs/architectures/networks/general/transunet.yaml \
    --checkpoint output/best_model.pth \
    --input ./data/test/images/ \
    --output vis_output/
```

### 5. Python API

```python
from medseg.utils.config import load_config
from medseg.model_builder import build_model

cfg = load_config("configs/architectures/networks/general/transunet.yaml")
model = build_model(cfg)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params: {trainable / 1e6:.2f}M")
```

---

## 📚 Tutorial

A step-by-step tutorial series covering deep learning medical image segmentation from fundamentals to advanced topics:

| Chapter | Title | Key Topics |
|---------|-------|------------|
| [01](docs/tutorial/01_introduction.md) | Introduction to Medical Image Segmentation | Concepts, clinical significance, metrics, method evolution |
| [02](docs/tutorial/02_unet.md) | U-Net in Detail | Architecture, skip connections, U-Net family variants |
| [03](docs/tutorial/03_data.md) | Data and Preprocessing | Formats, split strategies, augmentation pipeline |
| [04](docs/tutorial/04_training.md) | Training and Evaluation | Loss functions, optimizers, AMP/DDP, evaluation |

[Full tutorial index](docs/tutorial/README.md)

---

## 🏗️ Project Structure

```
segmentation_tool/
├── medseg/                                      # Core framework
│   ├── models/                                  # Model components
│   │   ├── encoders/                            #   169 encoders
│   │   │   ├── cnn/              (12 modules)   #     CNN: basic, ResNet, ConvNeXt, EfficientNet, MedNeXt, MEW, R2U, AttUNet, ...
│   │   │   ├── transformer/      (18 modules)   #     Transformer: TransUNet, SwinUNet, MISSFormer, DAEFormer, HiFormer, PVTv2, MaxViT, ...
│   │   │   ├── mamba/            (10 modules)   #     Mamba/SSM: VMUNet, UMamba, LKM, LoG-VMamba, UltraLight-VM, VMKLA, ...
│   │   │   ├── rwkv/             (4 modules)    #     RWKV: RWKV-UNet, U-RWKV, MD-RWKV, RIR-Zigzag
│   │   │   ├── linear_attn/      (5 modules)    #     Linear attention: RetNet, Linformer, Performer, TTT, xLSTM
│   │   │   ├── kan_mlp/          (4 modules)    #     KAN/MLP: UKAN, Rolling-UNet, UNeXt, Wav-KAN
│   │   │   ├── foundation/       (35 modules)   #     Foundation models (DPT head)
│   │   │   │   ├── general/      (5)            #       DINOv2, DINOv3, DINO, CLIP-ViT, SAM-ViT
│   │   │   │   ├── pathology/    (5)            #       Phikon, UNI, PLIP, MUSK, Phikon-v2
│   │   │   │   ├── radiology/    (3)            #       Rad-DINO, OmniRad, MedSigLIP
│   │   │   │   ├── ophthalmology/(4)            #       RETFound-DINOv2, FLAIR, OphMAE, RETFound
│   │   │   │   ├── dermatology/  (3)            #       PanDerm, DermCLIP, MonetDerm
│   │   │   │   ├── multimodal_med/(3)           #       BiomedCLIP, MedCLIP, KEEP
│   │   │   │   ├── mllm_vision/  (8)            #       Qwen3-VL, MedGemma, LLaVA-Med, HuatuoGPT, ...
│   │   │   │   ├── endoscopy/    (1)            #       EndoViT
│   │   │   │   └── ultrasound/   (3)            #       UltraDINO, UltraFedFM, USF-MAE
│   │   │   └── wrapper/          (1 module)     #     timm dynamic wrapper (1000+ models, timm_ prefix)
│   │   ├── decoders/                            #   40 decoders
│   │   │   ├── basic/            (4 registered) #     Basic upsampling: UNet, Bilinear, Deconv, DepthwiseSep
│   │   │   ├── dense/            (2 registered) #     Dense connections: UNet++, UNet3+
│   │   │   ├── cascade/          (10 registered)#     CASCADE, EMCAD (2 variants), G-CASCADE (2 variants), CFM, MERIT (2 variants), EDLDNet
│   │   │   ├── attention/        (3 registered) #     Attention Gate, HAM, Lawin
│   │   │   ├── transformer/      (5 registered) #     DAEFormer, MTUNet, nnFormer, SwinUNet, UCTransNet
│   │   │   ├── mlp/              (2 registered) #     SegFormer MLP, MLP Decoder
│   │   │   ├── specific/         (12 registered)#     TransUNet CUP, HiFormer, H2Former, MISSFormer, ScaleFormer, FAT-Net, MALUNet, EGE-UNet, ...
│   │   │   ├── pyramid/          (1 registered) #     UPerNet
│   │   │   └── mamba/            (1 registered) #     VM-UNet
│   │   ├── bottlenecks/          (17 modules)   #   17 bottlenecks: none, basic, ASPP, DenseASPP, PPM, Transformer, SE, CBAM, ...
│   │   ├── skip_connections/                    #   25 skip connections
│   │   │   ├── basic/            (2 modules)    #     Basic: concat, dense
│   │   │   ├── attention/        (10 modules)   #     Attention: AG, CAB, SAB, SCSE, CBAM, Gating, GRU, GAB, SC-Att, TA-MoSC
│   │   │   ├── transformer/      (5 modules)    #     Transformer: CrossAttn, TransFusion, AggAttn, MISSFormer, UCTrans
│   │   │   ├── mamba/            (1 module)     #     Mamba: SK-VM++
│   │   │   └── fusion/           (6 modules)    #     CNN fusion: BiFusion, Deformable, MultiScale, FeatureRefine, CCM, SDI
│   │   ├── networks/                            #   128 complete architectures (136 registered, size variants merged)
│   │   │   ├── cnn/              (35 registered)#     CNN: UNet3+, UNet++, AttUNet, nnUNet, MedNeXt, ACC-UNet, CMUNeXt, STUNet, ...
│   │   │   ├── transformer/      (36 registered)#     Transformer: TransUNet, SwinUNet, DAEFormer, PolypPVT, CASCADE, SEPNet, CTNet, ...
│   │   │   ├── mamba/            (25 registered)#     Mamba: VMUNet, UMamba, SwinUMamba, SkinMamba, DermoMamba, SerpMamba, ...
│   │   │   ├── sam/              (12 registered)#     SAM family: MedSAM, SAM-Med2D, SAM2, SAMUS, AutoSAM, MobileSAM, ...
│   │   │   ├── rwkv/             (4 registered) #     RWKV: U-RWKV, RWKV-UNet, MD-RWKV, RIR-Zigzag
│   │   │   ├── kan_mlp/          (7 registered) #     KAN/MLP: UKAN, Rolling-UNet (4 variants), UNeXt, Wav-KAN
│   │   │   └── linear_attn/      (4 registered) #     Linear attention: TTT-UNet, xLSTM-UNet (2 variants), U-VixLSTM
│   │   └── text_unet/            (13 modules)   #   Text-guided: CRIS, BiomedParse, LanGuideMedSeg, LViT, TGANet, TPRO, ...
│   ├── training/                                # Training paradigms
│   │   ├── semi/                 (23 modules)   #   21 semi-supervised: MeanTeacher, CPS, UniMatch, FixMatch, SSL4MIS-U, CorrMatch, ...
│   │   ├── domain_adaptation/    (18 modules)   #   18 domain adaptation: AdvEnt, DANN, TENT, FDA, MIC, HRDA, SePiCo, ...
│   │   ├── distillation/         (28 modules)   #   27 distillation: VanillaKD, DKD, MGD, DIST, CWD, ReviewKD, SimKD, NORM, ...
│   │   └── weakly_supervised/    (28 modules)   #   28 weakly supervised: Box, CAM, Point, Scribble, SEAM, PuzzleCAM, EPS, ...
│   ├── inference/                               # Inference
│   │   └── mllm/                 (16 modules)   #   MLLM pipeline: 5 detector × 4 segmenter = 20 combinations
│   │       │                                    #     Detector: GroundingDINO, Qwen2/2.5/3-VL, InternVL
│   │       │                                    #     Segmenter: SAM2, MedSAM, SAM-Med2D, LiteMedSAM
│   │       └── medisee/          (3 modules)    #     MediSee: LLM reasoning segmenter
│   ├── losses/                   (15 modules)   # 88 losses
│   │                                            #   Supervised: CE, Dice, Focal, Tversky, Lovász, Boundary, Hausdorff, ...
│   │                                            #   Distillation: VanillaKD, DKD, CWD, MGD, DIST, AT, RKD, ...
│   │                                            #   Domain adaptation: AdvEnt, DANN, FDA, MIC, TENT, ...
│   │                                            #   Weakly supervised: Box, CAM, Point, Scribble, TreeEnergy, GatedCRF, ...
│   ├── datasets/                 (10 modules)   # Data loading: Synapse, ACDC, Generic, QaTa-COV19, MosMedData+, 24 augmentations
│   │   ├── advanced_aug.py                      #   24 advanced augmentations (YAML configurable)
│   │   └── transforms.py                        #   Basic transforms (Resize, ToTensor, Normalize)
│   ├── utils/                    (8 modules)    # Utilities
│   │   ├── amp_ddp.py                           #   AMP mixed precision + DDP distributed + DataParallel
│   │   ├── logger.py                            #   TensorBoard / WandB unified logging
│   │   ├── config.py                            #   Config inheritance (_base_ field support)
│   │   ├── warmup.py                            #   Warmup scheduler + Lion/AdamW/SGD optimizers
│   │   ├── augmentation.py                      #   Augmentation builder (basic/albumentations/pipeline)
│   │   ├── reproducibility.py                   #   Reproducibility (global seed + cuDNN deterministic)
│   │   ├── weight_downloader.py                 #   Automatic weight download + manual URL hints
│   │   └── metrics.py                           #   Evaluation metrics: Dice, IoU, HD95, NSD
│   ├── model_builder.py                         # YAML → model auto-assembler
│   └── registry.py                              # 6 registries: ENCODER / DECODER / SKIP / BOTTLENECK / LOSS / AUGMENTATION
├── data/                                        # Dataset root (user datasets go here)
│   ├── YourDataset/                             #   Your custom dataset
│   ├── source/                                  #   Domain adaptation source
│   ├── target/                                  #   Domain adaptation target
│   ├── target_val/                              #   Domain adaptation validation
│   └── test_dummy/                              #   Dummy test data
├── figs/                                        # Figures & logos
│   └── logo.png                                 #   Project logo
├── examples/                                    # Usage examples
│   └── grounding_dino_example.py                #   GroundingDINO detection example
├── configs/                      (876 yamls)    # YAML configs
│   ├── architectures/            (749 yamls)    #   Network architecture configs
│   │   ├── networks/             (281 yamls)    #     Complete networks (128 arch across general/acdc/synapse)
│   │   ├── combinations/         (167 yamls)    #     Encoder+decoder free combinations
│   │   ├── decoder_study/        (121 yamls)    #     Decoder ablation (3 enc × 40 dec)
│   │   ├── skip_study/           (75 yamls)     #     Skip ablation (3 enc × 25 skip)
│   │   ├── bottleneck_study/     (51 yamls)     #     Bottleneck ablation (3 enc × 17 bn)
│   │   └── foundation/           (54 yamls)     #     Foundation models (9 modalities × 35 encoders)
│   ├── training_paradigms/       (99 yamls)     #   Training paradigm configs
│   │   ├── semi_supervision/     (21 yamls)     #     Semi-supervised (21 methods)
│   │   ├── domain_adaptation/    (18 yamls)     #     Domain adaptation (18 methods)
│   │   ├── distillation/         (22 yamls)     #     Distillation (27 methods)
│   │   ├── text_guided/          (19 yamls)     #     Text-guided (13 models + pipeline)
│   │   └── weak_supervision/     (19 yamls)     #     Weakly supervised (28 methods)
│   ├── intro_to_datasets/        (25 yamls)     #   25 dataset introductions + example configs
│   └── experiments/                             #   Experiment configs
├── scripts/                                     # Utility + experiment scripts
│   ├── experiments/              (14 scripts)   #   Experiment bash scripts
│   │   ├── run_sota_benchmark.sh                #     SOTA architecture comparison (11 models × 7 datasets)
│   │   ├── run_decoder_study.sh                 #     Decoder ablation (3 enc × 15 classic dec)
│   │   ├── run_bottleneck_study.sh              #     Bottleneck ablation (3 enc × 9 bn)
│   │   ├── run_skip_study.sh                    #     Skip ablation (3 enc × 12 skip)
│   │   ├── run_polyp_benchmark.sh               #     Polyp-specific models (16 models × 2 datasets)
│   │   ├── run_skin_benchmark.sh                #     Skin-specific models (16 models × 2 datasets + PH2 external)
│   │   ├── run_retinal_benchmark.sh             #     Retinal-specific models (7 models × 3 datasets)
│   │   ├── run_ultrasound_benchmark.sh          #     Ultrasound-specific models (8 models × BUSI)
│   │   ├── run_pathology_benchmark.sh           #     Pathology-specific models (5 models × GlaS)
│   │   ├── run_lightweight_skin.sh              #     Lightweight skin segmentation (8 models)
│   │   ├── run_semi_study.sh                    #     Semi-supervised paradigm comparison (6 methods)
│   │   ├── run_da_study.sh                      #     Domain adaptation paradigm comparison (8 methods)
│   │   ├── run_kd_study.sh                      #     Knowledge distillation comparison (7 methods)
│   │   └── run_weak_study.sh                    #     Weakly supervised paradigm comparison (6 methods)
│   ├── export_onnx.py                           #   ONNX model export (dynamic size + ORT verification)
│   ├── visualize.py                             #   Prediction visualization (input + pred + overlay)
│   ├── test_all_configs.py                      #   Config batch testing (build + forward + loss)
│   └── prepare_qata_mosmed.py                   #   QaTa-COV19 / MosMedData+ dataset validation
├── docs/                         (15 docs)      # Detailed documentation
│   ├── models/                                  #   Model docs: overview, networks, encoders, decoders, skip, bottleneck
│   ├── paradigms/                               #   Paradigm docs: infrastructure, semi, weak, DA, distillation, text-guided
│   ├── deployment/                              #   Deployment docs: ONNX, FLOPs, params, FPS
│   ├── data/                                    #   Data docs: 25 datasets, 5 types, 4 split modes
│   └── research_guide.md                        #   Research guide: 8 directions + 14 experiment scripts
├── train.py                                     # Supervised training (AMP + DDP + DataParallel + Logger + Warmup)
├── semi_train.py                                # Semi-supervised training (21 methods)
├── train_weakly_supervised.py                   # Weakly supervised training (28 methods)
├── train_domain_adaptation.py                   # Domain adaptation training (18 methods)
├── train_distillation.py                        # Knowledge distillation training (27 methods)
├── train_text_guided.py                         # Text-guided training (13 models)
├── test.py                                      # Inference / testing
├── profile_model.py                             # FLOPs / params / FPS profiling
├── setup.py                                     # Package installation
└── requirements.txt                             # Python dependencies
```

---

## 🧩 Model Components

> Detailed docs: [docs/models/](docs/models/README.md)

### Complete Networks — 128

| Category | Count | Examples |
|---|---|---|
| CNN | 35 | UNet3+, UNet++, Attention-UNet, nnU-Net, MedNeXt, ACC-UNet, CMUNeXt |
| Transformer | 35 | TransUNet, Swin-UNet, DAEFormer, MISSFormer, HiFormer, PolypPVT, CASCADE |
| Mamba / SSM | 24 | VM-UNet, U-Mamba, Swin-UMamba, LKM-UNet, LoG-VMamba, HC-Mamba |
| SAM family | 10 | MedSAM, SAM-Med2D, SAM2, SAMUS, AutoSAM, MobileSAM |
| KAN / MLP | 4 | U-KAN, Rolling-UNet, UNeXt, Wav-KAN |
| Linear Attention | 3 | TTT-UNet, xLSTM-UNet, U-VixLSTM |
| RWKV | 4 | U-RWKV, RWKV-UNet, MD-RWKV-UNet, RIR-Zigzag |
| Text-guided | 13 | CRIS, BiomedParse, LanGuideMedSeg, LViT, TGANet, TPRO, CausalCLIPSeg |

> Full list: [docs/models/networks.md](docs/models/networks.md)

### Encoders — 169

**Highlight: 35 foundation model encoders covering 9 medical modalities**

| Modality | Count | Models |
|---|---|---|
| General | 5 | DINOv2, DINOv3, DINO, CLIP-ViT, SAM-ViT |
| Pathology | 5 | Phikon, Phikon-v2, UNI, PLIP, MUSK |
| Radiology | 3 | Rad-DINO, OmniRad, MedSigLIP |
| Ophthalmology | 4 | RETFound-DINOv2, RETFound, FLAIR, OphMAE |
| Dermatology | 3 | DermCLIP, MoNet, PanDerm |
| Multimodal Medical | 3 | BiomedCLIP, MedCLIP, KEEP |
| MLLM Vision | 8 | Qwen2.5-VL, Qwen3-VL, MedGemma, LLaVA-Med, HuatuoGPT, HealthGPT, HuLuMed, LingShu |
| Ultrasound | 3 | UltraDINO, UltraFedFM, US-FMAE |
| Endoscopy | 1 | Endo-ViT |

All foundation ViTs use **DPT head** (multi-block multi-scale features), not naive FPN-from-tokens.

**Dynamic timm encoder**: any model from `timm.list_models()` with `timm_` prefix works directly.

```yaml
encoder:
  name: timm_efficientnet_b7    # or any timm model name
  pretrained: true
```

> Full list: [docs/models/encoders.md](docs/models/encoders.md)

### Decoders — 40

| Category | Count | Examples |
|---|---|---|
| Basic (upsampling) | 4 | UNet, Bilinear, Deconv, DepthwiseSep |
| Dense (connections) | 2 | UNet++, UNet3+ |
| Cascade | 10 | CASCADE, EMCAD (2 variants), G-CASCADE (2 variants), CFM, MERIT (2 variants), EDLDNet |
| Attention | 3 | Attention Gate, HAM, Lawin |
| Transformer | 5 | DAEFormer, MTUNet, SwinUNet, nnFormer, UCTransNet |
| MLP | 2 | SegFormer MLP, MLP Decoder |
| Specific (network) | 12 | TransUNet CUP, HiFormer, H2Former, MISSFormer, ScaleFormer, FAT-Net, MALUNet, EGE-UNet, ... |
| Mamba | 1 | VM-UNet |
| Pyramid | 1 | UPerNet |

> Full list: [docs/models/decoders.md](docs/models/decoders.md)

### Skip Connections — [docs/models/skip_connections.md](docs/models/skip_connections.md)

### Bottlenecks — [docs/models/bottlenecks.md](docs/models/bottlenecks.md)

---

## 🎓 Training Paradigms

> Detailed docs: [docs/paradigms/](docs/paradigms/README.md)

### Infrastructure

| Feature | YAML config |
|---|---|
| Mixed precision AMP | `training.amp: true` or CLI `--amp` |
| Multi-GPU DDP | `torchrun --nproc_per_node=N train.py` |
| DataParallel | `training.parallel: dp` |
| TensorBoard | `training.logger: tensorboard` |
| WandB | `training.logger: wandb` |
| Reproducibility Seed | `training.random_state: 42` + `training.deterministic: true` |
| Warmup scheduler | `training.scheduler.name: warmup_cosine` + `warmup_epochs: 10` |
| Config inheritance | `_base_: ../base.yaml` |
| Albumentations | `training.augmentation: albumentations` |
| YAML Aug Pipeline | `training.augmentation: pipeline` + `training.aug_pipeline: [...]` |

> Full config guide: [docs/paradigms/README.md](docs/paradigms/README.md)

### Augmentation Pipeline — 24 Methods

Freely combine 24 augmentation methods via YAML config, no code changes needed. All methods support intensity range parameters, randomly sampled per call.

```yaml
training:
  augmentation: pipeline        # enable pipeline mode
  aug_pipeline:                 # define augmentations in order
    - name: horizontal_flip
      params: { p: 0.5 }
    - name: vertical_flip
      params: { p: 0.5 }
    - name: random_rotate90
      params: { p: 0.5 }
    - name: random_rotate
      params: { p: 0.3, degrees_range: [-30, 30] }
    - name: random_affine
      params: { p: 0.3, degrees_range: [-15, 15], translate_range: [0.0, 0.1], scale_range: [0.8, 1.2] }
    - name: elastic_deform
      params: { p: 0.3, alpha_range: [20, 80], sigma_range: [3, 7] }
    - name: copy_paste
      params: { p: 0.3, max_objects: 2, scale_range: [0.5, 1.5] }
    - name: mosaic
      params: { p: 0.3, offset_range: [0.0, 0.2] }
    - name: clahe
      params: { p: 0.3, clip_limit_range: [1.0, 5.0], tile_size_range: [4, 16] }
    - name: gamma_correction
      params: { p: 0.3, gamma_range: [0.7, 1.5] }
    - name: gaussian_blur
      params: { p: 0.2, kernel_range: [3, 7], sigma_range: [0.1, 2.0] }
    - name: gaussian_noise
      params: { p: 0.2, std_range: [0.01, 0.08] }
```

**Supported Augmentation Methods (24)**:

| Category | Methods |
|---|---|
| Geometric | `horizontal_flip`, `vertical_flip`, `random_rotate90`, `random_rotate`, `random_affine`, `random_perspective`, `random_scale`, `elastic_deform`, `grid_mask` |
| Pixel-level | `photometric_distortion`, `color_jitter`, `brightness_contrast`, `gamma_correction`, `clahe`, `gaussian_blur`, `gaussian_noise`, `sharpness`, `posterize`, `random_solarize`, `channel_dropout` |
| Masking | `random_erasing`, `coarse_dropout`, `grid_mask` |
| Sample-level | `copy_paste`, `mosaic` |

> **Note**: All intensity parameters use `_range` suffix (e.g. `degrees_range`, `alpha_range`), randomly sampled per call.

> Full parameter docs for each method: [docs/data/README.md](docs/data/README.md#augmentation-pipeline--24-methods)
> Full config example: [resnet50_unet_advanced_aug.yaml](configs/architectures/decoder_study/general/resnet50_unet_advanced_aug.yaml)

### Semi-Supervised — 21 Methods

Mean Teacher · CPS · CCT · UniMatch · FixMatch · FlexMatch · FreeMatch · SoftMatch · UA-MT · URPC · Deep Co-Training · Pi-Model · Temporal Ensembling · Pseudo-Label · ICT · R-Drop · Cross-Teaching · CorrMatch · AllSpark · DiffRect · SSL4MIS-U

> Details: [docs/paradigms/semi_supervised.md](docs/paradigms/semi_supervised.md)

### Domain Adaptation — 18 Methods

Source Only · AdvEnt · DANN · TENT · DPL · CBMT · FDA · CRST · PixMatch · MIC · DAFormer · HRDA · PiPa · DDB · SePiCo · DiGA · MICDrop · SemiVL

> Details: [docs/paradigms/domain_adaptation.md](docs/paradigms/domain_adaptation.md)

### Knowledge Distillation — 27 Methods

Vanilla KD · FitNets · AT · FSP · NST · RKD · VID · DKD · MGD · DIST · CIRKD · CWD · ReviewKD · SimKD · NORM · SDD · AICSD · LSKD · TTM · CTKD · MLKD + 4 medical-specific

> Details: [docs/paradigms/distillation.md](docs/paradigms/distillation.md)

### Weakly Supervised — 28 Methods

Box · CAM · Point · Scribble · MIL · EM · GatedCRF · TreeEnergy · SEAM · PuzzleCAM · AdvCAM · EPS · BoxInst · ReCAM · ToCo · LPCAM · MARS · BACoN · WPGSeg · DuPL · MoRe · PSDPM · SemPLeS

> Details: [docs/paradigms/weakly_supervised.md](docs/paradigms/weakly_supervised.md)

### Text-Guided — 13 Models + Inference Pipeline

**Trainable models**: CRIS · BiomedParse · LanGuideMedSeg · LViT · TGANet · TPRO · CausalCLIPSeg · CLIP-Universal · CXR-CLIP-Seg · TP-DRSeg · MedCLIP-SAM · SaLIP · MediSee

**Inference Pipeline** (5 detector × 4 segmenter = 20 combinations):
- Detector: GroundingDINO · Qwen2-VL · Qwen2.5-VL · Qwen3-VL · InternVL
- Segmenter: SAM2 · MedSAM · SAM-Med2D · LiteMedSAM

> Details: [docs/paradigms/text_guided.md](docs/paradigms/text_guided.md)

---

## ⚡ Deployment & Efficiency

> Detailed docs: [docs/deployment/README.md](docs/deployment/README.md)

```bash
# ONNX Export
python scripts/export_onnx.py --config xxx.yaml --checkpoint best.pth --output model.onnx --verify

# FLOPs Calculation
python -c "
from fvcore.nn import FlopCountAnalysis
import torch
flops = FlopCountAnalysis(model, torch.randn(1,3,224,224))
print(f'FLOPs: {flops.total()/1e9:.2f}G')
"

# Params (trainable only)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable/1e6:.2f}M / Total: {total/1e6:.2f}M")
```

> Note: Frozen foundation encoder params are NOT counted as trainable.

---

## 📊 Datasets

> Detailed docs: [docs/data/README.md](docs/data/README.md)
> Dataset example configs: [configs/intro_to_datasets/](configs/intro_to_datasets/)

### Supported Dataset Types

| Type | Description |
|---|---|
| `synapse` | Synapse multi-organ CT (TransUNet format) |
| `acdc` | ACDC cardiac MRI (TransUNet format) |
| `generic` | Generic images/ + masks/ directories |
| `qata_covid19` | QaTa-COV19 chest X-ray + per-image text (LViT format) |
| `mosmed_plus` | MosMedData+ COVID CT + per-image text (LViT format) |

### Data Split Methods

```yaml
# Method 1: Explicit paths
data:
  train_dir: ./data/train
  val_dir: ./data/val
  test_dir: ./data/test       # optional

# Method 2: Ratio-based split
data:
  root_dir: ./data/all
  train_ratio: 0.7
  val_ratio: 0.15

# Method 3: N-fold cross validation
data:
  root_dir: ./data/all
  n_splits: 5
  fold_idx: 0
```

### Included Datasets (25)

**Skin**: ISIC 2016/2017/2018, PH2
**Polyp**: CVC-ClinicDB, CVC-ColonDB, Kvasir-SEG
**Pathology**: GlaS, PanNuke, MoNuSeg
**Retinal**: DRIVE, STARE, CHASE_DB1, HRF, ARIA, RITE, REFUGE, Drishti-GS
**Chest**: Montgomery+Shenzhen CXR, QaTa-COV19, COVID CT Seg
**Ultrasound**: BUSI
**Multi-organ**: Synapse, ACDC
**CT**: MosMedData+

---

## 🔧 Config System

### Two Model Config Modes

```yaml
# Mode 1: Modular combination (encoder + decoder + skip + bottleneck)
model:
  num_classes: 9
  img_size: 224
  encoder:
    name: timm_resnet50
    pretrained: true
  decoder:
    name: unet
  skip_connection:
    name: concat
  bottleneck:
    name: aspp

# Mode 2: Complete architecture (architecture key)
model:
  num_classes: 9
  img_size: 224
  architecture: transunet
  arch_params: {}
```

### Config Inheritance

```yaml
# child.yaml — only write overrides
_base_: ../base_resnet50.yaml
model:
  num_classes: 9
training:
  epochs: 300
```

### Complete Training Config Example

```yaml
model:
  num_classes: 9
  img_size: 224
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
  type: synapse
  img_size: 224
  train_dir: ./data/Synapse/train_npz
  val_dir: ./data/Synapse/test_vol_h5

training:
  random_state: 42
  deterministic: true
  amp: true
  parallel: auto
  logger: tensorboard
  augmentation: albumentations
  epochs: 200
  batch_size: 16
  num_workers: 4
  val_interval: 10
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

---

## 🔌 Custom Extensions

### Add New Encoder

```python
# medseg/models/encoders/cnn/my_encoder.py
from medseg.registry import ENCODER_REGISTRY

@ENCODER_REGISTRY.register("my_encoder")
class MyEncoder(nn.Module):
    def __init__(self, pretrained=False, in_channels=3, img_size=224, **kwargs):
        super().__init__()
        self.out_channels = [64, 128, 256, 512]
    def forward(self, x):
        return [f1, f2, f3, f4]  # multi-scale features
```

### Add New Decoder

```python
@DECODER_REGISTRY.register("my_decoder")
class MyDecoder(nn.Module):
    has_internal_skip = False
    def __init__(self, encoder_channels, bottleneck_channels, skip_connection=None, **kwargs):
        super().__init__()
        self.out_channels = encoder_channels[0]
    def forward(self, bottleneck_feat, skip_features):
        return decoded
```

### Add New Loss

```python
@LOSS_REGISTRY.register("my_loss")
class MyLoss(nn.Module):
    def forward(self, pred, target):
        return loss_value
```

### Add New Augmentation

```python
# medseg/datasets/advanced_aug.py
from medseg.registry import AUGMENTATION_REGISTRY

@AUGMENTATION_REGISTRY.register("my_augmentation")
class MyAugmentation:
    def __init__(self, p=0.5, **kwargs):
        self.p = p

    def set_dataset(self, dataset):
        """Optional: implement if dataset access needed"""
        self.dataset = dataset

    def __call__(self, sample: dict) -> dict:
        import random
        if random.random() > self.p:
            return sample
        image, label = sample['image'], sample['label']
        # ... implement augmentation logic ...
        return {'image': image, 'label': label}
```

After registration and import in `medseg/datasets/__init__.py`, use via `name: my_augmentation` in YAML.

After registration and import in `__init__.py`, use via `name: my_encoder` in YAML.

---

## 📜 Citation & License

```bibtex
@software{ultimatemedseg_2026,
  title  = {UltimateMedSeg: A Modern Modular 2D Medical Image Segmentation Toolbox},
  author = {Juntao Jiang and Jinsheng Bai and Linxuan Fan and Jiangning Zhang and Yong Liu},
  year   = {2026},
  url    = {https://github.com/juntaoJianggavin/UltimateMedSeg},
}
```

### License

Apache 2.0. For legitimate academic research and engineering use only.
Clinical deployment must comply with local regulations.

### Acknowledgements

Thanks to PyTorch, timm, MONAI, SSL4MIS, SAM, GroundingDINO, DINOv2/v3, CLIP, transformers, and all open-source projects that made this possible.
