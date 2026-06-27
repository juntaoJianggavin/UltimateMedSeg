# Chapter 01: Introduction to Medical Image Segmentation

[中文文档](01_introduction_CN.md) | [Next: U-Net in Detail](02_unet.md)

---

## 1. Background and Motivation

### What is Image Segmentation?

Image segmentation assigns a class label to every pixel in an image. There are three main paradigms:

| Paradigm | Goal | Example |
|----------|------|---------|
| **Semantic Segmentation** | Classify every pixel into a category | "This pixel is liver / spleen / background" |
| **Instance Segmentation** | Detect and delineate individual objects | "This pixel belongs to cell #3" |
| **Medical Segmentation** | Semantic segmentation applied to clinical images | Organ delineation, lesion detection, tissue classification |

Medical image segmentation is essentially **semantic segmentation with domain-specific challenges**: limited annotated data, high annotation cost (requires radiologists/pathologists), strong class imbalance, and diverse imaging modalities.

### Clinical Significance

Medical image segmentation directly impacts four critical clinical workflows:

**1. Computer-Aided Diagnosis (CAD)**
- Tumor detection: automatic localization of nodules in CT, lesions in fundus images
- Screening: automated polyp detection in colonoscopy video (CVC-ClinicDB, Kvasir-SEG)

**2. Quantitative Analysis**
- Organ volume measurement: left ventricle in cardiac MRI (ACDC), optic cup/disc in retinal images (REFUGE, Drishti-GS)
- Treatment response: tumor size tracking across time points

**3. Surgical Planning and Navigation**
- Pre-operative 3D reconstruction from CT/MRI
- Intra-operative boundary guidance

**4. Radiotherapy Planning**
- Automatic organ-at-risk (OAR) delineation
- Target volume contouring for treatment planning

---

## 2. Core Concepts

### Common Imaging Modalities and Datasets

| Modality | Typical Task | Built-in Datasets |
|----------|-------------|-------------------|
| CT | Multi-organ / Infection segmentation | Synapse (8 organs), COVID CT Seg, MosMedData+ |
| MRI | Cardiac structure | ACDC (RV/LV/MYO) |
| X-ray (CXR) | Lung / Infection segmentation | Montgomery-Shenzhen, QaTa-COV19 |
| Fundus Photography | Vessel / Optic disc/cup | DRIVE, STARE, CHASE_DB1, HRF, ARIA, RITE, REFUGE, Drishti-GS |
| Dermoscopy | Skin lesion | ISIC 2016/2017/2018, PH2 |
| Histopathology (WSI) | Nuclei/gland | MoNuSeg, GlaS, PanNuke |
| Ultrasound | Breast lesion | BUSI |
| Endoscopy | Polyp | CVC-ClinicDB, CVC-ColonDB, Kvasir-SEG |

### Evaluation Metrics

APRIL-MedSeg computes three families of metrics (see `medseg/utils/metrics.py`):

**Dice Similarity Coefficient (DSC)**

The most widely used overlap metric. For a predicted mask $P$ and ground truth $G$:

$$\text{Dice} = \frac{2|P \cap G|}{|P| + |G|}$$

Range: [0, 1]. Higher is better. Equivalent to F1-score.

**Intersection over Union (IoU / Jaccard)**

$$\text{IoU} = \frac{|P \cap G|}{|P \cup G|} = \frac{|P \cap G|}{|P| + |G| - |P \cap G|}$$

Range: [0, 1]. Higher is better. Always $\leq$ Dice for the same prediction.

**95th Percentile Hausdorff Distance (HD95)**

Measures the boundary distance between prediction and ground truth, using the 95th percentile (to avoid outlier sensitivity). Lower is better. Unit: pixels (or mm if spacing is provided).

```python
# From medseg/utils/metrics.py
metrics = compute_metrics(pred, target, num_classes)
# Returns: {"dice": {...}, "iou": {...}, "hd95": {...}}
```

### Method Evolution

```
Traditional Methods          Deep Learning Era                  Foundation Era
(Pre-2015)                   (2015-2022)                        (2023-present)
                                                                        
Thresholding        ──>      FCN (2015)                ──>      SAM (2023)
Region Growing      ──>      U-Net (2015, MICCAI)      ──>      DINOv2 (2024)
Watershed           ──>      UNet++ / Attention-UNet   ──>      BiomedCLIP (2024)
Active Contours     ──>      TransUNet / Swin-UNet     ──>      UNI / Phikon (2024)
Graph Cuts          ──>      Mamba-UNet / VM-UNet      ──>      MUSK / PLIP
```

Each generation addresses specific limitations:
- **FCN**: first end-to-end pixel-wise segmentation, but loses spatial detail
- **U-Net**: skip connections recover fine-grained boundaries, works with few samples
- **Transformers**: global self-attention captures long-range dependencies
- **State Space Models (Mamba)**: linear complexity with global receptive field
- **Foundation Models**: pre-trained on large-scale data, transfer to downstream with minimal fine-tuning

---

## 3. Framework Overview

### Architecture Philosophy

APRIL-MedSeg uses a **four-module free-combination** design:

```
Input Image ──> [Encoder] ──> [Bottleneck] ──> [Decoder] ──> Segmentation Output
                                  |                 ^
                                  └── [Skip Conn] ──┘
```

Each module is independently swappable via a single YAML line:

| Module | Registry Count | Examples |
|--------|---------------|----------|
| Encoder | 177 | `basic`, `timm_resnet50`, `timm_swin_tiny_patch4_window7_224`, `dinov2`, `dino` |
| Decoder | 45 | `bilinear`, `deconv`, `emcad`, `cascade_full`, `unetpp` |
| Skip Connection | 25 | `concat`, `add`, `cab`, `scse`, `gating` |
| Bottleneck | 17 | `none`, `aspp`, `dense_aspp`, `mamba`, `transformer` |
| Complete Network | 130 | `unet`, `transunet`, `swinunet`, `attention_unet`, `vmunet` |

### Two Configuration Modes

**Mode 1: Complete Architecture** (use a pre-defined network)

```yaml
model:
  architecture: transunet   # One name selects the full architecture
  num_classes: 9
  img_size: 224
```

**Mode 2: Free Combination** (mix-and-match any registered module)

```yaml
model:
  encoder:
    name: timm_resnet50
    pretrained: true
  decoder:
    name: emcad
  skip_connection:
    name: cab
  bottleneck:
    name: aspp
```

### Project Structure

```
APRIL-MedSeg/
├── train.py                    # Standard supervised training
├── test.py                     # Evaluation with TTA/ensemble
├── semi_train.py               # Semi-supervised training
├── train_domain_adaptation.py  # Domain adaptation
├── train_distillation.py       # Knowledge distillation
├── train_weakly_supervised.py  # Weakly supervised
├── train_text_guided.py        # Text-guided segmentation
├── configs/                    # 917 YAML configs
├── medseg/                     # Core library
│   ├── models/                 # 177 encoders, 45 decoders, 130 networks
│   ├── losses/                 # 15 loss functions
│   ├── datasets/               # 6 dataset classes
│   ├── training/               # Advanced training paradigms
│   ├── inference/              # TTA, ensemble
│   └── utils/                  # Metrics, augmentation, config
└── docs/                       # Bilingual documentation
```

---

## 4. Hands-On: Your First Training

### Step 1: Prepare Data

Create a simple dataset directory:

```
data/YourDataset/
├── train/
│   ├── image_001.png
│   └── image_002.png
├── val/
│   └── image_101.png
└── test/
    └── image_201.png
```

Each image must have a corresponding mask with the same filename.

### Step 2: Run Training

```bash
python train.py --config configs/architectures/combinations/general/unet_basic.yaml
```

This uses the default config:
- Encoder: `basic` (a simple CNN backbone)
- Decoder: `bilinear` (bilinear upsampling)
- Loss: Compound (0.4 CE + 0.6 Dice)
- Optimizer: AdamW (lr=0.001)
- Epochs: 300

### Step 3: Override Parameters

No need to edit the YAML file -- override from CLI:

```bash
python train.py --config configs/architectures/combinations/general/unet_basic.yaml \
    --override training.epochs=100 training.batch_size=8 model.num_classes=9
```

### Step 4: Enable AMP (Mixed Precision)

```bash
python train.py --config configs/architectures/combinations/general/unet_basic.yaml --amp
```

### Step 5: Evaluate

```bash
python test.py --config configs/architectures/combinations/general/unet_basic.yaml \
    --checkpoint output/best_model.pth
```

---

## 5. Recommended Experiments

**Experiment 1: Baseline**
- Run `unet_basic.yaml` on the Synapse dataset (8 abdominal organs + background)
- Record baseline Dice, IoU, and HD95

**Experiment 2: Module Swap**
- Change encoder to `timm_resnet50`: `--override model.encoder.name=timm_resnet50`
- Compare metrics with the basic encoder

**Experiment 3: Loss Ablation**
- Try Dice-only loss: create a variant YAML with `losses: [{name: dice, weight: 1.0}]`
- Compare convergence speed and final metrics

---

## 6. Further Reading

### Key Papers

| Paper | Year | Contribution |
|-------|------|-------------|
| [FCN (Long et al.)](https://arxiv.org/abs/1411.4038) | CVPR 2015 | First fully convolutional network for semantic segmentation |
| [U-Net (Ronneberger et al.)](https://arxiv.org/abs/1505.04597) | MICCAI 2015 | Skip connections + small-data paradigm for medical imaging |
| [TransUNet (Chen et al.)](https://arxiv.org/abs/2102.04306) | 2021 | CNN + Transformer hybrid for medical segmentation |
| [SAM (Kirillov et al.)](https://arxiv.org/abs/2304.02643) | ICCV 2023 | Segment Anything: universal segmentation model |
| [DINOv2 (Oquab et al.)](https://arxiv.org/abs/2304.07193) | 2024 | Self-supervised visual features without labels |

### Related Documentation

- [Encoder Guide](../models/encoders.md) -- All 177 encoders with HuggingFace model paths
- [Decoder Guide](../models/decoders.md) -- 45 decoders with design rationale
- [Loss Functions](../paradigms/README.md) -- 81 registered losses across 15 implementation files
- [Data Guide](../data/README.md) -- 25 built-in datasets and augmentation pipeline
- [Research Guide](../research_guide.md) -- Ablation study design and benchmarking protocols

---

[Next: U-Net in Detail](02_unet.md)
