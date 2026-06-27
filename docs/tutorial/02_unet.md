# Chapter 02: U-Net in Detail

[Previous: Introduction](01_introduction.md) | [中文文档](02_unet_CN.md) | [Next: Data and Preprocessing](03_data.md)

---

## 1. Background and Motivation

U-Net, published by Olaf Ronneberger et al. at MICCAI 2015, is arguably the most influential architecture in medical image segmentation. Its key innovations:

- **Encoder-decoder with skip connections**: recovers spatial detail lost during downsampling
- **Data-efficient**: trains well on as few as 30 annotated images through aggressive augmentation
- **Symmetric U-shape**: intuitive multi-scale feature hierarchy

U-Net remains the default baseline in virtually every medical segmentation benchmark. Understanding it thoroughly is essential before exploring more advanced architectures.

---

## 2. Core Concepts

### 2.1 Architecture Overview

```
                          Input (e.g., 572 x 572 x 1)
                                    │
                    ┌─────── Encoder (Contracting Path) ───────┐
                    │                                          │
                    │  Conv 3x3 → ReLU → Conv 3x3 → ReLU       │
                    │       ↓ MaxPool 2x2                      │
                    │  Conv 3x3 → ReLU → Conv 3x3 → ReLU       │
                    │       ↓ MaxPool 2x2                      │
                    │  Conv 3x3 → ReLU → Conv 3x3 → ReLU       │
                    │       ↓ MaxPool 2x2                      │
                    │  Conv 3x3 → ReLU → Conv 3x3 → ReLU       │
                    │       ↓ MaxPool 2x2                      │
                    └──────────── Bottleneck ──────────────────┘
                              Conv 3x3 → ReLU → Conv 3x3 → ReLU
                                    │
                    ┌─────── Decoder (Expanding Path) ─────────┐
                    │       ↑ UpConv 2x2 (or Bilinear)         │
                    │  [Skip Concat] → Conv 3x3 → ReLU → Conv  │
                    │       ↑ UpConv 2x2                       │
                    │  [Skip Concat] → Conv 3x3 → ReLU → Conv  │
                    │       ↑ UpConv 2x2                       │
                    │  [Skip Concat] → Conv 3x3 → ReLU → Conv  │
                    └──────────────────────────────────────────┘
                                    │
                              1x1 Conv → Softmax
                                    │
                          Output (e.g., 388 x 388 x C)
```

### 2.2 Key Design Decisions

**Why skip connections?**

The encoder progressively reduces spatial resolution (via pooling/strided conv) to expand receptive field. This loses fine-grained boundary information critical for segmentation. Skip connections bridge encoder features directly to the corresponding decoder stage, providing high-resolution spatial cues.

**Why U-shape?**

Multi-scale is the key. Different objects require different receptive fields:
- Small lesions need high-resolution features (early encoder layers)
- Large organs need semantic context (deep layers + bottleneck)

The U-shape naturally creates this multi-scale pyramid.

**Why two 3x3 convolutions per stage?**

Two stacked 3x3 convolutions have the same effective receptive field as one 5x5, but with:
- Fewer parameters (2 x 9 = 18 vs 25 channels)
- More non-linearities (two ReLU activations)
- Better representational capacity

### 2.3 Upsampling Strategies

The decoder needs to increase spatial resolution. Two main approaches:

| Strategy | Mechanism | Pros | Cons |
|----------|-----------|------|------|
| **Transposed Conv** (Deconv) | Learned upsampling via transposed convolution | Trainable upsampling kernel | Checkerboard artifacts possible |
| **Bilinear + Conv** | Fixed bilinear interpolation followed by convolution | No artifacts, simpler | Upsampling not learned |

In APRIL-MedSeg, both are available as decoders:
- `decoder: bilinear` -- bilinear upsampling + convolution
- `decoder: deconv` -- transposed convolution

---

## 3. Method Details

### 3.1 The U-Net Family

Since 2015, dozens of U-Net variants have been proposed. The framework includes the most important ones:

| Architecture | Year | Key Innovation | Config Name |
|-------------|------|---------------|-------------|
| **U-Net** | MICCAI 2015 | Original encoder-decoder + skip | `unet` |
| **UNet++** | DLMIA 2018 | Dense skip connections across scales | `unetpp` |
| **Attention U-Net** | MIDL 2018 | Attention gates on skip connections | `attention_unet` |
| **UNet 3+** | ICASSP 2020 | Full-scale skip connections (every encoder to every decoder) | `unet3plus` |
| **ResUNet++** | ISM 2019 | Residual blocks + attention + ASPP | `resunetpp` |
| **DenseUNet** | - | DenseNet-style dense connections | `denseunet` |
| **scSE-UNet** | MICCAI 2018 | Squeeze-Excitation channel/spatial attention | `scseunet` |
| **R2U-Net** | IEEE Access 2018 | Recurrent residual blocks | `r2unet` |
| **MultiResUNet** | Neural Networks 2020 | Multi-resolution residual blocks | `multiresunet` |
| **ResUNet-a** | ISPRS 2020 | Atrous convolutions + residual | `resunet_a` |
| **SA-UNet** | IEEE TIM 2021 | Spatial attention on skip | `sa_unet` |
| **KiU-Net** | MICCAI 2020 | Key-point guided U-Net | `kiunet` |
| **PAN** | BMVC 2018 | Pyramid Attention Network | `pan` |
| **LinkNet** | VCIP 2017 | Lightweight encoder-decoder | `linknet` |
| **PSPNet** | CVPR 2017 | Pyramid Spatial Pooling | `pspnet` |
| **FR-UNet** | IEEE TMI 2022 | Full-resolution skip connections | `fr_unet` |

### 3.2 Skip Connection Variants

The skip connection is one of U-Net's most studied components. The framework provides 25 options:

| Skip | Mechanism | When to Use |
|------|-----------|-------------|
| `concat` | Concatenate encoder features with decoder features | Default, works well universally |
| `add` | Element-wise addition | When feature dimensions match |
| `cab` | Channel Attention Block | Focus on informative channels |
| `scse` | Squeeze-and-Excitation + Spatial SE | Channel + spatial recalibration |
| `gating` | Attention gate (from Attention U-Net) | Suppress irrelevant skip features |
| `cross_attn` | Cross-attention between encoder and decoder | Long-range skip interaction |
| `feature_refine` | Learnable feature refinement | When skip features need processing |

### 3.3 Bottleneck Options

The bottleneck sits at the deepest point of the U, processing the most compressed features:

| Bottleneck | Mechanism | When to Use |
|-----------|-----------|-------------|
| `none` | No special processing | Default baseline |
| `aspp` | Atrous Spatial Pyramid Pooling | Multi-scale context |
| `dense_aspp` | Densely connected ASPP | Richer multi-scale features |
| `transformer` | Self-attention at bottleneck | Global context at deepest level |
| `mamba` | State space model at bottleneck | Linear-complexity global context |
| `rwkv` | RWKV at bottleneck | Efficient sequence modeling |
| `cbam` | Convolutional Block Attention | Channel + spatial attention |

---

## 4. Hands-On with APRIL-MedSeg

### 4.1 Mode 1: Complete Network

Use a pre-defined U-Net architecture directly:

```yaml
model:
  architecture: unet        # Complete U-Net
  num_classes: 2            # Binary: background + foreground
  img_size: 256
  encoder:
    in_channels: 3
```

This builds the full U-Net with its original encoder/decoder/skip design.

### 4.2 Mode 2: Free Combination

Mix any encoder with any decoder, skip, and bottleneck:

```yaml
model:
  num_classes: 2
  img_size: 256
  encoder:
    name: basic             # Built-in BasicEncoder (CNN)
    in_channels: 3
    params: {}
  decoder:
    name: bilinear          # Bilinear upsampling decoder
    params: {}
  skip_connection:
    name: concat            # Standard concatenation skip
    params: {}
  bottleneck:
    name: none              # No special bottleneck
```

This is the content of `configs/architectures/combinations/general/unet_basic.yaml`.

### 4.3 YAML Configuration Walkthrough

Let's dissect the full training config:

```yaml
# --- Model ---
model:
  num_classes: 2            # Number of output classes (background + target)
  img_size: 256             # Input image resolution
  encoder:
    name: basic             # Encoder backbone
    in_channels: 3          # Input channels (3 for RGB, 1 for grayscale)
    params: {}              # Encoder-specific parameters
  decoder:
    name: bilinear          # Decoder type
    params: {}
  bottleneck:
    name: none
  skip_connection:
    name: concat
    params: {}

# --- Data ---
data:
  type: generic             # Dataset type: generic, synapse, acdc
  img_size: 256
  train_dir: ./data/YourDataset/train
  val_dir: ./data/YourDataset/val
  test_dir: ./data/YourDataset/test

# --- Training ---
training:
  epochs: 300               # Total training epochs
  batch_size: 24            # Samples per GPU
  num_workers: 4            # DataLoader workers
  loss:
    name: compound          # Compound loss: weighted sum of multiple losses
    params:
      losses:
      - name: ce            # Cross-Entropy: good for initial convergence
        weight: 0.4
      - name: dice          # Dice loss: directly optimizes overlap metric
        weight: 0.6
  optimizer:
    name: adamw             # AdamW: Adam with decoupled weight decay
    lr: 0.001               # Initial learning rate
    weight_decay: 0.0001    # L2 regularization
  scheduler:
    name: cosine            # Cosine annealing: smooth LR decay
    min_lr: 1.0e-06         # Minimum learning rate
```

### 4.4 Training Commands

```bash
# Basic training
python train.py --config configs/architectures/combinations/general/unet_basic.yaml

# With AMP (mixed precision, ~1.5x faster)
python train.py --config configs/architectures/combinations/general/unet_basic.yaml --amp

# Override parameters from CLI
python train.py --config configs/architectures/combinations/general/unet_basic.yaml \
    --override training.epochs=200 training.batch_size=8 model.num_classes=9

# Resume from checkpoint
python train.py --config configs/architectures/combinations/general/unet_basic.yaml \
    --resume output/checkpoint_epoch100.pth

# Custom output directory
python train.py --config configs/architectures/combinations/general/unet_basic.yaml \
    --output_dir ./experiments/unet_baseline

# Set random seed for reproducibility
python train.py --config configs/architectures/combinations/general/unet_basic.yaml --seed 42
```

### 4.5 Trying U-Net Variants

Swap the complete network to a U-Net variant:

```bash
# Attention U-Net
python train.py --config configs/architectures/combinations/general/attention_unet_basic.yaml

# UNet++
python train.py --config configs/architectures/networks/general/unetpp.yaml

# Or override in free-combination mode: change decoder and skip
python train.py --config configs/architectures/combinations/general/unet_basic.yaml \
    --override model.decoder.name=unet model.skip_connection.name=gating
```

---

## 5. Recommended Experiments

### Experiment 1: U-Net Baseline on BUSI

Use the Breast Ultrasound dataset (binary: tumor vs background):

```bash
python train.py --config configs/intro_to_datasets/busi.yaml
```

Expected: Dice ~0.75-0.80 for baseline ResNet50 + U-Net decoder.

### Experiment 2: Skip Connection Ablation

Compare different skip connections with the same encoder/decoder:

| Skip | YAML Override | Expected Effect |
|------|---------------|-----------------|
| `concat` | (baseline) | Standard U-Net behavior |
| `add` | `model.skip_connection.name=add` | Slightly less expressive |
| `cab` | `model.skip_connection.name=cab` | Channel attention may improve |
| `gating` | `model.skip_connection.name=gating` | Attention-gated skips |

### Experiment 3: Decoder Comparison

Same encoder, different decoders:

| Decoder | Characteristics |
|---------|----------------|
| `bilinear` | Simple, artifact-free |
| `deconv` | Learned upsampling |
| `unet` | Full U-Net decoder (double conv per stage) |
| `emcad` | Efficient multi-scale cascade decoder |

---

## 6. Further Reading

### Key Papers

| Paper | Year | Venue | Key Idea |
|-------|------|-------|----------|
| [U-Net](https://arxiv.org/abs/1505.04597) | 2015 | MICCAI | Original encoder-decoder + skip |
| [UNet++](https://arxiv.org/abs/1807.10165) | 2018 | DLMIA | Dense nested skip connections |
| [Attention U-Net](https://arxiv.org/abs/1804.03999) | 2018 | MIDL | Attention gates on skip |
| [UNet 3+](https://arxiv.org/abs/2004.08790) | 2020 | ICASSP | Full-scale dense skip |
| [scSE-UNet](https://arxiv.org/abs/1803.02522) | 2018 | MICCAI | Channel + spatial recalibration |
| [TransUNet](https://arxiv.org/abs/2102.04306) | 2021 | - | CNN encoder + Transformer at bottleneck |
| [Swin-UNet](https://arxiv.org/abs/2105.05537) | 2022 | ECCV Workshop | Pure Transformer U-shape |

### Related Documentation

- [Networks Guide](../models/networks.md) -- All 130 complete network architectures
- [Encoder Guide](../models/encoders.md) -- 177 encoders including U-Net variants
- [Decoder Guide](../models/decoders.md) -- 45 decoders with design rationale
- [Skip Connections](../models/skip_connections.md) -- 25 skip connection implementations
- [Bottlenecks](../models/bottlenecks.md) -- 17 bottleneck modules

---

[Previous: Introduction](01_introduction.md) | [Next: Data and Preprocessing](03_data.md)
