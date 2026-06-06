# Skip Connections

[中文文档](skip_connections_CN.md)

This project provides **25** skip connection modules in 5 categories, transferring encoder features to the decoder at each level.

---

## Basic (2)

| Key | Description |
|---|---|
| `concat` | Channel-wise concatenation (default) |
| `dense` | Dense skip (UNet++ style nested connections) |

## Attention (10)

| Key | Source | Description |
|---|---|---|
| `attention_gate` | Attention U-Net (Oktay 2018) | Attention gate |
| `cab` | — | Channel Attention Bridge |
| `sab` | — | Spatial Attention Bridge |
| `scse` | Roy et al., TMI 2019 | Spatial-Channel Squeeze & Excitation |
| `cbam` | Woo et al., ECCV 2018 | Conv Block Attention Module |
| `gating` | — | Gating mechanism |
| `gru_gate` | — | GRU-based gating |
| `gab` | EGE-UNet, MICCAI 2023 Workshop, [GitHub](https://github.com/JCruan519/EGE-UNet) | Group Aggregation Bridge |
| `sc_att_bridge` | MALUNet, BIBM 2022, [GitHub](https://github.com/JCruan519/MALUNet) | Spatial-Channel Attention Bridge |
| `ta_mosc` | UTANet, AAAI 2025, [GitHub](https://github.com/AshleyLuo001/UTANet) | Task-Adaptive Mixture of Skip Connections |

## Transformer (5)

| Key | Source | Description |
|---|---|---|
| `cross_attn` | — | Cross-attention (decoder Q × encoder KV) |
| `transformer_fusion` | — | Transformer feature fusion |
| `aggregation_attention` | — | Aggregation attention |
| `missformer_bridge` | MISSFormer, 2022, [GitHub](https://github.com/ZhifangDeng/MISSFormer) | MISSFormer bridge module |
| `uctrans` | UCTransNet, AAAI 2022, [GitHub](https://github.com/McGregorWwww/UCTransNet) | Channel-wise Cross Transformer |

## Mamba (1)

| Key | Source | Description |
|---|---|---|
| `skvmpp` | SK-VM++, BSPC 2025, [GitHub](https://github.com/wurenkai/SK-VMPlusPlus) | Mamba SS2D-assisted skip (Pyramid Vision Mamba Layer) |

## Fusion — CNN Fusion (6)

| Key | Source | Description |
|---|---|---|
| `bifusion` | TransFuse style | Bi-directional fusion |
| `deformable` | — | Deformable convolution fusion |
| `multiscale` | — | Multi-scale fusion |
| `feature_refine` | — | Feature refinement with CBAM |
| `ccm` | — | Cross Channel Module |
| `sdi` | U-Net V2, ISBI 2025, [GitHub](https://github.com/yaoppeng/U-Net_v2) | Scale-Diverse Integration |

---

## YAML Usage

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
  skip_connection:
    name: skvmpp           # choose any of the 25
    params: {}
  bottleneck:
    name: none

data:
  type: generic
  img_size: 224
  train_dir: ./data/YourDataset/train
  val_dir: ./data/YourDataset/val

training:
  epochs: 200
  batch_size: 8
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
  scheduler:
    name: cosine
    min_lr: 0.000001
```

### Recommended Combinations

| Scenario | Recommended skip | Reason |
|---|---|---|
| Baseline | `concat` | Simplest, no extra params |
| Attention boost | `attention_gate` or `scse` | Classic and effective |
| Transformer encoder | `cross_attn` or `uctrans` | Leverages attention |
| Mamba encoder | `skvmpp` | Mamba SS2D enhances skip features |
| Lightweight | `add` or `gating` | Minimal overhead |
| Boundary focus | `ta_mosc` | Task-adaptive, AAAI 2025 |
