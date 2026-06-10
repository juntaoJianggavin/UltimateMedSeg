# Skip Connections

[中文文档](skip_connections_CN.md)

This project provides **25** skip connection modules in 5 categories, transferring encoder features to the decoder at each level.

---

## Basic (2)

| Key | Description | YAML |
|---|---|---|
| `concat` | Channel-wise concatenation (default) | [resnet50_concat.yaml](../../configs/architectures/skip_study/general/resnet50_concat.yaml) |
| `dense` | Dense skip (UNet++ style nested connections) | [resnet50_dense.yaml](../../configs/architectures/skip_study/general/resnet50_dense.yaml) |

## Attention (10)

| Key | Source | Description | YAML |
|---|---|---|---|
| `attention_gate` | Attention U-Net (Oktay 2018) | Attention gate | [resnet50_attention_gate.yaml](../../configs/architectures/skip_study/general/resnet50_attention_gate.yaml) |
| `cab` | — | Channel Attention Bridge | [resnet50_cab.yaml](../../configs/architectures/skip_study/general/resnet50_cab.yaml) |
| `sab` | — | Spatial Attention Bridge | [resnet50_sab.yaml](../../configs/architectures/skip_study/general/resnet50_sab.yaml) |
| `scse` | Roy et al., TMI 2019 | Spatial-Channel Squeeze & Excitation | [resnet50_scse.yaml](../../configs/architectures/skip_study/general/resnet50_scse.yaml) |
| `cbam` | Woo et al., ECCV 2018 | Conv Block Attention Module | [resnet50_cbam.yaml](../../configs/architectures/skip_study/general/resnet50_cbam.yaml) |
| `gating` | — | Gating mechanism | [resnet50_gating.yaml](../../configs/architectures/skip_study/general/resnet50_gating.yaml) |
| `gru_gate` | — | GRU-based gating | [resnet50_gru_gate.yaml](../../configs/architectures/skip_study/general/resnet50_gru_gate.yaml) |
| `gab` | EGE-UNet, MICCAI 2023 Workshop, [GitHub](https://github.com/JCruan519/EGE-UNet) | Group Aggregation Bridge | [resnet50_gab.yaml](../../configs/architectures/skip_study/general/resnet50_gab.yaml) |
| `sc_att_bridge` | MALUNet, BIBM 2022, [GitHub](https://github.com/JCruan519/MALUNet) | Spatial-Channel Attention Bridge | [resnet50_sc_att_bridge.yaml](../../configs/architectures/skip_study/general/resnet50_sc_att_bridge.yaml) |
| `ta_mosc` | UTANet, AAAI 2025, [GitHub](https://github.com/AshleyLuo001/UTANet) | Task-Adaptive Mixture of Skip Connections | [resnet50_ta_mosc.yaml](../../configs/architectures/skip_study/general/resnet50_ta_mosc.yaml) |

## Transformer (5)

| Key | Source | Description | YAML |
|---|---|---|---|
| `cross_attn` | — | Cross-attention (decoder Q × encoder KV) | [resnet50_cross_attn.yaml](../../configs/architectures/skip_study/general/resnet50_cross_attn.yaml) |
| `transformer_fusion` | — | Transformer feature fusion | [resnet50_transformer_fusion.yaml](../../configs/architectures/skip_study/general/resnet50_transformer_fusion.yaml) |
| `aggregation_attention` | — | Aggregation attention | [resnet50_aggregation_attention.yaml](../../configs/architectures/skip_study/general/resnet50_aggregation_attention.yaml) |
| `missformer_bridge` | MISSFormer, 2022, [GitHub](https://github.com/ZhifangDeng/MISSFormer) | MISSFormer bridge module | [resnet50_missformer_bridge.yaml](../../configs/architectures/skip_study/general/resnet50_missformer_bridge.yaml) |
| `uctrans` | UCTransNet, AAAI 2022, [GitHub](https://github.com/McGregorWwww/UCTransNet) | Channel-wise Cross Transformer | [resnet50_uctrans.yaml](../../configs/architectures/skip_study/general/resnet50_uctrans.yaml) |

## Mamba (1)

| Key | Source | Description | YAML |
|---|---|---|---|
| `skvmpp` | SK-VM++, BSPC 2025, [GitHub](https://github.com/wurenkai/SK-VMPlusPlus) | Mamba SS2D-assisted skip (Pyramid Vision Mamba Layer) | [resnet50_skvmpp.yaml](../../configs/architectures/skip_study/general/resnet50_skvmpp.yaml) |

## Fusion — CNN Fusion (6)

| Key | Source | Description | YAML |
|---|---|---|---|
| `bifusion` | TransFuse style | Bi-directional fusion | [resnet50_bifusion.yaml](../../configs/architectures/skip_study/general/resnet50_bifusion.yaml) |
| `deformable` | — | Deformable convolution fusion | [resnet50_deformable.yaml](../../configs/architectures/skip_study/general/resnet50_deformable.yaml) |
| `multiscale` | — | Multi-scale fusion | [resnet50_multiscale.yaml](../../configs/architectures/skip_study/general/resnet50_multiscale.yaml) |
| `feature_refine` | — | Feature refinement with CBAM | [resnet50_feature_refine.yaml](../../configs/architectures/skip_study/general/resnet50_feature_refine.yaml) |
| `ccm` | — | Cross Channel Module | [resnet50_ccm.yaml](../../configs/architectures/skip_study/general/resnet50_ccm.yaml) |
| `sdi` | U-Net V2, ISBI 2025, [GitHub](https://github.com/yaoppeng/U-Net_v2) | Scale-Diverse Integration | [resnet50_sdi.yaml](../../configs/architectures/skip_study/general/resnet50_sdi.yaml) |

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
