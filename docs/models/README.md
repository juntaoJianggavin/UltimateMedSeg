# Model Overview

[中文文档](README_CN.md)

## Introduction

This project provides a highly modular medical image segmentation model zoo, supporting:

| Module | Count |
|---|---|
| Complete Networks | 132 |
| Encoders | 178 |
| Decoders | 45 |
| Skip Connections | 25 |
| Bottlenecks | 17 |

## Modular Design

The project uses a four-module free-combination design: **encoder + decoder + skip connection + bottleneck**. You can freely mix-and-match any registered encoder with any decoder, skip connection, and bottleneck, or use a pre-defined complete network architecture directly.

```
Input Image ──> [Encoder] ──> [Bottleneck] ──> [Decoder] ──> Segmentation Output
                                  |                 ^
                                  └── [Skip Conn] ──┘
```

## Documentation Index

| Document | Content |
|---|---|
| [networks.md](networks.md) | 132 complete network architectures |
| [encoders.md](encoders.md) | 178 encoders (incl. foundation models) |
| [decoders.md](decoders.md) | 45 decoders |
| [skip_connections.md](skip_connections.md) | 25 skip connections |
| [bottlenecks.md](bottlenecks.md) | 17 bottlenecks |

## YAML Configuration Examples

### Mode 1: Complete Architecture

Use the `architecture` field to specify a complete network directly, without configuring encoder/decoder/skip/bottleneck.

```yaml
model:
  num_classes: 9
  img_size: 224
  architecture: transunet
  transfer_learning_path: null  # optional: full-model checkpoint for transfer learning
  encoder:
    in_channels: 3
    pretrained_path: null        # optional: manual backbone weight override
  arch_params: {}

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
    weight_decay: 0.01
  scheduler:
    name: cosine
    min_lr: 0.000001
```

### Mode 2: Encoder + Decoder Combination

Configure encoder, decoder, skip_connection, and bottleneck separately for free combination.

```yaml
model:
  num_classes: 9
  img_size: 224
  transfer_learning_path: null  # optional: full-model checkpoint for transfer learning
  encoder:
    name: timm_resnet50
    pretrained: true
    pretrained_path: null        # optional: manual backbone weight override
    in_channels: 3
    params: {}
  decoder:
    name: emcad
    params: {}
  skip_connection:
    name: cab
    params: {}
  bottleneck:
    name: aspp

data:
  type: synapse
  img_size: 224
  train_dir: ./data/Synapse/train_npz
  test_dir: ./data/Synapse/test_vol_h5
  train_list: ./data/Synapse/lists/lists_Synapse/train.txt
  test_list: ./data/Synapse/lists/lists_Synapse/test_vol.txt

training:
  epochs: 200
  batch_size: 24
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
    lr: 0.01
    weight_decay: 0.0001
  scheduler:
    name: cosine
    min_lr: 0.000001
```
