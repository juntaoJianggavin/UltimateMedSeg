# 模型总览

[English](README.md)

## 简介

本项目提供高度模块化的医学图像分割模型库，支持：

| 模块 | 数量 |
|---|---|
| 完整网络 | 130 |
| 编码器 | 177 |
| 解码器 | 45 |
| 跳跃连接 | 25 |
| 瓶颈层 | 17 |

## 模块化设计

项目采用 **encoder + decoder + skip connection + bottleneck** 四模块自由组合设计。你可以将任意注册的 encoder 与任意 decoder、skip connection、bottleneck 搭配使用，也可以直接使用预定义的完整网络架构。

```
输入图像 ──> [Encoder] ──> [Bottleneck] ──> [Decoder] ──> 分割输出
                                  |                 ^
                                  └── [Skip Conn] ──┘
```

## 目录索引

| 文档 | 内容 |
|---|---|
| [networks.md](networks.md) | 130 个完整网络架构 |
| [encoders.md](encoders.md) | 177 个编码器（含 Foundation 模型） |
| [decoders.md](decoders.md) | 45 个解码器 |
| [skip_connections.md](skip_connections.md) | 25 个跳跃连接 |
| [bottlenecks.md](bottlenecks.md) | 17 个瓶颈层 |

## YAML 配置示例

### 模式一：完整网络

使用 `architecture` 字段直接指定一个完整网络，无需配置 encoder/decoder/skip/bottleneck。

```yaml
model:
  num_classes: 9
  img_size: 224
  architecture: transunet
  transfer_learning_path: null  # 可选：完整模型检查点用于迁移学习
  encoder:
    in_channels: 3
    pretrained_path: null        # 可选：手动指定骨干权重路径
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

### 模式二：自由组合

分别配置 encoder、decoder、skip_connection、bottleneck，自由搭配。

```yaml
model:
  num_classes: 9
  img_size: 224
  transfer_learning_path: null  # 可选：完整模型检查点用于迁移学习
  encoder:
    name: timm_resnet50
    pretrained: true
    pretrained_path: null        # 可选：手动指定骨干权重路径
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
