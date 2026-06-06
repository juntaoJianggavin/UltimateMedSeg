# 瓶颈层

[English](bottlenecks.md)

本项目提供 17 个瓶颈层模块，位于 encoder 最深层与 decoder 之间，用于增强特征表达。

## 无操作 (1)

| 名称 | 说明 |
|---|---|
| `none` | 不使用瓶颈层（直通） |

## 基础 (1)

| 名称 | 说明 |
|---|---|
| `basic` | 基础卷积瓶颈 |

## 空洞卷积 (2)

| 名称 | 说明 |
|---|---|
| `aspp` | 空洞空间金字塔池化 (ASPP, DeepLab) |
| `dense_aspp` | 密集 ASPP |

## 池化 (1)

| 名称 | 说明 |
|---|---|
| `ppm` | 金字塔池化模块 (PPM, PSPNet) |

## 通道注意力 (3)

| 名称 | 说明 |
|---|---|
| `se` | Squeeze-and-Excitation |
| `eca` | 高效通道注意力 |
| `cbam` | CBAM 通道+空间注意力 |

## 空间注意力 (2)

| 名称 | 说明 |
|---|---|
| `coord_attn` | 坐标注意力 |
| `spatial_channel` | 空间-通道联合注意力 |

## 混合注意力 (3)

| 名称 | 说明 |
|---|---|
| `dual_attention` | 双注意力 (DANet)：位置 + 通道 |
| `acmix` | ACmix 注意力卷积混合 |
| `gated_attn` | 门控注意力 |

## Transformer (1)

| 名称 | 说明 |
|---|---|
| `transformer` | Transformer 瓶颈 |

## 坐标卷积 (1)

| 名称 | 说明 |
|---|---|
| `coordconv` | CoordConv 坐标卷积 |

## 专家混合 (1)

| 名称 | 说明 |
|---|---|
| `moe` | MoE 专家混合瓶颈 |

## LLM 增强 (1)

| 名称 | 说明 |
|---|---|
| `llm4seg` | LLM4Seg 大语言模型增强瓶颈 |

---

## YAML 使用示例

```yaml
model:
  num_classes: 9
  img_size: 224
  encoder:
    name: timm_resnet50
    pretrained: true
    in_channels: 3
  decoder:
    name: bilinear
    params: {}
  skip_connection:
    name: concat
    params: {}
  bottleneck:
    name: aspp            # 选择任意瓶颈层

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
