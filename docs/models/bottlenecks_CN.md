# 瓶颈层

[English](bottlenecks.md)

本项目提供 17 个瓶颈层模块，位于 encoder 最深层与 decoder 之间，用于增强特征表达。

## 无操作 (1)

| 名称 | 说明 | YAML |
|---|---|---|
| `none` | 不使用瓶颈层（直通） | [resnet50_none.yaml](../../configs/architectures/bottleneck_study/general/resnet50_none.yaml) |

## 基础 (1)

| 名称 | 说明 | YAML |
|---|---|---|
| `basic` | 基础卷积瓶颈 | [resnet50_basic.yaml](../../configs/architectures/bottleneck_study/general/resnet50_basic.yaml) |

## 空洞卷积 (2)

| 名称 | 说明 | YAML |
|---|---|---|
| `aspp` | 空洞空间金字塔池化 (ASPP, DeepLab) | [resnet50_aspp.yaml](../../configs/architectures/bottleneck_study/general/resnet50_aspp.yaml) |
| `dense_aspp` | 密集 ASPP | [resnet50_dense_aspp.yaml](../../configs/architectures/bottleneck_study/general/resnet50_dense_aspp.yaml) |

## 池化 (1)

| 名称 | 说明 | YAML |
|---|---|---|
| `ppm` | 金字塔池化模块 (PPM, PSPNet) | [resnet50_ppm.yaml](../../configs/architectures/bottleneck_study/general/resnet50_ppm.yaml) |

## 通道注意力 (3)

| 名称 | 说明 | YAML |
|---|---|---|
| `se` | Squeeze-and-Excitation | [resnet50_se.yaml](../../configs/architectures/bottleneck_study/general/resnet50_se.yaml) |
| `eca` | 高效通道注意力 | [resnet50_eca.yaml](../../configs/architectures/bottleneck_study/general/resnet50_eca.yaml) |
| `cbam` | CBAM 通道+空间注意力 | [resnet50_cbam.yaml](../../configs/architectures/bottleneck_study/general/resnet50_cbam.yaml) |

## 空间注意力 (2)

| 名称 | 说明 | YAML |
|---|---|---|
| `coord_attn` | 坐标注意力 | [resnet50_coord_attn.yaml](../../configs/architectures/bottleneck_study/general/resnet50_coord_attn.yaml) |
| `spatial_channel` | 空间-通道联合注意力 | [resnet50_spatial_channel.yaml](../../configs/architectures/bottleneck_study/general/resnet50_spatial_channel.yaml) |

## 混合注意力 (3)

| 名称 | 说明 | YAML |
|---|---|---|
| `dual_attention` | 双注意力 (DANet)：位置 + 通道 | [resnet50_dual_attention.yaml](../../configs/architectures/bottleneck_study/general/resnet50_dual_attention.yaml) |
| `acmix` | ACmix 注意力卷积混合 | [resnet50_acmix.yaml](../../configs/architectures/bottleneck_study/general/resnet50_acmix.yaml) |
| `gated_attn` | 门控注意力 | [resnet50_gated_attn.yaml](../../configs/architectures/bottleneck_study/general/resnet50_gated_attn.yaml) |

## Transformer (1)

| 名称 | 说明 | YAML |
|---|---|---|
| `transformer` | Transformer 瓶颈 | [resnet50_transformer.yaml](../../configs/architectures/bottleneck_study/general/resnet50_transformer.yaml) |

## 坐标卷积 (1)

| 名称 | 说明 | YAML |
|---|---|---|
| `coordconv` | CoordConv 坐标卷积 | [resnet50_coordconv.yaml](../../configs/architectures/bottleneck_study/general/resnet50_coordconv.yaml) |

## 专家混合 (1)

| 名称 | 说明 | YAML |
|---|---|---|
| `moe` | MoE 专家混合瓶颈 | [resnet50_moe.yaml](../../configs/architectures/bottleneck_study/general/resnet50_moe.yaml) |

## LLM 增强 (1)

| 名称 | 说明 | YAML |
|---|---|---|
| `llm4seg` | LLM4Seg 大语言模型增强瓶颈 | [resnet50_llm4seg.yaml](../../configs/architectures/bottleneck_study/general/resnet50_llm4seg.yaml) |

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
