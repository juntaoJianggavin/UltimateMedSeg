# Bottlenecks

[中文文档](bottlenecks_CN.md)

This project provides 17 bottleneck modules, placed between the deepest encoder layer and the decoder, to enhance feature representation.

## No-op (1)

| Name | Description | YAML |
|---|---|---|
| `none` | No bottleneck (pass-through) | [resnet50_none.yaml](../../configs/architectures/bottleneck_study/general/resnet50_none.yaml) |

## Basic (1)

| Name | Description | YAML |
|---|---|---|
| `basic` | Basic convolutional bottleneck | [resnet50_basic.yaml](../../configs/architectures/bottleneck_study/general/resnet50_basic.yaml) |

## Dilated Convolution (2)

| Name | Description | YAML |
|---|---|---|
| `aspp` | Atrous Spatial Pyramid Pooling | [resnet50_aspp.yaml](../../configs/architectures/bottleneck_study/general/resnet50_aspp.yaml) |
| `dense_aspp` | Dense ASPP | [resnet50_dense_aspp.yaml](../../configs/architectures/bottleneck_study/general/resnet50_dense_aspp.yaml) |

## Pooling (1)

| Name | Description | YAML |
|---|---|---|
| `ppm` | Pyramid Pooling Module | [resnet50_ppm.yaml](../../configs/architectures/bottleneck_study/general/resnet50_ppm.yaml) |

## Channel Attention (3)

| Name | Description | YAML |
|---|---|---|
| `se` | Squeeze-and-Excitation | [resnet50_se.yaml](../../configs/architectures/bottleneck_study/general/resnet50_se.yaml) |
| `eca` | Efficient Channel Attention | [resnet50_eca.yaml](../../configs/architectures/bottleneck_study/general/resnet50_eca.yaml) |
| `cbam` | CBAM Channel + Spatial Attention | [resnet50_cbam.yaml](../../configs/architectures/bottleneck_study/general/resnet50_cbam.yaml) |

## Spatial Attention (2)

| Name | Description | YAML |
|---|---|---|
| `coord_attn` | Coordinate Attention | [resnet50_coord_attn.yaml](../../configs/architectures/bottleneck_study/general/resnet50_coord_attn.yaml) |
| `spatial_channel` | Spatial-Channel joint attention | [resnet50_spatial_channel.yaml](../../configs/architectures/bottleneck_study/general/resnet50_spatial_channel.yaml) |

## Hybrid Attention (3)

| Name | Description | YAML |
|---|---|---|
| `dual_attention` | Dual Attention (position + channel) | [resnet50_dual_attention.yaml](../../configs/architectures/bottleneck_study/general/resnet50_dual_attention.yaml) |
| `acmix` | ACmix attention-convolution mixture | [resnet50_acmix.yaml](../../configs/architectures/bottleneck_study/general/resnet50_acmix.yaml) |
| `gated_attn` | Gated attention | [resnet50_gated_attn.yaml](../../configs/architectures/bottleneck_study/general/resnet50_gated_attn.yaml) |

## Transformer (1)

| Name | Description | YAML |
|---|---|---|
| `transformer` | Transformer bottleneck | [resnet50_transformer.yaml](../../configs/architectures/bottleneck_study/general/resnet50_transformer.yaml) |

## Coordinate Convolution (1)

| Name | Description | YAML |
|---|---|---|
| `coordconv` | CoordConv (coordinate convolution) | [resnet50_coordconv.yaml](../../configs/architectures/bottleneck_study/general/resnet50_coordconv.yaml) |

## Mixture of Experts (1)

| Name | Description | YAML |
|---|---|---|
| `moe` | Mixture of Experts bottleneck | [resnet50_moe.yaml](../../configs/architectures/bottleneck_study/general/resnet50_moe.yaml) |

## LLM-enhanced (1)

| Name | Description | YAML |
|---|---|---|
| `llm4seg` | LLM-enhanced bottleneck for segmentation | [resnet50_llm4seg.yaml](../../configs/architectures/bottleneck_study/general/resnet50_llm4seg.yaml) |

---

## YAML Usage Example

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
    name: aspp            # choose any bottleneck

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
