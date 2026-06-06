# Bottlenecks

[中文文档](bottlenecks_CN.md)

This project provides 17 bottleneck modules, placed between the deepest encoder layer and the decoder, to enhance feature representation.

## No-op (1)

| Name | Description |
|---|---|
| `none` | No bottleneck (pass-through) |

## Basic (1)

| Name | Description |
|---|---|
| `basic` | Basic convolutional bottleneck |

## Dilated Convolution (2)

| Name | Description |
|---|---|
| `aspp` | Atrous Spatial Pyramid Pooling |
| `dense_aspp` | Dense ASPP |

## Pooling (1)

| Name | Description |
|---|---|
| `ppm` | Pyramid Pooling Module |

## Channel Attention (3)

| Name | Description |
|---|---|
| `se` | Squeeze-and-Excitation |
| `eca` | Efficient Channel Attention |
| `cbam` | CBAM Channel + Spatial Attention |

## Spatial Attention (2)

| Name | Description |
|---|---|
| `coord_attn` | Coordinate Attention |
| `spatial_channel` | Spatial-Channel joint attention |

## Hybrid Attention (3)

| Name | Description |
|---|---|
| `dual_attention` | Dual Attention (position + channel) |
| `acmix` | ACmix attention-convolution mixture |
| `gated_attn` | Gated attention |

## Transformer (1)

| Name | Description |
|---|---|
| `transformer` | Transformer bottleneck |

## Coordinate Convolution (1)

| Name | Description |
|---|---|
| `coordconv` | CoordConv (coordinate convolution) |

## Mixture of Experts (1)

| Name | Description |
|---|---|
| `moe` | Mixture of Experts bottleneck |

## LLM-enhanced (1)

| Name | Description |
|---|---|
| `llm4seg` | LLM-enhanced bottleneck for segmentation |

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
