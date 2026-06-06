# 解码器

[English](decoders.md)

本项目提供 40 个解码器模块，按类别分组如下。

## 基础 (4)

基础上采样解码器。

| 名称 | 说明 |
|---|---|
| `unet` | 标准 UNet 反卷积解码器 |
| `bilinear` | 双线性插值上采样 |
| `deconv` | 转置卷积上采样 |
| `dw_sep` | 深度可分离卷积解码器 |

## 密集连接 (2)

密集连接解码器。

| 名称 | 说明 |
|---|---|
| `unetpp` | UNet++ 密集嵌套解码器 |
| `unet3plus` | UNet 3+ 全尺度跳跃连接解码器 |

## 级联 (8)

级联解码器，逐步细化分割结果。

| 名称 | 说明 |
|---|---|
| `cascade` | CASCADE 级联解码器 |
| `cascade_full` | CASCADE 完整版解码器 |
| `cascade_emcad` | CASCADE + EMCAD 混合 |
| `cfm` | CFM 级联特征融合 |
| `emcad` | EMCAD 高效多尺度级联注意力解码器 |
| `edldnet` | EDLDNet 解码器 |
| `gcascade` | G-CASCADE（add 融合） |
| `gcascade_cat` | G-CASCADE（concat 融合） |

## 金字塔 (1)

金字塔聚合解码器。

| 名称 | 说明 |
|---|---|
| `upernet` | UPerNet 统一感知金字塔 |

## MLP (2)

MLP 解码器。

| 名称 | 说明 |
|---|---|
| `mlp` | 通用 MLP 解码器 |
| `segformer` | SegFormer 风格 MLP 解码器 |

## 特定网络专属 (12)

特定网络专属解码器。

| 名称 | 对应网络 |
|---|---|
| `cfanet` | CFA-Net |
| `dcsaunet` | DCSAU-Net |
| `rwkv_unet` | RWKV-UNet |
| `kiunet` | KiU-Net |
| `transunet` | TransUNet (CUP) |
| `fatnet` | FAT-Net |
| `h2former` | H2Former |
| `hiformer` | HiFormer |
| `missformer` | MISSFormer |
| `scaleformer` | ScaleFormer |
| `malunet` | MALUNet |
| `ege_unet` | EGE-UNet |

## Transformer (5)

Transformer 解码器。

| 名称 | 说明 |
|---|---|
| `daeformer` | DAEFormer 解码器 |
| `mtunet` | MT-UNet 解码器 |
| `nnformer` | nnFormer 解码器 |
| `swinunet` | Swin-UNet 解码器 |
| `uctransnet` | UCTransNet 解码器 |

## 注意力 (3)

注意力机制解码器。

| 名称 | 说明 |
|---|---|
| `attention` | 注意力门控解码器 |
| `ham` | HAM 混合注意力 |
| `lawin` | Lawin 大窗口注意力 |

## Mamba (1)

| 名称 | 说明 |
|---|---|
| `vmunet` | VM-UNet Mamba 解码器 |

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
    name: emcad          # 选择任意解码器
    params: {}
  skip_connection:
    name: concat
  bottleneck:
    name: none

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
