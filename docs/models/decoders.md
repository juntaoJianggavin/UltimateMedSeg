# Decoders

[中文文档](decoders_CN.md)

This project provides 40 decoder modules, grouped by category as follows.

## Basic (4)

Basic upsampling decoders.

| Name | Description |
|---|---|
| `unet` | Standard UNet decoder with conv + upsample |
| `bilinear` | Bilinear interpolation upsampling |
| `deconv` | Transposed convolution upsampling |
| `dw_sep` | Depthwise separable convolution decoder |

## Dense (2)

Dense connection decoders.

| Name | Description |
|---|---|
| `unetpp` | UNet++ dense nested decoder |
| `unet3plus` | UNet 3+ full-scale skip connection decoder |

## Cascade (8)

Cascade decoders that progressively refine segmentation.

| Name | Description |
|---|---|
| `cascade` | CASCADE decoder |
| `cascade_full` | CASCADE full decoder |
| `cascade_emcad` | CASCADE + EMCAD hybrid |
| `cfm` | Cascaded Feature Merging |
| `emcad` | Efficient Multi-scale Cascaded Attention Decoder |
| `edldnet` | EDLDNet decoder |
| `gcascade` | G-CASCADE with add fusion |
| `gcascade_cat` | G-CASCADE with concat fusion |

## Pyramid (1)

Pyramid aggregation decoder.

| Name | Description |
|---|---|
| `upernet` | UPerNet Unified Perceptual Parsing |

## MLP (2)

MLP-based decoders.

| Name | Description |
|---|---|
| `mlp` | Generic MLP decoder |
| `segformer` | SegFormer-style MLP decoder |

## Specific (12)

Architecture-specific decoders.

| Name | Associated Network |
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

Transformer-based decoders.

| Name | Description |
|---|---|
| `daeformer` | DAEFormer decoder |
| `mtunet` | MT-UNet decoder |
| `nnformer` | nnFormer decoder |
| `swinunet` | Swin-UNet decoder |
| `uctransnet` | UCTransNet decoder |

## Attention (3)

Attention-based decoders.

| Name | Description |
|---|---|
| `attention` | Attention gate decoder |
| `ham` | Hybrid Attention Module |
| `lawin` | Large Window Attention decoder |

## Mamba (1)

| Name | Description |
|---|---|
| `vmunet` | VM-UNet Mamba decoder |

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
    name: emcad          # choose any decoder
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
