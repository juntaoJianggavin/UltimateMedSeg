# Decoders

[中文文档](decoders_CN.md)

This project provides 40 decoder modules, grouped by category as follows.

## Basic (4)

Basic upsampling decoders.

| Name | Description | YAML |
|---|---|---|
| `unet` | Standard UNet decoder with conv + upsample | [unet_basic.yaml](../../configs/architectures/combinations/general/unet_basic.yaml) |
| `bilinear` | Bilinear interpolation upsampling | [basic_bilinear.yaml](../../configs/architectures/decoder_study/general/basic_bilinear.yaml) |
| `deconv` | Transposed convolution upsampling | [deconv_resnet34.yaml](../../configs/architectures/combinations/general/deconv_resnet34.yaml) |
| `dw_sep` | Depthwise separable convolution decoder | [dwsep_resnet34.yaml](../../configs/architectures/combinations/general/dwsep_resnet34.yaml) |

## Dense (2)

Dense connection decoders.

| Name | Description | YAML |
|---|---|---|
| `unetpp` | UNet++ dense nested decoder | [basic_unetpp.yaml](../../configs/architectures/decoder_study/general/basic_unetpp.yaml) |
| `unet3plus` | UNet 3+ full-scale skip connection decoder | [basic_unet3plus.yaml](../../configs/architectures/decoder_study/general/basic_unet3plus.yaml) |

## Cascade (10)

Cascade decoders that progressively refine segmentation.

| Name | Description | YAML |
|---|---|---|
| `cascade` | CASCADE decoder | [cascade_resnet34.yaml](../../configs/architectures/combinations/general/cascade_resnet34.yaml) |
| `cascade_full` | CASCADE full decoder | [transunet_cascade_full.yaml](../../configs/architectures/combinations/general/transunet_cascade_full.yaml) |
| `cascade_emcad` | CASCADE + EMCAD hybrid | [mednext_cascade_emcad.yaml](../../configs/architectures/combinations/general/mednext_cascade_emcad.yaml) |
| `cfm` | Cascaded Feature Merging | [mednext_cfm.yaml](../../configs/architectures/combinations/general/mednext_cfm.yaml) |
| `emcad` | Efficient Multi-scale Cascaded Attention Decoder | [mednext_emcad.yaml](../../configs/architectures/combinations/general/mednext_emcad.yaml) |
| `edldnet` | EDLDNet decoder | [pvtv2_edldnet.yaml](../../configs/architectures/combinations/general/pvtv2_edldnet.yaml) |
| `gcascade` | G-CASCADE with add fusion | [pvtv2_gcascade.yaml](../../configs/architectures/combinations/general/pvtv2_gcascade.yaml) |
| `gcascade_cat` | G-CASCADE with concat fusion | [basic_gcascade_cat.yaml](../../configs/architectures/decoder_study/general/basic_gcascade_cat.yaml) |
| `merit_add` | MERIT decoder (add fusion) | [basic_merit_add.yaml](../../configs/architectures/decoder_study/general/basic_merit_add.yaml) |
| `merit_cat` | MERIT decoder (concat fusion) | [basic_merit_cat.yaml](../../configs/architectures/decoder_study/general/basic_merit_cat.yaml) |

## Pyramid (1)

Pyramid aggregation decoder.

| Name | Description | YAML |
|---|---|---|
| `upernet` | UPerNet Unified Perceptual Parsing | [basic_upernet.yaml](../../configs/architectures/decoder_study/general/basic_upernet.yaml) |

## MLP (2)

MLP-based decoders.

| Name | Description | YAML |
|---|---|---|
| `mlp` | Generic MLP decoder | [mlp_resnet34.yaml](../../configs/architectures/combinations/general/mlp_resnet34.yaml) |
| `segformer` | SegFormer-style MLP decoder | [swinunet_segformer.yaml](../../configs/architectures/combinations/general/swinunet_segformer.yaml) |

## Specific (12)

Architecture-specific decoders.

| Name | Associated Network | YAML |
|---|---|---|
| `cfanet` | CFA-Net | [basic_cfanet.yaml](../../configs/architectures/decoder_study/general/basic_cfanet.yaml) |
| `dcsaunet` | DCSAU-Net | [basic_dcsaunet.yaml](../../configs/architectures/decoder_study/general/basic_dcsaunet.yaml) |
| `rwkv_unet` | RWKV-UNet | [rwkv_unet.yaml](../../configs/architectures/combinations/general/rwkv_unet.yaml) |
| `kiunet` | KiU-Net | [basic_kiunet.yaml](../../configs/architectures/decoder_study/general/basic_kiunet.yaml) |
| `transunet` | TransUNet (CUP) | [transunet_cascade_full.yaml](../../configs/architectures/combinations/general/transunet_cascade_full.yaml) |
| `fatnet` | FAT-Net | [basic_fatnet.yaml](../../configs/architectures/decoder_study/general/basic_fatnet.yaml) |
| `h2former` | H2Former | [basic_h2former.yaml](../../configs/architectures/decoder_study/general/basic_h2former.yaml) |
| `hiformer` | HiFormer | [hiformer_cascade.yaml](../../configs/architectures/combinations/general/hiformer_cascade.yaml) |
| `missformer` | MISSFormer | [basic_missformer.yaml](../../configs/architectures/decoder_study/general/basic_missformer.yaml) |
| `scaleformer` | ScaleFormer | [scaleformer_cascade_full.yaml](../../configs/architectures/combinations/general/scaleformer_cascade_full.yaml) |
| `malunet` | MALUNet | [basic_malunet.yaml](../../configs/architectures/decoder_study/general/basic_malunet.yaml) |
| `ege_unet` | EGE-UNet | [basic_ege_unet.yaml](../../configs/architectures/decoder_study/general/basic_ege_unet.yaml) |

## Transformer (5)

Transformer-based decoders.

| Name | Description | YAML |
|---|---|---|
| `daeformer` | DAEFormer decoder | [daeformer_emcad.yaml](../../configs/architectures/combinations/general/daeformer_emcad.yaml) |
| `mtunet` | MT-UNet decoder | [basic_mtunet.yaml](../../configs/architectures/decoder_study/general/basic_mtunet.yaml) |
| `nnformer` | nnFormer decoder | [mednext_nnformer.yaml](../../configs/architectures/combinations/general/mednext_nnformer.yaml) |
| `swinunet` | Swin-UNet decoder | [swinunet_segformer.yaml](../../configs/architectures/combinations/general/swinunet_segformer.yaml) |
| `uctransnet` | UCTransNet decoder | [uctransnet.yaml](../../configs/architectures/combinations/general/uctransnet.yaml) |

## Attention (3)

Attention-based decoders.

| Name | Description | YAML |
|---|---|---|
| `attention` | Attention gate decoder | [attention_unet_basic.yaml](../../configs/architectures/combinations/general/attention_unet_basic.yaml) |
| `ham` | Hybrid Attention Module | [ham_resnet34.yaml](../../configs/architectures/combinations/general/ham_resnet34.yaml) |
| `lawin` | Large Window Attention decoder | [lawin_resnet50.yaml](../../configs/architectures/combinations/general/lawin_resnet50.yaml) |

## Mamba (1)

| Name | Description | YAML |
|---|---|---|
| `vmunet` | VM-UNet Mamba decoder | [vm_unet.yaml](../../configs/architectures/networks/general/vm_unet.yaml) |

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
