# Decoders (`medseg/decoders/`)

[中文文档](README_CN.md)

U-shape upsampling decoders that consume multi-scale encoder features and produce the final segmentation map.

## Registered Decoders

| Registry Key | Source File | Description |
|---|---|---|
| `bilinear` | `bilinear_decoder.py` | Simple bilinear upsample + 1×1 conv (fastest baseline) |
| `deconv` | `deconv_decoder.py` | Transposed-convolution decoder with skip concat |
| `attention` | `attention_decoder.py` | Attention gate at each skip level (Attention U-Net) |
| `cascade` | `cascade_decoder.py` | Cascaded refinement decoder |
| `dw_sep` | `dw_sep_decoder.py` | Depthwise-separable conv decoder (lightweight) |
| `emcad` | `emcad_decoder.py` | EMCad multi-scale aggregation decoder |
| `mlp` | `mlp_decoder.py` | MLP-based decoder (SegFormer-style) |
| `lawin` | `lawin_decoder.py` | Lawin multi-scale context decoder |
| `ham` | `ham_decoder.py` | HAM (Hierarchical Aggregation Module) decoder |
| `upernet` | `upernet_decoder.py` | UPerNet FPN+PPM decoder |
| `unetpp` | `unetpp_decoder.py` | UNet++ dense skip decoder |
| `transunet` | `transunet_decoder.py` | TransUNet cascaded up-conv decoder |
| `swinunet` | `swinunet_decoder.py` | Swin-UNet patch-expanding decoder |
| `missformer` | `missformer_decoder.py` | MISSFormer frequency-aware decoder |
| `dcsaunet` | `dcsaunet_decoder.py` | DCSAU-Net dual-path decoder |
| `rwkv_unet` | `rwkv_unet_decoder.py` | RWKV sequence-model decoder |
| `vmunet` | `vmunet_decoder.py` | VM-UNet Mamba-SSM decoder |

## Usage in YAML Config

```yaml
model:
  decoder:
    name: attention         # any registered key
    params:
      encoder_channels: [64, 256, 512, 1024, 2048]
      decoder_channels: [256, 128, 64, 32, 16]
```
