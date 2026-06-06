# Bottlenecks (`medseg/bottlenecks/`)

[中文文档](README_CN.md)

Plug-in modules placed between the encoder and decoder at the deepest feature level. They enrich the most compressed representation before the decoder upsamples.

## Registered Bottlenecks

| Registry Key | Source File | Description |
|---|---|---|
| `none` | `none_bottleneck.py` | Identity pass-through (no bottleneck) |
| `basic` | `basic_bottleneck.py` | Two 3×3 conv + residual (default) |
| `aspp` | `aspp_bottleneck.py` | Atrous Spatial Pyramid Pooling (DeepLabV3) |
| `dense_aspp` | `dense_aspp_bottleneck.py` | DenseASPP: cascaded atrous convolutions |
| `ppm` | `ppm_bottleneck.py` | Pyramid Pooling Module (PSPNet) |
| `transformer` | `transformer_bottleneck.py` | Transformer self-attention block |
| `se` | `se_bottleneck.py` | Squeeze-and-Excitation channel attention (CVPR 2018) |
| `dual_attention` | `da_bottleneck.py` | Dual Attention: position + channel (DANet) |
| `cbam` | `cbam_bottleneck.py` | CBAM: Convolutional Block Attention Module |
| `acmix` | `acmix_bottleneck.py` | ACmix: self-attention + conv fusion (CVPR 2022) |
| `coord_attn` | `coord_attn_bottleneck.py` | Coordinate Attention (CVPR 2021) |
| `eca` | `eca_bottleneck.py` | Efficient Channel Attention (ECA-Net, CVPR 2020) |
| `spatial_channel` | `spatial_channel_bottleneck.py` | Spatial-Channel cross-attention with gating |
| `gated_attn` | `gated_attn_bottleneck.py` | Gated self-attention (lightweight Transformer) |
| `moe` | `moe_bottleneck.py` | Mixture-of-Experts with top-k routing (ICLR 2017) |
| `coordconv` | `coordconv_bottleneck.py` | CoordConv: position-aware convolution (NeurIPS 2018) |

## Usage in YAML Config

```yaml
model:
  bottleneck:
    name: aspp              # any registered key
    params:
      atrous_rates: [6, 12, 18]   # ASPP-specific (example)
```

## Choosing a Bottleneck

| Scenario | Recommended |
|---|---|
| Quick baseline | `none` or `basic` |
| Multi-scale context | `aspp`, `dense_aspp`, `ppm` |
| Attention enrichment | `transformer`, `acmix`, `dual_attention` |
| Lightweight attention | `se`, `eca`, `cbam`, `coord_attn` |
| Spatial awareness | `spatial_channel`, `coordconv` |
| Sparse experts | `moe` |
| Global reasoning | `gated_attn` |
