# Skip Connections (`medseg/skip_connections/`)

[中文文档](README_CN.md)

Modules that process features at each encoder-to-decoder skip level before concatenation.

## Registered Skip Connections

| Registry Key | Source File | Description |
|---|---|---|
| `add` | `basic_skip.py` | Element-wise addition (ResNet-style) |
| `concat` | `basic_skip.py` | Channel-wise concatenation (U-Net default) |
| `cab` | `cab_skip.py` | Channel Attention Block |
| `sab` | `sab_skip.py` | Spatial Attention Block |
| `ccm` | `ccm_skip.py` | Cross-Channel Matching |
| `scse` | `scse_skip.py` | Concurrent Spatial + Channel SE (scSE) |
| `dense` | `dense_skip.py` | Dense connection (UNet++ style) |
| `multiscale` | `multiscale_skip.py` | Multi-scale feature fusion |
| `gating` | `gating_skip.py` | Gating mechanism (attention gate) |
| `cross_attn` | `cross_attn_skip.py` | Cross-attention between encoder and decoder features |
| `feature_refine` | `feature_refine_skip.py` | Learned feature refinement |

## Usage in YAML Config

```yaml
model:
  skip:
    name: scse               # any registered key
    params: {}               # extra kwargs
```

## Choosing a Skip Connection

| Scenario | Recommended |
|---|---|
| Baseline / speed | `concat` or `add` |
| Better boundary detail | `scse`, `cab`, `sab` |
| Transformer / attention models | `cross_attn`, `feature_refine` |
| Dense skip (UNet++) | `dense` |
| Multi-scale fusion | `multiscale` |
