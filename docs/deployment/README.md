# Deployment & Efficiency

[中文文档](README_CN.md)

## ONNX Export

Export models to ONNX format using `scripts/export_onnx.py`.

```bash
# Basic export
python scripts/export_onnx.py \
    --config configs/architectures/networks/general/transunet.yaml \
    --checkpoint output/best_model.pth \
    --output model.onnx \
    --img_size 224

# Export + verify
python scripts/export_onnx.py \
    --config configs/xxx.yaml \
    --checkpoint best.pth \
    --output model.onnx \
    --verify

# Custom input channels
python scripts/export_onnx.py \
    --config configs/xxx.yaml \
    --checkpoint best.pth \
    --output model.onnx \
    --img_size 256 \
    --in_channels 1
```

---

## FLOPs Calculation

Three libraries supported:

### fvcore (recommended)

```python
import torch
from fvcore.nn import FlopCountAnalysis

model.eval()
x = torch.randn(1, 3, 224, 224).to(device)
flops = FlopCountAnalysis(model, x)
print(f"FLOPs: {flops.total() / 1e9:.2f} G")
```

### ptflops

```python
from ptflops import get_model_complexity_info

macs, params = get_model_complexity_info(
    model, (3, 224, 224),
    as_strings=True, print_per_layer_stat=False
)
print(f"MACs: {macs}, Params: {params}")
```

### thop

```python
import torch
from thop import profile

x = torch.randn(1, 3, 224, 224).to(device)
macs, params = profile(model, inputs=(x,))
print(f"MACs: {macs / 1e9:.2f} G, Params: {params / 1e6:.2f} M")
```

> **Note**: FLOPs here means MACs (multiply-accumulate operations). Some papers report FLOPs = 2 * MACs.

---

## Parameter Count

**Important**: Only count `requires_grad=True` parameters. Frozen foundation encoders (e.g. pre-trained SAM, CLIP) should NOT be counted as trainable parameters.

```python
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
frozen = total - trainable

print(f"Trainable: {trainable / 1e6:.2f}M")
print(f"Total:     {total / 1e6:.2f}M")
print(f"Frozen:    {frozen / 1e6:.2f}M")
```

---

## FPS Measurement

GPU warmup is critical for accurate timing.

```python
import time
import torch

model.eval()
x = torch.randn(1, 3, 224, 224).to(device)

# Warmup (essential for accurate GPU timing)
with torch.no_grad():
    for _ in range(10):
        model(x)
    torch.cuda.synchronize()

# Benchmark
with torch.no_grad():
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        model(x)
    torch.cuda.synchronize()
    elapsed = time.time() - t0

fps = 100 / elapsed
latency = elapsed / 100 * 1000  # ms

print(f"FPS: {fps:.1f}")
print(f"Latency: {latency:.1f} ms")
```

---

## Important Notes

### Frozen Parameters

When using foundation models with frozen encoders (SAM, CLIP, DINOv2, etc.):

1. **Parameter count**: Report both trainable and total. Papers typically report trainable only.

2. **FLOPs**: Frozen layers still contribute to FLOPs during inference. Report full-model FLOPs.

3. **FPS**: Measures real inference speed including frozen layers.

### Profiling Script

The project includes `profile_model.py` for comprehensive model profiling:

```bash
python profile_model.py --config configs/xxx.yaml \
    --checkpoint output/best_model.pth \
    --img_size 224
```

This reports FLOPs, parameter count, and FPS in a single run.
