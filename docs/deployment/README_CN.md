# 部署与效率

[English](README.md)

## ONNX 导出

使用 `scripts/export_onnx.py` 导出模型为 ONNX 格式。

```bash
# 基础导出
python scripts/export_onnx.py \
    --config configs/architectures/networks/general/transunet.yaml \
    --checkpoint output/best_model.pth \
    --output model.onnx \
    --img_size 224

# 导出并验证
python scripts/export_onnx.py \
    --config configs/xxx.yaml \
    --checkpoint best.pth \
    --output model.onnx \
    --verify

# 自定义输入通道
python scripts/export_onnx.py \
    --config configs/xxx.yaml \
    --checkpoint best.pth \
    --output model.onnx \
    --img_size 256 \
    --in_channels 1
```

---

## FLOPs 计算

支持三种计算库：

### fvcore (推荐)

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

> **注意**: 这里的 FLOPs 指 MACs（乘加运算）。部分论文报告 FLOPs = 2 * MACs。

---

## 参数量计算

**重要**: 只计算 `requires_grad=True` 的参数。冻结的基础编码器（如预训练 SAM、CLIP）不计入可训练参数。

```python
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
frozen = total - trainable

print(f"可训练: {trainable / 1e6:.2f}M")
print(f"总计:   {total / 1e6:.2f}M")
print(f"冻结:   {frozen / 1e6:.2f}M")
```

---

## FPS 计算

GPU 预热对准确计时至关重要。

```python
import time
import torch

model.eval()
x = torch.randn(1, 3, 224, 224).to(device)

# 预热（对准确 GPU 计时至关重要）
with torch.no_grad():
    for _ in range(10):
        model(x)
    torch.cuda.synchronize()

# 计时
with torch.no_grad():
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        model(x)
    torch.cuda.synchronize()
    elapsed = time.time() - t0

fps = 100 / elapsed
latency = elapsed / 100 * 1000  # 毫秒

print(f"FPS: {fps:.1f}")
print(f"延迟: {latency:.1f} ms")
```

---

## 注意事项

### 冻结参数

使用冻结编码器的基础模型（SAM、CLIP、DINOv2 等）时：

1. **参数量**: 报告可训练和总参数量。论文通常只报告可训练参数。

2. **FLOPs**: 冻结层在推理时仍有计算量。报告完整模型 FLOPs。

3. **FPS**: FPS 衡量包含冻结层的实际推理速度。

### 性能分析脚本

项目提供 `profile_model.py` 进行综合模型分析：

```bash
python profile_model.py --config configs/xxx.yaml \
    --checkpoint output/best_model.pth \
    --img_size 224
```

一次运行即可报告 FLOPs、参数量和 FPS。
