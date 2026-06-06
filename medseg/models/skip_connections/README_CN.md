# 跳跃连接 (`medseg/skip_connections/`)

[English](README.md)

在编码器到解码器各层跳跃连接处处理特征的模块。

## 已注册跳跃连接

| 注册键 | 源文件 | 说明 |
|---|---|---|
| `add` | `basic_skip.py` | 逐元素相加（ResNet 风格） |
| `concat` | `basic_skip.py` | 通道拼接（U-Net 默认） |
| `cab` | `cab_skip.py` | 通道注意力块 |
| `sab` | `sab_skip.py` | 空间注意力块 |
| `ccm` | `ccm_skip.py` | 交叉通道匹配 |
| `scse` | `scse_skip.py` | 空间-通道并行 SE (scSE) |
| `dense` | `dense_skip.py` | 密集连接（UNet++ 风格） |
| `multiscale` | `multiscale_skip.py` | 多尺度特征融合 |
| `gating` | `gating_skip.py` | 门控机制（注意力门控） |
| `cross_attn` | `cross_attn_skip.py` | 编码器-解码器特征交叉注意力 |
| `feature_refine` | `feature_refine_skip.py` | 学习型特征精炼 |

## YAML 配置用法

```yaml
model:
  skip:
    name: scse               # 任意已注册键
    params: {}               # 额外参数
```

## 选择跳跃连接

| 场景 | 推荐 |
|---|---|
| 基线 / 速度 | `concat` 或 `add` |
| 更好的边界细节 | `scse`, `cab`, `sab` |
| Transformer / 注意力模型 | `cross_attn`, `feature_refine` |
| 密集跳跃 (UNet++) | `dense` |
| 多尺度融合 | `multiscale` |
