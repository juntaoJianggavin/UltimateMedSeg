# 瓶颈层 (`medseg/bottlenecks/`)

[English](README.md)

插入在编码器与解码器之间最深层特征级别的插件模块，用于在解码器上采样前增强最压缩的特征表示。

## 已注册瓶颈层

| 注册键 | 源文件 | 说明 |
|---|---|---|
| `none` | `none_bottleneck.py` | 直通（无瓶颈层） |
| `basic` | `basic_bottleneck.py` | 两个 3×3 卷积 + 残差（默认） |
| `aspp` | `aspp_bottleneck.py` | 空洞空间金字塔池化 (DeepLabV3) |
| `dense_aspp` | `dense_aspp_bottleneck.py` | DenseASPP：级联空洞卷积 |
| `ppm` | `ppm_bottleneck.py` | 金字塔池化模块 (PSPNet) |
| `transformer` | `transformer_bottleneck.py` | Transformer 自注意力块 |
| `se` | `se_bottleneck.py` | 压缩-激励通道注意力 (CVPR 2018) |
| `dual_attention` | `da_bottleneck.py` | 双注意力：位置 + 通道 (DANet) |
| `cbam` | `cbam_bottleneck.py` | CBAM：卷积块注意力模块 |
| `acmix` | `acmix_bottleneck.py` | ACmix：自注意力 + 卷积融合 (CVPR 2022) |
| `coord_attn` | `coord_attn_bottleneck.py` | 坐标注意力 (CVPR 2021) |
| `eca` | `eca_bottleneck.py` | 高效通道注意力 (ECA-Net, CVPR 2020) |
| `spatial_channel` | `spatial_channel_bottleneck.py` | 空间-通道交叉注意力 + 门控 |
| `gated_attn` | `gated_attn_bottleneck.py` | 门控自注意力（轻量级 Transformer） |
| `moe` | `moe_bottleneck.py` | 专家混合 + top-k 路由 (ICLR 2017) |
| `coordconv` | `coordconv_bottleneck.py` | CoordConv：位置感知卷积 (NeurIPS 2018) |

## YAML 配置用法

```yaml
model:
  bottleneck:
    name: aspp              # 任意已注册键
    params:
      atrous_rates: [6, 12, 18]   # ASPP 专属参数（示例）
```

## 选择瓶颈层

| 场景 | 推荐 |
|---|---|
| 快速基线 | `none` 或 `basic` |
| 多尺度上下文 | `aspp`, `dense_aspp`, `ppm` |
| 注意力增强 | `transformer`, `acmix`, `dual_attention` |
| 轻量级注意力 | `se`, `eca`, `cbam`, `coord_attn` |
| 空间感知 | `spatial_channel`, `coordconv` |
| 稀疏专家 | `moe` |
| 全局推理 | `gated_attn` |
