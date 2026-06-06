# 跳跃连接

[English](skip_connections.md)

本项目提供 **25** 个跳跃连接模块，分为 5 大类。负责将 encoder 各层特征传递给 decoder。

---

## 基础 (2)

| Key | 说明 |
|---|---|
| `concat` | 通道拼接（默认） |
| `dense` | 密集跳跃连接（UNet++ 风格） |

## 注意力 (10)

| Key | 来源 | 说明 |
|---|---|---|
| `attention_gate` | Attention U-Net (Oktay 2018) | 注意力门控 |
| `cab` | — | 通道注意力桥 |
| `sab` | — | 空间注意力桥 |
| `scse` | Roy et al., TMI 2019 | 空间-通道 SE |
| `cbam` | Woo et al., ECCV 2018 | CBAM 注意力 |
| `gating` | — | 门控机制（sigmoid 加权） |
| `gru_gate` | — | GRU 风格门控 |
| `gab` | EGE-UNet, MICCAI 2023 Workshop, [GitHub](https://github.com/JCruan519/EGE-UNet) | 分组聚合桥 |
| `sc_att_bridge` | MALUNet, BIBM 2022, [GitHub](https://github.com/JCruan519/MALUNet) | 空间+通道联合注意力桥 |
| `ta_mosc` | UTANet, AAAI 2025, [GitHub](https://github.com/AshleyLuo001/UTANet) | 任务自适应混合跳跃连接 |

## Transformer (5)

| Key | 来源 | 说明 |
|---|---|---|
| `cross_attn` | — | 交叉注意力（decoder Q × encoder KV） |
| `transformer_fusion` | — | Transformer 特征融合 |
| `aggregation_attention` | — | 聚合注意力 |
| `missformer_bridge` | MISSFormer, 2022, [GitHub](https://github.com/ZhifangDeng/MISSFormer) | MISSFormer 桥接模块 |
| `uctrans` | UCTransNet, AAAI 2022, [GitHub](https://github.com/McGregorWwww/UCTransNet) | 通道级交叉 Transformer |

## Mamba (1)

| Key | 来源 | 说明 |
|---|---|---|
| `skvmpp` | SK-VM++, BSPC 2025, [GitHub](https://github.com/wurenkai/SK-VMPlusPlus) | Mamba SS2D 辅助跳跃连接（金字塔视觉 Mamba 层） |

## CNN 融合 (6)

| Key | 来源 | 说明 |
|---|---|---|
| `bifusion` | TransFuse 风格 | 双向融合 |
| `deformable` | — | 可变形卷积融合 |
| `multiscale` | — | 多尺度融合 |
| `feature_refine` | — | CBAM 特征精炼 |
| `ccm` | — | 交叉通道模块 |
| `sdi` | U-Net V2, ISBI 2025, [GitHub](https://github.com/yaoppeng/U-Net_v2) | 尺度多样性整合 |

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
    name: unet
    params: {}
  skip_connection:
    name: skvmpp           # 任选 25 个之一
    params: {}
  bottleneck:
    name: none

data:
  type: generic
  img_size: 224
  train_dir: ./data/YourDataset/train
  val_dir: ./data/YourDataset/val

training:
  epochs: 200
  batch_size: 8
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
    lr: 0.0001
  scheduler:
    name: cosine
    min_lr: 0.000001
```

### 常用组合建议

| 场景 | 推荐 skip | 理由 |
|---|---|---|
| 基线对比 | `concat` | 最简单，无额外参数 |
| 注意力增强 | `attention_gate` 或 `scse` | 经典有效 |
| Transformer 编码器 | `cross_attn` 或 `uctrans` | 利用 self/cross-attention |
| Mamba 编码器 | `skvmpp` | Mamba SS2D 增强 skip 特征 |
| 轻量级 | `add` 或 `gating` | 最小计算开销 |
| 医学边界关注 | `ta_mosc` | 任务自适应，AAAI 2025 |
