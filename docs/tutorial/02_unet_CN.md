# 第 02 讲：U-Net 详解

[上一讲：概述](01_introduction_CN.md) | [English](02_unet.md) | [下一讲：数据与预处理](03_data_CN.md)

---

## 1. 背景与动机

U-Net 由 Olaf Ronneberger 等人于 MICCAI 2015 发表，是医学图像分割领域最具影响力的架构。其核心创新：

- **编码器-解码器 + 跳跃连接**：恢复下采样过程中丢失的空间细节
- **数据高效**：通过激进增强，30 张标注图像即可训练出有效模型
- **对称 U 形结构**：直观的多尺度特征层级

U-Net 至今仍是几乎所有医学分割基准的默认 baseline。深入理解它是探索更高级架构的前提。

---

## 2. 核心概念

### 2.1 架构总览

```
                          输入 (如 572 x 572 x 1)
                                    │
                    ┌─────── 编码器 (收缩路径) ────────┐
                    │                                  │
                    │  Conv 3x3 → ReLU → Conv 3x3 → ReLU │
                    │       ↓ MaxPool 2x2              │
                    │  Conv 3x3 → ReLU → Conv 3x3 → ReLU │
                    │       ↓ MaxPool 2x2              │
                    │  Conv 3x3 → ReLU → Conv 3x3 → ReLU │
                    │       ↓ MaxPool 2x2              │
                    │  Conv 3x3 → ReLU → Conv 3x3 → ReLU │
                    │       ↓ MaxPool 2x2              │
                    └────────── 瓶颈层 ────────────────┘
                          Conv 3x3 → ReLU → Conv 3x3 → ReLU
                                    │
                    ┌─────── 解码器 (扩张路径) ────────┐
                    │       ↑ UpConv 2x2 (或 Bilinear)│
                    │  [跳跃拼接] → Conv 3x3 → ReLU → Conv │
                    │       ↑ UpConv 2x2              │
                    │  [跳跃拼接] → Conv 3x3 → ReLU → Conv │
                    │       ↑ UpConv 2x2              │
                    │  [跳跃拼接] → Conv 3x3 → ReLU → Conv │
                    └──────────────────────────────────┘
                                    │
                              1x1 Conv → Softmax
                                    │
                          输出 (如 388 x 388 x C)
```

### 2.2 关键设计决策

**为什么需要跳跃连接？**

编码器通过池化/步长卷积逐步降低空间分辨率以扩大感受野，这导致了对分割至关重要的精细边界信息丢失。跳跃连接将编码器特征直接桥接到对应解码器层，提供高分辨率空间线索。

**为什么是 U 形？**

多尺度是关键。不同目标需要不同感受野：
- 小病灶需要高分辨率特征（早期编码器层）
- 大器官需要语义上下文（深层 + 瓶颈层）

U 形结构天然构建了这一多尺度金字塔。

**为什么每层两个 3x3 卷积？**

两个堆叠的 3x3 卷积与一个 5x5 卷积有相同的有效感受野，但：
- 参数更少（2 x 9 = 18 vs 25 通道）
- 更多非线性（两次 ReLU 激活）
- 更强的表达能力

### 2.3 上采样策略

解码器需要增加空间分辨率。两种主要方法：

| 策略 | 机制 | 优点 | 缺点 |
|------|------|------|------|
| **转置卷积** (Deconv) | 通过转置卷积学习上采样 | 可学习的上采样核 | 可能产生棋盘格伪影 |
| **双线性 + 卷积** | 固定双线性插值后接卷积 | 无伪影、更简单 | 上采样不可学习 |

APRIL-MedSeg 中两者均可用：
- `decoder: bilinear` -- 双线性上采样 + 卷积
- `decoder: deconv` -- 转置卷积

---

## 3. 方法详解

### 3.1 U-Net 家族

自 2015 年以来，已有数十种 U-Net 变体被提出。框架包含了最重要的一些：

| 架构 | 年份 | 核心创新 | 配置名 |
|------|------|----------|--------|
| **U-Net** | MICCAI 2015 | 原始编码器-解码器 + 跳跃 | `unet` |
| **UNet++** | DLMIA 2018 | 跨尺度密集跳跃连接 | `unetpp` |
| **Attention U-Net** | MIDL 2018 | 跳跃连接上的注意力门 | `attention_unet` |
| **UNet 3+** | ICASSP 2020 | 全尺度跳跃连接（每个编码器到每个解码器） | `unet3plus` |
| **ResUNet++** | ISM 2019 | 残差块 + 注意力 + ASPP | `resunetpp` |
| **DenseUNet** | - | DenseNet 风格的密集连接 | `denseunet` |
| **scSE-UNet** | MICCAI 2018 | 通道/空间 Squeeze-Excitation 注意力 | `scseunet` |
| **R2U-Net** | IEEE Access 2018 | 循环残差块 | `r2unet` |
| **MultiResUNet** | Neural Networks 2020 | 多分辨率残差块 | `multiresunet` |
| **ResUNet-a** | ISPRS 2020 | 空洞卷积 + 残差 | `resunet_a` |
| **SA-UNet** | IEEE TIM 2021 | 跳跃上的空间注意力 | `sa_unet` |
| **KiU-Net** | MICCAI 2020 | 关键点引导 U-Net | `kiunet` |
| **PAN** | BMVC 2018 | 金字塔注意力网络 | `pan` |
| **LinkNet** | VCIP 2017 | 轻量级编码器-解码器 | `linknet` |
| **PSPNet** | CVPR 2017 | 金字塔空间池化 | `pspnet` |
| **FR-UNet** | IEEE TMI 2022 | 全分辨率跳跃连接 | `fr_unet` |

### 3.2 跳跃连接变体

跳跃连接是 U-Net 中被研究最多的组件之一。框架提供 25 种选择：

| 跳跃 | 机制 | 适用场景 |
|------|------|----------|
| `concat` | 拼接编码器特征与解码器特征 | 默认选择，通用性好 |
| `add` | 逐元素相加 | 特征维度匹配时 |
| `cab` | 通道注意力块 | 关注信息量大的通道 |
| `scse` | Squeeze-and-Excitation + 空间 SE | 通道 + 空间重校准 |
| `gating` | 注意力门（来自 Attention U-Net） | 抑制无关的跳跃特征 |
| `cross_attn` | 编码器与解码器间的交叉注意力 | 长距离跳跃交互 |
| `feature_refine` | 可学习特征精炼 | 跳跃特征需要处理时 |

### 3.3 瓶颈层选项

瓶颈层位于 U 的最深处，处理最压缩的特征：

| 瓶颈层 | 机制 | 适用场景 |
|--------|------|----------|
| `none` | 无特殊处理 | 默认 baseline |
| `aspp` | 空洞空间金字塔池化 | 多尺度上下文 |
| `dense_aspp` | 密集连接的 ASPP | 更丰富的多尺度特征 |
| `transformer` | 瓶颈处的自注意力 | 最深层的全局上下文 |
| `mamba` | 瓶颈处的状态空间模型 | 线性复杂度全局上下文 |
| `rwkv` | 瓶颈处的 RWKV | 高效序列建模 |
| `cbam` | 卷积块注意力 | 通道 + 空间注意力 |

---

## 4. APRIL-MedSeg 实操

### 4.1 模式一：完整网络

直接使用预定义的 U-Net 架构：

```yaml
model:
  architecture: unet        # 完整 U-Net
  num_classes: 2            # 二分类：背景 + 前景
  img_size: 256
  encoder:
    in_channels: 3
```

### 4.2 模式二：自由组合

任意编码器 + 任意解码器 + 跳跃 + 瓶颈：

```yaml
model:
  num_classes: 2
  img_size: 256
  encoder:
    name: basic             # 内置 BasicEncoder (CNN)
    in_channels: 3
    params: {}
  decoder:
    name: bilinear          # 双线性上采样解码器
    params: {}
  skip_connection:
    name: concat            # 标准拼接跳跃
    params: {}
  bottleneck:
    name: none              # 无特殊瓶颈
```

这就是 `configs/architectures/combinations/general/unet_basic.yaml` 的内容。

### 4.3 YAML 配置逐项解读

```yaml
# --- 模型 ---
model:
  num_classes: 2            # 输出类别数（背景 + 目标）
  img_size: 256             # 输入图像分辨率
  encoder:
    name: basic             # 编码器 backbone
    in_channels: 3          # 输入通道（3=RGB, 1=灰度）
    params: {}              # 编码器特有参数
  decoder:
    name: bilinear          # 解码器类型
    params: {}
  bottleneck:
    name: none
  skip_connection:
    name: concat
    params: {}

# --- 数据 ---
data:
  type: generic             # 数据集类型: generic, synapse, acdc
  img_size: 256
  train_dir: ./data/YourDataset/train
  val_dir: ./data/YourDataset/val
  test_dir: ./data/YourDataset/test

# --- 训练 ---
training:
  epochs: 300               # 总训练轮次
  batch_size: 24            # 每 GPU 样本数
  num_workers: 4            # DataLoader 工作线程
  loss:
    name: compound          # 复合损失：多个损失的加权和
    params:
      losses:
      - name: ce            # 交叉熵：有利于初始收敛
        weight: 0.4
      - name: dice          # Dice 损失：直接优化重叠指标
        weight: 0.6
  optimizer:
    name: adamw             # AdamW：解耦权重衰减的 Adam
    lr: 0.001               # 初始学习率
    weight_decay: 0.0001    # L2 正则化
  scheduler:
    name: cosine            # 余弦退火：平滑 LR 衰减
    min_lr: 1.0e-06         # 最小学习率
```

### 4.4 训练命令

```bash
# 基础训练
python train.py --config configs/architectures/combinations/general/unet_basic.yaml

# 启用 AMP 混合精度（约 1.5 倍加速）
python train.py --config configs/architectures/combinations/general/unet_basic.yaml --amp

# 从 CLI 覆盖参数
python train.py --config configs/architectures/combinations/general/unet_basic.yaml \
    --override training.epochs=200 training.batch_size=8 model.num_classes=9

# 从检查点恢复训练
python train.py --config configs/architectures/combinations/general/unet_basic.yaml \
    --resume output/checkpoint_epoch100.pth

# 自定义输出目录
python train.py --config configs/architectures/combinations/general/unet_basic.yaml \
    --output_dir ./experiments/unet_baseline

# 设置随机种子以保证可复现性
python train.py --config configs/architectures/combinations/general/unet_basic.yaml --seed 42
```

### 4.5 尝试 U-Net 变体

将完整网络切换为 U-Net 变体：

```bash
# Attention U-Net
python train.py --config configs/architectures/combinations/general/attention_unet_basic.yaml

# UNet++
python train.py --config configs/architectures/networks/general/unetpp.yaml

# 或在自由组合模式下覆盖：更改解码器和跳跃
python train.py --config configs/architectures/combinations/general/unet_basic.yaml \
    --override model.decoder.name=unet model.skip_connection.name=gating
```

---

## 5. 推荐实验

### 实验 1：BUSI 上的 U-Net Baseline

使用乳腺超声数据集（二分类：肿瘤 vs 背景）：

```bash
python train.py --config configs/intro_to_datasets/busi.yaml
```

预期：ResNet50 + U-Net 解码器 baseline Dice ~0.75-0.80。

### 实验 2：跳跃连接消融

相同编码器/解码器，对比不同跳跃连接：

| 跳跃 | YAML 覆盖 | 预期效果 |
|------|-----------|----------|
| `concat` | (baseline) | 标准 U-Net 行为 |
| `add` | `model.skip_connection.name=add` | 表达力稍弱 |
| `cab` | `model.skip_connection.name=cab` | 通道注意力可能提升 |
| `gating` | `model.skip_connection.name=gating` | 注意力门控跳跃 |

### 实验 3：解码器对比

相同编码器，不同解码器：

| 解码器 | 特点 |
|--------|------|
| `bilinear` | 简单、无伪影 |
| `deconv` | 可学习上采样 |
| `unet` | 完整 U-Net 解码器（每层双卷积） |
| `emcad` | 高效多尺度级联解码器 |

---

## 6. 延伸阅读

### 关键论文

| 论文 | 年份 | 会议/期刊 | 核心思想 |
|------|------|-----------|----------|
| [U-Net](https://arxiv.org/abs/1505.04597) | 2015 | MICCAI | 原始编码器-解码器 + 跳跃 |
| [UNet++](https://arxiv.org/abs/1807.10165) | 2018 | DLMIA | 密集嵌套跳跃连接 |
| [Attention U-Net](https://arxiv.org/abs/1804.03999) | 2018 | MIDL | 跳跃上的注意力门 |
| [UNet 3+](https://arxiv.org/abs/2004.08790) | 2020 | ICASSP | 全尺度密集跳跃 |
| [scSE-UNet](https://arxiv.org/abs/1803.02522) | 2018 | MICCAI | 通道 + 空间重校准 |
| [TransUNet](https://arxiv.org/abs/2102.04306) | 2021 | - | CNN 编码器 + Transformer 瓶颈 |
| [Swin-UNet](https://arxiv.org/abs/2105.05537) | 2022 | ECCV Workshop | 纯 Transformer U 形结构 |

### 相关文档

- [网络指南](../models/networks_CN.md) -- 130 个完整网络架构
- [编码器指南](../models/encoders_CN.md) -- 177 个编码器（含 U-Net 变体）
- [解码器指南](../models/decoders_CN.md) -- 45 个解码器及设计理念
- [跳跃连接](../models/skip_connections_CN.md) -- 25 种跳跃连接实现
- [瓶颈层](../models/bottlenecks_CN.md) -- 17 个瓶颈层模块

---

[上一讲：概述](01_introduction_CN.md) | [下一讲：数据与预处理](03_data_CN.md)
