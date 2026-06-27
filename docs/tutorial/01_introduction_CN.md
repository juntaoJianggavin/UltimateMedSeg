# 第 01 讲：医学图像分割概述

[English](01_introduction.md) | [下一讲：U-Net 详解](02_unet_CN.md)

---

## 1. 背景与动机

### 什么是图像分割？

图像分割为图像中的每个像素分配一个类别标签。主要有三种范式：

| 范式 | 目标 | 示例 |
|------|------|------|
| **语义分割** | 将每个像素分类到某一类别 | "该像素是肝脏/脾脏/背景" |
| **实例分割** | 检测并勾勒出各个独立目标 | "该像素属于细胞 #3" |
| **医学分割** | 语义分割在医学影像中的应用 | 器官勾画、病灶检测、组织分类 |

医学图像分割本质上是**带有领域特有挑战的语义分割**：标注数据有限、标注成本高（需放射科/病理科医师）、类别严重不平衡、成像模态多样。

### 临床意义

医学图像分割直接影响四大关键临床工作流：

**1. 计算机辅助诊断 (CAD)**
- 肿瘤检测：CT 中自动定位肺结节、眼底图像中自动检测病灶
- 筛查：肠镜视频中的自动息肉检测（CVC-ClinicDB, Kvasir-SEG）

**2. 量化分析**
- 器官体积测量：心脏 MRI 中的左心室（ACDC）、视网膜图像中的视杯/视盘（REFUGE, Drishti-GS）
- 疗效评估：跨时间点的肿瘤大小追踪

**3. 手术规划与导航**
- CT/MRI 的术前三维重建
- 术中边界引导

**4. 放疗规划**
- 自动危及器官 (OAR) 勾画
- 治疗计划中的靶区勾画

---

## 2. 核心概念

### 常见成像模态与数据集

| 模态 | 典型任务 | 内置数据集 |
|------|----------|------------|
| CT | 多器官 / 感染分割 | Synapse（8 个器官）, COVID CT Seg, MosMedData+ |
| MRI | 心脏结构 | ACDC（RV/LV/MYO） |
| X-ray (CXR) | 肺部 / 感染分割 | Montgomery-Shenzhen, QaTa-COV19 |
| 眼底照相 | 血管 / 视盘视杯 | DRIVE, STARE, CHASE_DB1, HRF, ARIA, RITE, REFUGE, Drishti-GS |
| 皮肤镜 | 皮肤病灶 | ISIC 2016/2017/2018, PH2 |
| 组织病理 (WSI) | 细胞核/腺体 | MoNuSeg, GlaS, PanNuke |
| 超声 | 乳腺病灶 | BUSI |
| 内窥镜 | 息肉 | CVC-ClinicDB, CVC-ColonDB, Kvasir-SEG |

### 评价指标

APRIL-MedSeg 计算三大类指标（见 `medseg/utils/metrics.py`）：

**Dice 相似系数 (DSC)**

最常用的重叠度指标。对于预测掩码 $P$ 和真实标签 $G$：

$$\text{Dice} = \frac{2|P \cap G|}{|P| + |G|}$$

范围：[0, 1]，越高越好。等价于 F1-score。

**交并比 (IoU / Jaccard)**

$$\text{IoU} = \frac{|P \cap G|}{|P \cup G|} = \frac{|P \cap G|}{|P| + |G| - |P \cap G|}$$

范围：[0, 1]，越高越好。对于同一预测结果，IoU 始终 $\leq$ Dice。

**95% 豪斯多夫距离 (HD95)**

度量预测与真实标签之间的边界距离，取第 95 百分位以避免异常值干扰。越低越好。单位：像素（若提供间距则为 mm）。

```python
# 来自 medseg/utils/metrics.py
metrics = compute_metrics(pred, target, num_classes)
# 返回: {"dice": {...}, "iou": {...}, "hd95": {...}}
```

### 方法演进

```
传统方法                     深度学习时代                    Foundation 时代
(2015 年前)                  (2015-2022)                    (2023-至今)

阈值分割           ──>       FCN (2015)           ──>       SAM (2023)
区域生长           ──>       U-Net (2015)         ──>       DINOv2 (2024)
分水岭             ──>       UNet++ / Attention   ──>       BiomedCLIP (2024)
活动轮廓           ──>       TransUNet / Swin-UNet ──>      UNI / Phikon (2024)
图割               ──>       Mamba-UNet / VM-UNet  ──>      MUSK / PLIP
```

每一代都在解决前代的特定局限：
- **FCN**：首个端到端像素级分割，但丢失空间细节
- **U-Net**：跳跃连接恢复精细边界，少量样本即可训练
- **Transformer**：全局自注意力捕获长距离依赖
- **状态空间模型 (Mamba)**：线性复杂度 + 全局感受野
- **Foundation 模型**：大规模数据预训练，下游任务最少微调即可迁移

---

## 3. 框架概览

### 架构哲学

APRIL-MedSeg 采用**四模块自由组合**设计：

```
输入图像 ──> [Encoder] ──> [Bottleneck] ──> [Decoder] ──> 分割输出
                                |                 ^
                                └── [Skip Conn] ──┘
```

每个模块可独立通过一行 YAML 切换：

| 模块 | 注册数量 | 示例 |
|------|----------|------|
| 编码器 (Encoder) | 177 | `basic`, `timm_resnet50`, `timm_swin_tiny_patch4_window7_224`, `dinov2`, `dino` |
| 解码器 (Decoder) | 45 | `bilinear`, `deconv`, `emcad`, `cascade_full`, `unetpp` |
| 跳跃连接 (Skip) | 25 | `concat`, `add`, `cab`, `scse`, `gating` |
| 瓶颈层 (Bottleneck) | 17 | `none`, `aspp`, `dense_aspp`, `mamba`, `transformer` |
| 完整网络 | 130 | `unet`, `transunet`, `swinunet`, `attention_unet`, `vmunet` |

### 两种配置模式

**模式一：完整架构**（使用预定义网络）

```yaml
model:
  architecture: transunet   # 一个名字选定完整架构
  num_classes: 9
  img_size: 224
```

**模式二：自由组合**（任意注册的模块自由搭配）

```yaml
model:
  encoder:
    name: timm_resnet50
    pretrained: true
  decoder:
    name: emcad
  skip_connection:
    name: cab
  bottleneck:
    name: aspp
```

### 项目结构

```
APRIL-MedSeg/
├── train.py                    # 标准监督训练
├── test.py                     # 评估（支持 TTA/集成）
├── semi_train.py               # 半监督训练
├── train_domain_adaptation.py  # 域适应
├── train_distillation.py       # 知识蒸馏
├── train_weakly_supervised.py  # 弱监督
├── train_text_guided.py        # 文本引导分割
├── configs/                    # 917 个 YAML 配置
├── medseg/                     # 核心库
│   ├── models/                 # 177 编码器, 45 解码器, 130 完整网络
│   ├── losses/                 # 15 个损失函数
│   ├── datasets/               # 6 个数据集类
│   ├── training/               # 高级训练范式
│   ├── inference/              # TTA, 集成推理
│   └── utils/                  # 指标, 增强, 配置
└── docs/                       # 双语文档
```

---

## 4. 框架实操：第一次训练

### 第一步：准备数据

创建一个简单的数据集目录：

```
data/YourDataset/
├── train/
│   ├── image_001.png
│   └── image_002.png
├── val/
│   └── image_101.png
└── test/
    └── image_201.png
```

每张图像必须有同名的对应 mask 文件。

### 第二步：运行训练

```bash
python train.py --config configs/architectures/combinations/general/unet_basic.yaml
```

使用默认配置：
- 编码器：`basic`（简单 CNN backbone）
- 解码器：`bilinear`（双线性上采样）
- 损失：Compound（0.4 CE + 0.6 Dice）
- 优化器：AdamW (lr=0.001)
- Epochs：300

### 第三步：覆盖参数

无需修改 YAML 文件，直接从命令行覆盖：

```bash
python train.py --config configs/architectures/combinations/general/unet_basic.yaml \
    --override training.epochs=100 training.batch_size=8 model.num_classes=9
```

### 第四步：启用混合精度 (AMP)

```bash
python train.py --config configs/architectures/combinations/general/unet_basic.yaml --amp
```

### 第五步：评估

```bash
python test.py --config configs/architectures/combinations/general/unet_basic.yaml \
    --checkpoint output/best_model.pth
```

---

## 5. 推荐实验

**实验 1：Baseline**
- 在 Synapse 数据集（8 个腹部器官 + 背景）上运行 `unet_basic.yaml`
- 记录 baseline Dice、IoU、HD95

**实验 2：模块替换**
- 将编码器换成 `timm_resnet50`：`--override model.encoder.name=timm_resnet50`
- 对比指标与 basic 编码器的差异

**实验 3：损失函数消融**
- 尝试纯 Dice 损失：创建变体 YAML，`losses: [{name: dice, weight: 1.0}]`
- 对比收敛速度和最终指标

---

## 6. 延伸阅读

### 关键论文

| 论文 | 年份 | 贡献 |
|------|------|------|
| [FCN (Long et al.)](https://arxiv.org/abs/1411.4038) | CVPR 2015 | 首个全卷积网络语义分割 |
| [U-Net (Ronneberger et al.)](https://arxiv.org/abs/1505.04597) | MICCAI 2015 | 跳跃连接 + 小样本医学分割范式 |
| [TransUNet (Chen et al.)](https://arxiv.org/abs/2102.04306) | 2021 | CNN + Transformer 混合用于医学分割 |
| [SAM (Kirillov et al.)](https://arxiv.org/abs/2304.02643) | ICCV 2023 | Segment Anything：通用分割模型 |
| [DINOv2 (Oquab et al.)](https://arxiv.org/abs/2304.07193) | 2024 | 无监督视觉特征学习 |

### 相关文档

- [编码器指南](../models/encoders_CN.md) -- 177 个编码器及 HuggingFace 模型路径
- [解码器指南](../models/decoders_CN.md) -- 45 个解码器及设计理念
- [损失函数](../paradigms/README_CN.md) -- 81 个注册损失（15 个实现文件）
- [数据指南](../data/README.md) -- 25 个内置数据集及增强管线
- [研究指南](../research_guide_CN.md) -- 消融实验设计与基准测试协议

---

[下一讲：U-Net 详解](02_unet_CN.md)
