# 研究指南

[English](research_guide.md)

本文档为 5 个研究方向提供系统性研究建议，包括推荐基线、数据集、对比方案和实验脚本。

---

## 1. 通用 SOTA 架构基准测试

### 目标
在多个医学分割基准上公平比较不同架构。

### 数据集
| 数据集 | 模态 | 类别数 | 说明 |
|---------|------|--------|------|
| Synapse | 腹部 CT | 9 | 多器官分割，TransUNet 标准基准 |
| ACDC | 心脏 MRI | 4 | 心脏结构分割 |
| BUSI | 乳腺超声 | 2 | 肿瘤分割，5 折交叉验证 |
| CVC-ClinicDB | 结肠镜 | 2 | 息肉分割 |
| GlaS | 病理 H&E | 2 | 腺体分割 |
| Kvasir-SEG | 胃肠镜 | 2 | 息肉分割 |
| ISIC 2018 | 皮肤镜 | 2 | 皮肤病变分割 |

### 推荐基线

**必跑模型（覆盖所有架构类型）**：

| 架构 | 类型 | 论文 | 配置键 |
|------|------|------|--------|
| TransUNet | Transformer | Chen et al., 2021 | `transunet` |
| Swin-UNet | Transformer | Cao et al., 2022 | `swinunet` |
| VM-UNet | Mamba | Chen et al., 2024 | `vm_unet` |
| RWKV-UNet | RWKV | — | `rwkv_unet` |
| RIR-Zigzag | RWKV | TMI 2025 | `rir_zigzag` |
| Rolling-UNet | MLP | AAAI 2024 | `rolling_unet` |
| U-KAN | KAN | AAAI 2025 | `ukan` |
| Mobile-U-ViT | 轻量 Transformer | — | `mobile_u_vit` |
| UNet (basic) | CNN 基线 | Ronneberger 2015 | encoder `basic` + decoder `unet` |
| UNet++ | CNN Dense | Zhou 2018 | `unetpp` |
| Attention Unet | CNN 注意力 | Oktay 2018 | `attention_unet` |

**额外建议在 BUSI/CVC-ClinicDB/GlaS/Kvasir-SEG 上运行**：
- PolypPVT、CASCADE、HSNet、SSFormer（息肉/腺体专用方法）

### 实验脚本

```bash
# 运行所有基线
bash scripts/experiments/run_sota_benchmark.sh

# 单模型 × 单数据集
python train.py --config configs/architectures/networks/general/transunet.yaml \
    --output_dir output/sota/transunet_synapse --amp
```

---

## 2. 解码器消融实验

### 目标
固定编码器，比较所有解码器，寻找最佳编码器-解码器搭配。

### 实验设计

**3 个代表性编码器** × **全部 40 个解码器**：

| 编码器 | 类型 | 选择理由 |
|--------|------|----------|
| `basic` | 原始 UNet 卷积 | 最简基线，消除编码器影响 |
| `timm_resnet50` | CNN（ImageNet 预训练） | 经典 CNN 骨干 |
| `timm_pvt_v2_b2` | Transformer（ImageNet） | Transformer 代表 |

**数据集**：Synapse + ISIC 2018（一个多器官，一个二分类）

### YAML 示例

```yaml
# configs/architectures/decoder_study/general/resnet50_unet.yaml
model:
  num_classes: 2
  img_size: 256
  encoder:
    name: timm_resnet50
    pretrained: true
    in_channels: 3
  decoder:
    name: unet              # 替换此行以切换解码器
    params: {}
  skip_connection:
    name: concat
  bottleneck:
    name: none
```

### 实验脚本

```bash
# 所有解码器 × 3 个编码器
bash scripts/experiments/run_decoder_study.sh

# 单个组合
python train.py --config configs/architectures/decoder_study/general/resnet50_emcad.yaml \
    --output_dir output/decoder_study/resnet50_emcad --amp
```

### 可用 YAML

`configs/architectures/decoder_study/general/` 包含 120 个 YAML（3 个编码器 × 40 个解码器）。

---

## 3. 瓶颈层消融实验

### 实验设计

**3 个编码器 × 17 个瓶颈层**，解码器固定为 `bilinear`，跳跃连接固定为 `concat`。

```yaml
model:
  encoder: { name: timm_resnet50, pretrained: true }
  decoder: { name: bilinear }
  bottleneck: { name: aspp }      # 替换此行
  skip_connection: { name: concat }
```

### 实验脚本

```bash
bash scripts/experiments/run_bottleneck_study.sh
```

可用 YAML：`configs/architectures/bottleneck_study/general/`（51 个文件）。

---

## 4. 跳跃连接消融实验

### 实验设计

**3 个编码器 × 25 个跳跃连接**，解码器固定为 `unet`，瓶颈层固定为 `none`。

**重点关注新方法**：
- `skvmpp`（SK-VM++，BSPC 2025）— Mamba 辅助跳跃连接
- `ta_mosc`（UTANet，AAAI 2025）— 任务自适应混合跳跃连接
- `uctrans`（UCTransNet，AAAI 2022）— 通道级 Transformer 跳跃连接
- `sdi`（U-Net V2，ISBI 2025）— 尺度多样集成

```bash
bash scripts/experiments/run_skip_study.sh
```

可用 YAML：`configs/architectures/skip_study/general/`（75 个文件）。

---

## 5. 基础模型编码器研究

### 目标
比较通用基础模型与领域专用基础模型作为编码器的效果。

### 推荐对比

#### 通用 vs 专用（皮肤科/病理/放射/眼科）

| 通用编码器 | 专用编码器 | 数据集 |
|---|---|---|
| `dinov2` (base) | `panderm` | ISIC 2017/2018, PH2 |
| `dinov2` (base) | `phikon` / `uni` / `plip` | GlaS, PanNuke, MoNuSeg |
| `dinov2` (base) | `raddino` | Montgomery+Shenzhen CXR |
| `dinov2` (base) | `retfound_dinov2` / `flair` | DRIVE, CHASE_DB1, REFUGE |
| `clip_vit` (base) | `biomedclip` / `medclip` | 跨模态对比 |

#### MLLM 视觉塔 vs 传统基础模型

| MLLM 视觉 | 传统基础模型 | 数据集 |
|---|---|---|
| `qwen3_vl_vision` | `dinov2` (large) | Synapse, ACDC |
| `medgemma_vision` | `biomedclip` | 多模态基准 |

### 说明

- 所有基础编码器使用 **DPT head**（来自不同深度 block 的多尺度特征）
- 编码器冻结时仅训练解码器；参数量应区分可训练参数与冻结参数
- 建议统一使用 `unet` 解码器、`concat` 跳跃连接

### YAML 示例

```yaml
model:
  num_classes: 2
  img_size: native            # 基础模型推荐使用原生尺寸
  encoder:
    name: dinov2              # 替换为 phikon / panderm / raddino 等
    pretrained: true
    params:
      variant: base
    freeze_cfg:
      freeze: true
      unfreeze_last_n: 4      # 微调最后 4 个 block
  decoder:
    name: unet
  bottleneck:
    name: none
```

可用 YAML：`configs/architectures/foundation/`（57 个文件）。

---

## 6. 轻量级皮肤癌分割

### 目标
在皮肤病变分割上比较轻量级网络，评估参数量-精度权衡。

### 数据集
- **ISIC 2017** — 训练集，官方 train/val/test 划分
- **ISIC 2018** — 训练集，官方 train/val/test 划分
- **PH2** — 外部验证（200 张图像，5 折交叉验证）

### 基线

| 网络 | 参数量 | 论文 | 配置键 |
|------|--------|------|--------|
| EGE-UNet | ~50K | MICCAI 2023 W, [GitHub](https://github.com/JCruan519/EGE-UNet) | `ege_unet` |
| Lite-UNet | ~60K | 2023 | `lite_unet` |
| U-Lite | ~60K | 2023 | `u_lite` |
| MALUNet | ~170K | BIBM 2022, [GitHub](https://github.com/JCruan519/MALUNet) | `malunet` |
| LV-UNet | ~400K | BIBM 2024, [GitHub](https://github.com/juntaoJianggavin/LV-UNet) | `lv_unet` |
| UltraLight-VM-UNet | ~50K | 2024, [GitHub](https://github.com/wurenkai/UltraLight-VM-UNet) | `ultralight_vmunet` |
| UltraLBM-UNet | ~50K | 2024 | `ultralbm_unet` |
| MK-UNet | ~200K | ICCV 2025, [GitHub](https://github.com/SLDGroup/MK-UNet) | `mk_unet` |

### 实验脚本

```bash
bash scripts/experiments/run_lightweight_skin.sh
```

---

## 7. 训练范式研究

### 半监督学习

**数据集**：BUSI（5 折，10%/20%/50% 标注比例）
**骨干**：UNet（basic 编码器 + unet 解码器）+ RWKV-UNet

| 方法 | 配置 |
|------|------|
| Mean Teacher | `configs/training_paradigms/semi_supervision/mean_teacher.yaml` |
| CPS | `configs/training_paradigms/semi_supervision/cps.yaml` |
| UniMatch | `configs/training_paradigms/semi_supervision/unimatch.yaml` |
| FixMatch | `configs/training_paradigms/semi_supervision/fixmatch.yaml` |
| AugSeg (CVPR 2023) | —（暂无配置文件） |
| CorrMatch (CVPR 2024) | `configs/training_paradigms/semi_supervision/corrmatch.yaml` |

### 域适应

**场景**：Synapse → ACDC 跨模态（CT→MRI）
**方法**：AdvEnt / DANN / FDA / MIC / HRDA

### 知识蒸馏

**教师**：TransUNet（大模型）→ **学生**：U-Lite（轻量级）
**方法**：Vanilla KD / DKD / CWD / MGD / DIST

### 弱监督学习

**数据集**：BUSI（框标注）/ Kvasir-SEG（点标注）
**方法**：BoxSup / PointSup / CAM / GatedCRF / TreeEnergy

```bash
bash scripts/experiments/run_semi_study.sh
bash scripts/experiments/run_da_study.sh
bash scripts/experiments/run_kd_study.sh
bash scripts/experiments/run_weak_study.sh
```

---

## 8. 文本引导分割

> **（待续）**

### 可训练模型对比

在 QaTa-COV19 和 MosMedData+ 上比较 LanGuideMedSeg / LViT / MediSee 等方法（需要逐图像文本输入的方法）。

### 推理 Pipeline 对比

比较不同检测器 × 分割器组合在 Synapse 上的零样本分割效果。

---

## 如何注册新模型

### 1. 注册新编码器

```python
# medseg/models/encoders/cnn/my_encoder.py
from medseg.registry import ENCODER_REGISTRY

@ENCODER_REGISTRY.register("my_encoder")
class MyEncoder(nn.Module):
    def __init__(self, pretrained=False, in_channels=3, img_size=224, **kwargs):
        super().__init__()
        self.out_channels = [64, 128, 256, 512]
    def forward(self, x):
        return [f1, f2, f3, f4]
```

在 `medseg/models/encoders/cnn/__init__.py` 中添加 `from . import my_encoder`。

### 2. 注册新解码器

```python
# medseg/models/decoders/basic/my_decoder.py
from medseg.registry import DECODER_REGISTRY

@DECODER_REGISTRY.register("my_decoder")
class MyDecoder(nn.Module):
    has_internal_skip = False
    def __init__(self, encoder_channels, bottleneck_channels, skip_connection=None, **kwargs):
        super().__init__()
        self.out_channels = encoder_channels[0]
    def forward(self, bottleneck_feat, skip_features):
        return decoded
```

### 3. 注册新损失函数

```python
# medseg/losses/my_loss.py
from medseg.registry import LOSS_REGISTRY

@LOSS_REGISTRY.register("my_loss")
class MyLoss(nn.Module):
    def forward(self, pred, target):
        return loss_value
```

### 4. 注册新跳跃连接

```python
# medseg/models/skip_connections/attention/my_skip.py
from medseg.registry import SKIP_REGISTRY

@SKIP_REGISTRY.register("my_skip")
class MySkip(nn.Module):
    def get_out_channels(self, dec_ch, skip_ch):
        return dec_ch + skip_ch
    def forward(self, decoder_feat, skip_feat):
        return torch.cat([decoder_feat, skip_feat], dim=1)
```

### 5. 注册新瓶颈层

```python
from medseg.registry import BOTTLENECK_REGISTRY

@BOTTLENECK_REGISTRY.register("my_bottleneck")
class MyBottleneck(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.out_channels = in_channels
    def forward(self, x):
        return refined_x
```

### 6. 注册新半监督方法

继承 `BaseSemiMethod`，实现 `build()` / `train_step()` / `update()`，添加到 `medseg/training/semi/__init__.py` 中的 `_SEMI_METHODS` 字典。

### 注册后

1. 在对应的 `__init__.py` 中导入
2. 创建 YAML 配置文件（通过 `name: my_xxx` 引用）
3. 在对应的 `docs/` 文档中添加条目
4. 添加到本文档的对比表中
5. 运行 `python scripts/test_all_configs.py` 验证

---

## 9. 领域专用模型基准测试

以下实验按医学模态分组，每组包含通用基线 + 领域专用模型。所有组均包含 4 个通用基线：
UNet / Attention-UNet / UNet++ / ResNet50+UNet。SAM 家族模型被排除
（它们有独立的基于提示的评估范式）。

### 9.1 息肉分割

**数据集**：CVC-ClinicDB（5 折）+ Kvasir-SEG（5 折）

| 模型 | 架构创新 | 键 | 来源 |
|------|----------|-----|------|
| **基线** | | | |
| UNet | 标准 CNN 基线 | `encoder: basic` + `decoder: unet` | Ronneberger 2015 |
| Attention-UNet | 注意力门 | `attention_unet` | Oktay 2018 |
| UNet++ | 密集嵌套跳跃 | `unetpp` | Zhou 2018 |
| ResNet50+UNet | ImageNet 预训练 | `encoder: timm_resnet50` + `decoder: unet` | — |
| **领域专用** | | | |
| SEPNet | MAP(RFB) + CRC 渐进式细化 | `sepnet` | — |
| CTNet | SMIM 多尺度 + CIM 跨层融合 | `ctnet` | — |
| Polyper | Swin-T 双分支（区域+边界）+ BGM | `polyper` | — |
| PolypPVT | PVTv2 + CFM + 级联注意力 | `polyp_pvt` | AAAI 2023 |
| CASCADE | 级联注意力解码器 | `cascade` | MICCAI 2023 |
| HSNet | PVTv2 + 级联 CSA | `hsnet` | 2023 |
| SSFormer | MiT-B2 + PLD 解码器 | `ssformer` | 2023 |
| LDNet | 病变感知动态核 | `ldnet` | 2022 |
| ESFPNet | 高效稀疏 FPN | `esfpnet` | 2023 |
| MIST | 多任务分割 Transformer | `mist` | 2023 |
| FCBFormer | FCN + Transformer 融合 | `fcbformer` | 2022 |
| TransNetR | Transformer + 残差 | `transnetr` | 2022 |

```bash
bash scripts/experiments/run_polyp_benchmark.sh
```

### 9.2 皮肤分割

**训练**：ISIC 2017、ISIC 2018 | **外部验证**：PH2

| 模型 | 架构创新 | 键 | 参数量 |
|------|----------|-----|--------|
| **基线** |（同上）| | |
| **轻量级领域专用** | | | |
| EGE-UNet | 分组增强 + GHPA + 深监督 | `ege_unet` | ~50K |
| Lite-UNet | 轻量 Conv 编码器 | `lite_unet` | ~60K |
| U-Lite | 轴向深度卷积 | `u_lite` | ~60K |
| MALUNet | 多轴大核 + DGA | `malunet` | ~170K |
| LV-UNet | MobileNetV3 + VanillaNet 解码器 | `lv_unet` | ~400K |
| UltraLight-VM-UNet | 超轻量 Mamba | `ultralight_vmunet` | ~50K |
| UltraLBM-UNet | 超轻量双向 Mamba | `ultralbm_unet` | ~50K |
| MK-UNet | 多核 IRB + CBAM | `mk_unet` | ~200K |
| **Mamba 专用** | | | |
| MUCM-Net | UCMBlock（Mamba + 移位 MLP） | `mucm_net` | — |
| AC-MambaSeg | 自适应卷积 + Mamba 瓶颈 + CBAM 跳跃 | `ac_mambaseg` | — |
| SkinMamba | 跨尺度 Mamba + FFT 边界 | `skin_mamba` | — |
| DermoMamba | 跨尺度 Mamba + PCA + 三向 SweepMamba | `dermomamba` | — |

```bash
bash scripts/experiments/run_skin_benchmark.sh
```

### 9.3 视网膜血管分割

**数据集**：DRIVE（train20/test20）、STARE（5 折）、CHASE_DB1

| 模型 | 架构创新 | 键 |
|------|----------|-----|
| **基线** |（同上）| |
| FR-UNet | 全分辨率多分支血管分割 | `fr_unet` |
| SerpMamba | 蛇形扫描 4 向 SS2D | `serp_mamba` |
| MambaVesselNet++ | CNN-Mamba 混合 + 3 向扫描 | `mamba_vesselnet_pp` |

```bash
bash scripts/experiments/run_retinal_benchmark.sh
```

### 9.4 超声分割

**数据集**：BUSI（5 折）

| 模型 | 架构创新 | 键 |
|------|----------|-----|
| **基线** |（同上）| |
| AAU-Net | 自适应注意力（BUSI 专用） | `aau_net` |
| DCM-Net | 双编码器 CNN+Mamba + CBFM 跨分支融合 | `dcm_net` |
| UU-Mamba | U-Mamba + 不确定性感知输出 | `uu_mamba` |
| ViM-UNet | Vision Mamba 编码器 + UNet 解码器 | `vim_unet` |

```bash
bash scripts/experiments/run_ultrasound_benchmark.sh
```

### 9.5 病理分割

**数据集**：GlaS（train80%/test）

| 模型 | 架构创新 | 键 |
|------|----------|-----|
| **基线** |（同上）| |
| U-VixLSTM | Vision-xLSTM（mLSTM）编码器 + 跳跃解码器 | `u_vixlstm` |
| TransNuSeg | 多任务解码器 + 共享 QKV 注意力（MICCAI 2023） | `transnuseg` |
| HoverNetLite | NP+HV 双分支细胞核分割（轻量 HoVerNet） | `hovernet_lite` |
| NuLite | 轻量级细胞核分割 | `nulite` |

```bash
bash scripts/experiments/run_pathology_benchmark.sh
```

---

## 实验脚本总览

| 脚本 | 用途 | 模型数 |
|------|------|--------|
| `run_sota_benchmark.sh` | 通用 SOTA 架构对比 | 11 |
| `run_decoder_study.sh` | 解码器消融（3 编码 × 15 解码） | 45 |
| `run_bottleneck_study.sh` | 瓶颈层消融（3 编码 × 9 瓶颈） | 27 |
| `run_skip_study.sh` | 跳跃连接消融（3 编码 × 12 跳跃） | 36 |
| `run_lightweight_skin.sh` | 轻量级皮肤分割 | 8 |
| `run_polyp_benchmark.sh` | 息肉领域专用模型 | 16 |
| `run_skin_benchmark.sh` | 皮肤领域专用模型 | 16 |
| `run_retinal_benchmark.sh` | 视网膜领域专用模型 | 7 |
| `run_ultrasound_benchmark.sh` | 超声领域专用模型 | 8 |
| `run_pathology_benchmark.sh` | 病理领域专用模型 | 8 |
| `run_semi_study.sh` | 半监督范式 | 6 |
| `run_da_study.sh` | 域适应范式 | 8 |
| `run_kd_study.sh` | 知识蒸馏 | 7 |
| `run_weak_study.sh` | 弱监督范式 | 6 |

所有脚本位于 `scripts/experiments/`，第一个参数可指定折数或数据集。
