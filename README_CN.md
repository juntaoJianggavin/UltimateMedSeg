<div align="center">
  <img src="figs/logo.png" alt="UltimateMedSeg Logo" width="500"/>

  <p>
    <strong>Juntao Jiang</strong>,
    <strong>Jinsheng Bai</strong>,
    <strong>Linxuan Fan</strong>,
    <strong>Jiangning Zhang</strong>,
    <strong>Yong Liu</strong>
  </p>

  <p>
    <a href="README.md">English</a>
  </p>
</div>

> **136** 完整网络 · **172** 编码器 · **40** 解码器 · **88** 损失函数 · **25** 跳跃连接 · **17** 瓶颈层 · **6** 大训练范式 · **24** 种数据增强 · **878** YAML 配置 · 一行 YAML 完成切换

---

## 📑 目录

- [安装](#安装)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [模型组件](#模型组件)
- [训练范式](#训练范式)
- [部署与效率](#部署与效率)
- [数据集](#数据集)
- [配置系统](#配置系统)
- [自定义扩展](#自定义扩展)
- [引用与许可](#引用与许可)

---

## 📦 安装

### 环境要求

- Python >= 3.8
- PyTorch >= 2.0
- CUDA（推荐）/ CPU / Apple Silicon (MPS) 均可

### 基础安装

```bash
git clone <repo_url>
cd segmentation_tool

# 安装依赖
pip install -r requirements.txt

# 开发模式安装
pip install -e .
```

### 可选依赖

```bash
# Foundation 模型
pip install timm transformers huggingface_hub safetensors

# 数据增强
pip install albumentations

# 训练可视化
pip install tensorboard wandb

# MLLM 推理 pipeline
pip install groundingdino-py
pip install git+https://github.com/facebookresearch/segment-anything.git

# ONNX 导出与验证
pip install onnx onnxruntime

# Lion 优化器
pip install lion-pytorch
```

### 预训练权重自动下载

```bash
# 列出所有可自动下载的权重
python -m medseg.utils.weight_downloader list

# 下载指定权重
python -m medseg.utils.weight_downloader download medsam_vit_b

# 检查缓存状态
python -m medseg.utils.weight_downloader check
```

timm 编码器权重自动下载，无需手动管理。

---

## 🚀 快速开始

### 1. 标准监督训练

```bash
# ResNet50 + UNet decoder
python train.py --config configs/architectures/networks/general/aau_net.yaml \
    --output_dir output/aau_net

# 使用 AMP 混合精度
python train.py --config configs/architectures/networks/general/transunet.yaml \
    --output_dir output/transunet --amp

# 多卡 DDP 训练
torchrun --nproc_per_node=4 train.py \
    --config configs/architectures/networks/general/swinunet.yaml \
    --output_dir output/swinunet --amp
```

### 2. 半监督训练

```bash
python semi_train.py --config configs/training_paradigms/semi/mean_teacher.yaml \
    --output_dir output/semi_mt
```

### 3. ONNX 导出

```bash
python scripts/export_onnx.py \
    --config configs/architectures/networks/general/transunet.yaml \
    --checkpoint output/best_model.pth \
    --output model.onnx --verify
```

### 4. 预测可视化

```bash
python scripts/visualize.py \
    --config configs/architectures/networks/general/transunet.yaml \
    --checkpoint output/best_model.pth \
    --input ./data/test/images/ \
    --output vis_output/
```

### 5. Python API

```python
from medseg.utils.config import load_config
from medseg.model_builder import build_model

cfg = load_config("configs/architectures/networks/general/transunet.yaml")
model = build_model(cfg)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"可训练参数量: {trainable / 1e6:.2f}M")
```

---

## 🏗️ 项目结构

```
segmentation_tool/
├── medseg/                                      # 核心框架
│   ├── models/                                  # 模型组件
│   │   ├── encoders/                            #   172 个编码器
│   │   │   ├── cnn/              (12 modules)   #     CNN: basic, ResNet, ConvNeXt, EfficientNet, MedNeXt, MEW, R2U, AttUNet, ...
│   │   │   ├── transformer/      (18 modules)   #     Transformer: TransUNet, SwinUNet, MISSFormer, DAEFormer, HiFormer, PVTv2, MaxViT, ...
│   │   │   ├── mamba/            (10 modules)   #     Mamba/SSM: VMUNet, UMamba, LKM, LoG-VMamba, UltraLight-VM, VMKLA, ...
│   │   │   ├── rwkv/             (4 modules)    #     RWKV: RWKV-UNet, U-RWKV, MD-RWKV, RIR-Zigzag
│   │   │   ├── linear_attn/      (5 modules)    #     线性注意力: RetNet, Linformer, Performer, TTT, xLSTM
│   │   │   ├── kan_mlp/          (4 modules)    #     KAN/MLP: UKAN, Rolling-UNet, UNeXt, Wav-KAN
│   │   │   ├── foundation/       (38 modules)   #     Foundation 模型 (DPT head)
│   │   │   │   ├── general/      (5)            #       DINOv2, DINOv3, DINO, CLIP-ViT, SAM-ViT
│   │   │   │   ├── pathology/    (6)            #       Phikon, UNI, PLIP, MUSK, PathFoundation, Phikon-v2
│   │   │   │   ├── radiology/    (4)            #       Rad-DINO, CXR-Foundation, OmniRad, MedSigLIP
│   │   │   │   ├── ophthalmology/(4)            #       RETFound-DINOv2, FLAIR, OphMAE, RETFound
│   │   │   │   ├── dermatology/  (4)            #       DermFoundation, PanDerm, DermCLIP, MonetDerm
│   │   │   │   ├── multimodal_med/(3)           #       BiomedCLIP, MedCLIP, KEEP
│   │   │   │   ├── mllm_vision/  (8)            #       Qwen3-VL, MedGemma, LLaVA-Med, HuatuoGPT, ...
│   │   │   │   ├── endoscopy/    (1)            #       EndoViT
│   │   │   │   └── ultrasound/   (3)            #       UltraDINO, UltraFedFM, USF-MAE
│   │   │   └── wrapper/          (1 module)     #     timm 动态 wrapper (1000+ 模型，timm_ 前缀即用)
│   │   ├── decoders/                            #   40 个解码器
│   │   │   ├── basic/            (4 registered) #     基础上采样: UNet, Bilinear, Deconv, DepthwiseSep
│   │   │   ├── dense/            (2 registered) #     密集连接: UNet++, UNet3+
│   │   │   ├── cascade/          (5 registered) #     CASCADE, EMCAD (2 变体), G-CASCADE, CFM
│   │   │   ├── attention/        (3 registered) #     注意力门控, HAM, Lawin
│   │   │   ├── transformer/      (7 registered) #     DAEFormer, MTUNet, nnFormer, SwinUNet, H2Former, MISSFormer, ScaleFormer
│   │   │   ├── mlp/              (2 registered) #     SegFormer MLP, MLP 解码器
│   │   │   ├── specific/         (15 registered)#     TransUNet CUP, HiFormer, UCTransNet, FAT-Net, MALUNet, EGE-UNet, MERIT, ...
│   │   │   ├── pyramid/          (1 registered) #     金字塔: UPerNet
│   │   │   └── mamba/            (1 registered) #     Mamba: VM-UNet
│   │   ├── bottlenecks/          (17 modules)   #   17 个瓶颈层: none, basic, ASPP, DenseASPP, PPM, Transformer, SE, CBAM, ...
│   │   ├── skip_connections/                    #   25 个跳跃连接
│   │   │   ├── basic/            (2 modules)    #     基础: concat, dense
│   │   │   ├── attention/        (10 modules)   #     注意力: AG, CAB, SAB, SCSE, CBAM, Gating, GRU, GAB, SC-Att, TA-MoSC
│   │   │   ├── transformer/      (5 modules)    #     Transformer: CrossAttn, TransFusion, AggAttn, MISSFormer, UCTrans
│   │   │   ├── mamba/            (1 module)     #     Mamba: SK-VM++ (BSPC 2025)
│   │   │   └── fusion/           (6 modules)    #     CNN融合: BiFusion, Deformable, MultiScale, FeatureRefine, CCM, SDI
│   │   ├── networks/                            #   136 个完整网络
│   │   │   ├── cnn/              (40 registered)#     CNN: UNet3+, UNet++, AttUNet, nnUNet, MedNeXt, ACC-UNet, CMUNeXt, STUNet, ...
│   │   │   ├── transformer/      (33 registered)#     Transformer: TransUNet, SwinUNet, DAEFormer, PolypPVT, CASCADE, SEPNet, CTNet, ...
│   │   │   ├── mamba/            (23 registered)#     Mamba: VMUNet, UMamba, SwinUMamba, SkinMamba, DermoMamba, SerpMamba, ...
│   │   │   ├── sam/              (12 registered)#     SAM 家族: MedSAM, SAM-Med2D, SAM2, SAMUS, AutoSAM, MobileSAM, ...
│   │   │   ├── rwkv/             (4 registered) #     RWKV: U-RWKV, RWKV-UNet, MD-RWKV, RIR-Zigzag
│   │   │   ├── kan_mlp/          (7 registered) #     KAN/MLP: UKAN, Rolling-UNet (4 变体), UNeXt, Wav-KAN
│   │   │   └── linear_attn/      (4 registered) #     线性注意力: TTT-UNet, xLSTM-UNet (2 变体), U-VixLSTM
│   │   └── text_unet/            (13 modules)   #   文本引导: CRIS, BiomedParse, LanGuideMedSeg, LViT, TGANet, TPRO, ...
│   ├── training/                                # 训练范式
│   │   ├── semi/                 (23 modules)   #   21 个半监督: MeanTeacher, CPS, UniMatch, FixMatch, AugSeg, CorrMatch, ...
│   │   ├── domain_adaptation/    (18 modules)   #   18 个域适应: AdvEnt, DANN, TENT, FDA, MIC, HRDA, SePiCo, ...
│   │   ├── distillation/         (28 modules)   #   27 个蒸馏: VanillaKD, DKD, MGD, DIST, CWD, ReviewKD, SimKD, NORM, ...
│   │   └── weakly_supervised/    (28 modules)   #   28 个弱监督: Box, CAM, Point, Scribble, SEAM, PuzzleCAM, EPS, ...
│   ├── inference/                               # 推理
│   │   └── mllm/                 (16 modules)   #   MLLM pipeline: 5 detector × 4 segmenter = 20 种组合
│   │       │                                    #     Detector: GroundingDINO, Qwen2/2.5/3-VL, InternVL
│   │       │                                    #     Segmenter: SAM2, MedSAM, SAM-Med2D, LiteMedSAM
│   │       └── medisee/          (3 modules)    #     MediSee: LLM reasoning segmenter (ACM MM 2025)
│   ├── losses/                   (15 modules)   # 88 个损失函数
│   │                                            #   监督: CE, Dice, Focal, Tversky, Lovász, Boundary, Hausdorff, ...
│   │                                            #   蒸馏: VanillaKD, DKD, CWD, MGD, DIST, AT, RKD, ...
│   │                                            #   域适应: AdvEnt, DANN, FDA, MIC, TENT, ...
│   │                                            #   弱监督: Box, CAM, Point, Scribble, TreeEnergy, GatedCRF, ...
│   ├── datasets/                 (10 modules)   # 数据加载: Synapse, ACDC, Generic, QaTa-COV19, MosMedData+, 24 种增强
│   │   ├── advanced_aug.py                      #   24 种高级数据增强 (YAML 可配置)
│   │   └── transforms.py                        #   基础变换 (Resize, ToTensor, Normalize)
│   ├── utils/                    (8 modules)    # 工具
│   │   ├── amp_ddp.py                           #   AMP 混合精度 + DDP 分布式 + DataParallel 多卡
│   │   ├── logger.py                            #   TensorBoard / WandB 统一日志
│   │   ├── config.py                            #   配置继承 (_base_ 字段支持)
│   │   ├── warmup.py                            #   Warmup 调度器 + Lion/AdamW/SGD 优化器
│   │   ├── augmentation.py                      #   数据增强构建器 (basic/albumentations/pipeline)
│   │   ├── reproducibility.py                   #   可复现性 (全局 seed + cuDNN 确定性)
│   │   ├── weight_downloader.py                 #   权重自动下载 + 手动 URL 提示
│   │   └── metrics.py                           #   评估指标: Dice, IoU, HD95, NSD
│   ├── model_builder.py                         # YAML → 模型自动组装器
│   └── registry.py                              # 6 个注册表: ENCODER / DECODER / SKIP / BOTTLENECK / LOSS / AUGMENTATION
├── data/                                        # 数据集根目录（用户数据集放在这里）
│   ├── YourDataset/                             #   你的自定义数据集
│   ├── source/                                  #   域适应源域
│   ├── target/                                  #   域适应目标域
│   ├── target_val/                              #   域适应验证集
│   └── test_dummy/                              #   虚拟测试数据
├── figs/                                        # 图片与 logo
│   └── logo.png                                 #   项目 logo
├── examples/                                    # 使用示例
│   └── grounding_dino_example.py                #   GroundingDINO 检测示例
├── configs/                      (878 yamls)    # YAML 配置
│   ├── architectures/            (751 yamls)    #   网络结构配置
│   │   ├── networks/             (281 yamls)    #     完整网络 (general/acdc/synapse × 120+ arch)
│   │   ├── combinations/         (166 yamls)    #     encoder+decoder 自由组合
│   │   ├── decoder_study/        (121 yamls)    #     Decoder 消融 (3 enc × 40 dec)
│   │   ├── skip_study/           (75 yamls)     #     skip 消融 (3 enc × 25 skip)
│   │   ├── bottleneck_study/     (51 yamls)     #     bottleneck 消融 (3 enc × 17 bn)
│   │   └── foundation/           (57 yamls)     #     Foundation 模型 (9 模态 × 38 编码器)
│   ├── training_paradigms/       (99 yamls)     #   训练范式配置
│   │   ├── semi_supervision/     (21 yamls)     #     半监督 (21 方法)
│   │   ├── domain_adaptation/    (18 yamls)     #     域适应 (18 方法)
│   │   ├── distillation/         (22 yamls)     #     蒸馏 (27 方法)
│   │   ├── text_guided/          (19 yamls)     #     文本引导 (13 模型 + pipeline)
│   │   └── weak_supervision/     (19 yamls)     #     弱监督 (28 方法)
│   ├── intro_to_datasets/        (25 yamls)     #   25 个数据集介绍 + 示例配置
│   └── experiments/                             #   实验配置
├── scripts/                                     # 工具 + 实验脚本
│   ├── experiments/              (14 scripts)   #   实验 bash 脚本
│   │   ├── run_sota_benchmark.sh                #     通用 SOTA 架构对比 (11 模型 × 7 数据集)
│   │   ├── run_decoder_study.sh                 #     Decoder 消融 (3 enc × 15 经典 dec)
│   │   ├── run_bottleneck_study.sh              #     Bottleneck 消融 (3 enc × 9 bn)
│   │   ├── run_skip_study.sh                    #     Skip 消融 (3 enc × 12 skip)
│   │   ├── run_polyp_benchmark.sh               #     息肉专有模型 (16 模型 × 2 数据集)
│   │   ├── run_skin_benchmark.sh                #     皮肤专有模型 (16 模型 × 2 数据集 + PH2 外部验证)
│   │   ├── run_retinal_benchmark.sh             #     视网膜专有模型 (7 模型 × 3 数据集)
│   │   ├── run_ultrasound_benchmark.sh          #     超声专有模型 (8 模型 × BUSI)
│   │   ├── run_pathology_benchmark.sh           #     病理专有模型 (5 模型 × GlaS)
│   │   ├── run_lightweight_skin.sh              #     轻量化皮肤分割 (8 模型)
│   │   ├── run_semi_study.sh                    #     半监督范式对比 (6 方法)
│   │   ├── run_da_study.sh                      #     域适应范式对比 (8 方法)
│   │   ├── run_kd_study.sh                      #     知识蒸馏对比 (7 方法)
│   │   └── run_weak_study.sh                    #     弱监督范式对比 (6 方法)
│   ├── export_onnx.py                           #   ONNX 模型导出 (支持动态尺寸 + ORT 验证)
│   ├── visualize.py                             #   预测可视化 (input + pred + overlay)
│   ├── test_all_configs.py                      #   配置批量测试 (build + forward + loss)
│   └── prepare_qata_mosmed.py                   #   QaTa-COV19 / MosMedData+ 数据集验证
├── docs/                         (15 docs)      # 详细文档
│   ├── models/                                  #   模型文档: 总览, 网络, 编码器, 解码器, skip, bottleneck
│   ├── paradigms/                               #   范式文档: 基础设施, 半监督, 弱监督, 域适应, 蒸馏, 文本引导
│   ├── deployment/                              #   部署文档: ONNX, FLOPs, 参数量, FPS
│   ├── data/                                    #   数据文档: 25 个数据集, 5 种类型, 4 种划分
│   └── research_guide.md                        #   研究建议: 8 个研究方向 + 14 个实验脚本
├── train.py                                     # 监督训练 (AMP + DDP + DataParallel + Logger + Warmup)
├── semi_train.py                                # 半监督训练 (21 方法)
├── train_weakly_supervised.py                   # 弱监督训练 (28 方法)
├── train_domain_adaptation.py                   # 域适应训练 (18 方法)
├── train_distillation.py                        # 知识蒸馏训练 (27 方法)
├── train_text_guided.py                         # 文本引导训练 (13 模型)
├── test.py                                      # 推理 / 测试
├── profile_model.py                             # FLOPs / 参数量 / FPS 分析
├── setup.py                                     # 包安装配置
└── requirements.txt                             # Python 依赖
```

---

## 🧩 模型组件

> 详细文档: [docs/models/](docs/models/README_CN.md)

### 完整网络 — 136 个

| 类别 | 数量 | 代表模型 |
|---|---|---|
| CNN | 40 | UNet3+, UNet++, Attention-UNet, nnU-Net, MedNeXt, ACC-UNet, CMUNeXt |
| Transformer | 33 | TransUNet, Swin-UNet, DAEFormer, MISSFormer, HiFormer, PolypPVT, CASCADE |
| Mamba / SSM | 23 | VM-UNet, U-Mamba, Swin-UMamba, LKM-UNet, LoG-VMamba, HC-Mamba |
| SAM 家族 | 12 | MedSAM, SAM-Med2D, SAM2, SAMUS, AutoSAM, MobileSAM |
| KAN / MLP | 7 | U-KAN, Rolling-UNet (4 变体), UNeXt, Wav-KAN |
| 线性注意力 | 4 | TTT-UNet, xLSTM-UNet (2 变体), U-VixLSTM |
| RWKV | 4 | U-RWKV, RWKV-UNet, MD-RWKV-UNet, RIR-Zigzag |
| 文本引导 | 13 | CRIS, BiomedParse, LanGuideMedSeg, LViT, TGANet, TPRO, CausalCLIPSeg |

> 详细列表: [docs/models/networks.md](docs/models/networks.md)

### 编码器 — 172 个

**亮点：38 个 Foundation 模型编码器，覆盖 9 个医学模态**

| 模态 | 数量 | 模型 |
|---|---|---|
| 通用 | 5 | DINOv2, DINOv3, DINO, CLIP-ViT, SAM-ViT |
| 病理 | 6 | Phikon, Phikon-v2, UNI, PLIP, MUSK, PathFoundation |
| 放射 | 4 | Rad-DINO, CXR-Foundation, OmniRad, MedSigLIP |
| 眼科 | 4 | RETFound-DINOv2, RETFound, FLAIR, OphMAE |
| 皮肤 | 4 | DermFoundation, DermCLIP, MoNet, PanDerm |
| 多模态医学 | 3 | BiomedCLIP, MedCLIP, KEEP |
| MLLM视觉 | 8 | Qwen2.5-VL, Qwen3-VL, MedGemma, LLaVA-Med, HuatuoGPT, HealthGPT, HuLuMed, LingShu |
| 超声 | 3 | UltraDINO, UltraFedFM, US-FMAE |
| 内窥镜 | 1 | Endo-ViT |

所有 Foundation ViT 使用 **DPT head**（从不同深度 block 提取多尺度特征），而非简单的 FPN-from-tokens。

**timm 动态 encoder**：任何 `timm.list_models()` 中的模型加 `timm_` 前缀即可使用，无需预注册。

```yaml
encoder:
  name: timm_efficientnet_b7    # 或任何 timm 模型名
  pretrained: true
```

> 详细列表: [docs/models/encoders.md](docs/models/encoders.md)

### 解码器 — 40 个

| 类别 | 数量 | 代表模型 |
|---|---|---|
| 基础上采样 | 4 | UNet, Bilinear, Deconv, DepthwiseSep |
| 密集连接 | 2 | UNet++, UNet3+ |
| 级联 | 5 | CASCADE, EMCAD, G-CASCADE, CFM |
| 注意力 | 3 | Attention Gate, HAM, Lawin |
| Transformer | 7 | DAEFormer, MTUNet, SwinUNet, nnFormer, H2Former, MISSFormer, ScaleFormer |
| MLP | 2 | SegFormer MLP, MLP 解码器 |
| 网络专属 | 15 | TransUNet CUP, HiFormer, UCTransNet, FAT-Net, MALUNet, EGE-UNet, MERIT, ... |
| Mamba | 1 | VM-UNet |
| 金字塔 | 1 | UPerNet |

> 详细列表: [docs/models/decoders.md](docs/models/decoders.md)

### 跳跃连接 — [docs/models/skip_connections.md](docs/models/skip_connections.md)

### 瓶颈层 — [docs/models/bottlenecks.md](docs/models/bottlenecks.md)

---

## 🎓 训练范式

> 详细文档: [docs/paradigms/](docs/paradigms/README_CN.md)

### 基础设施

| 功能 | yaml 配置 |
|---|---|
| 混合精度 AMP | `training.amp: true` 或 CLI `--amp` |
| 多卡 DDP | `torchrun --nproc_per_node=N train.py` |
| DataParallel | `training.parallel: dp` |
| TensorBoard | `training.logger: tensorboard` |
| WandB | `training.logger: wandb` |
| 可复现性 Seed | `training.random_state: 42` + `training.deterministic: true` |
| Warmup 调度 | `training.scheduler.name: warmup_cosine` + `warmup_epochs: 10` |
| 配置继承 | `_base_: ../base.yaml` |
| Albumentations | `training.augmentation: albumentations` |
| YAML 增强管线 | `training.augmentation: pipeline` + `training.aug_pipeline: [...]` |

> 详细配置: [docs/paradigms/README.md](docs/paradigms/README_CN.md)

### 数据增强管线 — 24 种方法

通过 YAML 配置自由组合 24 种数据增强方法，无需修改代码。所有增强方法均支持强度范围参数，每次调用时随机采样。

```yaml
training:
  augmentation: pipeline        # 启用管线模式
  aug_pipeline:                 # 按顺序定义增强方法
    - name: horizontal_flip
      params: { p: 0.5 }
    - name: vertical_flip
      params: { p: 0.5 }
    - name: random_rotate90
      params: { p: 0.5 }
    - name: random_rotate
      params: { p: 0.3, degrees_range: [-30, 30] }
    - name: random_affine
      params: { p: 0.3, degrees_range: [-15, 15], translate_range: [0.0, 0.1], scale_range: [0.8, 1.2] }
    - name: elastic_deform
      params: { p: 0.3, alpha_range: [20, 80], sigma_range: [3, 7] }
    - name: copy_paste
      params: { p: 0.3, max_objects: 2, scale_range: [0.5, 1.5] }
    - name: mosaic
      params: { p: 0.3, offset_range: [0.0, 0.2] }
    - name: clahe
      params: { p: 0.3, clip_limit_range: [1.0, 5.0], tile_size_range: [4, 16] }
    - name: gamma_correction
      params: { p: 0.3, gamma_range: [0.7, 1.5] }
    - name: gaussian_blur
      params: { p: 0.2, kernel_range: [3, 7], sigma_range: [0.1, 2.0] }
    - name: gaussian_noise
      params: { p: 0.2, std_range: [0.01, 0.08] }
```

**支持的增强方法 (24 种)**:

| 类别 | 方法 |
|---|---|
| 几何变换 | `horizontal_flip`, `vertical_flip`, `random_rotate90`, `random_rotate`, `random_affine`, `random_perspective`, `random_scale`, `elastic_deform`, `grid_mask` |
| 像素变换 | `photometric_distortion`, `color_jitter`, `brightness_contrast`, `gamma_correction`, `clahe`, `gaussian_blur`, `gaussian_noise`, `sharpness`, `posterize`, `random_solarize`, `channel_dropout` |
| 遮挡 | `random_erasing`, `coarse_dropout`, `grid_mask` |
| 样本级 | `copy_paste`, `mosaic` |

> **注意**: 所有强度参数均使用 `_range` 后缀命名（如 `degrees_range`, `alpha_range`），每次调用时从范围内随机采样。

> 每个增强方法的完整参数说明: [docs/data/README.md](docs/data/README_CN.md#数据增强管线--augmentation-pipeline--24-种方法)
> 完整配置示例: [configs/architectures/decoder_study/general/resnet50_unet_advanced_aug.yaml](configs/architectures/decoder_study/general/resnet50_unet_advanced_aug.yaml)

### 半监督 — 21 个方法

Mean Teacher · CPS · CCT · UniMatch · FixMatch · FlexMatch · FreeMatch · SoftMatch · UA-MT · URPC · Deep Co-Training · Pi-Model · Temporal Ensembling · Pseudo-Label · ICT · R-Drop · Cross-Teaching · AugSeg · CorrMatch · AllSpark · DDFP · DiffRect · AD-MT · PMT

> 详细: [docs/paradigms/semi_supervised.md](docs/paradigms/semi_supervised.md)

### 域适应 — 18 个方法

AdvEnt · DANN · TENT · DPL · CBMT · FDA · CRST · PixMatch · MIC · DAFormer · HRDA · PiPa · DDB · SePiCo · DiGA · MICDrop · SemiVL

> 详细: [docs/paradigms/domain_adaptation.md](docs/paradigms/domain_adaptation.md)

### 知识蒸馏 — 27 个方法

Vanilla KD · FitNets · AT · FSP · NST · RKD · VID · DKD · MGD · DIST · CIRKD · CWD · ReviewKD · SimKD · NORM · SDD · AICSD · LSKD · TTM · CTKD · MLKD + 4 个医学专用

> 详细: [docs/paradigms/distillation.md](docs/paradigms/distillation.md)

### 弱监督 — 28 个方法

Box · CAM · Point · Scribble · MIL · EM · GatedCRF · TreeEnergy · SEAM · PuzzleCAM · AdvCAM · EPS · BoxInst · ReCAM · ToCo · LPCAM · MARS · BACoN · WPGSeg · DuPL · MoRe · PSDPM · SemPLeS

> 详细: [docs/paradigms/weakly_supervised.md](docs/paradigms/weakly_supervised.md)

### 文本引导 — 13 个模型 + 推理 Pipeline

**可训练模型**: CRIS · BiomedParse · LanGuideMedSeg · LViT · TGANet · TPRO · CausalCLIPSeg · CLIP-Universal · CXR-CLIP-Seg · TP-DRSeg · MedCLIP-SAM · SaLIP · MediSee

**推理 Pipeline** (5 detector × 4 segmenter = 20 种组合):
- Detector: GroundingDINO · Qwen2-VL · Qwen2.5-VL · Qwen3-VL · InternVL
- Segmenter: SAM2 · MedSAM · SAM-Med2D · LiteMedSAM

> 详细: [docs/paradigms/text_guided.md](docs/paradigms/text_guided.md)

---

## ⚡ 部署与效率

> 详细文档: [docs/deployment/README.md](docs/deployment/README_CN.md)

```bash
# ONNX 导出
python scripts/export_onnx.py --config xxx.yaml --checkpoint best.pth --output model.onnx --verify

# FLOPs 计算
python -c "
from fvcore.nn import FlopCountAnalysis
import torch
flops = FlopCountAnalysis(model, torch.randn(1,3,224,224))
print(f'FLOPs: {flops.total()/1e9:.2f}G')
"

# 参数量（只算可训练参数）
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"可训练: {trainable/1e6:.2f}M / 总计: {total/1e6:.2f}M")
```

> 注意：冻结的 Foundation encoder 参数不计入可训练参数量。

---

## 📊 数据集

> 详细文档: [docs/data/README.md](docs/data/README_CN.md)
> 数据集示例配置: [configs/intro_to_datasets/](configs/intro_to_datasets/)

### 支持的数据集类型

| 类型 | 说明 |
|---|---|
| `synapse` | Synapse 多器官 CT (TransUNet 格式) |
| `acdc` | ACDC 心脏 MRI (TransUNet 格式) |
| `generic` | 通用 images/ + masks/ 目录 |
| `qata_covid19` | QaTa-COV19 胸部 X 光 + per-image 文本 (LViT 格式) |
| `mosmed_plus` | MosMedData+ COVID CT + per-image 文本 (LViT 格式) |

### 数据划分方式

```yaml
# 方式1: 直接指定路径
data:
  train_dir: ./data/train
  val_dir: ./data/val
  test_dir: ./data/test       # 可选

# 方式2: 按比例自动划分
data:
  root_dir: ./data/all
  train_ratio: 0.7
  val_ratio: 0.15

# 方式3: N 折交叉验证
data:
  root_dir: ./data/all
  n_splits: 5
  fold_idx: 0
```

### 已收录数据集 (25 个)

**皮肤**: ISIC 2016/2017/2018, PH2
**息肉**: CVC-ClinicDB, CVC-ColonDB, Kvasir-SEG
**病理**: GlaS, PanNuke, MoNuSeg
**眼底**: DRIVE, STARE, CHASE_DB1, HRF, ARIA, RITE, REFUGE, Drishti-GS
**胸部**: Montgomery+Shenzhen CXR, QaTa-COV19, COVID CT Seg
**超声**: BUSI
**多器官**: Synapse, ACDC
**CT**: MosMedData+

---

## 🔧 配置系统

### 两种模型配置模式

```yaml
# 模式1: 模块组合（encoder + decoder + skip + bottleneck）
model:
  num_classes: 9
  img_size: 224
  encoder:
    name: timm_resnet50
    pretrained: true
  decoder:
    name: unet
  skip_connection:
    name: concat
  bottleneck:
    name: aspp

# 模式2: 完整架构（architecture key）
model:
  num_classes: 9
  img_size: 224
  architecture: transunet
  arch_params: {}
```

### 配置继承

```yaml
# child.yaml — 只写需要覆盖的部分
_base_: ../base_resnet50.yaml
model:
  num_classes: 9
training:
  epochs: 300
```

### 完整训练配置示例

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
  bottleneck:
    name: none

data:
  type: synapse
  img_size: 224
  train_dir: ./data/Synapse/train_npz
  val_dir: ./data/Synapse/test_vol_h5

training:
  random_state: 42
  deterministic: true
  amp: true
  parallel: auto
  logger: tensorboard
  augmentation: albumentations
  epochs: 200
  batch_size: 16
  num_workers: 4
  val_interval: 10
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
    weight_decay: 0.0001
  scheduler:
    name: warmup_cosine
    warmup_epochs: 10
    warmup_lr: 0.000001
    min_lr: 0.000001
```

---

## 🔌 自定义扩展

### 添加新编码器

```python
# medseg/models/encoders/cnn/my_encoder.py
from medseg.registry import ENCODER_REGISTRY

@ENCODER_REGISTRY.register("my_encoder")
class MyEncoder(nn.Module):
    def __init__(self, pretrained=False, in_channels=3, img_size=224, **kwargs):
        super().__init__()
        self.out_channels = [64, 128, 256, 512]
    def forward(self, x):
        return [f1, f2, f3, f4]  # 多尺度特征
```

### 添加新解码器

```python
@DECODER_REGISTRY.register("my_decoder")
class MyDecoder(nn.Module):
    has_internal_skip = False
    def __init__(self, encoder_channels, bottleneck_channels, skip_connection=None, **kwargs):
        super().__init__()
        self.out_channels = encoder_channels[0]
    def forward(self, bottleneck_feat, skip_features):
        return decoded
```

### 添加新损失

```python
@LOSS_REGISTRY.register("my_loss")
class MyLoss(nn.Module):
    def forward(self, pred, target):
        return loss_value
```

### 添加新数据增强

```python
# medseg/datasets/advanced_aug.py
from medseg.registry import AUGMENTATION_REGISTRY

@AUGMENTATION_REGISTRY.register("my_augmentation")
class MyAugmentation:
    def __init__(self, p=0.5, **kwargs):
        self.p = p

    def set_dataset(self, dataset):
        """可选：如果需要访问数据集，实现此方法"""
        self.dataset = dataset

    def __call__(self, sample: dict) -> dict:
        import random
        if random.random() > self.p:
            return sample
        image, label = sample['image'], sample['label']
        # ... 实现增强逻辑 ...
        return {'image': image, 'label': label}
```

注册后在 `medseg/datasets/__init__.py` 中 import，即可通过 YAML 中 `name: my_augmentation` 使用。

注册后在 `__init__.py` 中 import，即可通过 YAML 中 `name: my_encoder` 使用。

---

## 📜 引用与许可

```bibtex
@software{ultimatemedseg_2026,
  title  = {UltimateMedSeg: A Modern Modular 2D Medical Image Segmentation Toolbox},
  author = {Juntao Jiang and Jinsheng Bai and Linxuan Fan and Jiangning Zhang and Yong Liu},
  year   = {2026},
  url    = {https://github.com/juntaoJianggavin/UltimateMedSeg},
}
```

### 许可证

Apache 2.0. 仅限合法学术研究与工程应用，临床部署请遵循当地法规。

### 致谢

感谢 PyTorch、timm、MONAI、SSL4MIS、SAM、GroundingDINO、DINOv2/v3、CLIP、transformers 等开源项目。
