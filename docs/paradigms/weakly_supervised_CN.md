# 弱监督分割

[English](weakly_supervised.md)

本框架内置 **28** 种弱监督方法，位于 `medseg/training/weakly_supervised/`。

## 方法列表

### 核心方法 (16)

| 方法 | 论文 | 发表 | 说明 |
|------|------|------|------|
| `box_supervised` | BoxSup family | - | 框监督：框生成掩码 + 前景/背景 CE |
| `cam` | Zhou et al. / Selvaraju et al. | CVPR 2016 / ICCV 2017 | 类激活映射 (Grad-CAM) |
| `mil` | 多实例学习 | - | 图像级多实例学习 |
| `em_pseudo_label` | EM 优化 | - | EM 伪标签优化 |
| `point` | Bearman et al. | ECCV 2016 | 点监督 |
| `gated_crf` | Obukhov et al. | NeurIPS 2019 | 可微 CRF |
| `affinity` | AffinityNet 风格 | - | 像素亲和传播 |
| `tree_energy` | 树能量 | - | 树结构能量最小化 |
| `seam` | Wang et al. | CVPR 2020 | 自监督等变注意力 |
| `puzzle_cam` | Jo & Yu | ICIP 2021 | 拼图匹配 CAM |
| `advcam` | Lee et al. | CVPR 2021 | 对抗互补擦除 |
| `mctformer` | Xu et al. | CVPR 2022 | 多类 token transformer |
| `sam_guided_weak` | SAM 引导 | - | SAM 伪标签优化 |
| `iseg` | 交互式分割 | - | 基于点击的交互式监督 |
| `click_supervision` | 基于点击 | - | 点击点监督 |
| `scribble_sup` | 涂鸦监督 | - | 涂鸦标注监督 |

### 扩展方法 (12)

| 方法 | 论文 | 发表 | GitHub | 说明 |
|------|------|------|--------|------|
| `eps` | EPS | - | - | 显式伪标签监督 |
| `boxinst` | BoxInst | - | - | 框级实例分割 |
| `recam` | ReCAM | - | - | 重加权 CAM |
| `toco` | ToCo | - | - | Token 对比 |
| `lpcam` | LPCAM | - | - | 低通滤波 CAM |
| `mars` | MARS | - | - | 掩码感知精炼 |
| `bacon` | BACoN | - | - | 背景感知对比网络 |
| `wpgseg` | WPGSeg | - | - | 弱监督渐进引导 |
| `dupl` | DuPL | - | - | 双伪标签 |
| `more` | MoRe | - | - | 动量精炼 |
| `psdpm` | PSDPM | - | - | 先验伪标签去噪 |
| `semples` | SemPLeS | - | - | 语义伪标签选择 |

## 标注格式

弱监督方法使用不同类型的标注，均通过 JSON 文件加载。数据集类 (`WeaklySupervisedDataset`) 支持四种监督类型：

### 图像级标签 (`image_label`)

仅需图像级类别存在标签，无需空间标注。

```json
[
  {"image": "img_0001.png", "image_labels": [0, 2]},
  {"image": "img_0002.png", "image_labels": {"0": true, "1": false, "2": true}}
]
```

**使用方法：** `cam`、`mil`、`seam`、`puzzle_cam`、`advcam`、`mctformer`、`recam`、`lpcam`、`toco`、`mars`、`bacon`、`wpgseg`、`dupl`、`more`、`psdpm`、`semples`、`eps`、`em_pseudo_label`

### 边界框 (`box`)

归一化 `[x1, y1, x2, y2]` 格式的边界框（0–1 范围）。

```json
[
  {
    "image": "img_0001.png",
    "boxes": [[0.1, 0.2, 0.8, 0.9], [0.3, 0.4, 0.6, 0.7]]
  }
]
```

**使用方法：** `box_supervised`、`boxinst`

### 点 (`point`)

点击点以 `[x, y, class_id]` 元组表示（归一化坐标）。

```json
[
  {
    "image": "img_0001.png",
    "points": [[0.5, 0.3, 1], [0.2, 0.7, 0], [0.8, 0.6, 2]]
  }
]
```

**使用方法：** `point`、`click_supervision`、`iseg`

### 涂鸦 (`scribble`)

涂鸦线以 `[x, y]` 坐标对表示（归一化坐标）。

```json
[
  {
    "image": "img_0001.png",
    "scribbles": [[0.1, 0.2], [0.15, 0.25], [0.2, 0.3], [0.5, 0.5]]
  }
]
```

**使用方法：** `scribble_sup`

### 预计算 CAM（可选）

对于基于 CAM 的方法，预计算的类激活映射可存储为 `.npy` 文件，放在单独目录中。每个 `.npy` 文件形状为 `[num_classes, H, W]`，文件名应与图像匹配（如 `img_0001.npy` 对应 `img_0001.png`）。

```yaml
data:
  cam_dir: ./data/cams   # 存放 .npy CAM 文件的目录
```

### 监督类型汇总

| 类型 | 标注方式 | 数据集类 | 方法 |
|------|----------|----------|------|
| 图像级 | 类别标签 | `ImageLabelDataset` | cam、mil、seam、puzzle_cam、advcam、mctformer + 12 个扩展方法 |
| 框 | 边界框 | `BoxSupervisedDataset` | box_supervised、boxinst |
| 点 | 点击点 | `WeaklySupervisedDataset(point)` | point、click_supervision、iseg |
| 涂鸦 | 涂鸦线 | `WeaklySupervisedDataset(scribble)` | scribble_sup |
| 混合 | 多种类型 | — | sam_guided_weak、eps |

## 配置示例

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
  bottleneck:
    name: none

data:
  img_size: 224
  image_dir: ./data/images
  label_file: ./data/annotations/image_labels.json   # 图像级标签
  cam_dir: ./data/cams                                # 预计算 CAM（可选）
  val:
    image_dir: ./data/val/images
    mask_dir: ./data/val/masks
  test:
    image_dir: ./data/test/images
    mask_dir: ./data/test/masks

weak_supervision:
  method: affinity           # 上表任意方法名
  params:
    affinity_weight: 1.0
    propagation_steps: 3
    temperature: 0.5

training:
  epochs: 150
  batch_size: 16
  num_workers: 4
  val_interval: 10
  save_interval: 30
  loss:
    name: affinity_loss
    params:
      affinity_weight: 1.0
      propagation_steps: 3
      temperature: 0.5
  optimizer:
    name: adamw
    lr: 2e-4
    weight_decay: 1e-4
  scheduler:
    name: cosine
    min_lr: 1e-6
```

## 用法

```bash
# 训练
python train_weakly_supervised.py \
    --config configs/training_paradigms/weak_supervision/cam.yaml

# 框监督
python train_weakly_supervised.py \
    --config configs/training_paradigms/weak_supervision/box_supervised.yaml

# 点监督
python train_weakly_supervised.py \
    --config configs/training_paradigms/weak_supervision/point.yaml

# 涂鸦监督
python train_weakly_supervised.py \
    --config configs/training_paradigms/weak_supervision/scribble_sup.yaml

# 测试
python test.py --config configs/training_paradigms/weak_supervision/cam.yaml \
    --checkpoint output/best_model.pth
```

每个方法的配置位于 `configs/training_paradigms/weak_supervision/`。
