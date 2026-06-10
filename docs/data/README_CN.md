# 数据集

[English](README.md)

## 支持的数据集类型

| 类型 | 说明 | 加载器 |
|------|------|--------|
| `synapse` | Synapse 多器官 CT（TransUNet 格式：npz 训练 + h5 测试） | 专用 |
| `acdc` | ACDC 心脏 MRI（TransUNet 格式：npz + h5） | 专用 |
| `generic` | 任意包含 images/ + masks/ 文件夹的数据集 | 通用 |
| `qata_covid19` | QaTa-COV19 胸部 X 光 + 逐图文本（LViT 格式） | 文本感知 |
| `mosmed_plus` | MosMedData+ 新冠 CT + 逐图文本（LViT 格式） | 文本感知 |

---

## 数据划分方式

### 1. 显式目录

```yaml
data:
  type: generic
  img_size: 224
  train_dir: ./data/train
  val_dir: ./data/val
  test_dir: ./data/test
```

### 2. 比例划分

```yaml
data:
  type: generic
  img_size: 224
  image_dir: ./data/images
  mask_dir: ./data/masks
  split:
    mode: ratio
    train: 0.7
    val: 0.15
    test: 0.15
    seed: 42
```

### 3. N 折交叉验证

```yaml
data:
  type: generic
  img_size: 224
  image_dir: ./data/images
  mask_dir: ./data/masks
  split:
    mode: nfold
    n_folds: 5
    fold: 0              # 当前折（0-4）
    seed: 42
```

### 4. N 折 + 独立测试

```yaml
data:
  type: generic
  img_size: 224
  image_dir: ./data/images
  mask_dir: ./data/masks
  test_dir: ./data/test   # 固定测试集
  split:
    mode: nfold_test
    n_folds: 5
    fold: 0
    seed: 42
```

---

## 数据集汇总 (25)

提取自 `configs/intro_to_datasets/`。

### 腹部与心脏

| 数据集 | 模态 | 类别数 | 规模 | 划分 | YAML |
|--------|------|--------|------|------|------|
| Synapse | CT | 9（8 器官+背景） | 30 例 | 18 训练 / 12 测试 | [synapse.yaml](../../configs/intro_to_datasets/synapse.yaml) |
| ACDC | MRI | 4（RV/MYO/LV+背景） | 100 例 | 70/10/20 | [acdc.yaml](../../configs/intro_to_datasets/acdc.yaml) |

### 视网膜

| 数据集 | 任务 | 规模 | 类别数 | 划分 | YAML |
|--------|------|------|--------|------|------|
| DRIVE | 血管分割 | 40 张 | 2 | 20/20 官方 | [drive.yaml](../../configs/intro_to_datasets/drive.yaml) |
| STARE | 血管分割 | 20 张 | 2 | LOO 或跨数据集 | [stare.yaml](../../configs/intro_to_datasets/stare.yaml) |
| CHASE_DB1 | 血管分割 | 28 张 | 2 | 20 训练 / 8 测试 | [chase_db1.yaml](../../configs/intro_to_datasets/chase_db1.yaml) |
| HRF | 血管分割（高分辨率） | 45 张 | 2 | 5 折 CV | [hrf.yaml](../../configs/intro_to_datasets/hrf.yaml) |
| ARIA | 血管分割（多疾病） | 143 张 | 2 | 5 折 CV | [aria.yaml](../../configs/intro_to_datasets/aria.yaml) |
| RITE | 动脉/静脉分割 | 40 张 | 3 | 20/20（同 DRIVE） | [rite.yaml](../../configs/intro_to_datasets/rite.yaml) |
| REFUGE | 视盘/视杯分割 | 1200 张 | 3 | 400/400/400 | [refuge.yaml](../../configs/intro_to_datasets/refuge.yaml) |
| Drishti-GS | 视盘/视杯分割 | 101 张 | 3 | 50/51 | [drishti_gs.yaml](../../configs/intro_to_datasets/drishti_gs.yaml) |

### 皮肤病灶

| 数据集 | 规模 | 类别数 | 划分 | YAML |
|--------|------|--------|------|------|
| ISIC 2016 | 1279 张 | 2 | 900/379 | [isic2016.yaml](../../configs/intro_to_datasets/isic2016.yaml) |
| ISIC 2017 | 2750 张 | 2 | 2000/150/600 | [isic2017.yaml](../../configs/intro_to_datasets/isic2017.yaml) |
| ISIC 2018 | 3694 张 | 2 | 2594/100/1000 | [isic2018.yaml](../../configs/intro_to_datasets/isic2018.yaml) |
| PH2 | 200 张 | 2 | 5 折 CV 或外部测试 | [ph2.yaml](../../configs/intro_to_datasets/ph2.yaml) |

### 胃肠息肉

| 数据集 | 规模 | 分辨率 | 划分 | YAML |
|--------|------|--------|------|------|
| Kvasir-SEG | 1000 张 | 可变 | 5 折 CV | [kvasir_seg.yaml](../../configs/intro_to_datasets/kvasir_seg.yaml) |
| CVC-ClinicDB | 612 张 | 384x288 | 5 折 CV | [cvc_clinicdb.yaml](../../configs/intro_to_datasets/cvc_clinicdb.yaml) |
| CVC-ColonDB | 380 张 | 574x500 | 跨数据集测试 | [cvc_colondb.yaml](../../configs/intro_to_datasets/cvc_colondb.yaml) |

### 病理

| 数据集 | 组织 | 规模 | 类别数 | 划分 | YAML |
|--------|------|------|--------|------|------|
| GlaS | 结肠腺体 | 165 张 | 2 | 85 训练 / 80 测试 | [glas.yaml](../../configs/intro_to_datasets/glas.yaml) |
| MoNuSeg | 多器官细胞核 | 44 张 | 2 | 30/14 | [monuseg.yaml](../../configs/intro_to_datasets/monuseg.yaml) |
| PanNuke | 泛癌细胞核 | ~7900 patches | 6 | 3 折官方 | [pannuke.yaml](../../configs/intro_to_datasets/pannuke.yaml) |

### 胸部

| 数据集 | 模态 | 规模 | 任务 | YAML |
|--------|------|------|------|------|
| Montgomery+Shenzhen | CXR | 800 张 | 肺部分割 | [montgomery_shenzhen_cxr.yaml](../../configs/intro_to_datasets/montgomery_shenzhen_cxr.yaml) |
| COVID CT Seg | CT | 100 slices | 感染分割（GGO+实变） | [covid_ct_seg.yaml](../../configs/intro_to_datasets/covid_ct_seg.yaml) |
| QaTa-COV19 | CXR + 文本 | 10501 张 | 新冠感染分割 | [qata_covid19.yaml](../../configs/intro_to_datasets/qata_covid19.yaml) |
| MosMedData+ | CT + 文本 | 3674 slices | 新冠感染分割 | [mosmed_plus.yaml](../../configs/intro_to_datasets/mosmed_plus.yaml) |

### 超声

| 数据集 | 器官 | 规模 | 类别数 | YAML |
|--------|------|------|--------|------|
| BUSI | 乳腺 | 647 张 | 2（病灶） | [busi.yaml](../../configs/intro_to_datasets/busi.yaml) |

---

## Synapse / ACDC（TransUNet 预处理格式）

两种常见的 npz 布局（``SynapseDataset`` 会自动识别）：

| 布局 | ``image`` 形状 | 典型用途 | 示例 |
|------|----------------|----------|------|
| **单切片文件** | ``(H, W)`` | TransUNet 官方 train_npz / val_npz | ``case0005_slice000.npz`` |
| **3D 打包** | ``(D, H, W)`` | 自定义 smoke 数据、部分 3D 导出 | ``case0001.npz`` |

测试集 volume 使用 ``test_vol_h5/*.npy.h5`` 或 ``test_vol/*.npz``，形状 ``(D,H,W)``。

**下载**：推荐 [TransUNet 预处理 Google Drive](https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd)（Synapse）；ACDC 另见 [ACDC 预处理包](https://drive.google.com/drive/folders/1KQcrci7aKsYZi1hQoZ3T3QUtcy7b--n4)。

**训练前校验**：

```bash
python scripts/verify_synapse_acdc.py \
  --train-dir ./data/Synapse/train_npz \
  --val-dir ./data/Synapse/test_vol_h5
```

灰度 CT/MRI 会在 loader 内复制为 3 通道；``img_size`` 由 ``get_train_transforms`` 在线 resize。

---

## Synapse 配置示例

```yaml
model:
  num_classes: 9
  img_size: 224

data:
  type: synapse
  img_size: 224
  train_dir: ./data/Synapse/train_npz
  val_dir: ./data/Synapse/test_vol_h5
  test_dir: ./data/Synapse/test_vol_h5
  test_list: ./data/Synapse/lists/lists_Synapse/test_vol.txt
```

## ACDC 配置示例

```yaml
model:
  num_classes: 4
  img_size: 224

data:
  type: acdc
  img_size: 224
  train_dir: ./data/ACDC/train_npz
  val_dir: ./data/ACDC/val_npz
  test_dir: ./data/ACDC/test_vol
```

## 通用配置示例

```yaml
model:
  num_classes: 2
  img_size: 224

data:
  type: generic
  img_size: 224
  train_dir: ./data/DRIVE/training/images
  train_mask_dir: ./data/DRIVE/training/1st_manual
  val_dir: ./data/DRIVE/test/images
  val_mask_dir: ./data/DRIVE/test/1st_manual
```

## 文本感知配置

```yaml
model:
  num_classes: 1
  img_size: 224
  architecture: lvit

data:
  type: mosmed_plus
  img_size: 224
  data_root: ./data/MosMedDataPlus
  tokenizer_name: bert-base-uncased
  text_max_length: 10
  text_column: text
  text_source: dataset
```

所有数据集介绍配置位于 `configs/intro_to_datasets/`。

---

## 数据增强管线 — 24 种方法

> 所有增强方法均注册到 `AUGMENTATION_REGISTRY`，通过 YAML 配置使用。
> 所有强度参数均使用 `_range` 后缀命名，每次调用时从范围内随机采样。

### 启用方式

```yaml
training:
  augmentation: pipeline          # 切换到管线模式
  aug_pipeline:                   # 按顺序列出增强方法
    - name: horizontal_flip
      params: { p: 0.5 }
    - name: elastic_deform
      params: { p: 0.3, alpha_range: [20, 80], sigma_range: [3, 7] }
```

---

### 几何变换 (9 种)

#### `horizontal_flip`

随机水平翻转。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `p` | float | 0.5 | 应用概率 |

```yaml
- name: horizontal_flip
  params: { p: 0.5 }
```

#### `vertical_flip`

随机垂直翻转。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `p` | float | 0.5 | 应用概率 |

```yaml
- name: vertical_flip
  params: { p: 0.5 }
```

#### `random_rotate90`

随机 90°/180°/270° 旋转（保持图像尺寸不变）。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `p` | float | 0.5 | 应用概率 |

```yaml
- name: random_rotate90
  params: { p: 0.5 }
```

#### `random_rotate`

任意角度旋转（使用双线性插值，保持图像尺寸不变）。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `p` | float | 0.5 | 应用概率 |
| `degrees_range` | (float, float) | (-15, 15) | 旋转角度范围（度） |

```yaml
- name: random_rotate
  params: { p: 0.3, degrees_range: [-45, 45] }
```

#### `random_affine`

随机仿射变换（旋转 + 平移 + 缩放 + 剪切）。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `p` | float | 0.5 | 应用概率 |
| `degrees_range` | (float, float) | (-15, 15) | 旋转角度范围（度） |
| `translate_range` | (float, float) | (0.0, 0.1) | 平移范围（图像尺寸比例） |
| `scale_range` | (float, float) | (0.9, 1.1) | 缩放范围 |
| `shear_range` | (float, float) | (-5, 5) | 剪切角度范围（度） |

```yaml
- name: random_affine
  params:
    p: 0.3
    degrees_range: [-15, 15]
    translate_range: [0.0, 0.1]
    scale_range: [0.9, 1.1]
    shear_range: [-5, 5]
```

#### `random_perspective`

随机透视变换。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `p` | float | 0.5 | 应用概率 |
| `distortion_scale_range` | (float, float) | (0.05, 0.15) | 透视畸变强度范围（图像尺寸比例） |

```yaml
- name: random_perspective
  params: { p: 0.3, distortion_scale_range: [0.05, 0.2] }
```

#### `random_scale`

随机缩放后 resize 回原始尺寸（先放大/缩小，再插值回原尺寸）。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `p` | float | 0.5 | 应用概率 |
| `scale_range` | (float, float) | (0.8, 1.2) | 缩放因子范围 |

```yaml
- name: random_scale
  params: { p: 0.3, scale_range: [0.7, 1.3] }
```

#### `elastic_deform`

随机弹性形变（模拟组织变形）。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `p` | float | 0.3 | 应用概率 |
| `alpha_range` | (float, float) | (20, 80) | 形变幅度范围（越大形变越强） |
| `sigma_range` | (float, float) | (3, 7) | 高斯平滑 sigma 范围（越大越平滑） |

```yaml
- name: elastic_deform
  params: { p: 0.3, alpha_range: [20, 100], sigma_range: [3, 8] }
```

#### `grid_mask`

网格遮挡：按规律遮挡网格区域，迫使模型学习鲁棒特征。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `p` | float | 0.5 | 应用概率 |
| `d_range` | (float, float) | (0.05, 0.15) | 网格单元大小范围（图像尺寸比例） |
| `ratio_range` | (float, float) | (0.3, 0.7) | 每个单元内遮挡面积比例范围 |
| `rotate_range` | (float, float) | (0, 0) | 网格旋转角度范围（度） |

```yaml
- name: grid_mask
  params: { p: 0.3, d_range: [0.05, 0.2], ratio_range: [0.3, 0.7] }
```

---

### 像素变换 (11 种)

#### `photometric_distortion`

光度畸变：随机亮度、对比度、饱和度、色调变换（随机顺序）。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `p` | float | 0.5 | 应用概率 |
| `brightness_range` | (float, float) | (-0.3, 0.3) | 亮度偏移范围（加性） |
| `contrast_range` | (float, float) | (0.7, 1.3) | 对比度缩放范围（乘性） |
| `saturation_range` | (float, float) | (0.7, 1.3) | 饱和度缩放范围 |
| `hue_range` | (float, float) | (-18, 18) | 色调偏移范围（度） |

```yaml
- name: photometric_distortion
  params:
    p: 0.3
    brightness_range: [-0.3, 0.3]
    contrast_range: [0.7, 1.3]
```

#### `color_jitter`

颜色抖动：与 photometric_distortion 类似，但参数含义略有不同（brightness 为乘性因子范围）。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `p` | float | 0.5 | 应用概率 |
| `brightness_range` | (float, float) | (-0.2, 0.2) | 亮度加性偏移范围 |
| `contrast_range` | (float, float) | (0.8, 1.2) | 对比度乘性缩放范围 |
| `saturation_range` | (float, float) | (0.8, 1.2) | 饱和度乘性缩放范围 |
| `hue_range` | (float, float) | (-0.1, 0.1) | 色调偏移范围（×180°） |

```yaml
- name: color_jitter
  params: { p: 0.3, brightness_range: [-0.2, 0.2], contrast_range: [0.8, 1.2] }
```

#### `brightness_contrast`

随机亮度与对比度调整。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `p` | float | 0.3 | 应用概率 |
| `brightness_range` | (float, float) | (-0.2, 0.2) | 亮度加性偏移范围 |
| `contrast_range` | (float, float) | (0.8, 1.2) | 对比度乘性缩放范围 |

```yaml
- name: brightness_contrast
  params: { p: 0.3, brightness_range: [-0.3, 0.3], contrast_range: [0.7, 1.3] }
```

#### `gamma_correction`

随机 Gamma 校正。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `p` | float | 0.3 | 应用概率 |
| `gamma_range` | (float, float) | (0.7, 1.5) | Gamma 值范围（<1 提亮，>1 压暗） |

```yaml
- name: gamma_correction
  params: { p: 0.3, gamma_range: [0.5, 2.0] }
```

#### `clahe`

自适应直方图均衡化（Contrast Limited Adaptive Histogram Equalization）。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `p` | float | 0.5 | 应用概率 |
| `clip_limit_range` | (float, float) | (1.0, 5.0) | 对比度限制阈值范围 |
| `tile_size_range` | (int, int) | (4, 16) | 分块大小范围（像素） |

```yaml
- name: clahe
  params: { p: 0.3, clip_limit_range: [2.0, 6.0], tile_size_range: [4, 16] }
```

#### `gaussian_blur`

随机高斯模糊。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `p` | float | 0.3 | 应用概率 |
| `kernel_range` | (int, int) | (3, 7) | 卷积核大小范围（必须奇数） |
| `sigma_range` | (float, float) | (0.1, 2.0) | 标准差范围 |

```yaml
- name: gaussian_blur
  params: { p: 0.2, kernel_range: [3, 9], sigma_range: [0.1, 3.0] }
```

#### `gaussian_noise`

添加高斯噪声。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `p` | float | 0.3 | 应用概率 |
| `std_range` | (float, float) | (0.01, 0.08) | 噪声标准差范围 |

```yaml
- name: gaussian_noise
  params: { p: 0.3, std_range: [0.01, 0.1] }
```

#### `sharpness`

随机锐化（非锐化掩模法）。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `p` | float | 0.3 | 应用概率 |
| `factor_range` | (float, float) | (0.5, 2.0) | 锐化强度范围（<1 模糊，>1 锐化） |

```yaml
- name: sharpness
  params: { p: 0.3, factor_range: [0.5, 3.0] }
```

#### `posterize`

减少每个通道的位深度（色彩量化）。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `p` | float | 0.3 | 应用概率 |
| `bits_range` | (int, int) | (2, 6) | 保留位数范围（1-8，越小颜色越少） |

```yaml
- name: posterize
  params: { p: 0.2, bits_range: [2, 5] }
```

#### `random_solarize`

随机曝光反转：将高于阈值的像素取反。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `p` | float | 0.3 | 应用概率 |
| `threshold_range` | (float, float) | (0.3, 0.7) | 反转阈值范围（0-1） |

```yaml
- name: random_solarize
  params: { p: 0.2, threshold_range: [0.2, 0.8] }
```

#### `channel_dropout`

随机丢弃颜色通道。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `p` | float | 0.3 | 应用概率 |
| `drop_count_range` | (int, int) | (1, 1) | 每次丢弃的通道数范围 |
| `fill_value` | float | 0.0 | 填充值 |

```yaml
- name: channel_dropout
  params: { p: 0.2, drop_count_range: [1, 2], fill_value: 0.0 }
```

---

### 遮挡 (2 种)

#### `random_erasing`

随机擦除：随机选择矩形区域并填充（仅作用于图像，不影响标签）。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `p` | float | 0.5 | 应用概率 |
| `scale_range` | (float, float) | (0.02, 0.15) | 擦除面积占图像面积比例范围 |
| `ratio_range` | (float, float) | (0.3, 3.3) | 擦除区域宽高比范围 |
| `fill_value` | float/"random" | "random" | 填充值（"random" 表示随机填充） |
| `max_count` | int | 1 | 最多擦除区域数 |

```yaml
- name: random_erasing
  params: { p: 0.3, scale_range: [0.02, 0.2], max_count: 3 }
```

#### `coarse_dropout`

大块遮挡：丢弃多个大块矩形区域（图像和标签均不处理，仅图像填充）。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `p` | float | 0.5 | 应用概率 |
| `num_holes_range` | (int, int) | (1, 8) | 遮挡块数量范围 |
| `hole_height_range` | (float, float) | (0.02, 0.15) | 遮挡块高度范围（图像高度比例） |
| `hole_width_range` | (float, float) | (0.02, 0.15) | 遮挡块宽度范围（图像宽度比例） |
| `fill_value` | float | 0.0 | 填充值 |

```yaml
- name: coarse_dropout
  params:
    p: 0.3
    num_holes_range: [2, 8]
    hole_height_range: [0.05, 0.2]
    hole_width_range: [0.05, 0.2]
```

---

### 样本级变换 (2 种)

> 这两种增强需要访问数据集（随机采样其他样本），系统自动注入 `dataset`。

#### `copy_paste`

复制粘贴：从其他样本中复制前景目标并粘贴到当前图像。
参考：Ghiasi et al., CVPR 2021。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `p` | float | 0.5 | 应用概率 |
| `max_objects` | int | 3 | 每次最多粘贴目标数 |
| `scale_range` | (float, float) | (0.5, 1.5) | 粘贴目标缩放范围（相对原始尺寸） |
| `blend_ratio_range` | (float, float) | (0.0, 0.0) | 边界融合范围（0=硬粘贴，>0=alpha 混合） |

```yaml
- name: copy_paste
  params:
    p: 0.3
    max_objects: 3
    scale_range: [0.5, 1.5]
    blend_ratio_range: [0.0, 0.3]
```

#### `mosaic`

马赛克：将 4 张图像拼成 2×2 网格，再裁剪回原始尺寸。
参考：Bochkovskiy et al., 2020 (YOLOv4)。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `p` | float | 0.5 | 应用概率 |
| `mosaic_size` | int | 4 | 拼图数量（目前仅支持 4） |
| `offset_range` | (float, float) | (0.0, 0.2) | 中心点偏移范围（图像尺寸比例，0=正中） |

```yaml
- name: mosaic
  params: { p: 0.3, offset_range: [0.0, 0.3] }
```

---

### 完整 YAML 示例

```yaml
training:
  augmentation: pipeline
  aug_pipeline:
    # 空间变换
    - name: horizontal_flip
      params: { p: 0.5 }
    - name: vertical_flip
      params: { p: 0.3 }
    - name: random_rotate
      params: { p: 0.3, degrees_range: [-30, 30] }
    - name: random_affine
      params:
        p: 0.3
        degrees_range: [-15, 15]
        translate_range: [0.0, 0.1]
        scale_range: [0.8, 1.2]
        shear_range: [-5, 5]
    - name: elastic_deform
      params: { p: 0.3, alpha_range: [20, 80], sigma_range: [3, 7] }

    # 样本级
    - name: copy_paste
      params: { p: 0.3, scale_range: [0.5, 1.5], blend_ratio_range: [0.0, 0.3] }
    - name: mosaic
      params: { p: 0.2, offset_range: [0.0, 0.2] }

    # 外观变换
    - name: clahe
      params: { p: 0.3, clip_limit_range: [1.0, 5.0], tile_size_range: [4, 16] }
    - name: gamma_correction
      params: { p: 0.2, gamma_range: [0.7, 1.5] }
    - name: gaussian_blur
      params: { p: 0.2, kernel_range: [3, 7], sigma_range: [0.1, 2.0] }

    # 遮挡
    - name: grid_mask
      params: { p: 0.2, d_range: [0.05, 0.15], ratio_range: [0.3, 0.7] }
    - name: random_erasing
      params: { p: 0.2, scale_range: [0.02, 0.1], max_count: 2 }

    # 噪声
    - name: gaussian_noise
      params: { p: 0.2, std_range: [0.01, 0.05] }

  epochs: 200
  batch_size: 8
```

> 完整示例配置：`configs/architectures/decoder_study/general/resnet50_unet_advanced_aug.yaml`
