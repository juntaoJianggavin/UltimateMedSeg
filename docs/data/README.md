# Datasets

[中文文档](README_CN.md)

## Supported Dataset Types

| Type | Description | Loader |
|------|-------------|--------|
| `synapse` | Synapse multi-organ CT (TransUNet format: npz train + h5 test) | Specialized |
| `acdc` | ACDC cardiac MRI (TransUNet format: npz + h5) | Specialized |
| `generic` | Any dataset with images/ + masks/ folders | Generic |
| `qata_covid19` | QaTa-COV19 chest X-ray + per-image text (LViT format) | Text-aware |
| `mosmed_plus` | MosMedData+ COVID CT + per-image text (LViT format) | Text-aware |

---

## Data Split Modes

### 1. Explicit Directories

```yaml
data:
  type: generic
  img_size: 224
  train_dir: ./data/train
  val_dir: ./data/val
  test_dir: ./data/test
```

### 2. Ratio Split

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

### 3. N-Fold Cross Validation

```yaml
data:
  type: generic
  img_size: 224
  image_dir: ./data/images
  mask_dir: ./data/masks
  split:
    mode: nfold
    n_folds: 5
    fold: 0              # current fold (0-4)
    seed: 42
```

### 4. N-Fold + Hold-out Test

```yaml
data:
  type: generic
  img_size: 224
  image_dir: ./data/images
  mask_dir: ./data/masks
  test_dir: ./data/test   # fixed test set
  split:
    mode: nfold_test
    n_folds: 5
    fold: 0
    seed: 42
```

---

## Dataset Summary (25)

Extracted from `configs/intro_to_datasets/`.

### Abdominal & Cardiac

| Dataset | Modality | Classes | Size | Split | Link |
|---------|----------|---------|------|-------|------|
| Synapse | CT | 9 (8 organs+BG) | 30 cases | 18 train / 12 test | [TransUNet](https://github.com/Beckschen/TransUNet) |
| ACDC | MRI | 4 (RV/MYO/LV+BG) | 100 cases | 70/10/20 | [ACDC](https://github.com/Beckschen/TransUNet)|

### Retinal

| Dataset | Task | Size | Classes | Split |
|---------|------|------|---------|-------|
| DRIVE | Vessel seg | 40 images | 2 | 20/20 official |
| STARE | Vessel seg | 20 images | 2 | LOO or cross-dataset |
| CHASE_DB1 | Vessel seg | 28 images | 2 | 20 train / 8 test |
| HRF | Vessel seg (high-res) | 45 images | 2 | 5-fold CV |
| ARIA | Vessel seg (multi-disease) | 143 images | 2 | 5-fold CV |
| RITE | Artery/vein seg | 40 images | 3 | 20/20 (same as DRIVE) |
| REFUGE | OD/OC seg | 1200 images | 3 | 400/400/400 |
| Drishti-GS | OD/OC seg | 101 images | 3 | 50/51 |

### Skin Lesion

| Dataset | Size | Classes | Split |
|---------|------|---------|-------|
| ISIC 2016 | 1279 images | 2 | 900/379 |
| ISIC 2017 | 2750 images | 2 | 2000/150/600 |
| ISIC 2018 | 3694 images | 2 | 2594/100/1000 |
| PH2 | 200 images | 2 | 5-fold CV or external test |

### GI Polyp

| Dataset | Size | Resolution | Split |
|---------|------|-----------|-------|
| Kvasir-SEG | 1000 images | Variable | 5-fold CV |
| CVC-ClinicDB | 612 images | 384x288 | 5-fold CV |
| CVC-ColonDB | 380 images | 574x500 | Cross-dataset test |

### Pathology

| Dataset | Tissue | Size | Classes | Split |
|---------|--------|------|---------|-------|
| GlaS | Colon gland | 165 images | 2 | 85 train / 80 test |
| MoNuSeg | Multi-organ nuclei | 44 images | 2 | 30/14 |
| PanNuke | Pan-cancer nuclei | ~7900 patches | 6 | 3-fold official |

### Chest

| Dataset | Modality | Size | Task |
|---------|----------|------|------|
| Montgomery+Shenzhen | CXR | 800 images | Lung seg |
| COVID CT Seg | CT | 100 slices | Infection seg (GGO+consolidation) |
| QaTa-COV19 | CXR + text | 10501 images | COVID infection seg |
| MosMedData+ | CT + text | 3674 slices | COVID infection seg |

### Ultrasound

| Dataset | Organ | Size | Classes |
|---------|-------|------|---------|
| BUSI | Breast | 647 images | 2 (lesion) |

---

## Synapse Config Example

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

## ACDC Config Example

```yaml
model:
  num_classes: 4
  img_size: 224

data:
  type: acdc
  img_size: 224
  train_dir: ./data/ACDC/train
  val_dir: ./data/ACDC/val
  test_dir: ./data/ACDC/test
```

## Generic Config Example

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

## Text-Aware Config

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

All dataset intro configs are in `configs/intro_to_datasets/`.

---

## Augmentation Pipeline — 24 Methods

> All augmentation methods are registered to `AUGMENTATION_REGISTRY` and configured via YAML.
> All intensity parameters use `_range` suffix, randomly sampled per call.

### How to Enable

```yaml
training:
  augmentation: pipeline          # switch to pipeline mode
  aug_pipeline:                   # list augmentations in order
    - name: horizontal_flip
      params: { p: 0.5 }
    - name: elastic_deform
      params: { p: 0.3, alpha_range: [20, 80], sigma_range: [3, 7] }
```

---

### Geometric Transforms (9)

#### `horizontal_flip`

Random horizontal flip.

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `p` | float | 0.5 | Application probability |

```yaml
- name: horizontal_flip
  params: { p: 0.5 }
```

#### `vertical_flip`

Random vertical flip.

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `p` | float | 0.5 | Application probability |

```yaml
- name: vertical_flip
  params: { p: 0.5 }
```

#### `random_rotate90`

Random 90°/180°/270° rotation (preserves image dimensions).

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `p` | float | 0.5 | Application probability |

```yaml
- name: random_rotate90
  params: { p: 0.5 }
```

#### `random_rotate`

Arbitrary angle rotation (bilinear interpolation, preserves image dimensions).

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `p` | float | 0.5 | Application probability |
| `degrees_range` | (float, float) | (-15, 15) | Rotation angle range (degrees) |

```yaml
- name: random_rotate
  params: { p: 0.3, degrees_range: [-45, 45] }
```

#### `random_affine`

Random affine transform (rotation + translation + scale + shear).

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `p` | float | 0.5 | Application probability |
| `degrees_range` | (float, float) | (-15, 15) | Rotation angle range (degrees) |
| `translate_range` | (float, float) | (0.0, 0.1) | Translation range (fraction of image size) |
| `scale_range` | (float, float) | (0.9, 1.1) | Scale range |
| `shear_range` | (float, float) | (-5, 5) | Shear angle range (degrees) |

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

Random perspective transform.

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `p` | float | 0.5 | Application probability |
| `distortion_scale_range` | (float, float) | (0.05, 0.15) | Perspective distortion scale range (fraction of image size) |

```yaml
- name: random_perspective
  params: { p: 0.3, distortion_scale_range: [0.05, 0.2] }
```

#### `random_scale`

Random scale then resize back to original size (scale up/down, then interpolate back).

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `p` | float | 0.5 | Application probability |
| `scale_range` | (float, float) | (0.8, 1.2) | Scale factor range |

```yaml
- name: random_scale
  params: { p: 0.3, scale_range: [0.7, 1.3] }
```

#### `elastic_deform`

Random elastic deformation (simulates tissue deformation).

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `p` | float | 0.3 | Application probability |
| `alpha_range` | (float, float) | (20, 80) | Deformation amplitude range (higher = stronger) |
| `sigma_range` | (float, float) | (3, 7) | Gaussian smoothing sigma range (higher = smoother) |

```yaml
- name: elastic_deform
  params: { p: 0.3, alpha_range: [20, 100], sigma_range: [3, 8] }
```

#### `grid_mask`

Grid masking: occludes grid-pattern regions, forcing the model to learn robust features.

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `p` | float | 0.5 | Application probability |
| `d_range` | (float, float) | (0.05, 0.15) | Grid cell size range (fraction of image size) |
| `ratio_range` | (float, float) | (0.3, 0.7) | Occluded area ratio within each cell |
| `rotate_range` | (float, float) | (0, 0) | Grid rotation angle range (degrees) |

```yaml
- name: grid_mask
  params: { p: 0.3, d_range: [0.05, 0.2], ratio_range: [0.3, 0.7] }
```

---

### Pixel-level Transforms (11)

#### `photometric_distortion`

Photometric distortion: random brightness, contrast, saturation, hue transforms (random order).

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `p` | float | 0.5 | Application probability |
| `brightness_range` | (float, float) | (-0.3, 0.3) | Brightness offset range (additive) |
| `contrast_range` | (float, float) | (0.7, 1.3) | Contrast scale range (multiplicative) |
| `saturation_range` | (float, float) | (0.7, 1.3) | Saturation scale range |
| `hue_range` | (float, float) | (-18, 18) | Hue shift range (degrees) |

```yaml
- name: photometric_distortion
  params:
    p: 0.3
    brightness_range: [-0.3, 0.3]
    contrast_range: [0.7, 1.3]
```

#### `color_jitter`

Color jitter: similar to photometric_distortion, but brightness is a multiplicative factor.

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `p` | float | 0.5 | Application probability |
| `brightness_range` | (float, float) | (-0.2, 0.2) | Brightness additive offset range |
| `contrast_range` | (float, float) | (0.8, 1.2) | Contrast multiplicative scale range |
| `saturation_range` | (float, float) | (0.8, 1.2) | Saturation multiplicative scale range |
| `hue_range` | (float, float) | (-0.1, 0.1) | Hue shift range (×180°) |

```yaml
- name: color_jitter
  params: { p: 0.3, brightness_range: [-0.2, 0.2], contrast_range: [0.8, 1.2] }
```

#### `brightness_contrast`

Random brightness and contrast adjustment.

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `p` | float | 0.3 | Application probability |
| `brightness_range` | (float, float) | (-0.2, 0.2) | Brightness additive offset range |
| `contrast_range` | (float, float) | (0.8, 1.2) | Contrast multiplicative scale range |

```yaml
- name: brightness_contrast
  params: { p: 0.3, brightness_range: [-0.3, 0.3], contrast_range: [0.7, 1.3] }
```

#### `gamma_correction`

Random gamma correction.

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `p` | float | 0.3 | Application probability |
| `gamma_range` | (float, float) | (0.7, 1.5) | Gamma value range (<1 brightens, >1 darkens) |

```yaml
- name: gamma_correction
  params: { p: 0.3, gamma_range: [0.5, 2.0] }
```

#### `clahe`

Contrast Limited Adaptive Histogram Equalization.

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `p` | float | 0.5 | Application probability |
| `clip_limit_range` | (float, float) | (1.0, 5.0) | Contrast limit threshold range |
| `tile_size_range` | (int, int) | (4, 16) | Tile size range (pixels) |

```yaml
- name: clahe
  params: { p: 0.3, clip_limit_range: [2.0, 6.0], tile_size_range: [4, 16] }
```

#### `gaussian_blur`

Random Gaussian blur.

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `p` | float | 0.3 | Application probability |
| `kernel_range` | (int, int) | (3, 7) | Kernel size range (must be odd) |
| `sigma_range` | (float, float) | (0.1, 2.0) | Standard deviation range |

```yaml
- name: gaussian_blur
  params: { p: 0.2, kernel_range: [3, 9], sigma_range: [0.1, 3.0] }
```

#### `gaussian_noise`

Add Gaussian noise.

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `p` | float | 0.3 | Application probability |
| `std_range` | (float, float) | (0.01, 0.08) | Noise standard deviation range |

```yaml
- name: gaussian_noise
  params: { p: 0.3, std_range: [0.01, 0.1] }
```

#### `sharpness`

Random sharpening (unsharp masking).

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `p` | float | 0.3 | Application probability |
| `factor_range` | (float, float) | (0.5, 2.0) | Sharpening intensity range (<1 blur, >1 sharpen) |

```yaml
- name: sharpness
  params: { p: 0.3, factor_range: [0.5, 3.0] }
```

#### `posterize`

Reduce bit depth per channel (color quantization).

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `p` | float | 0.3 | Application probability |
| `bits_range` | (int, int) | (2, 6) | Bits to retain range (1-8, lower = fewer colors) |

```yaml
- name: posterize
  params: { p: 0.2, bits_range: [2, 5] }
```

#### `random_solarize`

Random solarize: invert pixels above a threshold.

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `p` | float | 0.3 | Application probability |
| `threshold_range` | (float, float) | (0.3, 0.7) | Inversion threshold range (0-1) |

```yaml
- name: random_solarize
  params: { p: 0.2, threshold_range: [0.2, 0.8] }
```

#### `channel_dropout`

Randomly drop color channels.

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `p` | float | 0.3 | Application probability |
| `drop_count_range` | (int, int) | (1, 1) | Number of channels to drop range |
| `fill_value` | float | 0.0 | Fill value |

```yaml
- name: channel_dropout
  params: { p: 0.2, drop_count_range: [1, 2], fill_value: 0.0 }
```

---

### Masking (2)

#### `random_erasing`

Random erasing: select rectangular regions and fill them (image only, labels unaffected).

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `p` | float | 0.5 | Application probability |
| `scale_range` | (float, float) | (0.02, 0.15) | Erasing area ratio range (relative to image) |
| `ratio_range` | (float, float) | (0.3, 3.3) | Erasing region aspect ratio range |
| `fill_value` | float/"random" | "random" | Fill value ("random" for random fill) |
| `max_count` | int | 1 | Maximum number of erased regions |

```yaml
- name: random_erasing
  params: { p: 0.3, scale_range: [0.02, 0.2], max_count: 3 }
```

#### `coarse_dropout`

Coarse dropout: drop multiple large rectangular regions (image filled, labels unchanged).

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `p` | float | 0.5 | Application probability |
| `num_holes_range` | (int, int) | (1, 8) | Number of holes range |
| `hole_height_range` | (float, float) | (0.02, 0.15) | Hole height range (fraction of image height) |
| `hole_width_range` | (float, float) | (0.02, 0.15) | Hole width range (fraction of image width) |
| `fill_value` | float | 0.0 | Fill value |

```yaml
- name: coarse_dropout
  params:
    p: 0.3
    num_holes_range: [2, 8]
    hole_height_range: [0.05, 0.2]
    hole_width_range: [0.05, 0.2]
```

---

### Sample-level Transforms (2)

> These two augmentations require dataset access (sample other images), automatically injected.

#### `copy_paste`

Copy-paste: copy foreground objects from other samples and paste onto current image.
Reference: Ghiasi et al., CVPR 2021.

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `p` | float | 0.5 | Application probability |
| `max_objects` | int | 3 | Maximum objects to paste per call |
| `scale_range` | (float, float) | (0.5, 1.5) | Pasted object scale range (relative to original) |
| `blend_ratio_range` | (float, float) | (0.0, 0.0) | Boundary blend range (0=hard paste, >0=alpha blend) |

```yaml
- name: copy_paste
  params:
    p: 0.3
    max_objects: 3
    scale_range: [0.5, 1.5]
    blend_ratio_range: [0.0, 0.3]
```

#### `mosaic`

Mosaic: tile 4 images in a 2×2 grid, then crop back to original size.
Reference: Bochkovskiy et al., 2020 (YOLOv4).

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `p` | float | 0.5 | Application probability |
| `mosaic_size` | int | 4 | Number of tiles (currently only 4 supported) |
| `offset_range` | (float, float) | (0.0, 0.2) | Center point offset range (fraction of image size, 0=center) |

```yaml
- name: mosaic
  params: { p: 0.3, offset_range: [0.0, 0.3] }
```

---

### Complete YAML Example

```yaml
training:
  augmentation: pipeline
  aug_pipeline:
    # Spatial transforms
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

    # Sample-level
    - name: copy_paste
      params: { p: 0.3, scale_range: [0.5, 1.5], blend_ratio_range: [0.0, 0.3] }
    - name: mosaic
      params: { p: 0.2, offset_range: [0.0, 0.2] }

    # Appearance transforms
    - name: clahe
      params: { p: 0.3, clip_limit_range: [1.0, 5.0], tile_size_range: [4, 16] }
    - name: gamma_correction
      params: { p: 0.2, gamma_range: [0.7, 1.5] }
    - name: gaussian_blur
      params: { p: 0.2, kernel_range: [3, 7], sigma_range: [0.1, 2.0] }

    # Masking
    - name: grid_mask
      params: { p: 0.2, d_range: [0.05, 0.15], ratio_range: [0.3, 0.7] }
    - name: random_erasing
      params: { p: 0.2, scale_range: [0.02, 0.1], max_count: 2 }

    # Noise
    - name: gaussian_noise
      params: { p: 0.2, std_range: [0.01, 0.05] }

  epochs: 200
  batch_size: 8
```

> Full example config: `configs/architectures/decoder_study/general/resnet50_unet_advanced_aug.yaml`
