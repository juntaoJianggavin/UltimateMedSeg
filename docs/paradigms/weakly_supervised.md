# Weakly Supervised Segmentation

[中文文档](weakly_supervised_CN.md)

28 built-in weakly supervised methods in `medseg/training/weakly_supervised/`.

## Methods

### Core Methods (16)

| Method | Paper | Published | Description |
|--------|-------|-------|-------------|
| `box_supervised` | BoxSup family | - | Box-only mask + FG/BG CE |
| `cam` | Zhou et al. / Selvaraju et al. | CVPR 2016 / ICCV 2017 | Class Activation Mapping (Grad-CAM) |
| `mil` | Multi-instance learning | - | Image-level label MIL |
| `em_pseudo_label` | EM refinement | - | EM pseudo mask refinement |
| `point` | Bearman et al. | ECCV 2016 | Point supervision |
| `gated_crf` | Obukhov et al. | NeurIPS 2019 | Differentiable CRF |
| `affinity` | AffinityNet style | - | Pixel affinity propagation |
| `tree_energy` | Tree energy | - | Tree-structured energy minimization |
| `seam` | Wang et al. | CVPR 2020 | Self-supervised equivariant attention |
| `puzzle_cam` | Jo & Yu | ICIP 2021 | Puzzle piece matching CAM |
| `advcam` | Lee et al. | CVPR 2021 | Adversarial complementary erasing |
| `mctformer` | Xu et al. | CVPR 2022 | Multi-class token transformer |
| `sam_guided_weak` | SAM-guided | - | SAM pseudo-mask refinement |
| `iseg` | Interactive seg | - | Interactive click-based supervision |
| `click_supervision` | Click-based | - | Click point supervision |
| `scribble_sup` | Scribble supervision | - | Scribble annotation supervision |

### Extended Methods (12)

| Method | Paper | Published | GitHub | Description |
|--------|-------|-------|--------|-------------|
| `eps` | EPS | - | - | Explicit pseudo-label supervision |
| `boxinst` | BoxInst | - | - | Box-level instance segmentation |
| `recam` | ReCAM | - | - | Re-weighted CAM |
| `toco` | ToCo | - | - | Token contrast |
| `lpcam` | LPCAM | - | - | Low-pass filtered CAM |
| `mars` | MARS | - | - | Mask-aware refinement |
| `bacon` | BACoN | - | - | Background-aware contrastive network |
| `wpgseg` | WPGSeg | - | - | Weakly-supervised progressive guided |
| `dupl` | DuPL | - | - | Dual pseudo label |
| `more` | MoRe | - | - | Momentum refinement |
| `psdpm` | PSDPM | - | - | Pseudo-label denoising with prior |
| `semples` | SemPLeS | - | - | Semantic pseudo label selection |

## Annotation Formats

Weakly supervised methods use different types of annotations, all loaded via a JSON file. The dataset class (`WeaklySupervisedDataset`) supports four supervision types:

### Image-Level Labels (`image_label`)

Only the image-level class presence is required — no spatial annotation.

```json
[
  {"image": "img_0001.png", "image_labels": [0, 2]},
  {"image": "img_0002.png", "image_labels": {"0": true, "1": false, "2": true}}
]
```

**Used by:** `cam`, `mil`, `seam`, `puzzle_cam`, `advcam`, `mctformer`, `recam`, `lpcam`, `toco`, `mars`, `bacon`, `wpgseg`, `dupl`, `more`, `psdpm`, `semples`, `eps`, `em_pseudo_label`

### Bounding Box (`box`)

Bounding boxes in normalised `[x1, y1, x2, y2]` format (0–1 range).

```json
[
  {
    "image": "img_0001.png",
    "boxes": [[0.1, 0.2, 0.8, 0.9], [0.3, 0.4, 0.6, 0.7]]
  }
]
```

**Used by:** `box_supervised`, `boxinst`

### Point (`point`)

Click points as `[x, y, class_id]` tuples (normalised coordinates).

```json
[
  {
    "image": "img_0001.png",
    "points": [[0.5, 0.3, 1], [0.2, 0.7, 0], [0.8, 0.6, 2]]
  }
]
```

**Used by:** `point`, `click_supervision`, `iseg`

### Scribble (`scribble`)

Scribble lines as `[x, y]` coordinate pairs (normalised).

```json
[
  {
    "image": "img_0001.png",
    "scribbles": [[0.1, 0.2], [0.15, 0.25], [0.2, 0.3], [0.5, 0.5]]
  }
]
```

**Used by:** `scribble_sup`

### Pre-computed CAMs (optional)

For CAM-based methods, pre-computed Class Activation Maps can be stored as `.npy` files in a separate directory. Each `.npy` file has shape `[num_classes, H, W]` and should match the image filename (e.g. `img_0001.npy` for `img_0001.png`).

```yaml
data:
  cam_dir: ./data/cams   # directory with .npy CAM files
```

### Supervision Type Summary

| Type | Annotation | Dataset Class | Methods |
|------|-----------|---------------|---------|
| Image-level | Class label | `ImageLabelDataset` | cam, mil, seam, puzzle_cam, advcam, mctformer, + 12 extended |
| Box | Bounding box | `BoxSupervisedDataset` | box_supervised, boxinst |
| Point | Click points | `WeaklySupervisedDataset(point)` | point, click_supervision, iseg |
| Scribble | Scribble lines | `WeaklySupervisedDataset(scribble)` | scribble_sup |
| Mixed | Multiple types | — | sam_guided_weak, eps |

## YAML Config

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
  label_file: ./data/annotations/image_labels.json   # image-level labels
  cam_dir: ./data/cams                                # pre-computed CAMs (optional)
  val:
    image_dir: ./data/val/images
    mask_dir: ./data/val/masks
  test:
    image_dir: ./data/test/images
    mask_dir: ./data/test/masks

weak_supervision:
  method: affinity           # any method name from tables above
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

## Usage

```bash
# Train
python train_weakly_supervised.py \
    --config configs/training_paradigms/weak_supervision/cam.yaml

# Box supervision
python train_weakly_supervised.py \
    --config configs/training_paradigms/weak_supervision/box_supervised.yaml

# Point supervision
python train_weakly_supervised.py \
    --config configs/training_paradigms/weak_supervision/point.yaml

# Scribble supervision
python train_weakly_supervised.py \
    --config configs/training_paradigms/weak_supervision/scribble_sup.yaml

# Test
python test.py --config configs/training_paradigms/weak_supervision/cam.yaml \
    --checkpoint output/best_model.pth
```

Each method config is in `configs/training_paradigms/weak_supervision/`.
