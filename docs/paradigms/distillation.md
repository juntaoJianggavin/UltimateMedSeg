# Knowledge Distillation

[中文文档](distillation_CN.md)

27 built-in KD methods in `medseg/training/distillation/`.

## Methods

### Classic KD (23)

| Method | Paper | Published | GitHub | Description |
|--------|-------|-------|--------|-------------|
| `vanilla_kd` | Hinton et al. | NeurIPS-W 2014 | [peterliht/knowledge-distillation-pytorch](https://github.com/peterliht/knowledge-distillation-pytorch) | Soft-label KD |
| `unet_distillation` | Hinton et al. | - | - | Logit/feature/attention multi-scale KD |
| `hint_distillation` | Romero et al. (FitNets) | ICLR 2015 | [adri-romsor/FitNets](https://github.com/adri-romsor/FitNets) | Intermediate hint layers |
| `attention_mimicry` | Simplified attention | - | - | Attention map mimicry baseline |
| `at` | Zagoruyko & Komodakis | ICLR 2017 | [szagoruyko/attention-transfer](https://github.com/szagoruyko/attention-transfer) | Attention transfer |
| `fsp` | Yim et al. (A Gift from KD) | CVPR 2017 | [yoshitomo-matsubara/torchdistill](https://github.com/yoshitomo-matsubara/torchdistill) | Flow of solution procedure |
| `nst` | Huang & Wang | 2017 | [HobbitLong/RepDistiller](https://github.com/HobbitLong/RepDistiller) | Neuron selectivity transfer |
| `rkd` | Park et al. | CVPR 2019 | [lenscloth/RKD](https://github.com/lenscloth/RKD) | Relational KD |
| `vid` | Ahn et al. | CVPR 2019 | [HobbitLong/RepDistiller](https://github.com/HobbitLong/RepDistiller) | Variational information distillation |
| `dkd` | Zhao et al. | CVPR 2022 | [megvii-research/mdistiller](https://github.com/megvii-research/mdistiller) | Decoupled KD |
| `mgd` | Yang et al. | ECCV 2022 | [yzd-v/MGD](https://github.com/yzd-v/MGD) | Masked generative distillation |
| `dist` | Huang et al. | NeurIPS 2022 | [hunto/DIST_KD](https://github.com/hunto/DIST_KD) | DIST distillation |
| `cirkd_minibatch` | Yang et al. | CVPR 2022 | [winycg/CIRKD](https://github.com/winycg/CIRKD) | Cross-image relational KD |
| `cwd` | Shu et al. | ICCV 2021 | [irfanICMLL/TorchDistiller](https://github.com/irfanICMLL/TorchDistiller) | Channel-wise distillation |
| `review_kd` | Chen et al. | CVPR 2021 | [dvlab-research/ReviewKD](https://github.com/dvlab-research/ReviewKD) | Knowledge review |
| `simkd` | Chen et al. | CVPR 2022 | [DefangChen/SimKD](https://github.com/DefangChen/SimKD) | Simple KD with projector |
| `norm_kd` | Liu et al. (NORM) | ICLR 2023 | [xyliu7/NORM](https://github.com/xyliu7/NORM) | Normalized logits KD |
| `sdd` | Wei et al. | CVPR 2024 | [shicaiwei123/SDD-CVPR2024](https://github.com/shicaiwei123/SDD-CVPR2024) | Scale decoupled distillation |
| `aicsd` | Mansurian et al. | TNNLS 2024 | [AmirMansurian/AICSD](https://github.com/AmirMansurian/AICSD) | Adaptive inter-class similarity |
| `logit_std_kd` | Sun et al. | CVPR 2024 | [sunshangquan/logit-standardization-KD](https://github.com/sunshangquan/logit-standardization-KD) | Logit standardization |
| `ttm_kd` | Zheng & Yang | ICLR 2024 | [zkxufo/TTM](https://github.com/zkxufo/TTM) | Transformed teacher matching |
| `ctkd` | Li et al. | AAAI 2023 | [zhengli97/CTKD](https://github.com/zhengli97/CTKD) | Curriculum temperature |
| `mlkd` | Jin et al. | CVPR 2023 | [Jin-Ying/Multi-Level-Logit-Distillation](https://github.com/Jin-Ying/Multi-Level-Logit-Distillation) | Multi-level logit distillation |

### Medical-Specific KD (4)

| Method | Description |
|--------|-------------|
| `anatomy_kd` | Organ topology distillation |
| `boundary_kd` | Boundary-aware KD |
| `multi_organ_kd` | Class-balanced multi-organ KD |
| `cross_modality_kd` | Cross-modality KD (CT/MRI) |

## YAML Config

**Teacher config** (`configs/training_paradigms/distillation/teacher_large.yaml`):

```yaml
model:
  num_classes: 4
  img_size: 224
  encoder:
    name: timm_resnet50
    pretrained: true
    in_channels: 3
  decoder:
    name: unet
```

**Student config** (e.g. `aicsd.yaml`):

```yaml
model:
  num_classes: 4
  img_size: 224
  encoder:
    name: timm_resnet34
    pretrained: true
    in_channels: 3
  decoder:
    name: bilinear
  bottleneck:
    name: none

data:
  img_size: 224
  source:
    image_dir: ./data/source/images
    mask_dir: ./data/source/masks
  target:
    image_dir: ./data/target/images
  val:
    image_dir: ./data/target_val/images
    mask_dir: ./data/target_val/masks

distillation:
  method: aicsd            # any method name from table
  weight: 1.0
  params:
    temperature: 1.0
    lambda_ics: 1.0
    lambda_icc: 1.0

training:
  epochs: 100
  batch_size: 8
  num_workers: 4
  val_interval: 10
  save_interval: 50
```

## Usage

```bash
# Train with distillation
python train_distillation.py \
    --teacher_config configs/training_paradigms/distillation/teacher_large.yaml \
    --student_config configs/training_paradigms/distillation/aicsd.yaml

# Specify teacher checkpoint
python train_distillation.py \
    --teacher_config configs/training_paradigms/distillation/teacher_large.yaml \
    --teacher_checkpoint output/teacher/best_model.pth \
    --student_config configs/training_paradigms/distillation/vanilla_kd.yaml

# Test student
python test.py --config configs/training_paradigms/distillation/aicsd.yaml \
    --checkpoint output/student/best_model.pth
```

## Pipeline

```
Pre-trained Teacher ──┐
                      ├──► KD Trainer ──► Compact Student
Training Data ────────┘
```

Each method config is in `configs/training_paradigms/distillation/`.
