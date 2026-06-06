# Domain Adaptation

[中文文档](domain_adaptation_CN.md)

18 built-in UDA methods in `medseg/training/domain_adaptation/`.

## Methods

| Method | Paper | Published | GitHub | Description |
|--------|-------|-------|--------|-------------|
| `source_only` | Baseline | - | [valeoai/ADVENT](https://github.com/valeoai/ADVENT) | Lower-bound: CE+Dice on source only |
| `advent` | Vu et al. | CVPR 2019 | [valeoai/ADVENT](https://github.com/valeoai/ADVENT) | Adversarial entropy minimization |
| `dann` | Ganin et al. | JMLR 2016 | [fungtion/DANN](https://github.com/fungtion/DANN) | Domain-adversarial neural net |
| `tent` | Wang et al. | ICLR 2021 | [DequanWang/tent](https://github.com/DequanWang/tent) | Test-time entropy minimization |
| `dpl` | Chen et al. | MICCAI 2021 | [cchen-cc/SFDA-DPL](https://github.com/cchen-cc/SFDA-DPL) | Dual pseudo label |
| `cbmt` | ADA4MIA benchmark | - | [whq-xxh/ADA4MIA](https://github.com/whq-xxh/ADA4MIA) | Class-balanced mean teacher |
| `fda` | Yang & Soatto | CVPR 2020 | [YanchaoYang/FDA](https://github.com/YanchaoYang/FDA) | Fourier domain adaptation |
| `crst` | Zou et al. | ICCV 2019 | [yzou2/CRST](https://github.com/yzou2/CRST) | Class-balanced self-training |
| `pixmatch` | Melas-Kyriazi & Manrai | CVPR 2021 | [lukemelas/pixmatch](https://github.com/lukemelas/pixmatch) | Pixel-level contrastive matching |
| `mic` | Hoyer et al. | CVPR 2023 | [lhoyer/MIC](https://github.com/lhoyer/MIC) | Masked image consistency |
| `daformer_fd` | Hoyer et al. | CVPR 2022 | [lhoyer/DAFormer](https://github.com/lhoyer/DAFormer) | Feature distance + rare class sampling |
| `hrda` | Hoyer et al. | ECCV 2022 | [lhoyer/HRDA](https://github.com/lhoyer/HRDA) | Multi-resolution scale attention |
| `pipa` | Chen et al. | ACM MM 2023 | [chen742/PiPa](https://github.com/chen742/PiPa) | Pixel + patch InfoNCE |
| `ddb` | Du et al. | CVPR 2023 | [xinyuelll/DDB](https://github.com/xinyuelll/DDB) | Dual-domain decoupled bridging |
| `sepico` | Xie et al. | TPAMI 2023 | [BIT-DA/SePiCo](https://github.com/BIT-DA/SePiCo) | Semantic pixel contrast + KL |
| `diga` | Shen et al. | CVPR 2023 | [BIT-DA/DiGA](https://github.com/BIT-DA/DiGA) | Distillation-guided adaptation |
| `micdrop` | Hoyer et al. | ECCV 2024 | [lhoyer/MICDrop](https://github.com/lhoyer/MICDrop) | MIC + complementary feature dropout |
| `semivl_da` | Karazija et al. | ECCV 2024 | [google-research/semivl](https://github.com/google-research/semivl) | Vision-language guided self-training |

## YAML Config

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
  source:                          # Source domain (labeled)
    image_dir: ./data/source/images
    mask_dir: ./data/source/masks
  target:                          # Target domain (unlabeled)
    image_dir: ./data/target/images
  val:                             # Target val (labeled subset)
    image_dir: ./data/target_val/images
    mask_dir: ./data/target_val/masks
  test:
    image_dir: ./data/target_test/images
    mask_dir: ./data/target_test/masks

domain_adaptation:
  method: advent                   # any method name from table
  params:
    entropy_weight: 0.1
    adversarial_weight: 0.1
    num_classes: 4

training:
  epochs: 100
  batch_size: 8
  num_workers: 4
  val_interval: 10
  save_interval: 50
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
    lr: 1e-4
    weight_decay: 1e-4
  scheduler:
    name: cosine
    min_lr: 1e-6
```

## Usage

```bash
# Train
python train_domain_adaptation.py --config configs/training_paradigms/domain_adaptation/advent.yaml

# Multi-GPU
torchrun --nproc_per_node=4 train_domain_adaptation.py \
    --config configs/training_paradigms/domain_adaptation/mic.yaml

# Test on target domain
python test.py --config configs/training_paradigms/domain_adaptation/advent.yaml \
    --checkpoint output/best_model.pth
```

## Pipeline

```
Source (labeled) ──┐
                   ├──► DA Trainer ──► Adapted Model ──► Target Inference
Target (unlabeled)─┘
```

Each method config is in `configs/training_paradigms/domain_adaptation/`.
