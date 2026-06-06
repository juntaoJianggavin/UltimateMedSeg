# Semi-Supervised Segmentation

[中文文档](semi_supervised_CN.md)

21 built-in semi-supervised methods in `medseg/training/semi/`.

## Methods

| Method | Paper | Published | GitHub | Description |
|--------|-------|-------|--------|-------------|
| `mean_teacher` | Tarvainen & Valpola | NeurIPS 2017 | [CuriousAI/mean-teacher](https://github.com/CuriousAI/mean-teacher) | EMA teacher consistency |
| `cps` | Chen et al. | CVPR 2021 | [charlesCXK/TorchSemiSeg](https://github.com/charlesCXK/TorchSemiSeg) | Cross pseudo supervision |
| `cct` | Ouali et al. | BMVC 2020 | [yassouali/CCT](https://github.com/yassouali/CCT) | Cross-consistency training |
| `unimatch` | Yang et al. | CVPR 2023 | [LiheYoung/UniMatch](https://github.com/LiheYoung/UniMatch) | Unified dual-stream matching |
| `fixmatch` | Sohn et al. | NeurIPS 2020 | [HiLab-git/SSL4MIS](https://github.com/HiLab-git/SSL4MIS) | Pseudo label + strong aug |
| `urpc` | Luo et al. | MIA 2022 | [HiLab-git/SSL4MIS](https://github.com/HiLab-git/SSL4MIS) | Uncertainty rectified PL |
| `deep_co_training` | Qiao et al. | ECCV 2018 | [AlanChou/Deep-Co-Training](https://github.com/AlanChou/Deep-Co-Training-for-Semi-Supervised-Image-Recognition) | Dual-net co-training |
| `flexmatch` | Zhang et al. | NeurIPS 2021 | [TorchSSL/TorchSSL](https://github.com/TorchSSL/TorchSSL) | Curriculum pseudo label |
| `softmatch` | Chen et al. | ICLR 2023 | [microsoft/Semi-supervised-learning](https://github.com/microsoft/Semi-supervised-learning) | Soft thresholding |
| `freematch` | Wang et al. | ICLR 2023 | [microsoft/Semi-supervised-learning](https://github.com/microsoft/Semi-supervised-learning) | Self-adaptive threshold |
| `ua_mt` | Yu et al. | MICCAI 2019 | [yulequan/UA-MT](https://github.com/yulequan/UA-MT) | Uncertainty-aware MT |
| `ssl4mis_u` | SSL4MIS uncertainty | - | [HiLab-git/SSL4MIS](https://github.com/HiLab-git/SSL4MIS) | MC-Dropout uncertainty |
| `pi_model` | Laine & Aila | ICLR 2017 | [smlaine2/tempens](https://github.com/smlaine2/tempens) | Stochastic perturbation |
| `temporal_ensembling` | Laine & Aila | ICLR 2017 | [smlaine2/tempens](https://github.com/smlaine2/tempens) | Per-sample EMA target |
| `pseudo_label` | Lee | ICML-W 2013 | [iBelieveCJM/pseudo_label](https://github.com/iBelieveCJM/pseudo_label-pytorch) | Hard pseudo labels |
| `ict` | Verma et al. | IJCAI 2019 | [vikasverma1077/ICT](https://github.com/vikasverma1077/ICT) | Interpolation consistency |
| `r_drop` | Wu et al. | NeurIPS 2021 | [dropreg/R-Drop](https://github.com/dropreg/R-Drop) | Regularized dropout |
| `cross_teaching` | Luo et al. | MIDL 2022 | [HiLab-git/SSL4MIS](https://github.com/HiLab-git/SSL4MIS) | CNN-Transformer cross teach |
| `augseg` | Zhao et al. | CVPR 2023 | [ZhenZHAO/AugSeg](https://github.com/ZhenZHAO/AugSeg) | Augmentation matters |
| `corrmatch` | Sun et al. | CVPR 2024 | [BBBBchan/CorrMatch](https://github.com/BBBBchan/CorrMatch) | Correlation matching |
| `allspark` | Wang et al. | CVPR 2024 | [xmed-lab/AllSpark](https://github.com/xmed-lab/AllSpark) | Reborn labeled tokens |
| `ddfp` | Wang et al. | CVPR 2024 | [Cuzyoung/DDFP](https://github.com/Cuzyoung/DDFP) | Diffusion-denoised PLs |
| `diffrect` | Liu et al. | MICCAI 2024 | [CUHK-AIM-Group/DiffRect](https://github.com/CUHK-AIM-Group/DiffRect) | Latent diffusion PL rectification |
| `ad_mt` | Zhao et al. | ECCV 2024 | [ZhenZHAO/AD-MT](https://github.com/ZhenZHAO/AD-MT) | Alternate diverse teaching |
| `pmt` | Gao et al. | ECCV 2024 | [Axi404/PMT](https://github.com/Axi404/PMT) | Progressive mean teacher |

## YAML Config

```yaml
model:
  num_classes: 9
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
  labeled_dir: ./data/labeled
  unlabeled_dir: ./data/unlabeled
  val_dir: ./data/val
  test_dir: ./data/test
  test_list: ./data/test/list.txt
  labeled_ratio: 0.1       # 5% / 10% / 20% / 50%
  split_mode: dir           # dir | ratio

semi:
  method: mean_teacher      # any method name from table above
  params:
    ema_decay: 0.999
    consistency_weight: 1.0
    rampup_epochs: 40

training:
  epochs: 200
  batch_size: 16
  labeled_batch_size: 8
  unlabeled_batch_size: 8
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
python semi_train.py --config configs/training_paradigms/semi_supervision/mean_teacher.yaml

# With AMP
python semi_train.py --config configs/training_paradigms/semi_supervision/unimatch.yaml --amp

# Multi-GPU
torchrun --nproc_per_node=4 semi_train.py --config configs/training_paradigms/semi_supervision/cps.yaml

# Test
python test.py --config configs/training_paradigms/semi_supervision/mean_teacher.yaml --checkpoint output/best_model.pth
```

## Key Parameters

- `labeled_ratio`: fraction of labeled data (0.05, 0.1, 0.2, 0.5)
- `split_mode`: `dir` (separate directories) or `ratio` (auto split)
- `semi.method`: method name from the table
- `semi.params`: method-specific hyperparameters (see each yaml in `configs/training_paradigms/semi_supervision/`)
