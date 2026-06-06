# Dataset Guide

[中文文档](README_CN.md)

This directory contains introductions to each medical image segmentation dataset, download instructions, and example YAML configurations.

## Dataset Overview

| Dataset | Modality | Classes | Official Split | Recommended Usage |
|---------|----------|---------|----------------|-------------------|
| ISIC 2016 | Dermoscopy | 2 (background+lesion) | train/test | Split val from train |
| ISIC 2017 | Dermoscopy | 2 | train/val/test | Use official split |
| ISIC 2018 | Dermoscopy | 2 | train/val/test | Use official split |
| BUSI | Breast Ultrasound | 3 (normal/benign/malignant) | No official split | 5-fold or 7:1:2 |
| CVC-ClinicDB | Colonoscopy Polyp | 2 | No official split | 5-fold or 8:1:1 |
| CVC-ColonDB | Colonoscopy Polyp | 2 | No official split | 5-fold or 8:1:1 |
| Kvasir-SEG | GI Endoscopy Polyp | 2 | No official split | 5-fold or 8:1:1 |
| GlaS | Gland Pathology | 2 | train/test | Split val from train |
| Synapse | Abdominal CT Multi-organ | 9 | TransUNet split | 18 train / 12 test |
| ACDC | Cardiac MRI | 4 | TransUNet split | 70 train / 10 val / 20 test |

## Directory Structure Convention

```
data/
├── ISIC2016/
│   ├── train/
│   │   ├── images/
│   │   └── masks/
│   └── test/
│       ├── images/
│       └── masks/
├── BUSI/
│   ├── images/
│   └── masks/
├── Synapse/
│   ├── train_npz/
│   └── test_vol_h5/
└── ...
```
