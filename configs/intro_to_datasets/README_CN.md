# 数据集介绍

[English](README.md)

本目录包含各医学图像分割数据集的介绍、下载方式、以及对应的示例 yaml 配置。

## 数据集总览

| 数据集 | 模态 | 类别 | 官方划分 | 推荐用法 |
|--------|------|------|----------|----------|
| ISIC 2016 | 皮肤镜 | 2 (背景+病灶) | train/test | 从 train 划出 val |
| ISIC 2017 | 皮肤镜 | 2 | train/val/test | 直接用官方划分 |
| ISIC 2018 | 皮肤镜 | 2 | train/val/test | 直接用官方划分 |
| BUSI | 乳腺超声 | 3 (正常/良性/恶性) | 无官方划分 | 5折 或 7:1:2 |
| CVC-ClinicDB | 结肠镜息肉 | 2 | 无官方划分 | 5折 或 8:1:1 |
| CVC-ColonDB | 结肠镜息肉 | 2 | 无官方划分 | 5折 或 8:1:1 |
| Kvasir-SEG | 胃肠镜息肉 | 2 | 无官方划分 | 5折 或 8:1:1 |
| GlaS | 腺体病理 | 2 | train/test | 从 train 划出 val |
| Synapse | 腹部CT多器官 | 9 | TransUNet划分 | 18例train/12例test |
| ACDC | 心脏MRI | 4 | TransUNet划分 | 70例train/10例val/20例test |

## 目录结构约定

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
