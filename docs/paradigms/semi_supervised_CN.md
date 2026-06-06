# 半监督分割

[English](semi_supervised.md)

本框架内置 **21** 种半监督方法，位于 `medseg/training/semi/`。

## 方法列表

| 方法 | 论文 | 发表 | GitHub | 说明 |
|------|------|------|--------|------|
| `mean_teacher` | Tarvainen & Valpola | NeurIPS 2017 | [CuriousAI/mean-teacher](https://github.com/CuriousAI/mean-teacher) | EMA 教师一致性 |
| `cps` | Chen et al. | CVPR 2021 | [charlesCXK/TorchSemiSeg](https://github.com/charlesCXK/TorchSemiSeg) | 交叉伪监督 |
| `cct` | Ouali et al. | BMVC 2020 | [yassouali/CCT](https://github.com/yassouali/CCT) | 交叉一致性 |
| `unimatch` | Yang et al. | CVPR 2023 | [LiheYoung/UniMatch](https://github.com/LiheYoung/UniMatch) | 统一双流匹配 |
| `fixmatch` | Sohn et al. | NeurIPS 2020 | [HiLab-git/SSL4MIS](https://github.com/HiLab-git/SSL4MIS) | 伪标签 + 强增强 |
| `urpc` | Luo et al. | MIA 2022 | [HiLab-git/SSL4MIS](https://github.com/HiLab-git/SSL4MIS) | 不确定性修正伪标签 |
| `deep_co_training` | Qiao et al. | ECCV 2018 | [AlanChou/Deep-Co-Training](https://github.com/AlanChou/Deep-Co-Training-for-Semi-Supervised-Image-Recognition) | 双网络协同训练 |
| `flexmatch` | Zhang et al. | NeurIPS 2021 | [TorchSSL/TorchSSL](https://github.com/TorchSSL/TorchSSL) | 课程伪标签 |
| `softmatch` | Chen et al. | ICLR 2023 | [microsoft/Semi-supervised-learning](https://github.com/microsoft/Semi-supervised-learning) | 软阈值 |
| `freematch` | Wang et al. | ICLR 2023 | [microsoft/Semi-supervised-learning](https://github.com/microsoft/Semi-supervised-learning) | 自适应阈值 |
| `ua_mt` | Yu et al. | MICCAI 2019 | [yulequan/UA-MT](https://github.com/yulequan/UA-MT) | 不确定性感知均值教师 |
| `ssl4mis_u` | SSL4MIS 不确定性 | - | [HiLab-git/SSL4MIS](https://github.com/HiLab-git/SSL4MIS) | MC-Dropout 不确定性 |
| `pi_model` | Laine & Aila | ICLR 2017 | [smlaine2/tempens](https://github.com/smlaine2/tempens) | 随机扰动 |
| `temporal_ensembling` | Laine & Aila | ICLR 2017 | [smlaine2/tempens](https://github.com/smlaine2/tempens) | 逐样本 EMA 目标 |
| `pseudo_label` | Lee | ICML-W 2013 | [iBelieveCJM/pseudo_label](https://github.com/iBelieveCJM/pseudo_label-pytorch) | 硬伪标签 |
| `ict` | Verma et al. | IJCAI 2019 | [vikasverma1077/ICT](https://github.com/vikasverma1077/ICT) | 插值一致性 |
| `r_drop` | Wu et al. | NeurIPS 2021 | [dropreg/R-Drop](https://github.com/dropreg/R-Drop) | 正则化 dropout |
| `cross_teaching` | Luo et al. | MIDL 2022 | [HiLab-git/SSL4MIS](https://github.com/HiLab-git/SSL4MIS) | CNN-Transformer 交叉教学 |
| `augseg` | Zhao et al. | CVPR 2023 | [ZhenZHAO/AugSeg](https://github.com/ZhenZHAO/AugSeg) | 增强策略研究 |
| `corrmatch` | Sun et al. | CVPR 2024 | [BBBBchan/CorrMatch](https://github.com/BBBBchan/CorrMatch) | 相关性匹配 |
| `allspark` | Wang et al. | CVPR 2024 | [xmed-lab/AllSpark](https://github.com/xmed-lab/AllSpark) | 重生有标签 token |
| `ddfp` | Wang et al. | CVPR 2024 | [Cuzyoung/DDFP](https://github.com/Cuzyoung/DDFP) | 扩散去噪伪标签 |
| `diffrect` | Liu et al. | MICCAI 2024 | [CUHK-AIM-Group/DiffRect](https://github.com/CUHK-AIM-Group/DiffRect) | 潜在扩散伪标签修正 |
| `ad_mt` | Zhao et al. | ECCV 2024 | [ZhenZHAO/AD-MT](https://github.com/ZhenZHAO/AD-MT) | 交替多样化教学 |
| `pmt` | Gao et al. | ECCV 2024 | [Axi404/PMT](https://github.com/Axi404/PMT) | 渐进均值教师 |

## 配置示例

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
  method: mean_teacher      # 上表任意方法名
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

## 用法

```bash
# 训练
python semi_train.py --config configs/training_paradigms/semi_supervision/mean_teacher.yaml

# 混合精度
python semi_train.py --config configs/training_paradigms/semi_supervision/unimatch.yaml --amp

# 多卡训练
torchrun --nproc_per_node=4 semi_train.py --config configs/training_paradigms/semi_supervision/cps.yaml

# 测试
python test.py --config configs/training_paradigms/semi_supervision/mean_teacher.yaml --checkpoint output/best_model.pth
```

## 关键参数

- `labeled_ratio`：标注数据比例（0.05, 0.1, 0.2, 0.5）
- `split_mode`：`dir`（独立目录）或 `ratio`（自动划分）
- `semi.method`：表中方法名
- `semi.params`：方法特定超参数（参见 `configs/training_paradigms/semi_supervision/` 中各 yaml）
