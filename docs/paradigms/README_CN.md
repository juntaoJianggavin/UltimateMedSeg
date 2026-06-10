# 训练基础设施

[English](README.md)

## 损失函数

本框架通过 `LOSS_REGISTRY` 注册了 **88 个** loss，按用途分为以下类别：

### 监督损失 (15)

| 名称 | 说明 | 源码 |
|------|------|------|
| `ce` | 交叉熵 | [ce_loss.py](../../medseg/losses/ce_loss.py) |
| `dice` | Dice 损失 | [dice_loss.py](../../medseg/losses/dice_loss.py) |
| `focal` | Focal 损失 (Lin et al.) | [focal_loss.py](../../medseg/losses/focal_loss.py) |
| `tversky` | Tversky 损失 | [tversky_loss.py](../../medseg/losses/tversky_loss.py) |
| `lovasz` | Lovasz-Softmax | [lovasz_loss.py](../../medseg/losses/lovasz_loss.py) |
| `boundary` | 边界损失 (距离图) | [boundary_loss.py](../../medseg/losses/boundary_loss.py) |
| `hausdorff` | Hausdorff 距离损失 | [hausdorff_loss.py](../../medseg/losses/hausdorff_loss.py) |
| `nsd` | 归一化表面距离 | [nsd_loss.py](../../medseg/losses/nsd_loss.py) |
| `edge` | 边缘损失 | [edge_loss.py](../../medseg/losses/edge_loss.py) |
| `el_loss` | 指数对数损失 | [el_loss.py](../../medseg/losses/el_loss.py) |
| `contrastive` | 监督对比损失 | [contrastive_loss.py](../../medseg/losses/contrastive_loss.py) |
| `wasserstein_dice` | Wasserstein Dice | [wasserstein_dice_loss.py](../../medseg/losses/wasserstein_dice_loss.py) |
| `kl_divergence` | KL 散度 | [kl_loss.py](../../medseg/losses/kl_loss.py) |
| `compound` | 组合损失 (加权组合) | [compound_loss.py](../../medseg/losses/compound_loss.py) |
| `deep_supervision` | 深度监督包装器 | [deep_supervision_loss.py](../../medseg/losses/deep_supervision_loss.py) |

### 半监督损失 (21)

半监督损失集成在 `medseg/training/semi/` 中，非独立损失类。详见 [semi_supervised.md](semi_supervised.md)。

### 域适应损失 (18)

详见 [domain_adaptation.md](domain_adaptation.md)。

### 知识蒸馏损失 (27)

详见 [distillation.md](distillation.md)。

### 弱监督损失 (28)

详见 [weakly_supervised.md](weakly_supervised.md)。

---

## 数据增强

支持两种增强模式：

### 基础模式

```yaml
augmentation:
  mode: basic
  params:
    random_flip: true
    random_rotate: 15
    random_scale: [0.8, 1.2]
    color_jitter: 0.2
```

### Albumentations 模式

```yaml
augmentation:
  mode: albumentations
  params:
    - name: HorizontalFlip
      p: 0.5
    - name: RandomBrightnessContrast
      p: 0.3
    - name: ElasticTransform
      alpha: 120
      sigma: 6
      p: 0.3
    - name: GaussNoise
      var_limit: [10, 50]
      p: 0.2
```

---

## 多卡训练

支持三种模式：

```yaml
training:
  parallel: auto    # auto | ddp | dp | none
```

| 值 | 模式 | 说明 |
|----|------|------|
| `auto` | 自动检测 | WORLD_SIZE>1 时用 DDP，否则单卡 |
| `ddp` | DistributedDataParallel | 多进程，推荐 |
| `dp` | DataParallel | 单进程多卡 |
| `none` | 单卡 | 无并行 |

DDP 启动：

```bash
torchrun --nproc_per_node=4 train.py --config configs/xxx.yaml
```

---

## 可复现性

```yaml
training:
  random_state: 42
  deterministic: true
```

设置 `torch.manual_seed`、`np.random.seed`、`random.seed`、`torch.cuda.manual_seed_all` 和 `torch.backends.cudnn.deterministic`。

---

## 日志系统

```yaml
training:
  logger: tensorboard    # tensorboard | wandb | both | none
  wandb:
    project: medseg
    entity: my_team
    name: exp_01
```

| 值 | 后端 |
|----|------|
| `tensorboard` | TensorBoard（默认） |
| `wandb` | Weights & Biases |
| `both` | TensorBoard + WandB |
| `none` | 禁用 |

---

## 混合精度 (AMP)

```yaml
training:
  amp: true
```

命令行覆盖：

```bash
python train.py --config configs/xxx.yaml --amp
```

使用 `torch.cuda.amp.GradScaler` + `autocast` 进行 FP16 训练。

---

## 优化器

```yaml
training:
  optimizer:
    name: adamw          # adamw | sgd | adam | lion
    lr: 1e-4
    weight_decay: 1e-4
    # SGD 专属
    momentum: 0.9
    nesterov: true
```

| 名称 | 论文 / 来源 |
|------|-------------|
| `adamw` | Loshchilov & Hutter, ICLR 2019 |
| `sgd` | 经典 SGD + 动量 |
| `adam` | Kingma & Ba, ICLR 2015 |
| `lion` | Chen et al., ICML 2024 |

---

## 学习率调度器

```yaml
training:
  scheduler:
    name: cosine         # cosine | step | poly | warmup_cosine | warmup_poly
    min_lr: 1e-6
    # step 专属
    step_size: 30
    gamma: 0.1
    # poly 专属
    power: 0.9
    # warmup 专属
    warmup_epochs: 10
    warmup_lr: 1e-6
```

| 名称 | 公式 |
|------|------|
| `cosine` | CosineAnnealingLR |
| `step` | StepLR (每 N 轮衰减) |
| `poly` | PolyLR: lr * (1 - iter/max_iter)^power |
| `warmup_cosine` | 线性预热 + 余弦衰减 |
| `warmup_poly` | 线性预热 + 多项式衰减 |

---

## 配置继承

使用 `_base_` 从父配置继承：

```yaml
_base_: configs/architectures/combinations/general/unet_resnet50.yaml

# 覆盖特定字段
training:
  epochs: 300
  optimizer:
    lr: 5e-5

data:
  type: acdc
  img_size: 256
```

子配置深度合并到基础配置中。列表被替换而非追加。

---

## 完整示例

```yaml
_base_: configs/architectures/combinations/general/unet_resnet50.yaml

model:
  num_classes: 9
  img_size: 224

data:
  type: synapse
  img_size: 224
  train_dir: ./data/Synapse/train_npz
  val_dir: ./data/Synapse/test_vol_h5

augmentation:
  mode: albumentations
  params:
    - name: HorizontalFlip
      p: 0.5

training:
  epochs: 200
  batch_size: 16
  amp: true
  random_state: 42
  deterministic: true
  parallel: auto
  logger: tensorboard

  optimizer:
    name: adamw
    lr: 1e-4
    weight_decay: 1e-4

  scheduler:
    name: warmup_cosine
    warmup_epochs: 10
    min_lr: 1e-6

  loss:
    name: compound
    params:
      losses:
        - name: ce
          weight: 0.4
        - name: dice
          weight: 0.6

  val_interval: 10
  save_interval: 50
```
