# Training Infrastructure

[中文文档](README_CN.md)

## Loss Functions

The framework registers **88** losses via `LOSS_REGISTRY`, grouped by usage:

### Supervised Losses (15)

| Name | Description |
|------|-------------|
| `ce` | Cross-Entropy |
| `dice` | Dice Loss |
| `focal` | Focal Loss (Lin et al.) |
| `tversky` | Tversky Loss |
| `lovasz` | Lovasz-Softmax |
| `boundary` | Boundary Loss (distance map) |
| `hausdorff` | Hausdorff Distance Loss |
| `nsd` | Normalized Surface Distance |
| `edge` | Edge Loss |
| `el_loss` | Exponential Logarithmic Loss |
| `contrastive` | Supervised Contrastive Loss |
| `wasserstein_dice` | Wasserstein Dice |
| `kl_divergence` | KL Divergence |
| `compound` | Compound (weighted combination) |
| `deep_supervision` | Deep Supervision wrapper |

### Semi-Supervised Losses (21)

Semi-supervised losses are integrated in `medseg/training/semi/`, not as standalone criterion classes. See [semi_supervised.md](semi_supervised.md).

### Domain Adaptation Losses (18)

See [domain_adaptation.md](domain_adaptation.md).

### Knowledge Distillation Losses (27)

See [distillation.md](distillation.md).

### Weakly Supervised Losses (28)

See [weakly_supervised.md](weakly_supervised.md).

---

## Data Augmentation

Two augmentation modes:

### Basic Mode

```yaml
augmentation:
  mode: basic
  params:
    random_flip: true
    random_rotate: 15
    random_scale: [0.8, 1.2]
    color_jitter: 0.2
```

### Albumentations Mode

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

## Multi-GPU Training

Three modes supported:

```yaml
training:
  parallel: auto    # auto | ddp | dp | none
```

| Value | Mode | Description |
|-------|------|-------------|
| `auto` | Auto-detect | DDP if WORLD_SIZE>1, else single GPU |
| `ddp` | DistributedDataParallel | Multi-process, recommended |
| `dp` | DataParallel | Single-process multi-GPU |
| `none` | Single GPU | No parallelism |

DDP launch:

```bash
torchrun --nproc_per_node=4 train.py --config configs/xxx.yaml
```

---

## Reproducibility

```yaml
training:
  random_state: 42
  deterministic: true
```

Sets `torch.manual_seed`, `np.random.seed`, `random.seed`, `torch.cuda.manual_seed_all`, and `torch.backends.cudnn.deterministic`.

---

## Logging

```yaml
training:
  logger: tensorboard    # tensorboard | wandb | both | none
  wandb:
    project: medseg
    entity: my_team
    name: exp_01
```

| Value | Backend |
|-------|---------|
| `tensorboard` | TensorBoard (default) |
| `wandb` | Weights & Biases |
| `both` | TensorBoard + WandB |
| `none` | Disabled |

---

## Mixed Precision (AMP)

```yaml
training:
  amp: true
```

CLI override:

```bash
python train.py --config configs/xxx.yaml --amp
```

Uses `torch.cuda.amp.GradScaler` + `autocast` for FP16 training.

---

## Optimizers

```yaml
training:
  optimizer:
    name: adamw          # adamw | sgd | adam | lion
    lr: 1e-4
    weight_decay: 1e-4
    # SGD-specific
    momentum: 0.9
    nesterov: true
```

| Name | Paper / Source |
|------|---------------|
| `adamw` | Loshchilov & Hutter, ICLR 2019 |
| `sgd` | Classic SGD with momentum |
| `adam` | Kingma & Ba, ICLR 2015 |
| `lion` | Chen et al., ICML 2024 |

---

## Schedulers

```yaml
training:
  scheduler:
    name: cosine         # cosine | step | poly | warmup_cosine | warmup_poly
    min_lr: 1e-6
    # step-specific
    step_size: 30
    gamma: 0.1
    # poly-specific
    power: 0.9
    # warmup-specific
    warmup_epochs: 10
    warmup_lr: 1e-6
```

| Name | Formula |
|------|---------|
| `cosine` | CosineAnnealingLR |
| `step` | StepLR (decay every N epochs) |
| `poly` | PolyLR: lr * (1 - iter/max_iter)^power |
| `warmup_cosine` | Linear warmup + cosine decay |
| `warmup_poly` | Linear warmup + poly decay |

---

## Config Inheritance

Use `_base_` to inherit from a parent config:

```yaml
_base_: configs/architectures/networks/general/unet_resnet50.yaml

# Override specific fields
training:
  epochs: 300
  optimizer:
    lr: 5e-5

data:
  type: acdc
  img_size: 256
```

The child config deep-merges into the base. Lists are replaced, not appended.

---

## Full Example

```yaml
_base_: configs/architectures/networks/general/unet_resnet50.yaml

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
