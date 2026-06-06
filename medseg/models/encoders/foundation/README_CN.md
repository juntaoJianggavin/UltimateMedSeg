# 基础模型编码器

[English](README.md)

本包将多个公开视觉基础模型封装为多阶段编码器，可插入通用 `build_model()` 流水线
（`encoder -> bottleneck -> decoder -> head`）。所有编码器均继承自
`BaseFoundationEncoder`（`_base.py`），共享相同的构造函数签名、
`out_channels: List[int]` 约定（最深阶段在**最后**），
以及 `FreezeMixin` API 用于冻结/部分解冻。

## 可用编码器

| `encoder.name` | 骨干网络（timm / HF） | 说明 |
|---|---|---|
| `clip_vit` | OpenAI CLIP ViT-B/16 或 ViT-L/14（timm `vit_*_clip_*`） | 仅视觉塔 |
| `biomedclip` | timm ViT-B/16（`vit_base_patch16_*`） | BiomedCLIP 权重（PMC），可用时加载 |
| `medclip` | timm `vit_base_patch16_224` | MedCLIP 权重，通过 `pretrained_path` 加载 |
| `dino` | timm DINO ViT-S/16 或 ViT-B/16 | 自然图像自监督预训练 |
| `dinov2` | timm `vit_<variant>_patch14_dinov2`（S/B/L/G） | 默认 `variant="small"` |
| `dinov3` | timm `vit_<variant>_patch16_dinov3` | DINOv3 系列 |
| `raddino` | timm ViT-B/14（DINOv2 架构） | Rad-DINO 胸部 X 光权重，通过 `pretrained_path` 加载 |
| `sam_vit` | timm 标准 ViT-B/L/H/16 | SAM 图像编码器部分；完整 SAM 栈请使用 `sam_*` 架构 |
| `conch` | timm ViT-B/16 | CONCH 病理权重 |
| `uni` | timm ViT-L/14（DINOv2 架构） | UNI 病理权重 |
| `usfm` | timm ViT-B/16 | USFM 超声权重 |
| `ctfm` | timm ViT-L/16 | CT-FM / Anatomy3D 权重，可用时加载 |
| `radimagenet` | timm ResNet50 | 通过 `feature_info` 获取多阶段 CNN 特征 |

所有编码器接受通用的 `BaseFoundationEncoder` 参数：

```python
(in_channels=3, img_size=224, pretrained=True,
 pretrained_path=None, freeze=True, unfreeze_last_n=0,
 inference_only=False, **subclass_kwargs)
```

子类特有参数（变体名称、FPN 通道等）放在 `model.encoder.params` 下。

## 通过 YAML 配置冻结行为

如果 `model.encoder.freeze_cfg` 存在且编码器暴露了 `set_freeze_policy`
（所有 `BaseFoundationEncoder` 子类均支持），`build_model()` 会在构造后调用
`encoder.set_freeze_policy(...)`。

```yaml
model:
  encoder:
    name: dinov2
    in_channels: 3
    pretrained: true
    params:
      variant: small
    freeze_cfg:
      freeze: true            # 先冻结整个骨干网络
      unfreeze_last_n: 4      # 然后解冻最后 4 个 Transformer block
      inference_only: false   # 若为 true，还将编码器切换到 eval() 并全局禁用梯度
```

语义说明（参见 `_base.py` 中的 `FreezeMixin.set_freeze_policy`）：

- `freeze: true`  -> 所有骨干参数设为 `requires_grad=False`。
- `freeze: false` -> 所有骨干参数设为 `requires_grad=True`。
- `unfreeze_last_n: N` -> 重新启用骨干最后 `N` 个 block 的梯度
  （查找 `backbone.blocks`/`layers`/`transformer_blocks`）以及最终归一化层。
  非分块结构的模型（如 RadImageNet 的 ResNet50）在其子类中覆盖
  `unfreeze_last_n_blocks`。
- `inference_only: true` -> 将编码器设为 `eval()` 并强制编码器中所有参数
  （包括 FPN/head 适配器）为 `requires_grad=False`。

如果省略 `freeze_cfg`，则使用 `BaseFoundationEncoder` 构造函数的默认策略
（`freeze=True, unfreeze_last_n=0, inference_only=False`）。

## 基于适配器的 PEFT

设置 `encoder.params.use_adapter=true`（+ 可选 `adapter_dim`，默认 64）可在骨干中
每个 Transformer block 后注入 Houlsby 瓶颈适配器。
与 `freeze_cfg.freeze=true` 配合使用时，仅训练适配器（约占骨干参数的 0.5-2%），
符合标准 ViT-Adapter PEFT 方案。适配器对 CNN 基础模型无效。
详见 `ADAPTER_PEFT.md`。

## SAM 家族网络：使用 `arch_params`，而非 `encoder.freeze_cfg`

SAM 家族（`networks/sam/*`：`samed`、`samus`、`sam_b`、`sam_l`、
`auto_sam`、`medical_sam_adapter`、`mobile_sam`、`sam_med2d` 等）通过
`architecture:` / `build_special_arch(...)` 构建，**不**通过通用编码器/解码器流水线。
它们继承自 `SAMBase`，拥有独立的三路冻结 API：

```yaml
model:
  architecture: samed
  arch_params:
    freeze_image_encoder: true
    freeze_prompt_encoder: true
    freeze_mask_decoder: false
    unfreeze_last_n_blocks: 0
```

`encoder.freeze_cfg` 对这些模型无效（不存在 `model.encoder` 块）。请使用 `arch_params`。

`sam_vit` *编码器*（本目录）是不同的东西：它只是 SAM ViT 图像编码器，
作为通用骨干使用，与上表中其他条目一样支持 `encoder.freeze_cfg`。
