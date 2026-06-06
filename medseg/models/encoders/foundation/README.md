# Foundation Encoders

[中文文档](README_CN.md)

This package wraps a number of public vision foundation models as multi-stage
encoders that plug into the generic `build_model()` pipeline
(`encoder -> bottleneck -> decoder -> head`). Every encoder here inherits from
`BaseFoundationEncoder` (`_base.py`), so they share the same constructor
signature, the same `out_channels: List[int]` contract (deepest stage **last**),
and the same `FreezeMixin` API for freeze/partial-unfreeze.

## Available encoders

| `encoder.name` | Backbone (timm / HF) | Notes |
|---|---|---|
| `clip_vit` | OpenAI CLIP ViT-B/16 or ViT-L/14 (timm `vit_*_clip_*`) | Vision tower only |
| `biomedclip` | timm ViT-B/16 (`vit_base_patch16_*`) | BiomedCLIP weights (PMC) when available |
| `medclip` | timm `vit_base_patch16_224` | MedCLIP weights via `pretrained_path` |
| `dino` | timm DINO ViT-S/16 or ViT-B/16 | Self-supervised on natural images |
| `dinov2` | timm `vit_<variant>_patch14_dinov2` (S/B/L/G) | Default `variant="small"` |
| `dinov3` | timm `vit_<variant>_patch16_dinov3` | DINOv3 family |
| `raddino` | timm ViT-B/14 (DINOv2 architecture) | Rad-DINO chest-x-ray weights via `pretrained_path` |
| `sam_vit` | timm plain ViT-B/L/H/16 | Image-encoder half of SAM; use `sam_*` archs for the full SAM stack |
| `conch` | timm ViT-B/16 | CONCH pathology weights |
| `uni` | timm ViT-L/14 (DINOv2 architecture) | UNI pathology weights |
| `usfm` | timm ViT-B/16 | USFM ultrasound weights |
| `ctfm` | timm ViT-L/16 | CT-FM / Anatomy3D weights when available |
| `radimagenet` | timm ResNet50 | Multi-stage CNN features from `feature_info` |

All encoders accept the common `BaseFoundationEncoder` kwargs:

```python
(in_channels=3, img_size=224, pretrained=True,
 pretrained_path=None, freeze=True, unfreeze_last_n=0,
 inference_only=False, **subclass_kwargs)
```

Subclass-specific knobs (variant name, FPN channels, etc.) go under
`model.encoder.params`.

## Configuring freeze behaviour from YAML

`build_model()` calls `encoder.set_freeze_policy(...)` after construction if
`model.encoder.freeze_cfg` is present and the encoder exposes
`set_freeze_policy` (every `BaseFoundationEncoder` subclass does).

```yaml
model:
  encoder:
    name: dinov2
    in_channels: 3
    pretrained: true
    params:
      variant: small
    freeze_cfg:
      freeze: true            # freeze the whole backbone first
      unfreeze_last_n: 4      # then re-enable grads on the last 4 transformer blocks
      inference_only: false   # if true, also flips the encoder to eval() and disables grads everywhere
```

Semantics (see `FreezeMixin.set_freeze_policy` in `_base.py`):

- `freeze: true`  -> all backbone params get `requires_grad=False`.
- `freeze: false` -> all backbone params get `requires_grad=True`.
- `unfreeze_last_n: N` -> re-enable grads on the last `N` blocks of the backbone
  (looks for `backbone.blocks`/`layers`/`transformer_blocks`) plus the final
  norm. Anything not block-structured (e.g. RadImageNet's ResNet50) overrides
  `unfreeze_last_n_blocks` in its own subclass.
- `inference_only: true` -> set encoder to `eval()` and force every parameter
  in the encoder (including FPN/head adapters) to `requires_grad=False`.

If `freeze_cfg` is omitted, the policy passed via `BaseFoundationEncoder`'s
constructor defaults (`freeze=True, unfreeze_last_n=0, inference_only=False`)
stays in effect.

## Adapter-based PEFT

Set `encoder.params.use_adapter=true` (+ optional `adapter_dim`, default 64) to
inject Houlsby bottleneck adapters after each transformer block in the backbone.
When combined with `freeze_cfg.freeze=true`, only the adapters (~0.5-2% of
backbone params) train, matching standard ViT-Adapter PEFT recipes. Adapter is
a no-op for CNN-based FMs. See `ADAPTER_PEFT.md` for usage details.

## SAM-family networks: use `arch_params`, not `encoder.freeze_cfg`

The SAM family (`networks/sam/*`: `samed`, `samus`, `sam_b`, `sam_l`,
`auto_sam`, `medical_sam_adapter`, `mobile_sam`, `sam_med2d`, ...) is built
through `architecture:` / `build_special_arch(...)`, **not** through the
generic encoder/decoder pipeline. They derive from `SAMBase`, which has its
own three-way freeze API:

```yaml
model:
  architecture: samed
  arch_params:
    freeze_image_encoder: true
    freeze_prompt_encoder: true
    freeze_mask_decoder: false
    unfreeze_last_n_blocks: 0
```

`encoder.freeze_cfg` is ignored for these models (there is no
`model.encoder` block). Use `arch_params` instead.

The `sam_vit` *encoder* (this directory) is a different thing: it's only the
SAM ViT image-encoder, used as a generic backbone, and it does honour
`encoder.freeze_cfg` like every other entry in the table above.
