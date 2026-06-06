# Adapter-based PEFT

[中文文档](ADAPTER_PEFT_CN.md)

Set `encoder.params.use_adapter=true` (+ optional `adapter_dim`, default 64) to
inject Houlsby bottleneck adapters after each transformer block in the backbone.
When combined with `freeze_cfg.freeze=true`, only the adapters (~0.5-2% of
backbone params) train, matching standard ViT-Adapter PEFT recipes. Adapter is
a no-op for CNN-based FMs.

## Example

```yaml
model:
  encoder:
    name: dinov2
    pretrained: true
    params:
      variant: base
      use_adapter: true
      adapter_dim: 64
    freeze_cfg:
      freeze: true
      unfreeze_last_n: 0
      inference_only: false
```

To use PEFT adapters, add the following to any foundation encoder config:

```yaml
model:
  encoder:
    params:
      use_adapter: true
      adapter_dim: 64
    freeze_cfg:
      freeze: true
      unfreeze_last_n: 0
      inference_only: false
```
