# 基于适配器的 PEFT

[English](ADAPTER_PEFT.md)

设置 `encoder.params.use_adapter=true`（+ 可选 `adapter_dim`，默认 64）可在骨干中
每个 Transformer block 后注入 Houlsby 瓶颈适配器。
与 `freeze_cfg.freeze=true` 配合使用时，仅训练适配器（约占骨干参数的 0.5-2%），
符合标准 ViT-Adapter PEFT 方案。适配器对 CNN 基础模型无效。

## 示例

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

要在任意基础编码器配置中使用 PEFT 适配器，添加以下内容：

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
