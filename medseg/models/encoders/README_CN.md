# 编码器 (`medseg/encoders/`)

[English](README.md)

模块化特征提取器，为 U 形解码器生成多尺度特征图。

## 已注册编码器

| 注册键 | 源文件 | 说明 |
|---|---|---|
| `basic` | `basic_encoder.py` | 轻量级 4 层卷积编码器（基线） |
| `timm_resnet18/34/50/101/152` | `timm_encoder.py` | ResNet 系列 (通过 timm, ImageNet 预训练) |
| `timm_resnext50_32x4d`, `timm_resnext101_32x8d` | `timm_encoder.py` | ResNeXt 系列 |
| `timm_wide_resnet50_2`, `timm_wide_resnet101_2` | `timm_encoder.py` | Wide ResNet |
| `timm_res2net50_26w_4s` | `timm_encoder.py` | Res2Net |
| `timm_vgg16/19`, `timm_vgg16_bn/19_bn` | `timm_encoder.py` | VGG 系列 |
| `timm_densenet121/161/169/201` | `timm_encoder.py` | DenseNet 系列 |
| `timm_efficientnet_b0`–`b5` | `timm_encoder.py` | EfficientNet 系列 |
| `timm_efficientnetv2_s/m` | `timm_encoder.py` | EfficientNetV2 |
| `timm_mobilenetv2_100`, `timm_mobilenetv3_*` | `timm_encoder.py` | MobileNet 系列 |
| `timm_convnext_tiny/small/base/large` | `timm_encoder.py` | ConvNeXt 系列 |
| `timm_convnextv2_tiny/base` | `timm_encoder.py` | ConvNeXtV2 |
| `timm_swin_tiny/small/base_*` | `timm_encoder.py` | Swin Transformer |
| `timm_swinv2_tiny_*` | `timm_encoder.py` | Swin Transformer V2 |
| `timm_pvt_v2_b0`–`b4` | `timm_encoder.py` | 金字塔视觉 Transformer v2 |
| `timm_mit_b0/b1/b2/b3/b5` | `timm_encoder.py` | MixTransformer (SegFormer 主干) |
| `timm_maxvit_tiny/small_*` | `timm_encoder.py` | MaxViT |
| `timm_senet154`, `timm_seresnet50` | `timm_encoder.py` | SENet 系列 |
| `timm_fastvit_t8`, `timm_mobilevit_s` | `timm_encoder.py` | 轻量级编码器 |
| `timm_coatnet_0_224` | `timm_encoder.py` | CoAtNet |
| `timm_*` (更多) | `timm_encoder.py` | 通用 timm 封装 — 通过 params 传入 `model_name` |
| `timm_vit_dino_base/small` | `vit_pyramid_encoder.py` | DINO ViT 金字塔适配 |
| `timm_vit_dinov2_base/large/giant` | `vit_pyramid_encoder.py` | DINOv2 ViT 金字塔 |
| `timm_vit_dinov3_small/base/large/huge_plus/7b` | `vit_pyramid_encoder.py` | DINOv3 (2025) ViT 金字塔 |
| `timm_vit_clip_base/large/huge` | `vit_pyramid_encoder.py` | CLIP ViT 金字塔 |
| `timm_vit_sam_base/large/huge` | `vit_pyramid_encoder.py` | SAM ViT 金字塔 |
| `timm_vit_mae_base/large` | `vit_pyramid_encoder.py` | MAE ViT 金字塔 |
| `timm_vit_deit_base/large` | `vit_pyramid_encoder.py` | DeiT3 ViT 金字塔 |
| `transunet` | `transunet_encoder.py` | TransUNet CNN+Transformer 混合编码器 |
| `swinunet` | `swinunet_encoder.py` | Swin-UNet 纯 Transformer 编码器 |
| `rwkv_unet` | `rwkv_encoder.py` | RWKV (线性 Transformer) 编码器 |
| `vmunet` | `vmunet_encoder.py` | VM-UNet (Visual Mamba) SSM 编码器 |
| `rir_zigzag` | `rir_zigzag_encoder.py` | RiR Zigzag 密集聚合编码器 |
| `uctransnet` | `ucransnet_encoder.py` | UCTransNet 通道感知 Transformer |
| `missformer` | `missformer_encoder.py` | MISSFormer (空间 + 频率) |
| `dcsaunet` | `dcsaunet_encoder.py` | DCSAU-Net 双路径编码器 |
| `medt` | `medt_encoder.py` | MedT 轴向注意力 Transformer |
| `mctrans` | `mctrans_encoder.py` | McTrans 多尺度 Transformer |
| `hiformer` | `hiformer_encoder.py` | HiFormer 层次化 Transformer |
| `daeformer` | `daeformer_encoder.py` | DAEFormer 可变形注意力 |
| `fatnet` | `fatnet_encoder.py` | FAT-Net 频率感知 Transformer |
| `h2former` | `h2former_encoder.py` | H2Former 混合层次 Transformer |
| `scaleformer` | `scaleformer_encoder.py` | ScaleFormer 多尺度注意力 |
| `cfanet` | `cfanet_encoder.py` | CFANet 跨频率注意力 |
| `mtunet` | `mtunet_encoder.py` | MT-UNet 多任务 Transformer |

## YAML 配置用法

```yaml
model:
  encoder:
    name: timm_resnet50       # 任意已注册键
    pretrained: true
    in_channels: 3
    params: {}                # 转发给构造函数的额外参数
```

## 添加新编码器

1. 在 `medseg/encoders/` 中创建 `my_encoder.py`。
2. 用 `@ENCODER_REGISTRY.register("my_key")` 装饰类。
3. 实现 `forward(x) -> List[Tensor]` 返回多尺度特征。
4. 添加 `out_channels` 属性列出每个尺度的通道数。
5. 在 `medseg/encoders/__init__.py` 中导入模块。
