# Encoders (`medseg/encoders/`)

[中文文档](README_CN.md)

Modular feature extractors that produce multi-scale feature maps for the U-shaped decoder.

## Registered Encoders

| Registry Key | Source File | Description |
|---|---|---|
| `basic` | `basic_encoder.py` | Lightweight 4-layer Conv encoder (baseline) |
| `timm_resnet18/34/50/101/152` | `timm_encoder.py` | ResNet family via timm (pretrained on ImageNet) |
| `timm_resnext50_32x4d`, `timm_resnext101_32x8d` | `timm_encoder.py` | ResNeXt family |
| `timm_wide_resnet50_2`, `timm_wide_resnet101_2` | `timm_encoder.py` | Wide ResNet |
| `timm_res2net50_26w_4s` | `timm_encoder.py` | Res2Net |
| `timm_vgg16/19`, `timm_vgg16_bn/19_bn` | `timm_encoder.py` | VGG family |
| `timm_densenet121/161/169/201` | `timm_encoder.py` | DenseNet family |
| `timm_efficientnet_b0`–`b5` | `timm_encoder.py` | EfficientNet family |
| `timm_efficientnetv2_s/m` | `timm_encoder.py` | EfficientNetV2 |
| `timm_mobilenetv2_100`, `timm_mobilenetv3_*` | `timm_encoder.py` | MobileNet family |
| `timm_convnext_tiny/small/base/large` | `timm_encoder.py` | ConvNeXt family |
| `timm_convnextv2_tiny/base` | `timm_encoder.py` | ConvNeXtV2 |
| `timm_swin_tiny/small/base_*` | `timm_encoder.py` | Swin Transformer |
| `timm_swinv2_tiny_*` | `timm_encoder.py` | Swin Transformer V2 |
| `timm_pvt_v2_b0`–`b4` | `timm_encoder.py` | Pyramid Vision Transformer v2 |
| `timm_mit_b0/b1/b2/b3/b5` | `timm_encoder.py` | MixTransformer (SegFormer backbone) |
| `timm_maxvit_tiny/small_*` | `timm_encoder.py` | MaxViT |
| `timm_senet154`, `timm_seresnet50` | `timm_encoder.py` | SENet family |
| `timm_fastvit_t8`, `timm_mobilevit_s` | `timm_encoder.py` | Lightweight encoders |
| `timm_coatnet_0_224` | `timm_encoder.py` | CoAtNet |
| `timm_*` (many more) | `timm_encoder.py` | Generic timm wrapper — pass `model_name` in params |
| `timm_vit_dino_base/small` | `vit_pyramid_encoder.py` | DINO ViT with pyramid adaptation |
| `timm_vit_dinov2_base/large/giant` | `vit_pyramid_encoder.py` | DINOv2 ViT pyramid |
| `timm_vit_dinov3_small/base/large/huge_plus/7b` | `vit_pyramid_encoder.py` | DINOv3 (2025) ViT pyramid |
| `timm_vit_clip_base/large/huge` | `vit_pyramid_encoder.py` | CLIP ViT pyramid |
| `timm_vit_sam_base/large/huge` | `vit_pyramid_encoder.py` | SAM ViT pyramid |
| `timm_vit_mae_base/large` | `vit_pyramid_encoder.py` | MAE ViT pyramid |
| `timm_vit_deit_base/large` | `vit_pyramid_encoder.py` | DeiT3 ViT pyramid |
| `transunet` | `transunet_encoder.py` | TransUNet CNN+Transformer hybrid |
| `swinunet` | `swinunet_encoder.py` | Swin-UNet pure-Transformer encoder |
| `rwkv_unet` | `rwkv_encoder.py` | RWKV (linear Transformer) encoder |
| `vmunet` | `vmunet_encoder.py` | VM-UNet (Visual Mamba) SSM encoder |
| `rir_zigzag` | `rir_zigzag_encoder.py` | RiR Zigzag dense aggregation encoder |
| `uctransnet` | `ucransnet_encoder.py` | UCTransNet channel-aware Transformer |
| `missformer` | `missformer_encoder.py` | MISSFormer (spatial + frequency) |
| `dcsaunet` | `dcsaunet_encoder.py` | DCSAU-Net dual-path encoder |
| `medt` | `medt_encoder.py` | MedT axial-attention Transformer |
| `mctrans` | `mctrans_encoder.py` | McTrans multi-scale Transformer |
| `hiformer` | `hiformer_encoder.py` | HiFormer hierarchical Transformer |
| `daeformer` | `daeformer_encoder.py` | DAEFormer deformable attention |
| `fatnet` | `fatnet_encoder.py` | FAT-Net frequency-aware Transformer |
| `h2former` | `h2former_encoder.py` | H2Former hybrid hierarchical Transformer |
| `scaleformer` | `scaleformer_encoder.py` | ScaleFormer multi-scale attention |
| `cfanet` | `cfanet_encoder.py` | CFANet cross-frequency attention |
| `mtunet` | `mtunet_encoder.py` | MT-UNet multi-task Transformer |

## Usage in YAML Config

```yaml
model:
  encoder:
    name: timm_resnet50       # any registered key
    pretrained: true
    in_channels: 3
    params: {}                # extra kwargs forwarded to constructor
```

## Adding a New Encoder

1. Create `my_encoder.py` in `medseg/encoders/`.
2. Decorate the class with `@ENCODER_REGISTRY.register("my_key")`.
3. Implement `forward(x) -> List[Tensor]` returning multi-scale features.
4. Add an `out_channels` property listing channel counts per scale.
5. Import the module in `medseg/encoders/__init__.py`.
