# 解码器 (`medseg/decoders/`)

[English](README.md)

U 形上采样解码器，接收多尺度编码器特征并生成分割图。

## 已注册解码器

| 注册键 | 源文件 | 说明 |
|---|---|---|
| `bilinear` | `bilinear_decoder.py` | 双线性上采样 + 1×1 卷积（最快基线） |
| `deconv` | `deconv_decoder.py` | 转置卷积解码器，带跳跃拼接 |
| `attention` | `attention_decoder.py` | 每层跳跃连接处注意力门控（Attention U-Net） |
| `cascade` | `cascade_decoder.py` | 级联细化解码器 |
| `dw_sep` | `dw_sep_decoder.py` | 深度可分离卷积解码器（轻量级） |
| `emcad` | `emcad_decoder.py` | EMCad 多尺度聚合解码器 |
| `mlp` | `mlp_decoder.py` | MLP 解码器（SegFormer 风格） |
| `lawin` | `lawin_decoder.py` | Lawin 多尺度上下文解码器 |
| `ham` | `ham_decoder.py` | HAM（层次聚合模块）解码器 |
| `upernet` | `upernet_decoder.py` | UPerNet FPN+PPM 解码器 |
| `unetpp` | `unetpp_decoder.py` | UNet++ 密集跳跃解码器 |
| `transunet` | `transunet_decoder.py` | TransUNet 级联上卷积解码器 |
| `swinunet` | `swinunet_decoder.py` | Swin-UNet 分块扩展解码器 |
| `missformer` | `missformer_decoder.py` | MISSFormer 频率感知解码器 |
| `dcsaunet` | `dcsaunet_decoder.py` | DCSAU-Net 双路径解码器 |
| `rwkv_unet` | `rwkv_unet_decoder.py` | RWKV 序列模型解码器 |
| `vmunet` | `vmunet_decoder.py` | VM-UNet Mamba-SSM 解码器 |

## YAML 配置用法

```yaml
model:
  decoder:
    name: attention         # 任意已注册键
    params:
      encoder_channels: [64, 256, 512, 1024, 2048]
      decoder_channels: [256, 128, 64, 32, 16]
```
