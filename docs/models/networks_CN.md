# 完整网络架构

[English](networks.md)

本项目支持 136 个完整网络架构（123 个标准架构 + 13 个文本引导架构），通过 `architecture` 字段直接使用。

## CNN (33)

卷积神经网络系列，经典 UNet 及其变体。

| 名称 | 论文 | 发表 | GitHub |
|---|---|---|---|
| `attention_unet` | Attention U-Net | MIDL 2018 | [ozan-oktay/Attention-Gating-Network](https://github.com/ozan-oktay/Attention-Gating-Network) |
| `unetpp` | UNet++ | DLMIA 2018 | [MrGiovanni/UNetPlusPlus](https://github.com/MrGiovanni/UNetPlusPlus) |
| `r2unet` | R2U-Net | IEEE Access 2018 | - |
| `multiresunet` | MultiResUNet | Neural Networks 2020 | - |
| `resunet_a` | ResUNet-a | ISPRS 2020 | - |
| `resunetpp` | ResUNet++ | ISM 2019 | - |
| `unet3plus` | UNet 3+ | ICASSP 2020 | [ZJUGiveLab/UNet-Version](https://github.com/ZJUGiveLab/UNet-Version) |
| `denseunet` | DenseUNet | - | - |
| `scseunet` | scSE-UNet (Squeeze-Excitation) | MICCAI 2018 | - |
| `sa_unet` | SA-UNet (Spatial Attention) | IEEE TIM 2021 | - |
| `kiunet` | KiU-Net | MICCAI 2020 | [jeya-maria-jose/KiU-Net-pytorch](https://github.com/jeya-maria-jose/KiU-Net-pytorch) |
| `pan` | PAN (Pyramid Attention Network) | BMVC 2018 | - |
| `linknet` | LinkNet | VCIP 2017 | - |
| `pspnet` | PSPNet | CVPR 2017 | - |
| `double_unet` | DoubleU-Net | CBMS 2020 | - |
| `fr_unet` | FR-UNet (Full-Resolution) | IEEE TMI 2022 | - |
| `dcsaunet` | DCSAU-Net | Computers in Biology and Medicine 2023 | [xq141839/DCSAU-Net](https://github.com/xq141839/DCSAU-Net) |
| `cfanet` | CFA-Net | Computers in Biology and Medicine 2024 | [ZhangJD-ong/CFA-Net](https://github.com/ZhangJD-ong/CFA-Net) |
| `mednext` | MedNeXt | MICCAI 2023 | [MIC-DKFZ/MedNeXt](https://github.com/MIC-DKFZ/MedNeXt) |
| `nnunet_2d` | nnU-Net (2D) | Nature Methods 2021 | [MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet) |
| `acc_unet` | ACC-UNet | MICCAI 2023 | - |
| `cmunext` | CMUNeXt | arXiv 2023 | - |
| `mew_unet` | MEW-UNet | arXiv 2024 | - |
| `lv_unet` | LV-UNet (Lightweight) | - | - |
| `ege_unet` | EGE-UNet | arXiv 2023 | [JCruan519/EGE-UNet](https://github.com/JCruan519/EGE-UNet) |
| `malunet` | MALUNet | arXiv 2022 | - |
| `lite_unet` | Lite-UNet | - | - |
| `mk_unet` | MK-UNet | - | - |
| `u_lite` | U-Lite | arXiv 2022 | - |
| `aau_net` | AAU-Net | IEEE JBHI 2023 | [CGPxy/AAU-net](https://github.com/CGPxy/AAU-net) |
| `cmu_net` | CMU-Net | Bioinformatics 2024 | - |
| `dscnet` | DSCNet | MICCAI 2023 | - |
| `dconnnet` | DconnNet | MICCAI 2023 | - |
| `stu_net` | STU-Net | arXiv 2023 | - |

## Transformer (32)

基于 Transformer 的分割网络。

| 名称 | 论文 | 发表 | GitHub |
|---|---|---|---|
| `transunet` | TransUNet | arXiv 2021 | [Beckschen/TransUNet](https://github.com/Beckschen/TransUNet) |
| `swinunet` | Swin-UNet | ECCV 2022 | [HuCaoFighting/Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet) |
| `medt` | MedT (Medical Transformer) | MICCAI 2021 | [jeya-maria-jose/Medical-Transformer](https://github.com/jeya-maria-jose/Medical-Transformer) |
| `daeformer` | DAEFormer | ICLR 2023 | - |
| `missformer` | MISSFormer | IEEE TMI 2022 | - |
| `h2former` | H2Former | IEEE TMI 2023 | - |
| `hiformer` | HiFormer | WACV 2023 | - |
| `mctrans` | MCTrans | MICCAI 2021 | - |
| `mtunet` | MT-UNet | MICCAI 2022 | - |
| `scaleformer` | ScaleFormer | MICCAI 2022 | - |
| `fatnet` | FAT-Net | IEEE TMI 2022 | - |
| `nnformer_2d` | nnFormer (2D) | MICCAI 2022 | - |
| `transfuse` | TransFuse | MICCAI 2021 | - |
| `levit_unet` | LeViT-UNet | ML4H 2022 | - |
| `transatt_unet` | TransAttUNet | arXiv 2022 | - |
| `da_transunet` | DA-TransUNet | arXiv 2023 | - |
| `ds_transunet` | DS-TransUNet | arXiv 2022 | - |
| `uctransnet_full` | UCTransNet (full) | AAAI 2022 | - |
| `uctransnet_enc` | UCTransNet (encoder-only) | AAAI 2022 | - |
| `mobile_u_vit` | Mobile-UViT | - | - |
| `cswin_unet` | CSWin-UNet | - | - |
| `fcbformer` | FCBFormer | MICCAI 2022 | - |
| `pvt_unet` | PVT-UNet | - | - |
| `transnetr` | TransNetR | IEEE Access 2023 | - |
| `polyp_pvt` | Polyp-PVT | MICCAI 2021 | - |
| `cascade` | CASCADE | MICCAI 2023 | - |
| `hsnet` | HSNet | MedIA 2023 | - |
| `ssformer` | SSFormer | MICCAI 2022 | - |
| `ldnet` | LDNet | MICCAI 2022 | - |
| `esfpnet` | ESFPNet | MICCAI 2022 | - |
| `mist` | MIST | IEEE TMI 2023 | - |

## Mamba / SSM (15)

基于 Mamba (Selective State Space Model) 的网络。

| 名称 | 论文 | 发表 |
|---|---|---|
| `mamba_unet` | Mamba-UNet | arXiv 2024 |
| `h_vmunet` | H-vmunet | arXiv 2024 |
| `lightm_unet` | LightM-UNet | arXiv 2024 |
| `swin_umamba` | Swin-UMamba | arXiv 2024 |
| `umamba_bot` | U-Mamba (bottleneck) | arXiv 2024 |
| `umamba_enc` | U-Mamba (encoder) | arXiv 2024 |
| `ultralight_vmunet` | UltraLight VM-UNet | arXiv 2024 |
| `vm_unet` | VM-UNet | arXiv 2024 |
| `vm_unet_v2` | VM-UNet V2 | arXiv 2024 |
| `lkm_unet` | LKM-UNet | arXiv 2024 |
| `log_vmamba` | LoG-VMamba | arXiv 2024 |
| `vmkla_unet` | VMKLA-UNet | arXiv 2024 |
| `ultralbm_unet` | UltraLBM-UNet | arXiv 2024 |
| `nnmamba_2d` | nnMamba (2D) | arXiv 2024 |
| `polyp_mamba` | Polyp-Mamba | arXiv 2024 |
| `hc_mamba` | HC-Mamba | arXiv 2024 |

## SAM (13)

基于 Segment Anything Model 的网络。

| 名称 | 论文 | 发表 |
|---|---|---|
| `sam_b` | SAM ViT-Base | ICCV 2023 |
| `sam_l` | SAM ViT-Large | ICCV 2023 |
| `mobile_sam` | MobileSAM | arXiv 2023 |
| `sam2` | SAM 2 | arXiv 2024 |
| `medsam` | MedSAM | Nature Comms 2024 |
| `samus` | SAMUS | arXiv 2023 |
| `sam_med2d` | SAM-Med2D | arXiv 2023 |
| `sammed2d_wrapper` | SAMMed2D (wrapper) | arXiv 2023 |
| `medical_sam_adapter` | Medical SAM Adapter | arXiv 2023 |
| `samed` | SAMed | arXiv 2023 |
| `auto_sam` | AutoSAM | arXiv 2023 |
| `lite_medsam` | Lite-MedSAM | arXiv 2024 |

## KAN / MLP (4)

| 名称 | 论文 | 发表 |
|---|---|---|
| `ukan` | U-KAN | arXiv 2024 |
| `wav_kan_unet` | Wav-KAN UNet | arXiv 2024 |
| `unext` | UNeXt | MICCAI 2022 |
| `rolling_unet` / `_m` / `_l` / `_s` | Rolling-UNet | arXiv 2024 |

## RWKV (4)

| 名称 | 论文 | 发表 |
|---|---|---|
| `u_rwkv` | U-RWKV | arXiv 2024 |
| `rwkv_unet` | RWKV-UNet | arXiv 2024 |
| `md_rwkv_unet` | MD-RWKV-UNet | arXiv 2024 |
| `rir_zigzag` | RIR-Zigzag | arXiv 2024 |

## Linear Attention (2)

| 名称 | 论文 | 发表 |
|---|---|---|
| `ttt_unet` | TTT-UNet | arXiv 2024 |
| `xlstm_unet_bot` / `xlstm_unet_enc` | xLSTM-UNet | arXiv 2024 |

## 文本引导 (13)

文本引导分割模型，forward 签名为 `(image, text=None)`。

| 名称 | 论文 | 发表 |
|---|---|---|
| `tganet` | TGANet | MICCAI 2022 |
| `lvit` | LViT | IEEE TMI 2023 |
| `languide` | LanGuideMedSeg | MICCAI 2023 |
| `clip_universal` | CLIP-Driven Universal Model | ICCV 2023 |
| `cris` | CRIS | CVPR 2022 |
| `biomedparse` | BiomedParse | Nature Methods 2024 |
| `tpro` | TPRO | ECCV 2024 |
| `salip` | SaLIP | arXiv 2024 |
| `causal_clipseg` | Causal CLIPSeg | arXiv 2024 |
| `medclip_sam` | MedCLIP-SAM | arXiv 2024 |
| `tp_drseg` | TP-DRSeg | arXiv 2024 |
| `cxrclipseg` | CXR-CLIPSeg | arXiv 2024 |
| `medisee` | MediSee (MLLM) | arXiv 2024 |

## YAML 使用示例

```yaml
model:
  num_classes: 9
  img_size: 224
  architecture: transunet
  encoder:
    in_channels: 3
  arch_params: {}

data:
  type: synapse
  img_size: 224
  train_dir: ./data/Synapse/train_npz
  test_dir: ./data/Synapse/test_vol_h5
  train_list: ./data/Synapse/lists/lists_Synapse/train.txt
  test_list: ./data/Synapse/lists/lists_Synapse/test_vol.txt

training:
  epochs: 200
  batch_size: 16
  num_workers: 4
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
    lr: 0.0001
    weight_decay: 0.01
  scheduler:
    name: cosine
    min_lr: 0.000001
```
