# 完整网络架构

[English](networks.md)

本项目支持 132 个完整网络架构（136 注册，合并尺寸变体；119 个标准 + 13 个文本引导），通过 `architecture` 字段直接使用。

## CNN (36)

卷积神经网络系列，经典 UNet 及其变体。

| 名称 | 论文 | 发表 | GitHub | YAML |
|---|---|---|---|---|
| `attention_unet` | Attention U-Net | MIDL 2018 | [ozan-oktay/Attention-Gating-Network](https://github.com/ozan-oktay/Attention-Gating-Network) | [attention_unet_basic.yaml](../../configs/architectures/combinations/general/attention_unet_basic.yaml) |
| `unetpp` | UNet++ | DLMIA 2018 | [MrGiovanni/UNetPlusPlus](https://github.com/MrGiovanni/UNetPlusPlus) | [unetpp.yaml](../../configs/architectures/networks/general/unetpp.yaml) |
| `r2unet` | R2U-Net | IEEE Access 2018 | - | [r2unet.yaml](../../configs/architectures/networks/general/r2unet.yaml) |
| `multiresunet` | MultiResUNet | Neural Networks 2020 | - | [multiresunet.yaml](../../configs/architectures/networks/general/multiresunet.yaml) |
| `resunet_a` | ResUNet-a | ISPRS 2020 | - | [resunet_a.yaml](../../configs/architectures/networks/general/resunet_a.yaml) |
| `resunetpp` | ResUNet++ | ISM 2019 | - | [resunetpp.yaml](../../configs/architectures/networks/general/resunetpp.yaml) |
| `unet3plus` | UNet 3+ | ICASSP 2020 | [ZJUGiveLab/UNet-Version](https://github.com/ZJUGiveLab/UNet-Version) | [unet3plus.yaml](../../configs/architectures/networks/general/unet3plus.yaml) |
| `denseunet` | DenseUNet | - | - | [denseunet.yaml](../../configs/architectures/networks/general/denseunet.yaml) |
| `scseunet` | scSE-UNet (Squeeze-Excitation) | MICCAI 2018 | - | [scseunet.yaml](../../configs/architectures/networks/general/scseunet.yaml) |
| `sa_unet` | SA-UNet (Spatial Attention) | IEEE TIM 2021 | - | [sa_unet.yaml](../../configs/architectures/networks/general/sa_unet.yaml) |
| `kiunet` | KiU-Net | MICCAI 2020 | [jeya-maria-jose/KiU-Net-pytorch](https://github.com/jeya-maria-jose/KiU-Net-pytorch) | [kiunet.yaml](../../configs/architectures/networks/general/kiunet.yaml) |
| `pan` | PAN (Pyramid Attention Network) | BMVC 2018 | - | [pan.yaml](../../configs/architectures/networks/general/pan.yaml) |
| `linknet` | LinkNet | VCIP 2017 | - | [linknet.yaml](../../configs/architectures/networks/general/linknet.yaml) |
| `pspnet` | PSPNet | CVPR 2017 | - | [pspnet.yaml](../../configs/architectures/networks/general/pspnet.yaml) |
| `fr_unet` | FR-UNet (Full-Resolution) | IEEE TMI 2022 | - | [fr_unet.yaml](../../configs/architectures/networks/general/fr_unet.yaml) |
| `dcsaunet` | DCSAU-Net | Computers in Biology and Medicine 2023 | [xq141839/DCSAU-Net](https://github.com/xq141839/DCSAU-Net) | [dcsaunet.yaml](../../configs/architectures/networks/general/dcsaunet.yaml) |
| `cfanet` | CFA-Net | Computers in Biology and Medicine 2024 | [ZhangJD-ong/CFA-Net](https://github.com/ZhangJD-ong/CFA-Net) | [cfanet.yaml](../../configs/architectures/networks/general/cfanet.yaml) |
| `mednext` | MedNeXt | MICCAI 2023 | [MIC-DKFZ/MedNeXt](https://github.com/MIC-DKFZ/MedNeXt) | [mednext_emcad.yaml](../../configs/architectures/combinations/general/mednext_emcad.yaml), [mednext_cascade_full.yaml](../../configs/architectures/combinations/general/mednext_cascade_full.yaml), [mednext_cfm.yaml](../../configs/architectures/combinations/general/mednext_cfm.yaml) |
| `nnunet_2d` | nnU-Net (2D) | Nature Methods 2021 | [MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet) | [nnunet_2d.yaml](../../configs/architectures/networks/general/nnunet_2d.yaml) |
| `acc_unet` | ACC-UNet | MICCAI 2023 | - | [acc_unet.yaml](../../configs/architectures/networks/general/acc_unet.yaml) |
| `cmunext` | CMUNeXt | arXiv 2023 | - | [cmunext.yaml](../../configs/architectures/networks/general/cmunext.yaml) |
| `mew_unet` | MEW-UNet | arXiv 2024 | - | [mew_unet.yaml](../../configs/architectures/networks/general/mew_unet.yaml) |
| `lv_unet` | LV-UNet (Lightweight) | - | - | [lv_unet.yaml](../../configs/architectures/networks/general/lv_unet.yaml) |
| `ege_unet` | EGE-UNet | arXiv 2023 | [JCruan519/EGE-UNet](https://github.com/JCruan519/EGE-UNet) | [ege_unet.yaml](../../configs/architectures/networks/general/ege_unet.yaml) |
| `malunet` | MALUNet | arXiv 2022 | - | [malunet.yaml](../../configs/architectures/networks/general/malunet.yaml) |
| `lite_unet` | Lite-UNet | - | - | [lite_unet.yaml](../../configs/architectures/networks/general/lite_unet.yaml) |
| `mk_unet` | MK-UNet | - | - | [mk_unet.yaml](../../configs/architectures/networks/general/mk_unet.yaml) |
| `u_lite` | U-Lite | arXiv 2022 | - | [u_lite.yaml](../../configs/architectures/networks/general/u_lite.yaml) |
| `aau_net` | AAU-Net | IEEE JBHI 2023 | [CGPxy/AAU-net](https://github.com/CGPxy/AAU-net) | [aau_net.yaml](../../configs/architectures/networks/general/aau_net.yaml) |
| `cmu_net` | CMU-Net | Bioinformatics 2024 | - | [cmu_net.yaml](../../configs/architectures/networks/general/cmu_net.yaml) |
| `dscnet` | DSCNet | MICCAI 2023 | - | [dscnet.yaml](../../configs/architectures/networks/general/dscnet.yaml) |
| `dconnnet` | DconnNet | MICCAI 2023 | - | [dconnnet.yaml](../../configs/architectures/networks/general/dconnnet.yaml) |
| `stu_net` | STU-Net | arXiv 2023 | - | [stu_net.yaml](../../configs/architectures/networks/general/stu_net.yaml) |
| `polyper` | Polyper | - | - | [polyper.yaml](../../configs/architectures/networks/general/polyper.yaml) |
| `hovernet_lite` | HoverNet Lite | - | - | [hovernet_lite.yaml](../../configs/architectures/networks/general/hovernet_lite.yaml) |
| `hrnet_w18` / `hrnet_w32` | HRNet（高分辨率网络） | CVPR 2019 | - | [hrnet_w18.yaml](../../configs/architectures/networks/general/hrnet_w18.yaml), [hrnet_w32.yaml](../../configs/architectures/networks/general/hrnet_w32.yaml) |

## Transformer (37)

基于 Transformer 的分割网络。

| 名称 | 论文 | 发表 | GitHub | YAML |
|---|---|---|---|---|
| `segformer_b0` ~ `segformer_b5` | SegFormer (MiT-B0\~B5) | NeurIPS 2021 | [NVlabs/SegFormer](https://github.com/NVlabs/SegFormer) | [segformer_b0.yaml](../../configs/architectures/networks/general/segformer_b0.yaml) |
| `transunet` | TransUNet | arXiv 2021 | [Beckschen/TransUNet](https://github.com/Beckschen/TransUNet) | [transunet_cascade_full.yaml](../../configs/architectures/combinations/general/transunet_cascade_full.yaml) |
| `swinunet` | Swin-UNet | ECCV 2022 | [HuCaoFighting/Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet) | [swinunet_segformer.yaml](../../configs/architectures/combinations/general/swinunet_segformer.yaml) |
| `medt` | MedT (Medical Transformer) | MICCAI 2021 | [jeya-maria-jose/Medical-Transformer](https://github.com/jeya-maria-jose/Medical-Transformer) | [medt.yaml](../../configs/architectures/networks/general/medt.yaml) |
| `daeformer` | DAEFormer | ICLR 2023 | - | [daeformer_emcad.yaml](../../configs/architectures/combinations/general/daeformer_emcad.yaml) |
| `missformer` | MISSFormer | IEEE TMI 2022 | - | [missformer.yaml](../../configs/architectures/networks/general/missformer.yaml) |
| `h2former` | H2Former | IEEE TMI 2023 | - | [h2former.yaml](../../configs/architectures/networks/general/h2former.yaml) |
| `hiformer` | HiFormer | WACV 2023 | - | [hiformer_cascade.yaml](../../configs/architectures/combinations/general/hiformer_cascade.yaml) |
| `mctrans` | MCTrans | MICCAI 2021 | - | [mctrans_cascade_emcad.yaml](../../configs/architectures/combinations/general/mctrans_cascade_emcad.yaml) |
| `mtunet` | MT-UNet | MICCAI 2022 | - | [mtunet.yaml](../../configs/architectures/networks/general/mtunet.yaml) |
| `scaleformer` | ScaleFormer | MICCAI 2022 | - | [scaleformer_cascade_full.yaml](../../configs/architectures/combinations/general/scaleformer_cascade_full.yaml) |
| `fatnet` | FAT-Net | IEEE TMI 2022 | - | [fatnet.yaml](../../configs/architectures/networks/general/fatnet.yaml) |
| `nnformer_2d` | nnFormer (2D) | MICCAI 2022 | - | [nnformer_2d.yaml](../../configs/architectures/networks/general/nnformer_2d.yaml) |
| `transfuse` | TransFuse | MICCAI 2021 | - | [transfuse.yaml](../../configs/architectures/networks/general/transfuse.yaml) |
| `levit_unet` | LeViT-UNet | ML4H 2022 | - | [levit_unet.yaml](../../configs/architectures/networks/general/levit_unet.yaml) |
| `transatt_unet` | TransAttUNet | arXiv 2022 | - | [transatt_unet.yaml](../../configs/architectures/networks/general/transatt_unet.yaml) |
| `da_transunet` | DA-TransUNet | arXiv 2023 | - | [da_transunet.yaml](../../configs/architectures/networks/acdc/da_transunet.yaml) |
| `ds_transunet` | DS-TransUNet | arXiv 2022 | - | [ds_transunet.yaml](../../configs/architectures/networks/acdc/ds_transunet.yaml) |
| `uctransnet_full` / `uctransnet_enc` | UCTransNet | AAAI 2022 | - | [uctransnet.yaml](../../configs/architectures/combinations/general/uctransnet.yaml) |
| `mobile_u_vit` | Mobile-UViT | - | - | [mobile_u_vit.yaml](../../configs/architectures/networks/general/mobile_u_vit.yaml) |
| `cswin_unet` | CSWin-UNet | - | - | [cswin_unet.yaml](../../configs/architectures/networks/general/cswin_unet.yaml) |
| `fcbformer` | FCBFormer | MICCAI 2022 | - | [fcbformer.yaml](../../configs/architectures/networks/general/fcbformer.yaml) |
| `pvt_unet` | PVT-UNet | - | - | [pvtv2_emcad.yaml](../../configs/architectures/combinations/general/pvtv2_emcad.yaml), [pvtv2_cascade_full.yaml](../../configs/architectures/combinations/general/pvtv2_cascade_full.yaml), [pvtv2_cfm.yaml](../../configs/architectures/combinations/general/pvtv2_cfm.yaml) |
| `pvtb2_emcad` | PVTb2-EMCAD | - | - | [pvtb2_emcad.yaml](../../configs/architectures/networks/general/pvtb2_emcad.yaml) |
| `transnetr` | TransNetR | IEEE Access 2023 | - | [transnetr.yaml](../../configs/architectures/networks/general/transnetr.yaml) |
| `polyp_pvt` | Polyp-PVT | MICCAI 2021 | - | [polyp_pvt.yaml](../../configs/architectures/networks/general/polyp_pvt.yaml) |
| `cascade` | CASCADE | MICCAI 2023 | - | [cascade_resnet34.yaml](../../configs/architectures/combinations/general/cascade_resnet34.yaml) |
| `hsnet` | HSNet | MedIA 2023 | - | [hsnet.yaml](../../configs/architectures/networks/general/hsnet.yaml) |
| `ssformer` | SSFormer | MICCAI 2022 | - | [ssformer.yaml](../../configs/architectures/networks/general/ssformer.yaml) |
| `ldnet` | LDNet | MICCAI 2022 | - | [ldnet.yaml](../../configs/architectures/networks/general/ldnet.yaml) |
| `esfpnet` | ESFPNet | MICCAI 2022 | - | [esfpnet.yaml](../../configs/architectures/networks/general/esfpnet.yaml) |
| `mist` | MIST | IEEE TMI 2023 | - | [mist.yaml](../../configs/architectures/networks/general/mist.yaml) |
| `double_unet` | DoubleU-Net | CBMS 2020 | - | [double_unet.yaml](../../configs/architectures/networks/general/double_unet.yaml) |
| `sepnet` | SEPNet | - | - | [sepnet.yaml](../../configs/architectures/networks/general/sepnet.yaml) |
| `ctnet` | CTNet | - | - | [ctnet.yaml](../../configs/architectures/networks/general/ctnet.yaml) |
| `nulite` | NuLite | - | - | [nulite.yaml](../../configs/architectures/networks/general/nulite.yaml) |

## Mamba / SSM (24)

基于 Mamba (Selective State Space Model) 的网络。

| 名称 | 论文 | 发表 | YAML |
|---|---|---|---|
| `mamba_unet` | Mamba-UNet | arXiv 2024 | [mamba_unet.yaml](../../configs/architectures/networks/general/mamba_unet.yaml) |
| `h_vmunet` | H-vmunet | arXiv 2024 | [h_vmunet.yaml](../../configs/architectures/networks/general/h_vmunet.yaml) |
| `lightm_unet` | LightM-UNet | arXiv 2024 | [lightm_unet.yaml](../../configs/architectures/networks/general/lightm_unet.yaml) |
| `swin_umamba` | Swin-UMamba | arXiv 2024 | [swin_umamba.yaml](../../configs/architectures/networks/general/swin_umamba.yaml) |
| `umamba_bot` / `umamba_enc` | U-Mamba | arXiv 2024 | [umamba_cascade_full.yaml](../../configs/architectures/combinations/general/umamba_cascade_full.yaml), [umamba_cfm.yaml](../../configs/architectures/combinations/general/umamba_cfm.yaml), [umamba_emcad.yaml](../../configs/architectures/combinations/general/umamba_emcad.yaml) |
| `ultralight_vmunet` | UltraLight VM-UNet | arXiv 2024 | [ultralight_vmunet.yaml](../../configs/architectures/networks/general/ultralight_vmunet.yaml) |
| `vm_unet` | VM-UNet | arXiv 2024 | [vm_unet.yaml](../../configs/architectures/networks/general/vm_unet.yaml) |
| `vm_unet_v2` | VM-UNet V2 | arXiv 2024 | [vm_unet_v2.yaml](../../configs/architectures/networks/general/vm_unet_v2.yaml) |
| `lkm_unet` | LKM-UNet | arXiv 2024 | [lkm_unet.yaml](../../configs/architectures/networks/general/lkm_unet.yaml) |
| `log_vmamba` | LoG-VMamba | arXiv 2024 | [log_vmamba.yaml](../../configs/architectures/networks/general/log_vmamba.yaml) |
| `vmkla_unet` | VMKLA-UNet | arXiv 2024 | [vmkla_unet.yaml](../../configs/architectures/networks/general/vmkla_unet.yaml) |
| `ultralbm_unet` | UltraLBM-UNet | arXiv 2024 | [ultralbm_unet.yaml](../../configs/architectures/networks/general/ultralbm_unet.yaml) |
| `nnmamba_2d` | nnMamba (2D) | arXiv 2024 | [nnmamba_2d.yaml](../../configs/architectures/networks/general/nnmamba_2d.yaml) |
| `polyp_mamba` | Polyp-Mamba | arXiv 2024 | [polyp_mamba.yaml](../../configs/architectures/networks/general/polyp_mamba.yaml) |
| `hc_mamba` | HC-Mamba | arXiv 2024 | [hc_mamba.yaml](../../configs/architectures/networks/general/hc_mamba.yaml) |
| `ac_mambaseg` | AC-MambaSeg | arXiv 2024 | [ac_mambaseg.yaml](../../configs/architectures/networks/general/ac_mambaseg.yaml) |
| `dcm_net` | DCM-Net | arXiv 2024 | [dcm_net.yaml](../../configs/architectures/networks/general/dcm_net.yaml) |
| `dermomamba` | DermoMamba | arXiv 2024 | [dermomamba.yaml](../../configs/architectures/networks/general/dermomamba.yaml) |
| `mucm_net` | MUCM-Net | arXiv 2024 | [mucm_net.yaml](../../configs/architectures/networks/general/mucm_net.yaml) |
| `serp_mamba` | Serp-Mamba | arXiv 2024 | [serp_mamba.yaml](../../configs/architectures/networks/general/serp_mamba.yaml) |
| `skin_mamba` | SkinMamba | arXiv 2024 | [skin_mamba.yaml](../../configs/architectures/networks/general/skin_mamba.yaml) |
| `mamba_vesselnet_pp` | Mamba-VesselNet++ | arXiv 2024 | [mamba_vesselnet_pp.yaml](../../configs/architectures/networks/general/mamba_vesselnet_pp.yaml) |
| `vim_unet` | ViM-UNet | arXiv 2024 | [vim_unet.yaml](../../configs/architectures/networks/general/vim_unet.yaml) |
| `uu_mamba` | UU-Mamba | arXiv 2024 | [uu_mamba.yaml](../../configs/architectures/networks/general/uu_mamba.yaml) |

## SAM (10)

基于 Segment Anything Model 的网络。

| 名称 | 论文 | 发表 | YAML |
|---|---|---|---|
| `sam_b` / `sam_l` | SAM ViT (Base/Large) | ICCV 2023 | [sam_vit_cascade_full.yaml](../../configs/architectures/combinations/general/sam_vit_cascade_full.yaml), [sam_vit_cfm.yaml](../../configs/architectures/combinations/general/sam_vit_cfm.yaml), [sam_vit_emcad.yaml](../../configs/architectures/combinations/general/sam_vit_emcad.yaml) |
| `mobile_sam` | MobileSAM | arXiv 2023 | [mobile_sam.yaml](../../configs/architectures/networks/general/mobile_sam.yaml) |
| `sam2` | SAM 2 | arXiv 2024 | [sam2.yaml](../../configs/architectures/networks/general/sam2.yaml) |
| `medsam` | MedSAM | Nature Comms 2024 | [medsam_encoder_emcad.yaml](../../configs/architectures/combinations/general/medsam_encoder_emcad.yaml) |
| `samus` | SAMUS | arXiv 2023 | [samus.yaml](../../configs/architectures/networks/general/samus.yaml) |
| `sam_med2d` / `sammed2d_wrapper` | SAM-Med2D | arXiv 2023 | [sam_med2d.yaml](../../configs/architectures/networks/general/sam_med2d.yaml), [qata_covid19_sammed2d.yaml](../../configs/architectures/foundation/sam/qata_covid19_sammed2d.yaml) |
| `medical_sam_adapter` | Medical SAM Adapter | arXiv 2023 | [medical_sam_adapter.yaml](../../configs/architectures/networks/general/medical_sam_adapter.yaml) |
| `samed` | SAMed | arXiv 2023 | [samed.yaml](../../configs/architectures/networks/general/samed.yaml) |
| `auto_sam` | AutoSAM | arXiv 2023 | [auto_sam.yaml](../../configs/architectures/networks/general/auto_sam.yaml) |
| `lite_medsam` | Lite-MedSAM | arXiv 2024 | [qata_covid19_lite_medsam.yaml](../../configs/architectures/foundation/sam/qata_covid19_lite_medsam.yaml) |

## KAN / MLP (4)

| 名称 | 论文 | 发表 | YAML |
|---|---|---|---|
| `ukan` | U-KAN | arXiv 2024 | [ukan.yaml](../../configs/architectures/networks/general/ukan.yaml) |
| `wav_kan_unet` | Wav-KAN UNet | arXiv 2024 | [wav_kan_unet.yaml](../../configs/architectures/networks/general/wav_kan_unet.yaml) |
| `unext` | UNeXt | MICCAI 2022 | [unext.yaml](../../configs/architectures/networks/general/unext.yaml) |
| `rolling_unet` | Rolling-UNet (S/M/L) | arXiv 2024 | [rolling_unet.yaml](../../configs/architectures/networks/general/rolling_unet.yaml), [rolling_unet_s.yaml](../../configs/architectures/networks/general/rolling_unet_s.yaml), [rolling_unet_m.yaml](../../configs/architectures/networks/general/rolling_unet_m.yaml), [rolling_unet_l.yaml](../../configs/architectures/networks/general/rolling_unet_l.yaml) |

## RWKV (5)

| 名称 | 论文 | 发表 | YAML |
|---|---|---|---|
| `u_rwkv` | U-RWKV（MICCAI 2025，方向自适应） | MICCAI 2025 | [u_rwkv.yaml](../../configs/architectures/networks/general/u_rwkv.yaml), [rwkv_unet.yaml](../../configs/architectures/combinations/general/rwkv_unet.yaml), [rwkv_unet_small.yaml](../../configs/architectures/combinations/general/rwkv_unet_small.yaml), [rwkv_unet_tiny.yaml](../../configs/architectures/combinations/general/rwkv_unet_tiny.yaml) |
| `u_rwkv_tip` | U-RWKV（TIP 2026，OmniShift + 卷积后 RWKV） | IEEE TIP 2026 | [u_rwkv_tip.yaml](../../configs/architectures/networks/general/u_rwkv_tip.yaml) |
| `rwkv_unet` | RWKV-UNet | arXiv 2024 | [rwkv_emcad.yaml](../../configs/architectures/combinations/general/rwkv_emcad.yaml), [rwkv_cascade_full.yaml](../../configs/architectures/combinations/general/rwkv_cascade_full.yaml), [rwkv_cfm.yaml](../../configs/architectures/combinations/general/rwkv_cfm.yaml) |
| `md_rwkv_unet` | MD-RWKV-UNet | arXiv 2024 | [md_rwkv_unet.yaml](../../configs/architectures/networks/general/md_rwkv_unet.yaml) |
| `rir_zigzag` | RIR-Zigzag | arXiv 2024 | [rir_zigzag.yaml](../../configs/architectures/combinations/general/rir_zigzag.yaml) |

## Linear Attention (3)

| 名称 | 论文 | 发表 | YAML |
|---|---|---|---|
| `ttt_unet` | TTT-UNet | arXiv 2024 | [ttt_unet.yaml](../../configs/architectures/networks/general/ttt_unet.yaml) |
| `xlstm_unet_bot` / `xlstm_unet_enc` | xLSTM-UNet | arXiv 2024 | [xlstm_unet_bot.yaml](../../configs/architectures/networks/general/xlstm_unet_bot.yaml), [xlstm_unet_enc.yaml](../../configs/architectures/networks/general/xlstm_unet_enc.yaml) |
| `u_vixlstm` | U-VixLSTM | arXiv 2024 | [u_vixlstm.yaml](../../configs/architectures/networks/general/u_vixlstm.yaml) |

## 文本引导 (13)

文本引导分割模型，forward 签名为 `(image, text=None)`。

| 名称 | 论文 | 发表 | YAML |
|---|---|---|---|
| `tganet` | TGANet | MICCAI 2022 | [synapse_clip.yaml](../../configs/training_paradigms/text_guided/synapse_clip.yaml) |
| `lvit` | LViT | IEEE TMI 2023 | [mosmed_plus_lvit.yaml](../../configs/training_paradigms/text_guided/mosmed_plus_lvit.yaml), [qata_covid19_lvit.yaml](../../configs/training_paradigms/text_guided/qata_covid19_lvit.yaml) |
| `languide` | LanGuideMedSeg | MICCAI 2023 | [mosmed_plus_languide.yaml](../../configs/training_paradigms/text_guided/mosmed_plus_languide.yaml), [qata_covid19_languide.yaml](../../configs/training_paradigms/text_guided/qata_covid19_languide.yaml) |
| `clip_universal` | CLIP-Driven Universal Model | ICCV 2023 | [synapse_clip_large.yaml](../../configs/training_paradigms/text_guided/synapse_clip_large.yaml) |
| `cris` | CRIS | CVPR 2022 | [synapse_clip.yaml](../../configs/training_paradigms/text_guided/synapse_clip.yaml) |
| `biomedparse` | BiomedParse | Nature Methods 2024 | - |
| `tpro` | TPRO | ECCV 2024 | - |
| `salip` | SaLIP | arXiv 2024 | - |
| `causal_clipseg` | Causal CLIPSeg | arXiv 2024 | - |
| `medclip_sam` | MedCLIP-SAM | arXiv 2024 | [synapse_grounding_dino_medsam.yaml](../../configs/training_paradigms/text_guided/synapse_grounding_dino_medsam.yaml) |
| `tp_drseg` | TP-DRSeg | arXiv 2024 | - |
| `cxrclipseg` | CXR-CLIPSeg | arXiv 2024 | - |
| `medisee` | MediSee (MLLM) | arXiv 2024 | [mosmed_plus_medisee.yaml](../../configs/training_paradigms/text_guided/mosmed_plus_medisee.yaml), [qata_covid19_medisee.yaml](../../configs/training_paradigms/text_guided/qata_covid19_medisee.yaml) |

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
