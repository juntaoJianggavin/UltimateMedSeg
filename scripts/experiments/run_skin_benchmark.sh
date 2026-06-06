#!/usr/bin/env bash
# =============================================================================
# 皮肤分割 Benchmark / Skin Segmentation Benchmark
# =============================================================================
# 训练 / Train: ISIC 2017, ISIC 2018
# 外部验证 / External validation: PH2
#
# 设计原则 / Design:
#   - 数据集配置来自 configs/intro_to_datasets/ (数据路径 + 训练超参)
#   - 模型架构通过 --override 覆盖
#
# 用法 / Usage:
#   bash scripts/experiments/run_skin_benchmark.sh
# =============================================================================
set -e
AMP="--amp"
BASE_OUT="output/skin_benchmark"
DATASET_CFG="configs/intro_to_datasets"

ARCHS=(
    # ---- 通用 Baseline / General Baselines ----
    attention_unet
    unetpp
    # ---- 皮肤专有（轻量级）/ Skin-Specific (Lightweight) ----
    ege_unet                # ~50K, MICCAI 2023 W
    lite_unet               # ~60K
    u_lite                  # ~60K
    malunet                 # ~170K, BIBM 2022
    lv_unet                 # ~400K, BIBM 2024
    ultralight_vmunet       # ~50K, 2024
    ultralbm_unet           # ~50K, 2024
    mk_unet                 # ~200K, ICCV 2025
    # ---- 皮肤专有（Mamba）/ Skin-Specific (Mamba) ----
    mucm_net                # MUCM-Net: UCMBlock
    ac_mambaseg             # AC-MambaSeg: Adaptive Conv + Mamba
    skin_mamba              # SkinMamba: Cross-scale + FFT
    dermomamba              # DermoMamba: Cross-scale + PCA + SweepMamba
)

for ds in isic2017 isic2018; do
    ds_cfg="${DATASET_CFG}/${ds}.yaml"
    [ ! -f "$ds_cfg" ] && echo "SKIP dataset config: $ds_cfg" && continue
    for arch in "${ARCHS[@]}"; do
        echo "=== ${arch} on ${ds} ==="
        python train.py --config "$ds_cfg" --output_dir "${BASE_OUT}/${ds}/${arch}" $AMP --seed 42 \
            --override model.architecture="${arch}" \
            || echo "FAILED: ${arch} on ${ds}"
    done
done

echo ""
echo "=== 完成 / Done ==="
echo "PH2 外部验证 / PH2 external validation:"
echo "  python test.py --config configs/intro_to_datasets/ph2.yaml --checkpoint <best_model.pth>"
