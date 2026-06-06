#!/usr/bin/env bash
# =============================================================================
# 皮肤癌轻量化分割 / Lightweight Skin Cancer Segmentation
# =============================================================================
# 在 ISIC 2017 / ISIC 2018 上训练，PH2 上外部验证
# Train on ISIC 2017/2018, external validation on PH2
#
# 设计原则 / Design:
#   - 数据集配置来自 configs/intro_to_datasets/ (数据路径 + 训练超参)
#   - 模型架构通过 --override 覆盖
#
# 用法 / Usage:
#   bash scripts/experiments/run_lightweight_skin.sh
#   bash scripts/experiments/run_lightweight_skin.sh isic2018   # 只跑一个数据集
# =============================================================================

set -e
AMP="--amp"
BASE_OUT="output/lightweight_skin"
DATASET_CFG="configs/intro_to_datasets"

# 轻量级网络 baseline
ARCHS=(
    ege_unet            # ~50K params, MICCAI 2023 W
    lite_unet           # ~60K params
    u_lite              # ~60K params
    malunet             # ~170K params, BIBM 2022
    lv_unet             # ~400K params, BIBM 2024
    ultralight_vmunet   # ~50K params, 2024
    ultralbm_unet       # ~50K params, 2024
    mk_unet             # ~200K params, ICCV 2025
)

DATASET=${1:-all}

run_dataset() {
    local ds=$1
    local ds_cfg="${DATASET_CFG}/${ds}.yaml"
    [ ! -f "$ds_cfg" ] && echo "SKIP dataset config: $ds_cfg" && return
    for arch in "${ARCHS[@]}"; do
        name="${arch}_${ds}"
        echo "=== $name ==="
        python train.py --config "$ds_cfg" --output_dir "${BASE_OUT}/${name}" $AMP --seed 42 \
            --override model.architecture="${arch}" \
            || echo "FAILED: $name"
    done
}

if [ "$DATASET" = "all" ] || [ "$DATASET" = "isic2017" ]; then
    echo "========== ISIC 2017 =========="
    run_dataset isic2017
fi

if [ "$DATASET" = "all" ] || [ "$DATASET" = "isic2018" ]; then
    echo "========== ISIC 2018 =========="
    run_dataset isic2018
fi

echo ""
echo "=== 完成 / Done ==="
echo "训练结果 / Results: ${BASE_OUT}/"
echo ""
echo "PH2 外部验证请在训练完成后运行 / Run PH2 external validation after training:"
echo "  python test.py --config configs/intro_to_datasets/ph2.yaml --checkpoint <best_model.pth>"
