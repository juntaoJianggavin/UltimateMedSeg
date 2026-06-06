#!/usr/bin/env bash
# =============================================================================
# 超声分割 Benchmark / Ultrasound Segmentation Benchmark
# =============================================================================
# 数据集 / Datasets: BUSI (5折)
#
# 设计原则 / Design:
#   - 数据集配置来自 configs/intro_to_datasets/ (数据路径 + 训练超参)
#   - 模型架构通过 --override 覆盖
#
# 用法 / Usage:
#   bash scripts/experiments/run_ultrasound_benchmark.sh
# =============================================================================
set -e
AMP="--amp"
BASE_OUT="output/ultrasound_benchmark"
DATASET_CFG="configs/intro_to_datasets"
FOLD=${1:-0}

ARCHS=(
    # ---- 通用 Baseline / General Baselines ----
    attention_unet
    unetpp
    # ---- 超声专有 / Ultrasound-Specific ----
    aau_net                 # AAU-Net: Adaptive Attention (BUSI 2023)
    dcm_net                 # DCM-Net: Dual CNN+Mamba + CBFM
    uu_mamba                # UU-Mamba: Uncertainty-aware
    vim_unet                # ViM-UNet: Vision Mamba encoder
)

ds_cfg="${DATASET_CFG}/busi.yaml"
[ ! -f "$ds_cfg" ] && echo "SKIP dataset config: $ds_cfg" && exit 1

for arch in "${ARCHS[@]}"; do
    echo "=== ${arch} on BUSI (fold=${FOLD}) ==="
    python train.py --config "$ds_cfg" --output_dir "${BASE_OUT}/busi/${arch}_fold${FOLD}" $AMP --seed 42 \
        --override model.architecture="${arch}" training.fold_idx="${FOLD}" \
        || echo "FAILED: ${arch}"
done
echo "=== Done: ${BASE_OUT}/ ==="
