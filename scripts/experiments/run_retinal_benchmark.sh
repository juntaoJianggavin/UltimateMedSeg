#!/usr/bin/env bash
# =============================================================================
# 视网膜血管分割 Benchmark / Retinal Vessel Segmentation Benchmark
# =============================================================================
# 数据集 / Datasets: DRIVE (train20/test20), STARE (5折), CHASE_DB1
#
# 设计原则 / Design:
#   - 数据集配置来自 configs/intro_to_datasets/ (数据路径 + 训练超参)
#   - 模型架构通过 --override 覆盖
#
# 用法 / Usage:
#   bash scripts/experiments/run_retinal_benchmark.sh
# =============================================================================
set -e
AMP="--amp"
BASE_OUT="output/retinal_benchmark"
DATASET_CFG="configs/intro_to_datasets"

ARCHS=(
    # ---- 通用 Baseline / General Baselines ----
    attention_unet
    unetpp
    # ---- 视网膜专有 / Retinal-Specific ----
    fr_unet                 # FR-UNet: Full-Resolution vessel seg
    serp_mamba              # SerpMamba: 蛇形扫描 4-dir SS2D
    mamba_vesselnet_pp      # MambaVesselNet++: CNN-Mamba 3-dir scan
)

for ds in drive stare chase_db1; do
    ds_cfg="${DATASET_CFG}/${ds}.yaml"
    [ ! -f "$ds_cfg" ] && echo "SKIP dataset config: $ds_cfg" && continue
    for arch in "${ARCHS[@]}"; do
        echo "=== ${arch} on ${ds} ==="
        python train.py --config "$ds_cfg" --output_dir "${BASE_OUT}/${ds}/${arch}" $AMP --seed 42 \
            --override model.architecture="${arch}" \
            || echo "FAILED: ${arch} on ${ds}"
    done
done
echo "=== Done: ${BASE_OUT}/ ==="
