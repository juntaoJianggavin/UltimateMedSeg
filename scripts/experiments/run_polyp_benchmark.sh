#!/usr/bin/env bash
# =============================================================================
# 息肉分割 Benchmark / Polyp Segmentation Benchmark
# =============================================================================
# 数据集 / Datasets: CVC-ClinicDB, Kvasir-SEG (5折)
#
# 设计原则 / Design:
#   - 数据集配置来自 configs/intro_to_datasets/ (数据路径 + 训练超参)
#   - 模型架构通过 --override 覆盖
#
# 用法 / Usage:
#   bash scripts/experiments/run_polyp_benchmark.sh
#   bash scripts/experiments/run_polyp_benchmark.sh 2    # fold=2
# =============================================================================
set -e
AMP="--amp"
BASE_OUT="output/polyp_benchmark"
FOLD=${1:-0}
DATASET_CFG="configs/intro_to_datasets"

ARCHS=(
    # ---- 通用 Baseline / General Baselines ----
    attention_unet          # Attention UNet (Oktay 2018)
    unetpp                  # UNet++ (Zhou 2018)
    # ---- 息肉专有 / Polyp-Specific ----
    sepnet                  # SEPNet: MAP(RFB) + CRC
    ctnet                   # CTNet: SMIM + CIM
    polyper                 # Polyper: Swin-T 双分支 + BGM
    polyp_pvt               # PolypPVT (AAAI 2023)
    cascade                 # CASCADE (MICCAI 2023)
    hsnet                   # HSNet (2023)
    ssformer                # SSFormer (2023)
    ldnet                   # LDNet (2022)
    esfpnet                 # ESFPNet (2023)
    mist                    # MIST (2023)
    fcbformer               # FCBFormer (2022)
    transnetr               # TransNetR (2022)
    pvt_unet                # PVT-Former (2022)
)

for ds in kvasir_seg cvc_clinicdb; do
    ds_cfg="${DATASET_CFG}/${ds}.yaml"
    [ ! -f "$ds_cfg" ] && echo "SKIP dataset config: $ds_cfg" && continue
    for arch in "${ARCHS[@]}"; do
        echo "=== ${arch} on ${ds} (fold=${FOLD}) ==="
        python train.py --config "$ds_cfg" --output_dir "${BASE_OUT}/${ds}/${arch}_fold${FOLD}" $AMP --seed 42 \
            --override model.architecture="${arch}" training.fold_idx="${FOLD}" \
            || echo "FAILED: ${arch} on ${ds}"
    done
done
echo "=== Done: ${BASE_OUT}/ ==="
