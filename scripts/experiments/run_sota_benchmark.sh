#!/usr/bin/env bash
# =============================================================================
# 通用 SOTA 架构对比 / General SOTA Architecture Benchmark
# =============================================================================
# 在 BUSI / CVC-ClinicDB / GlaS / Kvasir-SEG 上跑多个 SOTA 架构
# Run multiple SOTA architectures on BUSI / CVC-ClinicDB / GlaS / Kvasir-SEG
#
# 设计原则 / Design:
#   - 数据集配置来自 configs/intro_to_datasets/ (数据路径 + 训练超参)
#   - 模型架构通过 --override 覆盖
#
# 用法 / Usage:
#   bash scripts/experiments/run_sota_benchmark.sh
#   bash scripts/experiments/run_sota_benchmark.sh busi 0
# =============================================================================

set -e

DATASET=${1:-all}
FOLD=${2:-0}
AMP="--amp"
BASE_OUT="output/sota_benchmark"
DATASET_CFG="configs/intro_to_datasets"

# 必跑的 baseline 架构 / Required baseline architectures
ARCHS=(
    transunet swinunet vm_unet rwkv_unet rir_zigzag
    rolling_unet ukan mobile_u_vit attention_unet unetpp
)

if [ "$DATASET" = "all" ]; then
    datasets=(busi cvc_clinicdb glas kvasir_seg isic2018)
else
    datasets=($DATASET)
fi

for ds in "${datasets[@]}"; do
    ds_cfg="${DATASET_CFG}/${ds}.yaml"
    [ ! -f "$ds_cfg" ] && echo "SKIP dataset config: $ds_cfg" && continue
    for arch in "${ARCHS[@]}"; do
        out_dir="${BASE_OUT}/${ds}/${arch}_fold${FOLD}"
        echo "=== ${arch} on ${ds} (fold=${FOLD}) ==="
        python train.py \
            --config "$ds_cfg" \
            --output_dir "$out_dir" \
            $AMP \
            --seed 42 \
            --override model.architecture="${arch}" training.fold_idx="${FOLD}" \
            || echo "FAILED: ${arch} on ${ds}"
    done
done

echo ""
echo "=== 全部完成 / All done ==="
echo "结果在 / Results at: ${BASE_OUT}/"
