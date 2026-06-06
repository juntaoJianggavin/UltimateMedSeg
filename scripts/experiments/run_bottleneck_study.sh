#!/usr/bin/env bash
# =============================================================================
# Bottleneck 消融研究 / Bottleneck Ablation Study
# =============================================================================
# 3 encoder × 代表性 bottleneck，decoder 固定 bilinear，skip 固定 concat
# 3 encoders × representative bottlenecks, fixed bilinear decoder + concat skip
#
# 用法 / Usage:
#   bash scripts/experiments/run_bottleneck_study.sh
# =============================================================================

set -e
AMP="--amp"
BASE_OUT="output/bottleneck_study"
CFG_DIR="configs/architectures/bottleneck_study/general"

# 代表性 bottleneck
BOTTLENECKS=(
    none            # 无瓶颈（基线）/ No bottleneck (baseline)
    basic           # 两层 Conv-BN-ReLU
    aspp            # ASPP (DeepLabV3, Chen 2018)
    dense_aspp      # DenseASPP (多尺度密集膨胀)
    ppm             # PPM (PSPNet, Zhao 2017)
    transformer     # Transformer bottleneck (自注意力)
    se              # Squeeze-and-Excitation (Hu 2018)
    cbam            # CBAM (Woo 2018)
    dual_attention  # 双注意力 (位置+通道)
)

ENCODERS=(basic resnet50 pvtv2)

for enc in "${ENCODERS[@]}"; do
    for bn in "${BOTTLENECKS[@]}"; do
        cfg="${CFG_DIR}/${enc}_${bn}.yaml"
        [ ! -f "$cfg" ] && echo "SKIP: $cfg" && continue
        name="${enc}_${bn}"
        echo "=== $name ==="
        python train.py --config "$cfg" --output_dir "${BASE_OUT}/${name}" $AMP --seed 42 \
            || echo "FAILED: $name"
    done
done

echo "=== Done. Results: ${BASE_OUT}/ ==="
