#!/usr/bin/env bash
# =============================================================================
# 知识蒸馏研究 / Knowledge Distillation Study
# =============================================================================
# Teacher: TransUNet → Student: U-Lite，对比不同蒸馏方法
# Teacher: TransUNet → Student: U-Lite, compare distillation methods
#
# 用法 / Usage:
#   bash scripts/experiments/run_kd_study.sh
# =============================================================================

set -e
BASE_OUT="output/kd_study"
TEACHER="configs/architectures/networks/general/transunet.yaml"
STUDENT="configs/architectures/networks/general/u_lite.yaml"

METHODS=(vanilla_kd dkd cwd mgd dist at rkd)

for method in "${METHODS[@]}"; do
    echo "=== KD: $method ==="
    python train_distillation.py \
        --teacher_config "$TEACHER" \
        --student_config "$STUDENT" \
        --teacher_ckpt output/sota_benchmark/transunet/best_model.pth \
        --distillation_type "$method" \
        --output_dir "${BASE_OUT}/${method}" \
        --seed 42 || echo "FAILED: $method"
done

echo "=== Done. Results: ${BASE_OUT}/ ==="
