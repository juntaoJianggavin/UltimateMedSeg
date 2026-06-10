#!/usr/bin/env bash
# =============================================================================
# 半监督训练范式研究 / Semi-Supervised Training Paradigm Study
# =============================================================================
# 在 BUSI 上用不同标注比例（10%/20%/50%）对比半监督方法
# Compare semi-supervised methods on BUSI with different labeling ratios
#
# 用法 / Usage:
#   bash scripts/experiments/run_semi_study.sh
# =============================================================================

set -e
AMP="--amp"
BASE_OUT="output/semi_study"
CFG_DIR="configs/training_paradigms/semi_supervision"

METHODS=(
    mean_teacher    # NeurIPS 2017 — EMA 教师基线
    cps             # CVPR 2021 — 双模型互伪标签
    unimatch        # CVPR 2023 — 双强增强+特征扰动
    fixmatch        # NeurIPS 2020 — 强弱增强+阈值伪标签
    corrmatch       # CVPR 2024 — 相关性匹配传播
)

for method in "${METHODS[@]}"; do
    cfg="${CFG_DIR}/${method}.yaml"
    [ ! -f "$cfg" ] && echo "SKIP: $cfg" && continue
    echo "=== Semi: $method ==="
    python semi_train.py --config "$cfg" \
        --output_dir "${BASE_OUT}/${method}" \
        --seed 42 || echo "FAILED: $method"
done

echo "=== Done. Results: ${BASE_OUT}/ ==="
