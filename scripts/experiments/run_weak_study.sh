#!/usr/bin/env bash
# =============================================================================
# 弱监督训练范式研究 / Weakly Supervised Paradigm Study
# =============================================================================
# 用法 / Usage:
#   bash scripts/experiments/run_weak_study.sh
# =============================================================================

set -e
BASE_OUT="output/weak_study"
CFG_DIR="configs/training_paradigms/weak_supervision"

METHODS=(box_supervised cam point gated_crf tree_energy eps)

for method in "${METHODS[@]}"; do
    cfg="${CFG_DIR}/${method}.yaml"
    [ ! -f "$cfg" ] && echo "SKIP: $cfg" && continue
    echo "=== Weak: $method ==="
    python train_weakly_supervised.py --config "$cfg" \
        --supervision_type "${method}" \
        --output_dir "${BASE_OUT}/${method}" \
        --seed 42 || echo "FAILED: $method"
done

echo "=== Done. Results: ${BASE_OUT}/ ==="
