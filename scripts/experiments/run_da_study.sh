#!/usr/bin/env bash
# =============================================================================
# 域适应训练范式研究 / Domain Adaptation Paradigm Study
# =============================================================================
# 用法 / Usage:
#   bash scripts/experiments/run_da_study.sh
# =============================================================================

set -e
BASE_OUT="output/da_study"
CFG_DIR="configs/training_paradigms/domain_adaptation"

METHODS=(advent dann fda mic hrda tent dpl cbmt)

for method in "${METHODS[@]}"; do
    cfg="${CFG_DIR}/${method}.yaml"
    [ ! -f "$cfg" ] && echo "SKIP: $cfg" && continue
    echo "=== DA: $method ==="
    python train_domain_adaptation.py --config "$cfg" \
        --output_dir "${BASE_OUT}/${method}" || echo "FAILED: $method"
done

echo "=== Done. Results: ${BASE_OUT}/ ==="
