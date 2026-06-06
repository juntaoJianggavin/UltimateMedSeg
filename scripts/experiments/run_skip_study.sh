#!/usr/bin/env bash
# =============================================================================
# Skip Connection 消融研究 / Skip Connection Ablation Study
# =============================================================================
# 3 encoder × 代表性 skip，decoder 固定 unet，bottleneck 固定 none
# 3 encoders × representative skips, fixed unet decoder + none bottleneck
#
# 用法 / Usage:
#   bash scripts/experiments/run_skip_study.sh
# =============================================================================

set -e
AMP="--amp"
BASE_OUT="output/skip_study"
CFG_DIR="configs/architectures/skip_study/general"

# 代表性 skip（经典 + 以 skip 为创新点的）
SKIPS=(
    concat              # 基线：通道拼接 / Baseline: channel concat
    add                 # 逐元素相加 / Element-wise add
    attention_gate      # Attention Gate (Oktay 2018)
    scse                # SCSE (Roy, TMI 2019)
    cbam                # CBAM (Woo, ECCV 2018)
    cross_attn          # Cross-Attention
    uctrans             # UCTransNet (AAAI 2022)
    skvmpp              # SK-VM++ Mamba skip (BSPC 2025)
    ta_mosc             # TA-MoSC (UTANet, AAAI 2025)
    gab                 # GAB (EGE-UNet, MICCAI 2023)
    sdi                 # SDI (U-Net V2, ISBI 2025)
    dense               # Dense skip (UNet++ style)
)

ENCODERS=(basic resnet50 pvtv2)

for enc in "${ENCODERS[@]}"; do
    for sk in "${SKIPS[@]}"; do
        cfg="${CFG_DIR}/${enc}_${sk}.yaml"
        [ ! -f "$cfg" ] && echo "SKIP: $cfg" && continue
        name="${enc}_${sk}"
        echo "=== $name ==="
        python train.py --config "$cfg" --output_dir "${BASE_OUT}/${name}" $AMP --seed 42 \
            || echo "FAILED: $name"
    done
done

echo "=== Done. Results: ${BASE_OUT}/ ==="
