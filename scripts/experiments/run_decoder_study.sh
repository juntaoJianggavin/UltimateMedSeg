#!/usr/bin/env bash
# =============================================================================
# Decoder 消融研究 / Decoder Ablation Study
# =============================================================================
# 3 encoder × 代表性 decoder（经典 + 以 decoder 为创新点的工作）
# 3 encoders × representative decoders (classic + decoder-innovation papers)
#
# 不跑那些以 encoder 或整体架构为创新点的 decoder（如 swinunet/hiformer/missformer）
# Skip decoders from encoder-innovation papers (swinunet/hiformer/missformer etc.)
#
# 用法 / Usage:
#   bash scripts/experiments/run_decoder_study.sh
# =============================================================================

set -e

AMP="--amp"
BASE_OUT="output/decoder_study"
CFG_DIR="configs/architectures/decoder_study/general"

# 代表性 decoder（经典 + 以 decoder 为创新点的）
# Representative decoders (classic + decoder-innovation papers)
DECODERS=(
    # 经典基线 / Classic baselines
    bilinear        # 最简双线性上采样
    unet            # 标准 UNet decoder (Ronneberger 2015)
    deconv          # 转置卷积上采样
    # 密集连接 / Dense
    unetpp          # UNet++ (Zhou 2018) — 密集嵌套 skip
    unet3plus       # UNet 3+ (Huang 2020) — 全尺度 skip
    # 级联创新 / Cascade innovations
    cascade         # CASCADE (MICCAI 2023) — 级联注意力
    emcad           # EMCAD (2024) — 多尺度卷积注意力
    cascade_emcad   # CASCADE + EMCAD 组合
    cfm             # Polyp-PVT CFM (2021) — 级联融合
    # 注意力 / Attention
    attention       # Attention gate decoder
    ham             # HAM/Hamburger (SegNeXt) — 矩阵分解注意力
    lawin           # Lawin (2022) — 大窗口注意力
    # 金字塔 / Pyramid
    upernet         # UPerNet (2018) — FPN + PPM
    # MLP / Transformer
    mlp             # SegFormer All-MLP decoder
    segformer       # SegFormer decoder
)

ENCODERS=(basic resnet50 pvtv2)

for enc in "${ENCODERS[@]}"; do
    for dec in "${DECODERS[@]}"; do
        cfg="${CFG_DIR}/${enc}_${dec}.yaml"
        [ ! -f "$cfg" ] && echo "SKIP (no yaml): $cfg" && continue
        name="${enc}_${dec}"
        echo "=== $name ==="
        python train.py --config "$cfg" --output_dir "${BASE_OUT}/${name}" $AMP --seed 42 \
            || echo "FAILED: $name"
    done
done

echo "=== Done. Results: ${BASE_OUT}/ ==="
