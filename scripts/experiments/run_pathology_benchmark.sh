#!/usr/bin/env bash
# =============================================================================
# 病理分割 Benchmark / Pathology Segmentation Benchmark
# =============================================================================
# 数据集 / Datasets: GlaS (train80%/test), MoNuSeg, PanNuke
#
# 设计原则 / Design:
#   - 数据集配置来自 configs/intro_to_datasets/ (数据路径 + 训练超参)
#   - Dataset configs from configs/intro_to_datasets/ (data paths + training)
#   - 模型架构通过 --override 覆盖 / Model architecture via --override
#
# 用法 / Usage:
#   bash scripts/experiments/run_pathology_benchmark.sh
# =============================================================================
set -e
AMP="--amp"
BASE_OUT="output/pathology_benchmark"
DATASET_CFG="configs/intro_to_datasets"

# 独立架构 / Standalone architectures (override model.architecture)
STANDALONE_ARCHS=(
    # ---- 通用 Baseline / General Baselines ----
    unet
    attention_unet
    unetpp
    # ---- 病理专有 / Pathology-Specific ----
    u_vixlstm               # U-VixLSTM: Vision-xLSTM (mLSTM) encoder
    transnuseg              # TransNuSeg: 多任务解码器+共享QKV注意力 (MICCAI 2023)
    hovernet_lite           # HoverNetLite: NP+HV双分支核分割 (经典HoVerNet轻量版)
    nulite                  # NuLite: 轻量核分割
)

DATASETS=(glas)

for ds in "${DATASETS[@]}"; do
    ds_cfg="${DATASET_CFG}/${ds}.yaml"
    [ ! -f "$ds_cfg" ] && echo "SKIP dataset config: $ds_cfg" && continue

    # 独立架构 / Standalone architectures
    for arch in "${STANDALONE_ARCHS[@]}"; do
        echo "=== ${arch} on ${ds} ==="
        python train.py --config "$ds_cfg" --output_dir "${BASE_OUT}/${ds}/${arch}" $AMP --seed 42 \
            --override model.architecture="${arch}" \
            || echo "FAILED: ${arch} on ${ds}"
    done

    # 模块化架构 / Modular architectures
    echo "=== resnet50_unet on ${ds} ==="
    python train.py --config "$ds_cfg" --output_dir "${BASE_OUT}/${ds}/resnet50_unet" $AMP --seed 42 \
        --override model.encoder.name=timm_resnet50 model.encoder.pretrained=true model.decoder.name=unet \
        || echo "FAILED: resnet50_unet on ${ds}"
done
echo "=== Done: ${BASE_OUT}/ ==="
