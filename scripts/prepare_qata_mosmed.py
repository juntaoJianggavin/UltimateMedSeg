#!/usr/bin/env python3
"""QaTa-COV19 和 MosMedData+ 数据集准备脚本。
QaTa-COV19 and MosMedData+ dataset preparation script.

两个数据集来自 LViT 论文 (https://github.com/HUANGLIZI/LViT)，
下载后需要统一整理成本项目 TextImageDataset 期望的格式。

用法 / Usage:
    # 准备 QaTa-COV19
    python scripts/prepare_qata_mosmed.py --dataset qata --raw_dir /path/to/downloaded/QaTa-COV19

    # 准备 MosMedData+
    python scripts/prepare_qata_mosmed.py --dataset mosmed --raw_dir /path/to/downloaded/MosMedDataPlus

    # 验证已有数据是否正确
    python scripts/prepare_qata_mosmed.py --dataset qata --raw_dir ./data/QaTa-COV19 --verify

下载方式 / How to download:
    1. 访问 https://github.com/HUANGLIZI/LViT#datasets
    2. 下载 Google Drive 链接中的 QaTa-COV19 和 MosMedData+ 压缩包
    3. 解压后运行本脚本

下载后的原始目录结构 / Raw directory structure after download:
    QaTa-COV19/
    ├── Train Folder/
    │   ├── Img/          (*.png — 胸部X光灰度/RGB图像)
    │   ├── GT/           (*.png — 二值mask，0=背景 255=感染)
    │   └── Train_text.xlsx (Excel: 列1=文件名, 列2=放射报告文本)
    ├── Val Folder/
    │   ├── Img/
    │   ├── GT/
    │   └── Val_text.xlsx
    └── Test Folder/
        ├── Img/
        ├── GT/
        └── Test_text.xlsx

    MosMedDataPlus/
    ├── Train Folder/
    │   ├── Img/          (*.png — CT切片灰度图)
    │   ├── GT/           (*.png — 二值mask)
    │   └── Train_text.xlsx
    ├── Val Folder/
    │   ├── Img/
    │   ├── GT/
    │   └── Val_text.xlsx
    └── Test Folder/
        ├── Img/
        ├── GT/
        └── Test_text.xlsx

Excel 文本标注格式 / Excel text annotation format:
    第一列: 图像文件名 (如 "case_001.png" 或 "case_001")
    第二列: 放射报告/医学描述文本
    Column 1: image filename (e.g. "case_001.png" or "case_001")
    Column 2: radiology report / medical description text

    QaTa-COV19 文本示例 / QaTa-COV19 text examples:
        "bilateral ground-glass opacities in both lungs"
        "consolidation in right lower lobe with pleural effusion"
        "patchy infiltrates in bilateral lung fields"
        "no significant lung abnormality"

    MosMedData+ 文本示例 / MosMedData+ text examples:
        "ground-glass opacity in the right lung"
        "bilateral consolidation with air bronchogram"
        "small area of infection in left lower lobe"
        "normal lung parenchyma without lesion"
"""

import argparse
import os
import sys
from pathlib import Path


def verify_structure(data_root: str, dataset: str) -> bool:
    """验证数据集目录结构是否正确。
    Verify dataset directory structure is correct."""
    root = Path(data_root)
    ok = True
    issues = []

    expected_splits = {
        "Train Folder": "Train_text",
        "Val Folder": "Val_text",
        "Test Folder": "Test_text",
    }

    for split_dir, text_prefix in expected_splits.items():
        split_path = root / split_dir
        if not split_path.is_dir():
            issues.append(f"  缺少目录 / Missing dir: {split_dir}/")
            ok = False
            continue

        img_dir = split_path / "Img"
        gt_dir = split_path / "GT"

        if not img_dir.is_dir():
            issues.append(f"  缺少图像目录 / Missing: {split_dir}/Img/")
            ok = False
        else:
            n_img = len([f for f in os.listdir(img_dir) if f.endswith('.png')])
            if n_img == 0:
                issues.append(f"  图像目录为空 / Empty: {split_dir}/Img/ (0 PNG files)")
                ok = False
            else:
                print(f"  ✓ {split_dir}/Img/: {n_img} images")

        if not gt_dir.is_dir():
            issues.append(f"  缺少mask目录 / Missing: {split_dir}/GT/")
            ok = False
        else:
            n_gt = len([f for f in os.listdir(gt_dir) if f.endswith('.png')])
            print(f"  ✓ {split_dir}/GT/: {n_gt} masks")

        # 检查文本标注
        text_found = False
        for ext in (".xlsx", ".csv", ".tsv"):
            text_path = split_path / (text_prefix + ext)
            if text_path.exists():
                text_found = True
                try:
                    import pandas as pd
                    df = pd.read_excel(text_path) if ext == ".xlsx" else pd.read_csv(text_path)
                    print(f"  ✓ {text_prefix}{ext}: {len(df)} rows, columns={list(df.columns)}")
                    # 显示前 3 条文本示例
                    cols = list(df.columns)
                    if len(cols) >= 2:
                        for i, row in df.head(3).iterrows():
                            print(f"    示例 / Example: \"{row[cols[1]][:60]}...\"")
                except ImportError:
                    print(f"  ✓ {text_prefix}{ext} exists (install pandas+openpyxl to inspect)")
                break
        if not text_found:
            issues.append(f"  缺少文本标注 / Missing: {split_dir}/{text_prefix}.xlsx")
            # 不算错误——没有文本标注也能用普通方法训练
            print(f"  ⚠ {split_dir}/{text_prefix}.xlsx 不存在 (非文本引导方法仍可使用)")

    if issues:
        print("\n问题 / Issues:")
        for issue in issues:
            print(issue)

    return ok


def check_image_mask_alignment(data_root: str):
    """检查图像和mask是否一一对应。
    Check image-mask alignment."""
    root = Path(data_root)
    for split_dir in ["Train Folder", "Val Folder", "Test Folder"]:
        img_dir = root / split_dir / "Img"
        gt_dir = root / split_dir / "GT"
        if not img_dir.is_dir() or not gt_dir.is_dir():
            continue

        img_names = set(os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.endswith('.png'))
        gt_names = set(os.path.splitext(f)[0] for f in os.listdir(gt_dir) if f.endswith('.png'))

        only_img = img_names - gt_names
        only_gt = gt_names - img_names
        matched = img_names & gt_names

        print(f"\n  {split_dir}: {len(matched)} matched, {len(only_img)} img-only, {len(only_gt)} gt-only")
        if only_img:
            print(f"    无mask的图像 / Images without mask: {list(only_img)[:5]}...")
        if only_gt:
            print(f"    无图像的mask / Masks without image: {list(only_gt)[:5]}...")


def create_symlinks(raw_dir: str, target_dir: str):
    """如果数据不在默认位置，创建符号链接。
    Create symlinks if data is not at the default location."""
    raw = Path(raw_dir).resolve()
    target = Path(target_dir)
    if target.exists():
        print(f"  目标已存在 / Target exists: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(raw)
    print(f"  创建链接 / Created symlink: {target} -> {raw}")


def main():
    parser = argparse.ArgumentParser(
        description="QaTa-COV19 / MosMedData+ 数据集准备与验证"
    )
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["qata", "mosmed"],
                        help="数据集类型 / Dataset type")
    parser.add_argument("--raw_dir", type=str, required=True,
                        help="下载解压后的数据集根目录 / Raw dataset root after download")
    parser.add_argument("--verify", action="store_true",
                        help="只验证不处理 / Verify only, no processing")
    parser.add_argument("--link", action="store_true",
                        help="创建符号链接到 data/ 目录 / Create symlink to data/")
    args = parser.parse_args()

    dataset_names = {
        "qata": "QaTa-COV19",
        "mosmed": "MosMedDataPlus",
    }
    default_targets = {
        "qata": "data/QaTa-COV19",
        "mosmed": "data/MosMedDataPlus",
    }

    name = dataset_names[args.dataset]
    print(f"{'='*60}")
    print(f"  {name} 数据集验证 / {name} Dataset Verification")
    print(f"  目录 / Directory: {args.raw_dir}")
    print(f"{'='*60}\n")

    print("1. 检查目录结构 / Checking directory structure...")
    ok = verify_structure(args.raw_dir, args.dataset)

    print("\n2. 检查图像-mask对齐 / Checking image-mask alignment...")
    check_image_mask_alignment(args.raw_dir)

    if args.link:
        print("\n3. 创建符号链接 / Creating symlink...")
        create_symlinks(args.raw_dir, default_targets[args.dataset])

    print(f"\n{'='*60}")
    if ok:
        print(f"  ✓ {name} 数据集已就绪！/ Dataset is ready!")
        print(f"\n  使用方式 / Usage:")
        print(f"    # 用于通用分割（不需要文本）")
        print(f"    python train.py --config configs/intro_to_datasets/{args.dataset}_covid19.yaml" if args.dataset == "qata" else
              f"    python train.py --config configs/intro_to_datasets/mosmed_plus.yaml")
        print(f"\n    # 用于文本引导分割")
        if args.dataset == "qata":
            print(f"    python train_text_guided.py --config configs/training_paradigms/text_guided/qata_covid19_languide.yaml")
            print(f"    python train_text_guided.py --config configs/training_paradigms/text_guided/qata_covid19_cris.yaml")
        else:
            print(f"    python train_text_guided.py --config configs/training_paradigms/text_guided/mosmed_plus_languide.yaml")
            print(f"    python train_text_guided.py --config configs/training_paradigms/text_guided/mosmed_plus_clip_universal.yaml")
    else:
        print(f"  ✗ {name} 数据集有问题，请检查上述错误。")
        print(f"    下载地址 / Download: https://github.com/HUANGLIZI/LViT#datasets")
    print(f"{'='*60}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
