#!/usr/bin/env python3
"""Download medical segmentation datasets from Hugging Face Hub.

Respects HF mirror settings via ``medseg.utils.hf_hub``::

    export MEDSEG_HF_MIRROR=1
    python scripts/download_hf_dataset.py --list
    python scripts/download_hf_dataset.py medseg7d --output-dir ~/datasets/MedSeg-7D

Note: TransUNet-preprocessed Synapse/ACDC (train_npz / test_vol_h5) is **not**
hosted as a standard HF dataset repo. For that format, use TransUNet Google Drive
or preprocess locally after downloading a raw/community variant from HF.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from medseg.utils.hf_hub import (  # noqa: E402
    HF_DATASET_CATALOG,
    configure_hf_hub,
    hf_snapshot_download,
    resolve_hf_endpoint,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download HF datasets for medseg")
    parser.add_argument(
        "dataset",
        nargs="?",
        choices=sorted(HF_DATASET_CATALOG.keys()),
        help="catalog key (see --list)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="local directory (default: ./data/<key>)",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="use https://hf-mirror.com for this run (sets MEDSEG_HF_MIRROR=1)",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default=None,
        help="override HF endpoint for this run (e.g. https://hf-mirror.com)",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="optional git revision / branch / tag",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list catalog entries and exit",
    )
    args = parser.parse_args()

    if args.mirror:
        import os

        os.environ.setdefault("MEDSEG_HF_MIRROR", "1")
    if args.endpoint:
        import os

        os.environ["HF_ENDPOINT"] = args.endpoint.rstrip("/")

    endpoint = configure_hf_hub()
    print(f"HF endpoint: {endpoint or 'https://huggingface.co (default)'}")

    if args.list or args.dataset is None:
        print("\nAvailable HF dataset catalog entries:\n")
        for key, meta in HF_DATASET_CATALOG.items():
            print(f"  {key}")
            print(f"    repo:   {meta['repo_id']} ({meta['repo_type']})")
            print(f"    about:  {meta['description']}")
            print(f"    covers: {', '.join(meta['includes'])}")
            print()
        if args.dataset is None and not args.list:
            parser.error("provide a dataset key or use --list")
        return 0

    meta = HF_DATASET_CATALOG[args.dataset]
    out = Path(args.output_dir or ROOT / "data" / args.dataset)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {meta['repo_id']} -> {out}")
    local = hf_snapshot_download(
        meta["repo_id"],
        repo_type=meta["repo_type"],
        revision=args.revision,
        local_dir=str(out),
    )
    print(f"Done: {local}")
    print(
        "\nNote: Synapse/ACDC in TransUNet npz format are not in this catalog. "
        "See configs/intro_to_datasets/synapse.yaml and acdc.yaml for expected layout."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
