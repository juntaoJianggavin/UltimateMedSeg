#!/usr/bin/env python3
"""Pre-download timm ImageNet pretrained weights into the medseg local cache.

Default backend is Hugging Face Hub (standard timm path). ModelScope is
available as an explicit alternative::

    python scripts/download_timm_pretrained.py resnet50
    python scripts/download_timm_pretrained.py resnet50 --source modelscope
    python scripts/download_timm_pretrained.py resnet50 --mirror
    python scripts/download_timm_pretrained.py --list-hf-id resnet50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from medseg.utils.hf_hub import configure_hf_hub, resolve_hf_endpoint  # noqa: E402
from medseg.utils.timm_pretrained import (  # noqa: E402
    ensure_timm_pretrained_via_hf,
    ensure_timm_pretrained_via_modelscope,
    timm_cache_dir,
    timm_hf_hub_id,
)


def main():
    parser = argparse.ArgumentParser(description="Pre-download timm pretrained weights")
    parser.add_argument("models", nargs="*", help="timm model names, e.g. resnet50")
    parser.add_argument(
        "--source",
        choices=("hf", "modelscope"),
        default="hf",
        help="download backend (default: hf)",
    )
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    parser.add_argument("--list-hf-id", metavar="MODEL", help="print hub repo id and exit")
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
    args = parser.parse_args()

    if args.mirror:
        import os

        os.environ.setdefault("MEDSEG_HF_MIRROR", "1")
    if args.endpoint:
        import os

        os.environ["HF_ENDPOINT"] = args.endpoint.rstrip("/")

    endpoint = configure_hf_hub()
    if endpoint:
        print(f"HF endpoint: {endpoint}")
    elif resolve_hf_endpoint() is None:
        print("HF endpoint: https://huggingface.co (default)")

    if args.list_hf_id:
        print(timm_hf_hub_id(args.list_hf_id) or "")
        return

    if not args.models:
        parser.error("provide at least one model name, e.g. resnet50")

    download_fn = (
        ensure_timm_pretrained_via_modelscope
        if args.source == "modelscope"
        else ensure_timm_pretrained_via_hf
    )

    for name in args.models:
        hub_id = timm_hf_hub_id(name)
        if not hub_id:
            print(f"[skip] {name}: no hf_hub_id")
            continue
        print(f"[download] {name} <- {args.source}/{hub_id}")
        path = download_fn(name, force=args.force)
        print(f"  weight: {path}")
        print(f"  cache:  {timm_cache_dir(hub_id)}")


if __name__ == "__main__":
    main()
