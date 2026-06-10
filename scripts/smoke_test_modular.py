#!/usr/bin/env python3
"""Smoke-test modular model assembly (encoder + skip + bottleneck + decoder).

Verifies that YAML-based combinations and programmatic configs can be
built, run forward, and compute loss.

Usage:
    conda activate medseg
    cd /path/to/2026_medseg
    python scripts/smoke_test_modular.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.smoke_test_all import BATCH, create_dummy_data, test_one_config  # noqa: E402


CASES = [
    (
        "basic combo",
        "configs/architectures/combinations/general/unet_resnet50.yaml",
        "timm_resnet50 + concat + none + bilinear",
    ),
    (
        "decoder swap",
        "configs/architectures/decoder_study/general/resnet50_emcad.yaml",
        "timm_resnet50 + concat + none + emcad",
    ),
    (
        "skip swap",
        "configs/architectures/skip_study/general/resnet50_cbam.yaml",
        "timm_resnet50 + cbam + none + bilinear",
    ),
    (
        "bottleneck swap",
        "configs/architectures/bottleneck_study/general/resnet50_cbam.yaml",
        "timm_resnet50 + concat + cbam + bilinear",
    ),
    (
        "full modular stack",
        "configs/architectures/combinations/general/mednext_cbam_emcad.yaml",
        "mednext + concat + cbam + emcad",
    ),
    (
        "standalone network",
        "configs/architectures/networks/general/r2unet.yaml",
        "architecture=r2unet (non-modular path)",
    ),
]


def test_programmatic_combo() -> tuple[str, str]:
    """Build a model purely from a Python dict (no yaml file)."""
    import medseg.models.encoders  # noqa: F401
    import medseg.models.decoders  # noqa: F401
    import medseg.models.skip_connections  # noqa: F401
    import medseg.models.bottlenecks  # noqa: F401
    from medseg.model_builder import build_model
    from medseg.registry import LOSS_REGISTRY

    img_size = 64
    cfg = {
        "model": {
            "num_classes": 3,
            "img_size": img_size,
            "encoder": {
                "name": "timm_resnet34",
                "pretrained": False,
                "in_channels": 3,
                "params": {},
            },
            "skip_connection": {"name": "scse", "params": {}},
            "bottleneck": {"name": "aspp", "params": {}},
            "decoder": {"name": "unet", "params": {}},
        }
    }

    model = build_model(cfg)
    model.train()
    x = torch.randn(BATCH, 3, img_size, img_size)
    out = model(x)
    if isinstance(out, (list, tuple)):
        out = out[0]

    loss_fn = LOSS_REGISTRY.get("compound")(
        losses=[
            {"name": "ce", "weight": 0.5},
            {"name": "dice", "weight": 0.5},
        ]
    )
    target = torch.randint(0, 3, (BATCH, img_size, img_size))
    if out.shape[2:] != target.shape[1:]:
        out = F.interpolate(out, size=target.shape[1:], mode="bilinear", align_corners=False)
    loss = loss_fn(out, target)
    loss.backward()

    return "OK", f"loss={float(loss):.4f} shape={tuple(out.shape)}"


def main() -> int:
    dummy_root = create_dummy_data()
    print("Modular assembly smoke test")
    print(f"Project root: {ROOT}")
    print(f"Dummy data:   {dummy_root}\n")

    results: list[tuple[str, str, str, str]] = []
    t0 = time.time()

    # Programmatic API path
    try:
        status, msg = test_programmatic_combo()
    except Exception as exc:
        status, msg = "FAIL", f"{type(exc).__name__}: {exc}"
    results.append(("programmatic", "inline dict", "resnet34 + scse + aspp + unet", f"{status} | {msg}"))
    mark = "+" if status == "OK" else "x"
    print(f"  [{mark}] programmatic | resnet34 + scse + aspp + unet")
    print(f"       {msg}\n")

    # YAML-based representative cases
    for label, yaml_path, components in CASES:
        full_path = ROOT / yaml_path
        if not full_path.exists():
            status, msg = "FAIL", f"missing yaml: {yaml_path}"
        else:
            try:
                status, msg = test_one_config(str(full_path), dummy_root)
            except Exception as exc:
                status, msg = "FAIL", f"{type(exc).__name__}: {exc}"
        results.append((label, yaml_path, components, f"{status} | {msg}"))
        mark = "+" if status == "OK" else ("~" if status == "SKIP" else "x")
        print(f"  [{mark}] {label}")
        print(f"       {components}")
        print(f"       {yaml_path}")
        print(f"       {msg}\n")

    ok = sum(1 for *_, r in results if r.startswith("OK"))
    fail = sum(1 for *_, r in results if r.startswith("FAIL"))
    skip = sum(1 for *_, r in results if r.startswith("SKIP"))
    elapsed = time.time() - t0

    print("=" * 70)
    print(f"RESULTS ({elapsed:.1f}s): {ok} OK, {fail} FAIL, {skip} SKIP")
    print("=" * 70)

    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
