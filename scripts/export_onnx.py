#!/usr/bin/env python3
"""ONNX 模型导出脚本。/ ONNX model export script.

用法 / Usage:
    python scripts/export_onnx.py \
        --config configs/architectures/networks/general/transunet.yaml \
        --checkpoint output/best_model.pth \
        --output model.onnx \
        --img_size 224

    # 验证导出的模型 / Verify exported model
    python scripts/export_onnx.py \
        --config configs/xxx.yaml \
        --checkpoint best.pth \
        --output model.onnx \
        --verify
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def export(args):
    import torch
    import yaml
    from medseg.model_builder import build_model

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg.get("model", cfg)
    if args.img_size:
        model_cfg["img_size"] = args.img_size
    img_size = model_cfg.get("img_size", 224)
    if img_size == "native":
        img_size = 224
    nc = model_cfg.get("num_classes", 2)

    # 构建模型 / Build model
    model = build_model(cfg)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint: {args.checkpoint}")
    model.eval()

    # 导出 / Export
    dummy = torch.randn(1, 3, img_size, img_size)
    output_path = args.output or "model.onnx"

    print(f"Exporting to {output_path} (img_size={img_size}, num_classes={nc})...")
    torch.onnx.export(
        model, dummy, output_path,
        input_names=["image"],
        output_names=["mask"],
        dynamic_axes={
            "image": {0: "batch", 2: "height", 3: "width"},
            "mask": {0: "batch", 2: "height", 3: "width"},
        } if args.dynamic else None,
        opset_version=args.opset,
        do_constant_folding=True,
    )
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"OK: {output_path} ({size_mb:.1f} MB)")

    # 验证 / Verify
    if args.verify:
        try:
            import onnxruntime as ort
            import numpy as np
            sess = ort.InferenceSession(output_path)
            inp = {sess.get_inputs()[0].name: dummy.numpy()}
            out = sess.run(None, inp)[0]
            print(f"ONNX Runtime verify: input={dummy.shape} → output={out.shape}")

            # 对比 PyTorch 输出 / Compare with PyTorch output
            with torch.no_grad():
                pt_out = model(dummy).numpy()
            diff = np.abs(pt_out - out).max()
            print(f"Max diff (PyTorch vs ONNX): {diff:.6f}")
            if diff < 1e-4:
                print("✓ 验证通过 / Verification passed")
            else:
                print(f"⚠ 差异较大，请检查 / Large diff, please check")
        except ImportError:
            print("Install onnxruntime to verify: pip install onnxruntime")


def main():
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default="model.onnx")
    parser.add_argument("--img_size", type=int, default=None)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--dynamic", action="store_true", help="动态输入尺寸 / Dynamic input shapes")
    parser.add_argument("--verify", action="store_true", help="用 onnxruntime 验证 / Verify with ORT")
    export(parser.parse_args())


if __name__ == "__main__":
    main()
