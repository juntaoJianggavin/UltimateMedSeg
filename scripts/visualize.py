#!/usr/bin/env python3
"""预测可视化脚本：输入图像 → 模型推理 → overlay 输出。
Prediction visualization: input image → model inference → overlay output.

用法 / Usage:
    # 单张图像 / Single image
    python scripts/visualize.py \
        --config configs/xxx.yaml \
        --checkpoint output/best_model.pth \
        --input image.png \
        --output vis_output/

    # 整个目录 / Entire directory
    python scripts/visualize.py \
        --config configs/xxx.yaml \
        --checkpoint output/best_model.pth \
        --input ./data/test/images/ \
        --output vis_output/

输出 / Output:
    每张图生成 3 个文件 / 3 files per image:
    - xxx_input.png      原始图像 / Original image
    - xxx_pred.png       预测 mask（彩色）/ Predicted mask (colorized)
    - xxx_overlay.png    叠加图（图像 + 半透明 mask）/ Overlay (image + translucent mask)
"""

import argparse
import os
import sys
import glob

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 类别调色板（最多 20 类）/ Class color palette (up to 20 classes)
PALETTE = [
    (0, 0, 0),       # 0: 背景 / background
    (255, 0, 0),     # 1: 红 / red
    (0, 255, 0),     # 2: 绿 / green
    (0, 0, 255),     # 3: 蓝 / blue
    (255, 255, 0),   # 4: 黄 / yellow
    (255, 0, 255),   # 5: 洋红 / magenta
    (0, 255, 255),   # 6: 青 / cyan
    (255, 128, 0),   # 7: 橙 / orange
    (128, 0, 255),   # 8: 紫 / purple
    (0, 128, 255),   # 9: 天蓝 / sky blue
    (255, 128, 128), (128, 255, 128), (128, 128, 255),
    (255, 255, 128), (255, 128, 255), (128, 255, 255),
    (192, 64, 0), (0, 192, 64), (64, 0, 192), (192, 192, 0),
]


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """将类别 mask 转为彩色图。/ Convert class mask to color image."""
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id in range(mask.max() + 1):
        if cls_id < len(PALETTE):
            color[mask == cls_id] = PALETTE[cls_id]
    return color


def overlay(image: np.ndarray, mask_color: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """图像 + 半透明 mask 叠加。/ Image + translucent mask overlay."""
    if image.shape[:2] != mask_color.shape[:2]:
        mask_color = np.array(Image.fromarray(mask_color).resize(
            (image.shape[1], image.shape[0]), Image.NEAREST))
    fg = mask_color.sum(axis=2) > 0
    out = image.copy()
    out[fg] = (image[fg] * (1 - alpha) + mask_color[fg] * alpha).astype(np.uint8)
    return out


def predict_single(model, image_path: str, img_size: int, device, output_dir: str):
    """对单张图预测并保存可视化。/ Predict and save visualization for one image."""
    import torch
    import torch.nn.functional as F

    img_pil = Image.open(image_path).convert("RGB")
    img_np = np.array(img_pil)

    # 预处理 / Preprocess
    img_resized = img_pil.resize((img_size, img_size), Image.BILINEAR)
    img_t = torch.from_numpy(np.array(img_resized)).float().permute(2, 0, 1) / 255.0
    img_t = img_t.unsqueeze(0).to(device)

    # 推理 / Inference
    with torch.no_grad():
        out = model(img_t)
        if isinstance(out, (list, tuple)):
            out = out[0]
        pred = out.argmax(dim=1).squeeze(0).cpu().numpy()

    # Resize 回原始尺寸 / Resize back to original size
    pred_resized = np.array(Image.fromarray(pred.astype(np.uint8)).resize(
        (img_np.shape[1], img_np.shape[0]), Image.NEAREST))

    # 保存 / Save
    base = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(output_dir, exist_ok=True)

    Image.fromarray(img_np).save(os.path.join(output_dir, f"{base}_input.png"))

    mask_color = colorize_mask(pred_resized)
    Image.fromarray(mask_color).save(os.path.join(output_dir, f"{base}_pred.png"))

    ov = overlay(img_np, mask_color)
    Image.fromarray(ov).save(os.path.join(output_dir, f"{base}_overlay.png"))

    print(f"  {base}: classes={np.unique(pred_resized).tolist()}")


def main():
    import torch
    import yaml

    parser = argparse.ArgumentParser(description="Prediction visualization")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input", type=str, required=True, help="图像文件或目录 / Image file or directory")
    parser.add_argument("--output", type=str, default="vis_output/")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--img_size", type=int, default=None)
    args = parser.parse_args()

    # 加载模型 / Load model
    from medseg.utils.config import load_config
    from medseg.model_builder import build_model
    import medseg.models.encoders, medseg.models.decoders, medseg.models.bottlenecks, medseg.models.skip_connections, medseg.losses

    cfg = load_config(args.config)
    model = build_model(cfg)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state, strict=False)
    model = model.to(args.device).eval()

    img_size = args.img_size or cfg.get("model", {}).get("img_size", 224)
    if img_size == "native":
        img_size = 224

    # 收集图像 / Collect images
    if os.path.isdir(args.input):
        images = sorted(glob.glob(os.path.join(args.input, "*.png")) +
                        glob.glob(os.path.join(args.input, "*.jpg")) +
                        glob.glob(os.path.join(args.input, "*.tif")))
    else:
        images = [args.input]

    print(f"Predicting {len(images)} images → {args.output}")
    for img_path in images:
        predict_single(model, img_path, img_size, args.device, args.output)

    print(f"Done. Results saved to {args.output}")


if __name__ == "__main__":
    main()
