"""Grounding DINO example for medical image organ detection.

This example demonstrates how to use Grounding DINO to detect organs
in medical images using text prompts.

Usage:
    python examples/grounding_dino_example.py --image path/to/image.jpg
"""

import argparse
import numpy as np
import torch
import cv2
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medseg.grounding_dino_wrapper import (
    GroundingDINODetector,
    build_medical_text_prompt,
)


def visualize_detections(
    image: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    phrases: list,
    output_path: str = "output.jpg",
):
    """Visualize detection results.

    Args:
        image: Input image (H, W, 3)
        boxes: Detected boxes (N, 4) in normalized [0, 1]
        scores: Confidence scores (N,)
        phrases: Detected phrases (N,)
        output_path: Path to save output image
    """
    H, W = image.shape[:2]
    vis_image = image.copy()

    for box, score, phrase in zip(boxes, scores, phrases):
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)

        # Draw box
        color = (0, 255, 0)  # Green
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label = f"{phrase}: {score:.2f}"
        cv2.putText(
            vis_image,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    cv2.imwrite(output_path, vis_image)
    print(f"Saved visualization to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Grounding DINO Medical Example")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to medical image (optional, uses demo image if not provided)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="tiny",
        choices=["tiny", "base"],
        help="Grounding DINO model variant",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="grounding_dino_output.jpg",
        help="Output image path",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.35,
        help="Detection threshold",
    )
    args = parser.parse_args()

    # Define organ classes to detect
    class_names = [
        "liver",
        "spleen",
        "kidney",
        "stomach",
        "aorta",
    ]

    # Build text prompt
    text_prompt = build_medical_text_prompt(class_names, add_organ_suffix=True)
    print(f"Text prompt: {text_prompt}")

    # Initialize detector
    detector = GroundingDINODetector(model_type=args.model_type)

    # Load or create demo image
    if args.image:
        image = cv2.imread(args.image)
        if image is None:
            print(f"Error: Could not load image {args.image}")
            return
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        # Create demo image
        print("No image provided, creating demo image...")
        image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

    # Run detection
    print(f"\nDetecting organs with threshold {args.threshold}...")
    boxes, scores, phrases = detector.detect(
        image=image,
        text_prompt=text_prompt,
        box_threshold=args.threshold,
        text_threshold=args.threshold * 0.7,
    )

    print(f"\nDetected {len(boxes)} objects:")
    for box, score, phrase in zip(boxes, scores, phrases):
        print(f"  {phrase}: {score:.3f} at [{box[0]:.2f}, {box[1]:.2f}, {box[2]:.2f}, {box[3]:.2f}]")

    # Visualize
    visualize_detections(image, boxes, scores, phrases, args.output)

    # Generate region proposals
    print("\nGenerating region proposals...")
    proposals = detector.generate_region_proposals(image, class_names, args.threshold)
    print(f"Proposals shape: {proposals.shape}")

    # Save proposals visualization
    vis_proposals = visualize_proposals(image, proposals, class_names)
    proposal_output = args.output.replace(".jpg", "_proposals.jpg")
    cv2.imwrite(proposal_output, vis_proposals)
    print(f"Saved proposals visualization to {proposal_output}")


def visualize_proposals(
    image: np.ndarray,
    proposals: torch.Tensor,
    class_names: list,
) -> np.ndarray:
    """Visualize region proposals.

    Args:
        image: Input image (H, W, 3)
        proposals: Proposal masks (num_classes, H, W)
        class_names: Class names

    Returns:
        Visualization image
    """
    vis = image.copy()
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
    ]

    for idx, (mask, name) in enumerate(zip(proposals, class_names)):
        if mask.sum() > 0:
            color = colors[idx % len(colors)]
            # Create colored overlay
            overlay = vis.copy()
            mask_np = mask.numpy().astype(bool)
            overlay[mask_np] = color
            vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)

            # Add label
            nonzero = mask.nonzero()
            y_coord = int(nonzero[0].float().mean()) if mask.sum() > 0 else 50
            x_coord = int(nonzero[1].float().mean()) if mask.sum() > 0 else 50
            cv2.putText(
                vis,
                name,
                (x_coord, y_coord),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )

    return vis


if __name__ == "__main__":
    main()
