"""Evaluate 3D reference grounding (ScanRefer/ReferIt3D)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute mAcc@IoU for referential grounding.")
    parser.add_argument("--predictions", type=Path, required=True, help="Predicted boxes JSONL.")
    parser.add_argument("--references", type=Path, required=True, help="Ground-truth boxes JSONL.")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    return parser.parse_args()


def load_boxes(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def iou_3d(box_a, box_b) -> float:
    def volume(box):
        sizes = [max(0.0, box["max"][i] - box["min"][i]) for i in range(3)]
        return sizes[0] * sizes[1] * sizes[2]

    inter_min = [max(box_a["min"][i], box_b["min"][i]) for i in range(3)]
    inter_max = [min(box_a["max"][i], box_b["max"][i]) for i in range(3)]
    inter = {"min": inter_min, "max": inter_max}
    inter_vol = volume(inter)
    union = volume(box_a) + volume(box_b) - inter_vol
    return inter_vol / max(union, 1e-6)


def main() -> None:
    args = parse_args()
    preds = load_boxes(args.predictions)
    refs = load_boxes(args.references)
    correct = 0
    for pred, ref in zip(preds, refs):
        if iou_3d(pred["box"], ref["box"]) >= args.iou_threshold:
            correct += 1
    metric = correct / max(len(refs), 1)
    print(f"mAcc@IoU{args.iou_threshold}: {metric * 100:.2f}%")


if __name__ == "__main__":
    main()
