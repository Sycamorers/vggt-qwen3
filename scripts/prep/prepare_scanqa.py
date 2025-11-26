#!/usr/bin/env python
"""Convert ScanQA/SQA3D style annotations into multi-view JSON samples.

This script demonstrates how to:
1. Load ScanNet-based RGB frames and camera intrinsics/extrinsics.
2. Sample N views per scene.
3. Attach geometry tokens (pose, intrinsics, depth histogram).
4. Emit a JSON array that the training dataloader can ingest.

Usage:
    python scripts/prep/prepare_scanqa.py \
        --dataset scanqa \
        --scan-root data/raw/scannet \
        --qa-file data/raw/scanqa/questions.json \
        --output data/processed/scanqa/train.json \
        --num-views 8
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


def _default_depth_hist(depth_map: np.ndarray, num_bins: int = 16) -> List[float]:
    """Convert a dense depth map to a normalized histogram."""
    vals = depth_map[np.isfinite(depth_map)]
    if vals.size == 0:
        return [0.0] * num_bins
    counts, _ = np.histogram(vals, bins=num_bins, range=(vals.min(), vals.max()))
    total = counts.sum() + 1e-6
    return (counts / total).astype(np.float32).tolist()


def _load_pose(pose_file: Path) -> Tuple[List[float], List[float]]:
    """Load a 4x4 pose matrix and split into rotation + translation."""
    mat = np.loadtxt(pose_file).reshape(4, 4)
    rot = mat[:3, :3].flatten().tolist()
    trans = mat[:3, 3].tolist()
    return rot, trans


def _load_intrinsics(intr_file: Path) -> List[float]:
    mat = np.loadtxt(intr_file).reshape(3, 3)
    return mat.flatten().tolist()


@dataclass
class Sample:
    images: List[str]
    geom_token: Dict[str, List[float]]
    task: str
    question: str
    answer: str

    def to_dict(self) -> Dict:
        return {
            "images": self.images,
            "geom_token": self.geom_token,
            "task": self.task,
            "question": self.question,
            "answer": self.answer,
        }


def iter_examples(args: argparse.Namespace) -> Iterable[Sample]:
    qa_data = json.loads(Path(args.qa_file).read_text())
    rng = random.Random(args.seed)
    for entry in qa_data:
        scene_id = entry["scene_id"]
        view_ids = rng.sample(entry["available_views"], k=min(args.num_views, len(entry["available_views"])))
        images = []
        rot_list, trans_list, intr_list, depth_hist = [], [], [], []
        for vid in view_ids:
            img_rel = f"{scene_id}/color/{vid:06d}.jpg"
            images.append(img_rel)
            pose_file = Path(args.scan_root) / scene_id / "pose" / f"{vid:06d}.txt"
            intr_file = Path(args.scan_root) / scene_id / "intrinsic" / f"{vid:06d}.txt"
            depth_file = Path(args.scan_root) / scene_id / "depth" / f"{vid:06d}.png"
            rot, trans = _load_pose(pose_file)
            rot_list.append(rot)
            trans_list.append(trans)
            intr_list.append(_load_intrinsics(intr_file))
            depth_map = np.asarray(depth_reader(depth_file))
            depth_hist.append(_default_depth_hist(depth_map))
        geom = {
            "R": rot_list,
            "t": trans_list,
            "K": intr_list,
            "depth_hist": depth_hist,
        }
        yield Sample(
            images=images,
            geom_token=geom,
            task="3d_qa",
            question=entry["question"],
            answer=entry["answer"],
        )


def depth_reader(path: Path) -> np.ndarray:
    """Read depth images as float arrays (placeholder)."""
    import imageio.v2 as imageio

    depth_raw = imageio.imread(path).astype(np.float32)
    depth_raw[depth_raw == 0] = math.nan
    return depth_raw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ScanQA/SQA3D data.")
    parser.add_argument("--dataset", default="scanqa", choices=["scanqa", "sqa3d"], help="Dataset alias.")
    parser.add_argument("--scan-root", type=Path, required=True, help="Path to ScanNet frame dumps.")
    parser.add_argument("--qa-file", type=Path, required=True, help="Path to QA annotations JSON.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON file.")
    parser.add_argument("--num-views", type=int, default=8, help="Number of sampled views per scene.")
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    num = 0
    with args.output.open("w", encoding="utf-8") as f:
        f.write("[\n")
        first = True
        for sample in iter_examples(args):
            if not first:
                f.write(",\n")
            json.dump(sample.to_dict(), f, ensure_ascii=False)
            first = False
            num += 1
        f.write("\n]\n")
    print(f"Wrote {num} samples to {args.output}")


if __name__ == "__main__":
    main()
