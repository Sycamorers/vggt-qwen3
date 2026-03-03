#!/usr/bin/env python
"""Create a tiny synthetic multi-view dataset for smoke-testing the pipeline.

The generated samples live under:
  data/raw/toy/<scene_id>/color/<view>.jpg
  data/processed/toy/train.json

Each JSON entry contains the fields expected by the training dataloader.
This avoids any external downloads and keeps VRAM usage low.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a synthetic toy dataset.")
    parser.add_argument("--scenes", type=int, default=4, help="Number of toy scenes.")
    parser.add_argument("--views-per-scene", type=int, default=2, help="Views per scene.")
    parser.add_argument("--samples-per-scene", type=int, default=3, help="How many Q/A pairs per scene.")
    parser.add_argument("--image-size", type=int, default=256, help="Height/width of generated images.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/toy/train.json"),
        help="Output JSON file relative to repo root.",
    )
    parser.add_argument("--seed", type=int, default=7, help="RNG seed.")
    return parser.parse_args()


def make_image(path: Path, size: int, color: tuple[int, int, int], text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (size, size), color)
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), text, fill=(255, 255, 255))
    img.save(path, format="JPEG", quality=85)


def build_geom(num_views: int) -> Dict[str, List[List[float]]]:
    rot = [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] for _ in range(num_views)]
    trans = [[0.0, 0.0, float(i) * 0.1] for i in range(num_views)]
    intr = [[500.0, 0.0, 128.0, 0.0, 500.0, 128.0, 0.0, 0.0, 1.0] for _ in range(num_views)]
    depth_hist = [[0.0] * 16 for _ in range(num_views)]
    return {"R": rot, "t": trans, "K": intr, "depth_hist": depth_hist}


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw_root = Path("data/raw/toy")

    samples = []
    for scene_idx in range(args.scenes):
        scene_id = f"scene_{scene_idx:03d}"
        geom = build_geom(args.views_per_scene)
        view_paths: List[str] = []
        for view_idx in range(args.views_per_scene):
            rel_path = Path("toy") / scene_id / "color" / f"{view_idx:06d}.jpg"
            abs_path = raw_root / scene_id / "color" / f"{view_idx:06d}.jpg"
            color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
            make_image(abs_path, args.image_size, color, text=f"{scene_id} v{view_idx}")
            view_paths.append(str(rel_path))
        for sample_idx in range(args.samples_per_scene):
            question = f"[toy] Scene {scene_id}: what do you see near view {sample_idx}?"
            answer = f"This is a synthetic sample {sample_idx} from {scene_id}."
            samples.append(
                {
                    "images": view_paths,
                    "geom_token": geom,
                    "task": "toy_qa",
                    "question": question,
                    "answer": answer,
                }
            )

    with args.output.open("w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(samples)} samples to {args.output}")
    print(f"Raw images under {raw_root}")


if __name__ == "__main__":
    main()
