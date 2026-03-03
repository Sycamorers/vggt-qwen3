#!/usr/bin/env python
"""Build ARKit synthetic instruction data from ARKit 3DOD-style annotations.

This script is tailored to the layout you already have:
  data/processed/ARKit/Training/<scene_id>/
    ├── <scene_id>_3dod_annotation.json
    ├── lowres_wide/
    │     ├── *.png
    ├── lowres_wide_intrinsics/
    └── lowres_wide.traj

It will generate a JSON file compatible with the training/inference
pipeline under:
  data/processed/arkit_synth/train.json

Each record looks like:
  {
    "images": [...],
    "geom_token": null,
    "task": "arkit_actions",
    "instruction": "...",
    "action_json": {...}
  }

You can then point Stage 3 configs and the ARKit inference script at
`data/processed/arkit_synth/train.json`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare ARKit synthetic data from 3DOD annotations."
    )
    parser.add_argument(
        "--arkit-training-root",
        type=Path,
        default=Path("data/processed/ARKit/Training"),
        help="Root directory containing per-scene 3DOD annotations.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/arkit_synth/train.json"),
        help="Output JSON file (single array).",
    )
    parser.add_argument(
        "--num-views",
        type=int,
        default=10,
        help="Number of RGB views to use per sample.",
    )
    parser.add_argument(
        "--max_scenes",
        type=int,
        default=None,
        help="Optional: limit to the first N scenes.",
    )
    return parser.parse_args()


def discover_scenes(root: Path) -> List[Path]:
    scenes: List[Path] = []
    if not root.exists():
        print(f"⚠️  ARKit training root {root} does not exist.")
        return scenes
    for scene_dir in sorted(root.glob("*")):
        if not scene_dir.is_dir():
            continue
        ann_files = list(scene_dir.glob("*_3dod_annotation.json"))
        if not ann_files:
            continue
        img_dir = scene_dir / "lowres_wide"
        if not img_dir.is_dir():
            continue
        scenes.append(scene_dir)
    print(f"Discovered {len(scenes)} ARKit 3DOD scenes under {root}")
    return scenes


def load_annotation(scene_dir: Path) -> Dict:
    ann_files = list(scene_dir.glob("*_3dod_annotation.json"))
    if not ann_files:
        return {}
    path = ann_files[0]
    try:
        return json.loads(path.read_text())
    except Exception as e:
        print(f"⚠️  Failed to read {path}: {e}")
        return {}


def pick_views(scene_dir: Path, num_views: int) -> List[str]:
    img_dir = scene_dir / "lowres_wide"
    images = sorted(img_dir.glob("*.png"))
    return [str(p) for p in images[:num_views]]


def make_instruction(scene_id: str, label: str) -> str:
    """Build an English instruction for a single 3D object."""
    return (
        f"In scene {scene_id}, find an object belonging to the category '{label}' "
        f"and place a virtual anchor at the center of that object."
    )


def build_action_json(scene_id: str, obj: Dict) -> Dict:
    seg = obj.get("segments", {})
    obb = seg.get("obbAligned") or seg.get("obb") or {}
    center = obb.get("centroid", [0, 0, 0])
    extent = obb.get("axesLengths", [1, 1, 1])
    normal = obb.get("dominantNormal", [0, 1, 0])
    return {
        "action": "place_anchor",
        "scene": scene_id,
        "center": center,
        "normal": normal,
        "extent": extent,
    }


def iter_samples(
    root: Path,
    num_views: int,
    max_scenes: int | None = None,
) -> Iterable[Dict]:
    scenes = discover_scenes(root)
    if max_scenes is not None:
        scenes = scenes[:max_scenes]

    for scene_idx, scene_dir in enumerate(scenes):
        scene_id = scene_dir.name
        ann = load_annotation(scene_dir)
        objs = ann.get("data", [])
        image_paths = pick_views(scene_dir, num_views)
        if not objs or not image_paths:
            continue
        for obj in objs:
            label = obj.get("label", "object")
            yield {
                "images": image_paths,
                "geom_token": None,  # geometry can be added later if needed
                "task": "arkit_actions",
                "instruction": make_instruction(scene_id, label),
                "action_json": build_action_json(scene_id, obj),
            }


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with args.output.open("w", encoding="utf-8") as f:
        f.write("[\n")
        first = True
        for sample in iter_samples(
            root=args.arkit_training_root,
            num_views=args.num_views,
            max_scenes=args.max_scenes,
        ):
            if not first:
                f.write(",\n")
            json.dump(sample, f, ensure_ascii=False)
            first = False
            count += 1
        f.write("\n]\n")

    print(f"Generated {count} ARKit synthetic instructions into {args.output}")


if __name__ == "__main__":
    main()
