#!/usr/bin/env python
"""Generate synthetic ARKit-style instructions from ARKitScenes metadata.

Inputs:
    --arkit-root data/raw/arkitscenes
Outputs:
    data/processed/arkit_synth/train.json

The JSON array contains objects of the form:
  {
    "images": [...],
    "geom_token": {...},
    "task": "arkit_actions",
    "instruction": "...",
    "action_json": {...}
  }
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthesize RoomPlan instructions.")
    parser.add_argument("--arkit-root", type=Path, required=True, help="Root directory of ARKitScenes.")
    parser.add_argument("--output", type=Path, default=Path("data/processed/arkit_synth/train.json"))
    parser.add_argument("--num-views", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2024)
    return parser.parse_args()


def load_planes(scene_dir: Path) -> List[Dict]:
    """Load plane annotations for a single scene.

    This assumes the ARKitScenes layout:
      <scene_dir>/annotations/planes.json
    """
    plane_file = scene_dir / "annotations" / "planes.json"
    if not plane_file.exists():
        return []
    return json.loads(plane_file.read_text())


def load_cameras(scene_dir: Path) -> List[Dict]:
    """Load camera metadata for a single scene.

    Expected layout:
      <scene_dir>/cameras.json
    """
    file = scene_dir / "cameras.json"
    if not file.exists():
        return []
    return json.loads(file.read_text())


def pick_views(cameras: List[Dict], num_views: int, rng: random.Random) -> List[Dict]:
    if len(cameras) <= num_views:
        return cameras
    return rng.sample(cameras, num_views)


def make_instruction(scene_id: str, plane: Dict) -> str:
    """Build an English RoomPlan-style instruction for a single plane."""
    normal = plane.get("normal", [0, 1, 0])
    size = plane.get("extent", [1.0, 1.0])
    kind = plane.get("label", "plane")
    return (
        f"In scene {scene_id}, find a {kind} plane whose normal is approximately "
        f"({normal[0]:.2f}, {normal[1]:.2f}, {normal[2]:.2f}) with size about "
        f"{size[0]:.2f}×{size[1]:.2f} meters, and place a virtual anchor at its center."
    )


def action_json(scene_id: str, plane: Dict) -> Dict:
    return {
        "action": "place_anchor",
        "scene": scene_id,
        "center": plane.get("center", [0, 0, 0]),
        "normal": plane.get("normal", [0, 1, 0]),
        "extent": plane.get("extent", [1, 1]),
    }


def build_geom_token(cameras: List[Dict]) -> Dict:
    poses = [cam["pose"] for cam in cameras]
    intr = [cam["intrinsics"] for cam in cameras]
    depth_stats = [cam.get("depth_hist", [0] * 16) for cam in cameras]
    return {"R": poses, "t": [[0, 0, 0]] * len(poses), "K": intr, "depth_hist": depth_stats}


def iter_samples(args: argparse.Namespace) -> Iterable[Dict]:
    """Iterate over ARKit scenes and yield synthetic RoomPlan samples.

    This now searches recursively under --arkit-root for any directory that
    contains `annotations/planes.json` and `cameras.json`, so it will work
    whether your scenes live directly under the root or in nested folders
    like `Training/<scene_id>/`.
    """
    rng = random.Random(args.seed)

    # Find all scene directories that have both planes and cameras metadata.
    root: Path = args.arkit_root
    candidate_scenes: List[Path] = []
    for plane_file in root.rglob("annotations/planes.json"):
        scene_dir = plane_file.parent.parent
        cam_file = scene_dir / "cameras.json"
        if cam_file.exists():
            candidate_scenes.append(scene_dir)

    candidate_scenes = sorted(set(candidate_scenes))
    print(f"Discovered {len(candidate_scenes)} ARKit scenes with planes + cameras under {root}")

    for scene_dir in candidate_scenes:
        planes = load_planes(scene_dir)
        cameras = load_cameras(scene_dir)
        if not planes or not cameras:
            continue
        views = pick_views(cameras, args.num_views, rng)
        geom = build_geom_token(views)
        image_paths = [view["rgb_path"] for view in views]
        for plane in planes:
            yield {
                "images": image_paths,
                "geom_token": geom,
                "task": "arkit_actions",
                "instruction": make_instruction(scene_dir.name, plane),
                "action_json": action_json(scene_dir.name, plane),
            }


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("w", encoding="utf-8") as f:
        f.write("[\n")
        first = True
        for sample in iter_samples(args):
            if not first:
                f.write(",\n")
            json.dump(sample, f, ensure_ascii=False)
            first = False
            count += 1
        f.write("\n]\n")
    print(f"Generated {count} ARKit synthetic instructions into {args.output}")


if __name__ == "__main__":
    main()
