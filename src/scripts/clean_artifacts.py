#!/usr/bin/env python3
"""Remove training / inference artifacts (checkpoints, logs, outputs).

This script is intentionally conservative: it only deletes well-known
artifact directories in the current repository:

- ckpts/
- outputs/
- logs/
- runs/
- results/

It never touches source code or data directories.

Safety:
- By default the script runs in DRY-RUN mode and only prints what would
  be deleted.
- To actually delete, pass `--yes` or set the environment variable
  `CONFIRM=1`.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import List


ARTIFACT_DIRS = ["ckpts", "outputs", "logs", "runs", "results"]
ARTIFACT_GLOBS = ["pytorchdist_*.out"]


def discover_targets(root: Path) -> List[Path]:
    targets: List[Path] = []
    for name in ARTIFACT_DIRS:
        path = root / name
        if path.exists():
            targets.append(path)
    for pattern in ARTIFACT_GLOBS:
        for path in root.glob(pattern):
            targets.append(path)
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean training/inference artifacts.")
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Repository root to clean (default: current directory).",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Actually delete artifacts. Without this flag, runs in dry-run mode.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    targets = discover_targets(root)

    print(f"Repository root: {root}")
    print("Known artifact directories:", ", ".join(ARTIFACT_DIRS))
    print("Known artifact file patterns:", ", ".join(ARTIFACT_GLOBS))
    print("")

    if not targets:
        print("Nothing to delete. No artifact directories/files found.")
        return

    print("The following paths are considered artifacts:")
    for path in targets:
        kind = "dir" if path.is_dir() else "file"
        print(f"  [{kind}] {path}")

    confirm = args.yes or os.environ.get("CONFIRM") == "1"
    if not confirm:
        print(
            "\nDRY RUN: nothing has been deleted.\n"
            "Re-run with --yes or set CONFIRM=1 to actually remove these paths."
        )
        return

    print("\nProceeding with deletion...")
    for path in targets:
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            print(f"  ✓ removed {path}")
        except Exception as exc:
            print(f"  ⚠️  failed to remove {path}: {exc}")

    print("\nCleanup complete.")


if __name__ == "__main__":
    main()

