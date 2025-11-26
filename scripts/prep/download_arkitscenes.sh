#!/usr/bin/env bash
set -euo pipefail

# Placeholder downloader for ARKitScenes. You must accept Apple's license
# before triggering this script. The official instructions live at:
# https://github.com/apple/ARKitScenes

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TARGET="${ROOT_DIR}/data/raw/arkitscenes"
mkdir -p "${TARGET}"

echo "[ARKitScenes] Please make sure you have accepted the dataset EULA."
echo "[ARKitScenes] Downloading metadata into ${TARGET}"

# The official repo provides a python downloader. Invoke it if available.
if ! command -v python >/dev/null 2>&1; then
  echo "python command not found; aborting." >&2
  exit 1
fi

python - <<'PY'
import pathlib
import textwrap

root = pathlib.Path(__file__).resolve().parents[2]
target = root / "data" / "raw" / "arkitscenes"
target.mkdir(parents=True, exist_ok=True)
readme = target / "README_download.txt"
readme.write_text(textwrap.dedent("""
ARKitScenes download placeholder.
Follow https://github.com/apple/ARKitScenes#dataset-download to:
  1. Log into the Apple download portal.
  2. Use the official script to fetch RGB-D frames, poses, meshes, and annotations.
  3. Extract archives under data/raw/arkitscenes.
""").strip() + "\n")
print(f"Wrote instructions to {readme}")
PY

echo "[ARKitScenes] Downloader completed. Populate data/raw/arkitscenes manually if needed."
