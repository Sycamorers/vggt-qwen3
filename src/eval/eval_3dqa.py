"""Evaluation harness for ScanQA/SQA3D exact match accuracy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from src.models.vggt_qwen3_vlm import VGGTQwen3VLM, VisionLanguageConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate 3D QA datasets.")
    parser.add_argument("--predictions", type=Path, required=True, help="Model prediction JSON.")
    parser.add_argument("--references", type=Path, required=True, help="Ground-truth JSON.")
    return parser.parse_args()


def load_json_array(path: Path):
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        data = data.get("data") or data.get("samples") or []
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}, got {type(data)}")
    return data


def main() -> None:
    args = parse_args()
    preds = load_json_array(args.predictions)
    refs = load_json_array(args.references)
    correct = 0
    for pred, ref in zip(preds, refs):
        correct += int(pred["answer"].strip().lower() == ref["answer"].strip().lower())
    acc = correct / max(len(refs), 1)
    print(f"Accuracy: {acc * 100:.2f}% ({correct}/{len(refs)})")


if __name__ == "__main__":
    main()
