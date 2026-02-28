#!/usr/bin/env python
"""
Simple evaluation script for Stage-1 ScanQA predictions.

Reads a predictions JSONL file (as produced by qa_inference.py) and
writes metrics.json with basic statistics and exact-match accuracy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Stage-1 ScanQA predictions.")
    parser.add_argument(
        "--predictions",
        type=str,
        default="outputs/qa/scanqa_predictions_test.jsonl",
        help="Path to predictions JSONL file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/stage1_scanqa",
        help="Directory where metrics.json will be written.",
    )
    return parser.parse_args()


def _normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def main() -> None:
    args = parse_args()
    pred_path = Path(args.predictions)
    if not pred_path.exists():
        raise SystemExit(f"Predictions file not found: {pred_path}")

    total = 0
    non_empty = 0
    em = 0

    with pred_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            total += 1
            pred = (rec.get("prediction") or "").strip()
            ref = (rec.get("reference") or "").strip()
            if pred:
                non_empty += 1
            if pred and ref and _normalize(pred) == _normalize(ref):
                em += 1

    metrics: Dict[str, float] = {}
    metrics["num_examples"] = total
    metrics["num_non_empty_predictions"] = non_empty
    metrics["exact_match"] = float(em) / float(total) if total > 0 else 0.0

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)

    print(f"Wrote metrics to {metrics_path}")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

