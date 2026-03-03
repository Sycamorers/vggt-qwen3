#!/usr/bin/env python3
"""Tiny Stage-1 QA format smoke test.

Runs inference on a small number of ScanQA/SQA3D samples and reports:
- question, prediction, reference
- normalized prediction/reference (lowercased, punctuation-stripped, collapsed whitespace)
- whether the normalized strings exactly match

This script is meant to catch obvious formatting mismatches (e.g., extra
phrases around the short answer) without running a full evaluation suite.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List

import torch

from vggt_qwen3.inference.qa_inference import (
    build_model_from_config,
    build_tokenizer,
    load_checkpoint_if_available,
    load_yaml,
    run_inference,
)
from vggt_qwen3.dataio.dataset_builder import DatasetConfig, MultiViewJsonDataset


def normalize_answer(text: str) -> str:
    """Conservative normalization to improve exact-match robustness."""
    text = text.lower().strip()
    # Remove basic punctuation.
    text = re.sub(r"[^\w\s]", " ", text)
    # Collapse whitespace.
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-1 QA short-answer format smoke test.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stage1_3d.yaml",
        help="Stage config used for model reconstruction.",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="data/processed/scanqa/*.jsonl",
        help="Glob pattern for QA JSONL files.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Path to a trained checkpoint directory (e.g. ckpts/stage1_3d).",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of samples to run.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (cuda, cpu). Defaults to cuda if available.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for sampling scenes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    stage_cfg = load_yaml(args.config)
    model_name = stage_cfg["model"]["name_or_path"]
    tokenizer_path = stage_cfg["model"].get("tokenizer_path")

    tokenizer = build_tokenizer(model_name, tokenizer_path)
    model, full_cfg = build_model_from_config(args.config, device=device)
    model.text_model.resize_token_embeddings(len(tokenizer))
    load_checkpoint_if_available(model, args.checkpoint_dir)

    data_cfg = full_cfg["data"]
    num_views = data_cfg.get("num_views", 1)
    image_size = data_cfg.get("image_size", 448)

    ds_cfg = DatasetConfig(
        path_glob=args.glob,
        num_views=num_views,
        image_size=image_size,
        task="qa",
    )
    dataset = MultiViewJsonDataset(ds_cfg)
    # Take the first N scenes deterministically; for quick smoke tests this is ok.
    indices = list(range(min(args.num_samples, len(dataset))))
    samples: List[Dict] = [dataset[i] for i in indices]

    print(f"Running format smoke test on {len(samples)} samples...")
    preds = run_inference(
        model=model,
        tokenizer=tokenizer,
        samples=samples,
        device=device,
        image_size=image_size,
        max_new_tokens=32,
        output_path=None,
    )

    total = 0
    matched = 0
    for rec in preds:
        question = rec["question"]
        pred = rec["prediction"] or ""
        ref = rec.get("reference") or ""
        norm_pred = normalize_answer(pred)
        norm_ref = normalize_answer(ref)
        is_match = bool(norm_pred) and (norm_pred == norm_ref)
        total += 1
        matched += int(is_match)

        print("\n" + "=" * 80)
        print(f"Question:   {question}")
        print(f"Prediction: {pred!r}")
        print(f"Reference:  {ref!r}")
        print(f"Norm pred:  {norm_pred!r}")
        print(f"Norm ref:   {norm_ref!r}")
        print(f"Exact EM?:  {is_match}")

    acc = matched / total if total > 0 else 0.0
    print("\n" + "=" * 80)
    print(f"Normalized exact-match on {total} samples: {acc:.3f}")


if __name__ == "__main__":
    main()
