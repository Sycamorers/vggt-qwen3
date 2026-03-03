#!/usr/bin/env python
"""
Minimal sanity check for Stage-1 ScanQA inference.

This script runs a few ScanQA samples end-to-end and asserts that:
  - A checkpoint is successfully loaded (unless --allow_base_fallback is set).
  - Multimodal conditioning tensors are non-empty and have non-trivial variance.
  - Predictions are short answers (length-limited).
  - Predictions do not contain obvious meta-hallucination phrases.

Usage:
  python scripts/scanqa_sanity_check.py \
    --config configs/stage1_3d.yaml \
    --checkpoint_dir ckpts/stage1_3d_debug
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch

from vggt_qwen3.inference.qa_inference import (
    DATASET_DEFAULTS,
    build_model_from_config,
    build_tokenizer,
    load_checkpoint_if_available,
    load_yaml,
    log_model_and_checkpoint_summary,
    run_inference,
)
from vggt_qwen3.dataio.dataset_builder import DatasetConfig, MultiViewJsonDataset
from vggt_qwen3.dataio.collate_multiview import build_default_transform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-1 ScanQA sanity check.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stage1_3d.yaml",
        help="Stage config used for model reconstruction.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="ckpts/stage1_3d_debug",
        help="Path to a trained Stage-1 checkpoint root (e.g. ckpts/stage1_3d_debug).",
    )
    parser.add_argument(
        "--allow_base_fallback",
        action="store_true",
        help=(
            "Allow running with base weights if the checkpoint cannot be loaded. "
            "By default this is disabled so that broken checkpoint paths are surfaced."
        ),
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3,
        help="Number of ScanQA samples to run for the sanity check.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32,
        help="Maximum tokens to generate per sample.",
    )
    parser.add_argument(
        "--max_answer_chars",
        type=int,
        default=40,
        help="Maximum answer length in characters for short-answer mode.",
    )
    parser.add_argument(
        "--max_answer_tokens",
        type=int,
        default=12,
        help="Maximum answer length in whitespace-separated tokens.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stage_cfg = load_yaml(args.config)
    model_name = stage_cfg["model"]["name_or_path"]
    tokenizer_path = stage_cfg["model"].get("tokenizer_path")

    tokenizer = build_tokenizer(model_name, tokenizer_path)
    model, full_cfg = build_model_from_config(args.config, device=device)
    model.text_model.resize_token_embeddings(len(tokenizer))

    ckpt_info = load_checkpoint_if_available(
        model,
        args.checkpoint_dir,
        allow_base_fallback=args.allow_base_fallback,
    )
    log_model_and_checkpoint_summary(model, model_name, ckpt_info)
    if not args.allow_base_fallback and ckpt_info.get("used_base_weights"):
        raise SystemExit(
            "Sanity check expected a fine-tuned checkpoint to be loaded, but "
            "inference fell back to base HF weights. Double-check --checkpoint_dir."
        )

    data_cfg = full_cfg["data"]
    scanqa_defaults = DATASET_DEFAULTS["scanqa"]
    ds_cfg = DatasetConfig(
        path_glob=scanqa_defaults["glob"],
        num_views=data_cfg.get("num_views", 1),
        image_size=data_cfg.get("image_size", 448),
        task="scanqa",
    )
    dataset = MultiViewJsonDataset(ds_cfg)

    if len(dataset) == 0:
        raise RuntimeError(
            f"ScanQA dataset appears empty for glob '{ds_cfg.path_glob}'. "
            "Ensure data/processed/scanqa/test_split.jsonl exists."
        )

    num_samples = min(args.num_samples, len(dataset))
    indices: List[int] = list(range(num_samples))
    samples = [dataset[i] for i in indices]

    # ---------------------------------------------------------------------
    # Conditioning sanity: VGGT image features must be non-empty and non-degenerate.
    # ---------------------------------------------------------------------
    transform = build_default_transform(ds_cfg.image_size)
    with torch.no_grad():
        for i, sample in enumerate(samples):
            images = sample["images"]
            if not images:
                raise AssertionError(f"Sample {i} has no images; Stage-1 expects visual context.")
            vis = torch.stack([transform(img) for img in images], dim=0).unsqueeze(0).to(device)
            vis_tokens = model.encode_images(vis)
            if vis_tokens.numel() == 0:
                raise AssertionError(f"Sample {i} produced empty vis_tokens.")
            std = vis_tokens.float().std().item()
            if std <= 1e-6:
                raise AssertionError(
                    f"Sample {i} vis_tokens std too small ({std:.3e}); conditioning may be degenerate."
                )

    # ---------------------------------------------------------------------
    # Run a tiny inference pass (short-answer mode).
    # ---------------------------------------------------------------------
    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "num_beams": 1,
        "repetition_penalty": 1.1,
    }

    results = run_inference(
        model=model,
        tokenizer=tokenizer,
        samples=samples,
        device=device,
        image_size=ds_cfg.image_size,
        max_new_tokens=args.max_new_tokens,
        output_path=None,
        dataset_name="scanqa",
        debug=False,
        events_path=None,
        short_answer_only=True,
        debug_one=0,
        generation_kwargs=gen_kwargs,
    )

    if not results:
        raise AssertionError("run_inference returned no results.")

    bad_phrases = [
        "not a real photo",
        "cannot view images",
        "can't view images",
        "as an ai",
        "as a language model",
        "brain",
        "pollen",
        "plastic",
    ]

    for rec in results:
        pred = (rec.get("prediction") or "").strip()
        qid = rec.get("question_id")

        if not pred:
            raise AssertionError(f"Empty prediction for question_id={qid}")

        if len(pred) > args.max_answer_chars:
            raise AssertionError(
                f"Prediction too long ({len(pred)} chars) for question_id={qid}: {pred!r}"
            )

        if len(pred.split()) > args.max_answer_tokens:
            raise AssertionError(
                f"Prediction too long ({len(pred.split())} tokens) for question_id={qid}: {pred!r}"
            )

        lower = pred.lower()
        for phrase in bad_phrases:
            if phrase in lower:
                raise AssertionError(
                    f"Meta-hallucination phrase '{phrase}' detected in prediction for "
                    f"question_id={qid}: {pred!r}"
                )

    print("✅ Stage-1 ScanQA sanity check passed.")


if __name__ == "__main__":
    main()

