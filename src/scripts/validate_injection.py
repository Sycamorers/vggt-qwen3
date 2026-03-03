#!/usr/bin/env python3
"""CPU-only preflight to validate Stage-1 <image> injection contract.

Checks (without loading the heavy text/vision models):
- Every sample has at least one `<image>` token in the prompt.
- Injection span length implied by config (num_vis_tokens + geom_tokens) fits
  inside the padded sequence length.
- The worst-case injection region `[image_pos, image_pos + span_len)` lies
  entirely within the `labels == -100` region (prompt/padding), i.e. it never
  overlaps answer tokens.
- Labels for answer tokens (labels != -100) are disjoint from the injection
  region.

If any assertion fails, the script exits with a non-zero status code.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from vggt_qwen3.dataio.dataset_builder import DatasetConfig, MultiSourceDataset, MultiViewJsonDataset
from vggt_qwen3.dataio.collate_multiview import MultiViewCollator


def load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_tokenizer(model_name: str, tokenizer_path: str | None = None):
    path = tokenizer_path or model_name
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if "<image>" not in tokenizer.get_vocab():
        tokenizer.add_tokens(["<image>"])
    tokenizer.padding_side = "right"
    return tokenizer


def build_batch(cfg_path: str, num_samples: int) -> tuple:
    """Construct a single collated batch from the Stage-1 config."""
    cfg = load_yaml(cfg_path)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]

    tokenizer = build_tokenizer(
        model_cfg["name_or_path"],
        model_cfg.get("tokenizer_path"),
    )

    datasets: Dict[str, MultiViewJsonDataset] = {}
    for name, glob_path in data_cfg["datasets"].items():
        ds_cfg = DatasetConfig(
            path_glob=glob_path,
            num_views=data_cfg["num_views"],
            image_size=data_cfg["image_size"],
            task=name,
        )
        datasets[name] = MultiViewJsonDataset(ds_cfg)

    multi = MultiSourceDataset(datasets, data_cfg["mix_ratio"])
    collator = MultiViewCollator(
        data_cfg["image_size"],
        tokenizer,
        data_cfg["max_length"],
        num_vis_tokens=model_cfg["num_vis_tokens"],
        geom_tokens=model_cfg.get("geom_tokens", 0),
    )
    loader = DataLoader(
        multi,
        batch_size=min(num_samples, len(multi)),
        shuffle=False,
        collate_fn=collator,
    )
    batch = next(iter(loader))
    return batch, tokenizer, model_cfg


def validate_injection(
    batch: Dict[str, torch.Tensor],
    tokenizer,
    model_cfg: Dict,
) -> None:
    input_ids: torch.Tensor = batch["input_ids"]
    labels: torch.Tensor = batch["labels"]
    image_token_id = tokenizer.convert_tokens_to_ids("<image>")

    if image_token_id is None:
        raise AssertionError("Tokenizer does not contain '<image>' token ID.")

    span_len = int(model_cfg["num_vis_tokens"] + model_cfg.get("geom_tokens", 0))

    bsz, seqlen = input_ids.shape
    print(f"Batch size: {bsz}, sequence length: {seqlen}")
    print(f"Configured num_vis_tokens: {model_cfg['num_vis_tokens']}")
    print(f"Configured geom_tokens: {model_cfg.get('geom_tokens', 0)}")
    print(f"Expected (max) injection span length: {span_len}")
    print(f"Image token ID: {image_token_id}")

    # Decode first prompt for quick visual inspection
    first_ids = input_ids[0].tolist()
    first_labels = labels[0].tolist()
    prompt_tokens: List[int] = []
    for tid, lab in zip(first_ids, first_labels):
        if lab == -100:
            prompt_tokens.append(tid)
        else:
            break
    decoded_prompt = tokenizer.decode(prompt_tokens, skip_special_tokens=False)
    print("\n[Sample 0] Decoded prompt (labels == -100 region):")
    print(repr(decoded_prompt))

    errors: List[str] = []

    for i in range(bsz):
        ids_i = input_ids[i].tolist()
        labels_i = labels[i].tolist()

        positions = [idx for idx, t in enumerate(ids_i) if t == image_token_id]
        print(f"\n[Sample {i}]")
        print(f"  <image> occurrences: {len(positions)} at indices {positions}")

        if not positions:
            errors.append(f"Sample {i}: no <image> token found in input_ids.")
            continue
        if len(positions) > 1:
            print(f"  ⚠️  Multiple <image> tokens found; using the first for checks.")

        pos = positions[0]
        inj_start = pos
        inj_end = pos + span_len  # exclusive

        print(f"  Injection start index: {inj_start}")
        print(f"  Injection end index (exclusive): {inj_end}")

        if inj_end > seqlen:
            errors.append(
                f"Sample {i}: injection span [{inj_start}, {inj_end}) exceeds sequence length {seqlen}."
            )
            continue

        # Answer region: labels != -100
        answer_indices = [j for j, lab in enumerate(labels_i) if lab != -100]
        if not answer_indices:
            print("  ⚠️  No answer tokens (labels != -100) found; skipping overlap check.")
            continue
        ans_start = min(answer_indices)
        ans_end = max(answer_indices) + 1  # exclusive
        print(f"  Answer token range: [{ans_start}, {ans_end})")

        # Verify that injection span does not intersect answer region.
        overlaps = not (inj_end <= ans_start or inj_start >= ans_end)
        print(f"  Injection/answer overlap: {overlaps}")
        if overlaps:
            errors.append(
                f"Sample {i}: injection span [{inj_start}, {inj_end}) overlaps answer tokens [{ans_start}, {ans_end})."
            )

        # Verify labels == -100 on entire injection span (prompt/padding only).
        inj_labels = labels_i[inj_start:inj_end]
        if any(lab != -100 for lab in inj_labels):
            errors.append(
                f"Sample {i}: labels in injection span are not all -100 "
                f"(found {set(inj_labels)})."
            )

    if errors:
        print("\n❌ Injection validation FAILED with the following issues:")
        for err in errors:
            print(f"  - {err}")
        raise AssertionError("Injection validation failed; see errors above.")

    print("\n✅ Injection validation PASSED for this batch.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Stage-1 <image> injection contract on CPU.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stage1_3d.yaml",
        help="Stage config YAML used for dataset/collator construction.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=4,
        help="Number of samples to draw for the validation batch.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"Config not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    batch, tokenizer, model_cfg = build_batch(str(cfg_path), args.num_samples)
    validate_injection(batch, tokenizer, model_cfg)


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

