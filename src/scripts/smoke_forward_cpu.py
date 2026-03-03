#!/usr/bin/env python3
"""Single-batch smoke forward pass for Stage 1.

This script runs a tiny supervised forward pass through VGGT-Qwen3 using the
Stage-1 config, primarily to catch shape / injection bugs before long runs.

It is designed to be as light as possible:
- Uses a small batch (default 1 sample).
- Uses the `mock` VGGT backbone by default to avoid loading the full 3D model.
- Can run on CPU or GPU. On CPU we force float32 to avoid bf16 issues.

Expected usage on HiPerGator:
  conda activate roomplan
  export VGGT_DEBUG_INJECT=1
  python scripts/smoke_forward_cpu.py --config configs/stage1_3d.yaml --device cpu --batch-size 1
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from vggt_qwen3.dataio.dataset_builder import DatasetConfig, MultiSourceDataset, MultiViewJsonDataset
from vggt_qwen3.dataio.collate_multiview import MultiViewCollator
from vggt_qwen3.models.projector_perceiver import PerceiverConfig
from vggt_qwen3.models.vggt_qwen3_vlm import VGGTQwen3VLM, VisionLanguageConfig


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


def build_batch(cfg: Dict, tokenizer, batch_size: int):
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]

    datasets = {}
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
        batch_size=min(batch_size, len(multi)),
        shuffle=False,
        collate_fn=collator,
    )
    return next(iter(loader))


def build_model(stage_cfg: Dict, device: torch.device, use_mock_vision: bool) -> VGGTQwen3VLM:
    model_cfg = stage_cfg["model"]
    proj_cfg = load_yaml(model_cfg["projector"]) if isinstance(model_cfg["projector"], str) else model_cfg["projector"]

    dtype = model_cfg.get("dtype", "bfloat16")
    if device.type == "cpu":
        # Use float32 on CPU for maximum compatibility.
        dtype = "float32"

    vlm_cfg = VisionLanguageConfig(
        text_model_name=model_cfg["name_or_path"],
        vision_ckpt_dir="mock" if use_mock_vision else model_cfg["vision_backbone"],
        num_vis_tokens=model_cfg["num_vis_tokens"],
        geom_tokens=model_cfg.get("geom_tokens", 0),
        projector_cfg=PerceiverConfig(**proj_cfg),
        freeze_vision=True,
        dtype=dtype,
    )
    model = VGGTQwen3VLM(vlm_cfg).to(device)
    model.eval()
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-1 single-batch smoke forward (CPU/GPU-friendly).")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stage1_3d.yaml",
        help="Stage config YAML for Stage 1.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run on (e.g. cpu, cuda, cuda:0).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for the smoke forward pass.",
    )
    parser.add_argument(
        "--use-mock-vision",
        action="store_true",
        help="Use a tiny mock VGGT backbone instead of loading the full checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Stage config not found: {cfg_path}")

    stage_cfg = load_yaml(str(cfg_path))
    device = torch.device(args.device)

    tokenizer = build_tokenizer(
        stage_cfg["model"]["name_or_path"],
        stage_cfg["model"].get("tokenizer_path"),
    )
    batch = build_batch(stage_cfg, tokenizer, args.batch_size)

    print(f"Building model on device={device} (use_mock_vision={args.use_mock_vision})...")
    model = build_model(stage_cfg, device=device, use_mock_vision=args.use_mock_vision)

    # Move batch tensors to the chosen device.
    images = batch["pixel_values"].to(device)
    geom_token = batch["geom_token"]
    if geom_token is not None:
        geom_token = {k: v.to(device) for k, v in geom_token.items()}
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    print("Running single forward pass...")
    loss = model(
        images=images,
        geom_token=geom_token,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
    print(f"✅ Smoke forward completed. Loss: {loss.item():.4f}")


if __name__ == "__main__":
    main()

