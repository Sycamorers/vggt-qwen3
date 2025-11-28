"""Stage-wise supervised fine-tuning entrypoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict

import time

import torch
import yaml
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import DistributedDataParallelKwargs
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from src.dataio.collate_multiview import MultiViewCollator
from src.dataio.dataset_builder import DatasetConfig, MultiSourceDataset, MultiViewJsonDataset
from src.models.projector_perceiver import PerceiverConfig
from src.models.vggt_qwen3_vlm import VGGTQwen3VLM, VisionLanguageConfig


os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")


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


def build_dataloader(cfg: Dict, tokenizer) -> DataLoader:
    datasets = {}
    for name, glob_path in cfg["datasets"].items():
        ds_cfg = DatasetConfig(
            path_glob=glob_path,
            num_views=cfg["num_views"],
            image_size=cfg["image_size"],
            task=name,
        )
        datasets[name] = MultiViewJsonDataset(ds_cfg)
    multi = MultiSourceDataset(datasets, cfg["mix_ratio"])
    collator = MultiViewCollator(cfg["image_size"], tokenizer, cfg["max_length"])
    loader = DataLoader(
        multi,
        batch_size=cfg["train_batch_size"],
        shuffle=True,
        collate_fn=collator,
    )
    return loader


def build_model(cfg: Dict) -> VGGTQwen3VLM:
    proj_cfg = (
        load_yaml(cfg["model"]["projector"])
        if isinstance(cfg["model"]["projector"], str)
        else cfg["model"]["projector"]
    )
    vlm_cfg = VisionLanguageConfig(
        text_model_name=cfg["model"]["name_or_path"],
        vision_ckpt_dir=cfg["model"]["vision_backbone"],
        num_vis_tokens=cfg["model"]["num_vis_tokens"],
        geom_tokens=cfg["model"].get("geom_tokens", 0),
        projector_cfg=PerceiverConfig(**proj_cfg),
        freeze_vision=cfg["model"].get("freeze_vision", True),
        dtype=cfg["model"].get("dtype", "bfloat16"),
    )
    return VGGTQwen3VLM(vlm_cfg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VGGT-Qwen3 training harness.")
    parser.add_argument("--config", required=True, help="Path to stage config YAML.")
    parser.add_argument("--deepspeed", default=None, help="Optional DeepSpeed config JSON.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_steps", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stage_cfg = load_yaml(args.config)

    tokenizer = build_tokenizer(
        stage_cfg["model"]["name_or_path"],
        stage_cfg["model"].get("tokenizer_path"),
    )

    # --------------------
    # Data & model
    # --------------------
    data_cfg = stage_cfg["data"]
    data_cfg["train_batch_size"] = stage_cfg["train"]["batch_size_per_gpu"]
    dataloader = build_dataloader(data_cfg, tokenizer)
    model = build_model(stage_cfg)

    train_cfg = stage_cfg["train"]
    max_steps = args.max_steps or train_cfg["max_steps"]
    grad_accum = train_cfg["grad_accum"]
    precision = train_cfg["precision"]

    # --------------------
    # Accelerator & DDP / DeepSpeed
    # --------------------
    deepspeed_plugin = None
    if args.deepspeed:
        deepspeed_plugin = DeepSpeedPlugin(zero_stage=3, config_file=args.deepspeed)

    logging_dir = Path(args.output_dir) / "logs"
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=grad_accum,
        mixed_precision=precision,
        log_with="tensorboard",
        project_dir=str(logging_dir),
        deepspeed_plugin=deepspeed_plugin,
        kwargs_handlers=[ddp_kwargs],
    )

    # --------------------
    # Optimizer & scheduler
    # --------------------
    projector_params, base_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "projector" in name or "geom_head" in name:
            projector_params.append(param)
        else:
            base_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": base_params, "lr": train_cfg["lr"]},
            {
                "params": projector_params,
                "lr": train_cfg.get("proj_lr", train_cfg["lr"]),
            },
        ],
        weight_decay=train_cfg["weight_decay"],
    )

    num_warmup_steps = int(train_cfg["warmup_ratio"] * max_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps,
        max_steps,
    )

    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model,
        optimizer,
        dataloader,
        scheduler,
    )

    accelerator.init_trackers("roomplan", config={"stage": args.config})

    # --------------------
    # 定义一个安全的保存函数：所有 rank 都参与 save_state
    # --------------------
    def save_checkpoint(save_path: Path):
        """
        所有 rank 都调用 accelerator.save_state，
        但只有 main_process 负责创建目录 / 写文件。
        """
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            save_path.mkdir(parents=True, exist_ok=True)
        accelerator.wait_for_everyone()

        # 这一行必须让所有 rank 都执行，避免 collective 顺序不一致
        accelerator.save_state(str(save_path))

        accelerator.wait_for_everyone()

    # --------------------
    # Training loop
    # --------------------
    step = 0
    model.train()

    if accelerator.is_main_process:
        print(f"\n{'=' * 80}")
        print("🚀 Starting training loop")
        print(f"   Max steps:  {max_steps}")
        print(f"   Log every:  {train_cfg['log_every_steps']} steps")
        print(f"   Save every: {train_cfg.get('save_every_steps', 'disabled')} steps")
        print(f"{'=' * 80}\n")

    start_time = time.time()

    for batch in dataloader:
        with accelerator.accumulate(model):
            loss = model(
                images=batch["pixel_values"],
                geom_token=batch["geom_token"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if accelerator.is_main_process and step % train_cfg["log_every_steps"] == 0:
            elapsed = time.time() - start_time
            steps_per_sec = (step + 1) / elapsed if elapsed > 0 else 0.0
            eta_seconds = (
                (max_steps - step - 1) / steps_per_sec if steps_per_sec > 0 else 0.0
            )
            eta_hours = eta_seconds / 3600.0

            base_lr = optimizer.param_groups[0]["lr"]
            proj_lr = (
                optimizer.param_groups[1]["lr"]
                if len(optimizer.param_groups) > 1
                else base_lr
            )

            print(
                f"Step {step:5d}/{max_steps} | "
                f"Loss: {loss.item():.4f} | "
                f"LR: {base_lr:.2e}/{proj_lr:.2e} | "
                f"Speed: {steps_per_sec:.2f} steps/s | "
                f"ETA: {eta_hours:.1f}h"
            )

            accelerator.log({"loss": loss.item(), "step": step})

        step += 1

        # 周期性保存 checkpoint：所有 rank 参与，避免 NCCL timeout
        if train_cfg.get("save_every_steps") and step % train_cfg["save_every_steps"] == 0:
            save_dir = Path(args.output_dir) / f"step_{step}"
            save_checkpoint(save_dir)

        if step >= max_steps:
            break

    save_checkpoint(Path(args.output_dir))


if __name__ == "__main__":
    main()
