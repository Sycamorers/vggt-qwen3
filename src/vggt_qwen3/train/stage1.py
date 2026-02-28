"""Stage-wise supervised fine-tuning entrypoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict

import platform
import subprocess
import time

import torch
import yaml
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import DistributedDataParallelKwargs
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from vggt_qwen3.dataio.collate_multiview import MultiViewCollator
from vggt_qwen3.dataio.dataset_builder import (
    DatasetConfig,
    MultiSourceDataset,
    MultiViewJsonDataset,
)
from vggt_qwen3.models.projector_perceiver import PerceiverConfig
from vggt_qwen3.models.vggt_qwen3_vlm import VGGTQwen3VLM, VisionLanguageConfig


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
    num_workers = int(os.getenv("DATALOADER_NUM_WORKERS", "0"))
    loader = DataLoader(
        multi,
        batch_size=cfg["train_batch_size"],
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
    )
    return loader


def _find_git_root(start: Path) -> Path | None:
    """Walk upwards from `start` to find a `.git` directory, if any."""
    for path in [start, *start.parents]:
        if (path / ".git").exists():
            return path
    return None


def write_repro_metadata(output_dir: Path, args: argparse.Namespace, stage_cfg: Dict) -> None:
    """Write minimal but useful reproducibility metadata alongside checkpoints."""
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # 1) CLI args
    args_path = meta_dir / "args.yaml"
    with args_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "config": args.config,
                "output_dir": args.output_dir,
                "max_steps": args.max_steps,
            },
            f,
        )

    # 2) Environment summary (Python, Torch, CUDA, pip freeze if available)
    env_path = meta_dir / "env.txt"
    with env_path.open("w", encoding="utf-8") as f:
        f.write(f"Python: {platform.python_version()}\n")
        f.write(f"Platform: {platform.platform()}\n")
        f.write(f"Torch: {torch.__version__}\n")
        f.write(f"CUDA available: {torch.cuda.is_available()}\n")
        if torch.cuda.is_available():
            f.write(f"CUDA version: {torch.version.cuda}\n")
            f.write(f"Num GPUs: {torch.cuda.device_count()}\n")
            for i in range(torch.cuda.device_count()):
                f.write(f"  GPU {i}: {torch.cuda.get_device_name(i)}\n")
        f.write("\n")

        # Best-effort pip freeze
        try:
            proc = subprocess.run(
                ["pip", "freeze"],
                check=False,
                capture_output=True,
                text=True,
            )
            f.write("pip freeze:\n")
            f.write(proc.stdout)
        except Exception as exc:  # pragma: no cover - diagnostics only
            f.write(f"\n[pip freeze failed: {exc}]\n")

    # 3) Git metadata (commit + short status) if inside a git repo.
    git_root = _find_git_root(Path.cwd())
    git_path = meta_dir / "git.txt"
    with git_path.open("w", encoding="utf-8") as f:
        if git_root is None:
            f.write("No .git directory found; skipping git metadata.\n")
            return

        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=git_root,
                check=False,
                capture_output=True,
                text=True,
            ).stdout.strip()
            f.write(f"Commit: {commit}\n")

            status = subprocess.run(
                ["git", "status", "--short", "--branch"],
                cwd=git_root,
                check=False,
                capture_output=True,
                text=True,
            ).stdout
            f.write("\nStatus:\n")
            f.write(status)
        except Exception as exc:  # pragma: no cover - diagnostics only
            f.write(f"Git metadata collection failed: {exc}\n")


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

    # ------------------------------------------------------------------
    # Lightweight overrides from the environment (used by train_fixed.sh)
    # ------------------------------------------------------------------
    # Allow external launch scripts to safely shrink the per-GPU batch size
    # and gradient accumulation without editing the YAML config.
    train_cfg = stage_cfg["train"]
    batch_override = os.getenv("BATCH_PER_GPU")
    if batch_override is not None:
        try:
            train_cfg["batch_size_per_gpu"] = int(batch_override)
            print(f"[stage1] Overriding batch_size_per_gpu to {train_cfg['batch_size_per_gpu']} from BATCH_PER_GPU")
        except ValueError:
            print(f"[stage1] Ignoring invalid BATCH_PER_GPU={batch_override!r}")

    grad_override = os.getenv("GRAD_ACCUM")
    if grad_override is not None:
        try:
            train_cfg["grad_accum"] = int(grad_override)
            print(f"[stage1] Overriding grad_accum to {train_cfg['grad_accum']} from GRAD_ACCUM")
        except ValueError:
            print(f"[stage1] Ignoring invalid GRAD_ACCUM={grad_override!r}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = build_tokenizer(
        stage_cfg["model"]["name_or_path"],
        stage_cfg["model"].get("tokenizer_path"),
    )

    # --------------------
    # Data & model
    # --------------------
    data_cfg = stage_cfg["data"]
    # Use the (possibly overridden) training batch size.
    data_cfg["train_batch_size"] = train_cfg["batch_size_per_gpu"]
    dataloader = build_dataloader(data_cfg, tokenizer)

    # Basic dataset sanity checks (sizes, sample keys/shapes).
    dataset = dataloader.dataset
    total_samples = len(dataset)
    if total_samples == 0:
        raise RuntimeError(
            "Training dataloader is empty. Check that your data.globs in the "
            "stage config match existing JSON/JSONL files and contain records."
        )
    if isinstance(dataset, MultiSourceDataset):
        per_source = {name: len(ds) for name, ds in dataset.datasets.items()}
        print(f"Loaded datasets: {per_source} (total={total_samples})")
    else:
        print(f"Loaded dataset with {total_samples} samples.")

    # One tiny CPU batch to log tensor shapes / keys.
    sample_batch = next(iter(dataloader))
    print(
        "Sample batch keys and shapes: "
        + ", ".join(
            f"{k}: {tuple(v.shape) if hasattr(v, 'shape') else type(v)}"
            for k, v in sample_batch.items()
        )
    )

    model = build_model(stage_cfg)
    max_steps = args.max_steps or train_cfg["max_steps"]
    grad_accum = train_cfg["grad_accum"]
    precision = train_cfg["precision"]

    # --------------------
    # Accelerator & DDP / DeepSpeed
    # --------------------
    deepspeed_plugin = None
    if args.deepspeed:
        deepspeed_plugin = DeepSpeedPlugin(zero_stage=3, config_file=args.deepspeed)

    logging_dir = output_dir / "logs"
    logging_dir.mkdir(parents=True, exist_ok=True)
    events_path = logging_dir / "events.jsonl"
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

    if accelerator.is_main_process:
        write_repro_metadata(output_dir, args, stage_cfg)

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

            # Structured JSONL event log (one line per logging step).
            event = {
                "step": int(step),
                "loss": float(loss.item()),
                "lr_base": float(base_lr),
                "lr_proj": float(proj_lr),
                "steps_per_sec": float(steps_per_sec),
                "eta_hours": float(eta_hours),
            }
            with events_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")

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
