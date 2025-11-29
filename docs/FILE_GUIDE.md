# File Guide (Current Stack)

This file maps key files/folders to their purposes, aligned with the current training workflow (Stage 2 via `train_fixed.sh`).

---

## Top-Level Docs
- `README.md` – Primary overview and quickstart.
- `docs/COMPLETE_TRAINING_GUIDE.md` – Detailed walkthrough (env, data, training, troubleshooting).
- `docs/MONITORING_GUIDE.md` / `docs/TRAINING_MONITORING_SUMMARY.md` – Monitoring options and commands.
- `docs/SLURM_TRAINING_GUIDE.md` – HiPerGator/Slurm usage.

## Configs (`configs/`)
- `stage2_3d.yaml` – Active Stage 2 config (ScanQA/SQA3D).
- `stage1_sft.yaml`, `stage3_arkit.yaml` – Other stages (Stage 1 needs data; Stage 3 is a reference config only—trainer does not yet support structured `action_json`).
- `perceiver_small.yaml` – Vision projector architecture.
- `deepspeed_zero3.json` – DeepSpeed settings.
- `accelerate_*.yaml` – Sample Accelerate configs.

## Launch / Scripts
- `train_fixed.sh` – Hardened launcher (caches, NCCL, ZeRO-3, auto batch probe).
- `run.sh` – Slurm template that calls `train_fixed.sh`.
- `scripts/slurm/*.sbatch` – Stage-specific Slurm job scripts.
- `demo_monitoring.sh` – Interactive demo for monitoring commands.

## Source Code (`src/`)
- `models/`
  - `vggt_qwen3_vlm.py` – VGGT aggregator + Perceiver projector + Qwen3 wrapper.
  - `projector_perceiver.py` – Perceiver resampler config/implementation.
- `dataio/`
  - `dataset_builder.py` – JSON/JSONL dataset loader and multi-source mixer.
  - `collate_multiview.py` – Image transforms, prompt building, padding.
- `train/train_sft.py` – Accelerate/DeepSpeed training loop.
- `eval/` – Placeholder evaluation scripts.

## Data & Outputs
- `data/processed/` – Current ScanQA/SQA3D JSONL shards, DocVQA JSON.
- `data/raw/` – Placeholders plus ARKit README for downloads.
- `ckpts/`, `logs/` – Training outputs (checkpoints, TensorBoard).

## Third-Party
- `third_party/vggt/` – VGGT code + `vggt_1B_commercial.pt` checkpoint (user-provided).
- `third_party/Qwen3/` – Optional local Qwen3 mirror (otherwise use HF cache).
