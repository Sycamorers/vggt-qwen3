## Overview

This repository implements **Stage-1** of the VGGT‑Qwen3 RoomPlan pipeline:

- A frozen **VGGT** multi‑view 3D vision backbone.
- A lightweight **Perceiver** projector that maps VGGT tokens into the Qwen3 hidden space.
- Optional **geometry tokens** capturing camera/extrinsics/intrinsics + depth histograms.
- A **Qwen3‑4B‑Instruct** text decoder fine‑tuned with LoRA on 3D QA datasets
  (ScanQA + SQA3D).

Stage‑1 focuses on **3D visual question answering** over multi‑view RGB frames,
training a vision‑language model that produces grounded, short free‑form answers.
Later stages (not covered in depth here) extend this model to ARKit‑based
RoomPlan style outputs.

High‑level components:

- `src/vggt_qwen3/train/stage1.py` — supervised Stage‑1 training entrypoint.
- `src/vggt_qwen3/inference/qa_inference.py` — ScanQA/SQA3D QA inference.
- `configs/stage1_3d.yaml` — model + training hyperparameters.
- `configs/deepspeed_zero3.json` / `configs/accelerate_*.yaml` — distributed training.
- `scripts/prep/*.py` — data preparation utilities for ScanQA/SQA3D.
- `train_fixed.sh` / `scripts/train_stage1.sh` — Slurm‑friendly training launcher.
- `infer_stage1.sh` / `scripts/infer_stage1_scanqa.sh` — Stage‑1 QA inference.

