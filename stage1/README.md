# Stage 1 Overview

This directory is a lightweight marker for Stage-1–only workflows.

All Stage-1 code and configs live in:
- `configs/stage1_3d.yaml` – Stage-1 training and inference config.
- `src/vggt_qwen3/` – Python package implementing data loading, model, training, and inference.
- `train_stage1.sh` / `infer_stage1.sh` – Canonical entry points for training and QA inference.

For full details, see:
- Top-level `README.md` (quickstart and commands).
- `docs/stage1_quickstart.md` (extended walkthrough).
- `docs/model/architecture.md` (model and injection details).

