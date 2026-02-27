# Repository Structure and Conventions

This document explains how the VGGT-Qwen3 RoomPlan repository is organized after the Stage-1 cleanup, and how to extend it in a maintainable way.

## Top-level layout

- `README.md`  
  High-level project overview, quickstart, and canonical Stage-1 commands.

- `LICENSE`  
  Project license (Apache-2.0). Third-party components under `third_party/` retain their own licenses.

- `CHANGELOG.md`  
  Human-readable log of changes, including migration notes for refactors.

- `CITATION.cff`  
  Basic citation metadata for referencing this repo in academic work.

- `pyproject.toml`, `requirements.txt`  
  Packaging and dependency metadata. The project can be installed in editable mode via:
  ```bash
  pip install -e .
  ```

- `configs/`  
  YAML configs for models, data, optimizer, and Accelerate/DeepSpeed.
  - `stage1_3d.yaml` – primary Stage-1 config (ScanQA + SQA3D).
  - `perceiver_small.yaml` – Perceiver projector config.
  - `accelerate_*.yaml` – Accelerate launcher configs for different GPU counts.
  - `deepspeed_zero3.json` – optional ZeRO-3 config.
  - `stage2_arkit.yaml` – Stage 2 (RoomPlan) config; currently **future work**.

- `stage1/`  
  Lightweight marker for Stage-1 workflows with a short README pointing to the main docs and entry points.

- `src/vggt_qwen3/`  
  The Python package containing all Stage-1 code (see below).

- `scripts/`  
  Thin command-line helpers and utilities:
  - `eval_baseline.sh`, `eval_baseline_quick.py` – QA evaluation helpers.
  - `test_dataloader.py`, `check_init.py`, etc. – engineering utilities.
  - `prep/` – data preparation scripts (ScanQA/SQA3D/ARKit).
  - `slurm/` – example Slurm scripts (Stage 2/3; not part of canonical Stage-1 workflows).

- `train_stage1.sh`, `infer_stage1.sh`  
  Canonical shell wrappers for Stage-1 training and inference.

- `docs/`  
  Markdown documentation:
  - `index.md` – documentation landing page.
  - `stage1_quickstart.md` – extended Stage-1 quickstart.
  - `model/architecture.md` – model and injection mechanism details.
  - `dev/debug_history.md` – maintainer-facing debug notes.
  - `dev/repo_structure.md` – this file.

- `data/`  
  Local data directory:
  - `data/raw/` – raw files (ignored by git except for `.gitkeep`).
  - `data/processed/` – preprocessed ScanQA/SQA3D/ARKit JSON/JSONL files.

- `ckpts/`, `logs/`, `outputs/`  
  Default locations for checkpoints, logs, and evaluation outputs. These are ignored by git, except for `.gitkeep` placeholders.

- `third_party/`  
  External code and resources:
  - `third_party/vggt/` – VGGT repo and checkpoint(s).
  - `third_party/Qwen3/` – Qwen3 docs and artifacts (if present).

## Python package: `vggt_qwen3`

Under `src/vggt_qwen3/`:

- `__init__.py`  
  Marks the root package.

- `dataio/`  
  - `dataset_builder.py` – dataset definitions:
    - `DatasetConfig`, `MultiViewJsonDataset`, `MultiSourceDataset`.
  - `collate_multiview.py` – multi-view image transforms and `MultiViewCollator` (handles `<image>` injection span and label masking).

- `models/`  
  - `projector_perceiver.py` – Perceiver-based projector and config.
  - `vggt_qwen3_vlm.py` – `VGGTQwen3VLM` wrapper combining VGGT, projector, and Qwen3, including the token-level injection logic.

- `train/`  
  - `stage1.py` – Stage-1 supervised fine-tuning harness:
    - Builds tokenizer, datasets, collator, model, optimizer, and scheduler.
    - Uses Accelerate (and optional DeepSpeed) for distributed training.
  - `losses.py` – supplementary loss helpers (e.g., for future extensions).

- `inference/`  
  - `qa_inference.py` – Stage-1 QA inference:
    - Rebuilds the model from a config YAML.
    - Loads checkpoints (including ZeRO-3 merged weights).
    - Runs short-answer generation with visual token injection at `<image>`.
  - `arkit_inference.py` – ARKit/RoomPlan inference (Stage 2/3 flavored, **future work**).

- `eval/`  
  - `eval_3dqa.py`, `eval_ref3d.py` – simple evaluation utilities for 3D QA and referring-3D tasks.

### Import conventions

- Always import from the package namespace, not from `src.*`. For example:

  ```python
  from vggt_qwen3.dataio.dataset_builder import DatasetConfig, MultiViewJsonDataset
  from vggt_qwen3.dataio.collate_multiview import MultiViewCollator
  from vggt_qwen3.models.vggt_qwen3_vlm import VGGTQwen3VLM, VisionLanguageConfig
  ```

- For new modules, add them under `vggt_qwen3` and avoid creating additional top-level packages.

## Canonical entry points

Stage-1 workflows should use the following entry points:

- **Training**

  ```bash
  python -m vggt_qwen3.train.stage1 --config configs/stage1_3d.yaml --output_dir ckpts/stage1_3d
  ```

  or via wrapper:

  ```bash
  ./train_stage1.sh
  ```

- **Inference / evaluation (Stage 1 QA)**

  ```bash
  python -m vggt_qwen3.inference.qa_inference \
    --config configs/stage1_3d.yaml \
    --checkpoint_dir ckpts/stage1_3d \
    --glob "data/processed/scanqa/test_split.jsonl" \
    --num_samples 200 \
    --max_new_tokens 32 \
    --output_jsonl outputs/qa/scanqa_predictions_test.jsonl
  ```

  or via wrapper:

  ```bash
  ./infer_stage1.sh
  ```

Other scripts under `scripts/` are helpers built on top of these entry points and should be kept thin (i.e., they should call `python -m vggt_qwen3...` rather than duplicating logic).

## Conventions for new code

When adding new features or stages:

- Keep **user-facing** and **maintainer-facing** documentation separate:
  - User: `README.md`, `docs/index.md`, `docs/stage1_quickstart.md`.
  - Maintainer: `docs/dev/debug_history.md`, `docs/dev/repo_structure.md`.
- Prefer extending the existing package structure under `vggt_qwen3` rather than creating new top-level packages.
- For new training/inference flows:
  - Create a Python module under `vggt_qwen3.train` or `vggt_qwen3.inference`.
  - Add a thin shell wrapper in `./scripts/` or top-level if it’s a canonical command.
- Avoid site-specific paths or environment assumptions:
  - Use configs for paths.
  - Document environment variables clearly (e.g., `HF_HOME`, `CUDA_VISIBLE_DEVICES`).

## Stage 2 and future work

- Stage-2 configs and scripts (e.g., `configs/stage2_arkit.yaml`, `scripts/slurm/stage2_3d_2xb200.sbatch`) are retained but considered **future work**.
- The current docs and entry points are Stage-1–only; when Stage 2 is promoted to a first-class workflow, mirror the Stage-1 patterns:
  - Dedicated config files and package modules.
  - Canonical training/inference entry points.
  - Separate user vs. maintainer docs.

