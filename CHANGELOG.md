# Changelog

All notable changes to this project will be documented in this file.

The format is inspired by [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- Initial `pyproject.toml` and `requirements.txt` for editable installation via `pip install -e .`.
- Top-level `LICENSE` (Apache-2.0) and `CITATION.cff`.
- Package layout under `src/vggt_qwen3/` with submodules for data loading, models, training, inference, and evaluation.
- Stage-1–focused documentation:
  - `docs/index.md` (documentation landing page).
  - `docs/stage1_quickstart.md` (extended quickstart for Stage 1).
  - `docs/model/architecture.md` (model and injection details).
  - `docs/dev/debug_history.md` and `docs/dev/repo_structure.md` (maintainer-facing notes).
- Canonical training and inference entry points:
  - `train_stage1.sh` → launches `python -m vggt_qwen3.train.stage1`.
  - `infer_stage1.sh` → launches `python -m vggt_qwen3.inference.qa_inference`.

### Changed
- Moved core Python modules from `src/` into the `vggt_qwen3` package under `src/vggt_qwen3/`:
  - `src/dataio/*` → `src/vggt_qwen3/dataio/*`
  - `src/models/*` → `src/vggt_qwen3/models/*`
  - `src/inference/*` → `src/vggt_qwen3/inference/*`
  - `src/train/train_sft.py` → `src/vggt_qwen3/train/stage1.py`
  - `src/train/losses.py` → `src/vggt_qwen3/train/losses.py`
  - `src/eval/*` → `src/vggt_qwen3/eval/*`
- Updated all imports and scripts to use the `vggt_qwen3` package namespace instead of `src.*`.
- Standardized Stage-1 inference (`qa_inference`) to:
  - Use a short-answer hint in the prompt.
  - Default to `max_new_tokens=32`, `temperature=0`, `top_p=1.0`.

### Migration notes
- **Installing the package**
  - Old: rely on `PYTHONPATH=.` and import from `src.*`.
  - New: install in editable mode and import `vggt_qwen3.*`:
    - `pip install -e .`
    - Then `python -m vggt_qwen3.train.stage1 ...` or `python -m vggt_qwen3.inference.qa_inference ...`.

- **Training entry point**
  - Old: `python -m src.train.train_sft --config configs/stage1_3d.yaml --output_dir ckpts/stage1_3d`.
  - New: `python -m vggt_qwen3.train.stage1 --config configs/stage1_3d.yaml --output_dir ckpts/stage1_3d`.

- **Inference entry point (Stage 1 QA)**
  - Old: `python -m src.inference.qa_inference ...`.
  - New: `python -m vggt_qwen3.inference.qa_inference ...`.

- **Scripts under `scripts/`**
  - Any script that previously invoked `src.*` modules has been updated to call `vggt_qwen3.*`.
  - Existing SLURM scripts for Stage 2/3 keep working but are considered **future work**; the README and docs focus on Stage 1 only.

