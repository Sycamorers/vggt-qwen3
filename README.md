# VGGT-Qwen3 RoomPlan (Stage 1)

VGGT-Qwen3 RoomPlan is a Stage-1 multi-view vision–language model for 3D indoor scene understanding. It combines a frozen VGGT visual backbone with a Perceiver projector and Qwen3-4B to answer questions about 3D scenes (ScanQA/SQA3D) using a token-level visual injection mechanism.

Stage 2 (RoomPlan action JSON prediction) is **future work** and intentionally out of scope for this README.

## Overview

High-level Stage-1 pipeline:

```text
Multi-view Images
    └─▶ VGGT Aggregator (frozen)
             └─▶ Perceiver Projector
                      └─▶ Visual tokens
                               └─▶ Injected at <image> in Qwen3-4B
                                        └─▶ Text answer
```

- VGGT: multi-view visual aggregator, kept frozen in Stage 1.
- Perceiver projector: maps VGGT features to a fixed-length sequence of visual tokens in Qwen3’s hidden dimension.
- Qwen3-4B: causal LM, partially frozen + LoRA, trained to produce short answers.
- Token-level injection: visual tokens overwrite embeddings at the `<image>` placeholder in the prompt; loss is computed only on answer tokens.

For a deeper architectural description (including label masking and injection span rules), see:

- `docs/model/architecture.md`

## Supported Stage-1 features

- Training on ScanQA + SQA3D JSONL shards using:
  - Multi-view images (currently often single “bird-view” crops).
  - Textual questions and short answers.
- Visual token injection at a dedicated `<image>` placeholder in the prompt.
- Stage-1 QA inference with a short-answer constraint (for better exact-match behavior).
- Configurable training via YAML:
  - Model name, Perceiver config, number of visual/geom tokens.
  - Dataset mix, views, sequence length.
  - Optimizer and scheduler parameters.
- Canonical entry points:
  - `train_stage1.sh` / `python -m vggt_qwen3.train.stage1`
  - `infer_stage1.sh` / `python -m vggt_qwen3.inference.qa_inference`

Stage 2 / RoomPlan JSON actions are not documented here beyond brief “future work” mentions.

## Installation

### Python and environment

- Recommended Python: **3.9+**.
- Recommended: conda environment.

Using conda (general GPUs like V100/A100/H100):

```bash
conda env create -f env/environment.yml
conda activate roomplan
```

Using venv + pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

#### NVIDIA B200 / Blackwell GPUs (sm_100)

If you are running on NVIDIA B200 (compute capability `sm_100`), the default `env/environment.yml` (PyTorch 2.4.0) may not support this architecture. In that case, create a dedicated environment using a newer PyTorch build that includes B200 support:

```bash
# From the repo root, on a node with CUDA 12.9+ available
# (adjust the module name to your cluster)
module load cuda/12.9.1   # or similar

conda create -n roomplan-b200 python=3.10 -y
conda activate roomplan-b200

# Install a PyTorch build that supports NVIDIA B200 (sm_100)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install remaining dependencies without overriding the CUDA-enabled PyTorch
pip install -r requirements.txt --no-deps

# Install this repo and VGGT in editable mode
pip install -e .
pip install -e third_party/vggt

# (Optional) logging / training extras
pip install --no-build-isolation deepspeed tensorboard
```

You can verify that your PyTorch build sees the B200 architecture via:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.get_arch_list())"
```

Look for `"sm_100"` in the printed list. Once this environment is active, use the standard training commands (for example, single-GPU DeepSpeed):

```bash
ACCELERATE_CONFIG=configs/accelerate_1gpu.yaml ./train_stage1.sh
```

### Third-party dependencies

- VGGT:
  - Place the VGGT checkpoint at `third_party/vggt/vggt_1B_commercial.pt`.
  - (Optional) install VGGT in editable mode:
    ```bash
    pip install -e third_party/vggt
    ```
- Hugging Face cache (recommended on shared filesystems):
  ```bash
  export HF_HOME="$PWD/.cache/huggingface"
  export TRANSFORMERS_CACHE="$HF_HOME"
  export HF_DATASETS_CACHE="$HF_HOME"
  ```

Verify CUDA/GPU:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Data

Expected on-disk layout for Stage 1:

```text
data/
  raw/
    scannet/      # optional; only needed to regenerate from raw
    scanqa/
    sqa3d/
  processed/
    scanqa/
      train_split.jsonl
      test_split.jsonl
    sqa3d/
      train_split.jsonl
      test_split.jsonl
```

Each JSONL line in `train_split.jsonl` / `test_split.jsonl` is a record with:

- `images`: list of image paths (relative or absolute).
- `geom_token`: `null` for current Stage-1 shards (geometry bypassed).
- `task`: `"scanqa"` or `"sqa3d"`.
- `question`: question string.
- `answer`: short textual answer.
- `scene_id`: scene identifier.

The Stage-1 config `configs/stage1_3d.yaml` mixes ScanQA and SQA3D (default 0.7 / 0.3) via `MultiSourceDataset`.

Data loading and collation:

- `vggt_qwen3.dataio.dataset_builder.MultiViewJsonDataset`:
  - Loads JSON/JSONL, normalizes fields, caps to `num_views`, and resolves image paths.
- `vggt_qwen3.dataio.collate_multiview.MultiViewCollator`:
  - Resizes/crops images.
  - Builds prompts like:
    ```text
    "{question}
<image>
"
    ```
  - Inserts a reserved padding span right after `<image>` sized to `num_vis_tokens + geom_tokens`, with labels set to `-100`.
  - Appends answer tokens, labels only the answer region (labels != -100).

This guarantees a safe injection span for the visual tokens (no overlap with answer tokens).

## Training Stage 1

The Stage-1 trainer lives at:

- `src/vggt_qwen3/train/stage1.py`
- Config: `configs/stage1_3d.yaml`

### Single-GPU

Canonical command:

```bash
accelerate launch   --config_file configs/accelerate_single_gpu.yaml   -m vggt_qwen3.train.stage1   --config configs/stage1_3d.yaml   --output_dir ckpts/stage1_3d
```

Or, via the wrapper:

```bash
./train_stage1.sh
```

You can override key parameters via environment variables:

- `CONFIG` – path to a Stage-1 config (default `configs/stage1_3d.yaml`).
- `OUTPUT_DIR` – checkpoint directory (default `ckpts/stage1_3d`).
- `ACCELERATE_CONFIG` – Accelerate config (default `configs/accelerate_single_gpu.yaml`).

### Multi-GPU (data parallel)

Use one of the provided Accelerate configs (e.g., `configs/accelerate_4gpu.yaml`):

```bash
ACCELERATE_CONFIG=configs/accelerate_4gpu.yaml ./train_stage1.sh
```

On Slurm/HiPerGator, wrap this command in an `sbatch` script that requests the appropriate number of GPUs and nodes. See the existing `scripts/slurm/` files and `docs/SLURM_TRAINING_GUIDE.md` for patterns (note: Stage 2 scripts are future work).

#### 4-GPU `torchrun` example

If you prefer launching the Stage-1 harness directly with `torchrun` (using Accelerate only as a library), you can run:

```bash
torchrun --standalone --nproc_per_node=4 -m vggt_qwen3.train.stage1 \
  --config configs/stage1_3d.yaml \
  --output_dir ckpts/stage1_3d \
  --max_steps 30000
```

This uses standard DDP instead of the `accelerate launch` CLI, but the training loop and logging behavior are identical.

### Optional: DeepSpeed ZeRO-3

If you want DeepSpeed ZeRO-3, pass a DeepSpeed config via `DEEPSPEED_CONFIG`:

```bash
DEEPSPEED_CONFIG=configs/deepspeed_zero3.json ACCELERATE_CONFIG=configs/accelerate_4gpu.yaml ./train_stage1.sh
```

Internally, `vggt_qwen3.train.stage1`:

- Builds the tokenizer and ensures a dedicated `<image>` token.
- Creates datasets and `MultiViewCollator`.
- Builds `VGGTQwen3VLM` with:
  - Frozen VGGT.
  - Perceiver projector.
  - Qwen3-4B with partial freezing and LoRA.
- Splits parameters into base vs projector/geom parameter groups with separate LRs.
- Uses a cosine LR schedule with warmup.
- Writes reproducibility metadata under the chosen `--output_dir`:
  - `logs/events.jsonl` – structured training metrics (loss, LR, speed, ETA).
  - `meta/args.yaml` – CLI arguments for the run.
  - `meta/env.txt` – Python/Torch/CUDA summary and `pip freeze`.
  - `meta/git.txt` – git commit and status (when available).

## Inference / Evaluation (Stage 1)

Stage-1 QA inference uses:

- `vggt_qwen3.inference.qa_inference`
- The shell wrapper `infer_stage1.sh`

### ScanQA only

Wrapper (recommended):

```bash
CONFIG=configs/stage1_3d.yaml \
CHECKPOINT_DIR=ckpts/stage1_3d \
DATASET=scanqa \
./infer_stage1.sh
```

Equivalent Python command:

```bash
python -m vggt_qwen3.inference.qa_inference \
  --config configs/stage1_3d.yaml \
  --checkpoint_dir ckpts/stage1_3d \
  --dataset scanqa \
  --num_samples 200 \
  --max_new_tokens 32
```

Defaults:

- Glob: `data/processed/scanqa/test_split.jsonl`
- Predictions: `outputs/qa/scanqa/scanqa_predictions_test.jsonl`
- Events log: `outputs/qa/scanqa/scanqa_predictions_test.jsonl.events.jsonl`

### SQA3D only

Wrapper:

```bash
CONFIG=configs/stage1_3d.yaml \
CHECKPOINT_DIR=ckpts/stage1_3d \
DATASET=sqa3d \
./infer_stage1.sh
```

Equivalent Python:

```bash
python -m vggt_qwen3.inference.qa_inference \
  --config configs/stage1_3d.yaml \
  --checkpoint_dir ckpts/stage1_3d \
  --dataset sqa3d \
  --num_samples 200 \
  --max_new_tokens 32
```

Defaults:

- Glob: `data/processed/sqa3d/test_split.jsonl`
- Predictions: `outputs/qa/sqa3d/sqa3d_predictions_test.jsonl`
- Events log: `outputs/qa/sqa3d/sqa3d_predictions_test.jsonl.events.jsonl`

### Combined ScanQA + SQA3D

Run both datasets in a single call:

```bash
python -m vggt_qwen3.inference.qa_inference \
  --config configs/stage1_3d.yaml \
  --checkpoint_dir ckpts/stage1_3d \
  --dataset scanqa+sqa3d \
  --num_samples 200 \
  --max_new_tokens 32
```

This produces both prediction files listed above (one per dataset) plus per-dataset `.events.jsonl` logs.

### Short-answer constraint & prompts

The QA inference script builds prompts using the Qwen3 chat template with a short-answer hint:

```text
ScanQA (no situation text):
  {question}
  <image>
  Answer with a short phrase only.

SQA3D (includes “situation” when present):
  Situation: {situation}
  Question: {question}
  <image>
  Answer with a short phrase only.
```

Generation settings:

- `temperature = 0.0`
- `top_p = 1.0`
- `num_beams = 1`
- Small `max_new_tokens` (default 32)

This combination encourages deterministic, concise answers suitable for exact-match evaluation.

### Outputs, logs, and metrics

- Per-sample predictions are written as JSONL records with:
  - `question`, `prediction`, `reference`, `scene_id`, `task`, etc.
- A companion `*.events.jsonl` file logs:
  - Dataset name, number of samples, number of generations, and how many predictions were non-empty.
  - One event per sample plus a final summary event.

### Published example results

This repository ships small, versioned inference artifacts for reproducibility:

- `results/stage1_scanqa/`
  - `predictions_test.jsonl` – full ScanQA predictions for a Stage-1 debug checkpoint.
  - `predictions_test.sample.jsonl` – first 200 predictions for quick inspection.
  - `metrics.json` – basic metrics (`num_examples`, `num_non_empty_predictions`, `exact_match`).
  - `command.txt` – exact commands + git commit hash used to generate the run.

- `results/stage1_sqa3d/`
  - `predictions_test.jsonl` – full SQA3D predictions.
  - `predictions_test.sample.jsonl` – first 200 predictions.
  - `metrics.json` – same metric schema as above.
  - `command.txt` – commands + commit hash.

To regenerate these artifacts on a new machine:

```bash
# ScanQA
CHECKPOINT_DIR=ckpts/stage1_3d_debug \
OUTPUT_JSONL=outputs/qa/scanqa_predictions_test.jsonl \
scripts/infer_stage1_scanqa.sh

scripts/eval_scanqa.sh \
  --predictions outputs/qa/scanqa_predictions_test.jsonl \
  --output_dir results/stage1_scanqa

# SQA3D
CHECKPOINT_DIR=ckpts/stage1_3d_debug \
OUTPUT_JSONL=outputs/qa/sqa3d_predictions_test.jsonl \
DATASET=sqa3d \
./infer_stage1.sh

scripts/eval_scanqa.sh \
  --predictions outputs/qa/sqa3d_predictions_test.jsonl \
  --output_dir results/stage1_sqa3d
```

For more detailed evaluation beyond exact match, see `docs/evaluation.md`.

## Repository layout

Key paths:

- `src/vggt_qwen3/`
  - `dataio/` – dataset and collator.
  - `models/` – VGGTQwen3 wrapper and Perceiver projector.
  - `train/stage1.py` – Stage-1 training harness.
  - `inference/qa_inference.py` – Stage-1 QA inference.
  - `eval/` – lightweight evaluation helpers.
- `configs/`
  - `stage1_3d.yaml` – Stage-1 config.
  - `perceiver_small.yaml` – projector config.
  - `accelerate_*.yaml`, `deepspeed_zero3.json` – launcher configs.
- `scripts/`
  - `eval_baseline.sh`, `eval_baseline_quick.py` – evaluation helpers.
  - `prep/` – data preparation scripts (ScanQA/SQA3D/ARKit).
  - `slurm/` – Slurm examples (Stage 2/3 future work).
- `train_stage1.sh`, `infer_stage1.sh` – canonical CLI wrappers.
- `docs/`
  - `index.md` – docs landing page.
  - `stage1_quickstart.md` – extended Stage-1 guide.
  - `model/architecture.md` – model internals.
  - `dev/debug_history.md` – maintainer debug notes.
  - `dev/repo_structure.md` – repo organization rationale.

## Troubleshooting

Common issues and tips:

- **Stage-1 inference returns empty predictions**
  - Symptom: every line in the JSONL file has an empty `prediction` string, or the script raises a `RuntimeError` stating that no non-empty predictions were produced.
  - Checklist:
    - Verify that test splits exist and are non-empty:
      - `data/processed/scanqa/test_split.jsonl`
      - `data/processed/sqa3d/test_split.jsonl`
    - Re-run with debug mode to inspect prompts and token IDs:
      - `python -m vggt_qwen3.inference.qa_inference --dataset scanqa --debug --debug_max_samples 4 ...`
    - Inspect the companion `*.events.jsonl` file to confirm `total_non_empty_predictions > 0`.
    - If `total_generations_attempted > 0` but all decoded strings are empty, suspect a tokenizer decode issue (special-tokens-only outputs); the debug logs print both `skip_special_tokens=True` and `False` decodes and raw token IDs to help diagnose.

- **Missing `<image>` token**
  - Symptom: `MultiViewCollator` or model raises an error about missing `<image>` or uses `unk` ID.
  - Fix: use the provided tokenizer builders in `train.stage1` and `qa_inference`; avoid custom tokenization that bypasses them.

- **Injection span overlapping answer tokens**
  - Stage-1 collator inserts a reserved padding span after `<image>` sized to `num_vis_tokens + geom_tokens` and labels it as `-100`. If you change `num_vis_tokens` or `geom_tokens`, keep `data.max_length` sufficiently large; `MultiViewCollator` enforces a sanity check and will raise if `max_length` is too small.

- **dtype or device mismatches**
  - On GPU, Stage 1 uses bf16 by default; ensure your hardware supports it.
  - On CPU, you may want to run with float32 (adjust `model.dtype` in the config or via the wrapper if needed).

- **Cleaning artifacts (checkpoints, logs, outputs)**
  - To safely remove training and inference artifacts without touching code or data:
    ```bash
    # Dry run – prints what would be removed
    python scripts/clean_artifacts.py

    # Actual deletion (requires confirmation flag or env var)
    python scripts/clean_artifacts.py --yes
    # or
    CONFIRM=1 python scripts/clean_artifacts.py
    ```
  - This script only targets `ckpts/`, `outputs/`, `logs/`, `runs/`, `results/`, and `pytorchdist_*.out` files under the repo root.

For a deeper engineering/debugging history (including injection fixes), see:

- `docs/dev/debug_history.md`

## Citing

If you use this codebase or model in your research, please cite it using the metadata in `CITATION.cff`.

## License

- This repository is licensed under the **Apache License 2.0** (see `LICENSE`).
- VGGT, Qwen3, and any other third-party components in `third_party/` are subject to their respective licenses. Please review those files before redistribution or commercial use.
