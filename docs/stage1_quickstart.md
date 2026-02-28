# Stage-1 Quickstart (ScanQA + SQA3D)

This guide walks through Stage-1 training and inference for VGGT-Qwen3 on ScanQA and SQA3D. It assumes you have access to GPUs for full training, but most steps (data checks, config inspection) can be done on CPU.

Stage 2 (RoomPlan action JSON prediction) is **future work** and not covered here.

## 1. Environment setup

### Python and environment

- Python: **3.9+** (tested with 3.9)

Create and activate a conda environment:

```bash
conda env create -f env/environment.yml
conda activate roomplan
```

Alternatively, for a minimal pip-only setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Dependencies and external repos

- Qwen3 model and tokenizer will be downloaded on first use:
  - Default: `Qwen/Qwen3-4B-Instruct-2507`
  - Cache directory can be controlled via `HF_HOME`, e.g.:
    ```bash
    export HF_HOME="$PWD/.cache/huggingface"
    ```
- VGGT backbone:
  - Place the VGGT checkpoint under `third_party/vggt/vggt_1B_commercial.pt`.
  - Install VGGT in editable mode if you plan to inspect or modify it:
    ```bash
    pip install -e third_party/vggt
    ```

## 2. Data layout (Stage 1)

Stage 1 expects preprocessed ScanQA and SQA3D data in JSONL format:

- `data/processed/scanqa/train_split.jsonl`
- `data/processed/scanqa/test_split.jsonl`
- `data/processed/sqa3d/train_split.jsonl`
- `data/processed/sqa3d/test_split.jsonl`

Each JSONL line is a JSON object with at least:

- `images`: list of image paths (relative or absolute)
- `question`: question text
- `answer`: short textual answer
- `scene_id`: scene identifier
- `geom_token`: `null` for Stage 1 (geometry bypassed)

You can regenerate or customize these files via the prep scripts under `scripts/prep/` (e.g., `prepare_scanqa.py`), but the default Stage-1 config assumes the layout above.

## 3. Configs

Primary Stage-1 config:

- `configs/stage1_3d.yaml`

Key fields:

- `model.name_or_path`: HF identifier for the Qwen3 model.
- `model.tokenizer_path`: optional tokenizer override (usually same as model).
- `model.vision_backbone`: path to VGGT repo/checkpoint (e.g., `third_party/vggt`).
- `model.num_vis_tokens`: number of visual tokens injected per sample.
- `model.geom_tokens`: geometry token count (Stage 1 uses `geom_token: null`, so effectively only visual tokens are injected).
- `data.datasets`: glob patterns for ScanQA and SQA3D JSONL files.
- `data.num_views`: number of views per sample (Stage 1 can run single-view “bird” crops).
- `data.max_length`: maximum text sequence length (must be >= `num_vis_tokens + geom_tokens + margin`).
- `train.*`: learning rate, batch size, warmup, max_steps, gradient accumulation, etc.

You can create new configs (e.g., different view counts or image size) by copying `configs/stage1_3d.yaml` and adjusting fields.

## 4. Training Stage 1

### Single-GPU (Accelerate)

For a single GPU, use the provided accelerate config:

```bash
accelerate launch \
  --config_file configs/accelerate_single_gpu.yaml \
  -m vggt_qwen3.train.stage1 \
  --config configs/stage1_3d.yaml \
  --output_dir ckpts/stage1_3d
```

This is equivalent to:

```bash
./train_stage1.sh
```

(`train_stage1.sh` is a thin wrapper around the command above; you can override `CONFIG`, `OUTPUT_DIR`, and `ACCELERATE_CONFIG` via environment variables.)

### Multi-GPU (data parallel with Accelerate)

To use multiple GPUs, choose one of the provided accelerate configs:

- `configs/accelerate_4gpu.yaml`
- `configs/accelerate_8gpu.yaml`
- etc.

Example:

```bash
ACCELERATE_CONFIG=configs/accelerate_4gpu.yaml \
./train_stage1.sh
```

On HiPerGator or other Slurm clusters, you can wrap this inside an `sbatch` script that reserves the appropriate number of GPUs and nodes.

### Optional DeepSpeed / ZeRO

Stage 1 supports ZeRO-3 via Accelerate’s DeepSpeed integration. Pass a DeepSpeed config JSON (e.g., `configs/deepspeed_zero3.json`) to the Stage-1 trainer:

```bash
accelerate launch \
  --config_file configs/accelerate_4gpu.yaml \
  -m vggt_qwen3.train.stage1 \
  --config configs/stage1_3d.yaml \
  --output_dir ckpts/stage1_3d_zero3 \
  --deepspeed configs/deepspeed_zero3.json
```

## 5. Inference and evaluation (Stage 1 QA)

Stage-1 QA inference uses the trained VGGT-Qwen3 model to answer ScanQA/SQA3D questions with short textual answers.

### Canonical command

```bash
python -m vggt_qwen3.inference.qa_inference \
  --config configs/stage1_3d.yaml \
  --checkpoint_dir ckpts/stage1_3d \
  --dataset scanqa \
  --num_samples 200 \
  --max_new_tokens 32
```

Or, using the thin wrapper:

```bash
./infer_stage1.sh
```

By default, the Stage-1 inference script:

- Builds prompts of the form:
  - ScanQA: `"{question}\n<image>\nAnswer with a short phrase only."`
  - SQA3D: `"Situation: {situation}\nQuestion: {question}\n<image>\nAnswer with a short phrase only."`
- Applies the Qwen3 chat template (via `tokenizer.apply_chat_template`).
- Injects visual tokens at the `<image>` placeholder.
- Generates with:
  - `temperature=0.0`, `top_p=1.0`, `num_beams=1`
  - `max_new_tokens` configurable (default 32)

### Short-answer constraint

The prompt suffix “Answer with a short phrase only.” plus the deterministic generation settings strongly encourage short answers suitable for exact-match metrics. If your task requires longer explanations, you can remove or modify this suffix in `vggt_qwen3/inference/qa_inference.py`.

### Outputs

`qa_inference` writes a JSONL file with one record per sample, including:

- `question`
- `prediction`
- `reference`
- `scene_id`
- `task`

You can compute exact-match metrics using:

- `scripts/eval_baseline.sh` (shell wrapper).
- `scripts/eval_baseline_quick.py` (Python script using `qa_inference` under the hood).

## 6. Repository layout (Stage 1)

Key directories and files:

- `src/vggt_qwen3/` – Python package
  - `dataio/` – dataset and collator utilities.
  - `models/` – VGGT-Qwen3 wrapper and Perceiver projector.
  - `train/stage1.py` – Stage-1 training harness (Accelerate + DeepSpeed).
  - `inference/qa_inference.py` – Stage-1 QA inference.
  - `eval/` – small helpers for evaluation.
- `configs/`
  - `stage1_3d.yaml` – main Stage-1 config.
  - `accelerate_*.yaml` – Accelerate configs for 1–8 GPUs.
  - `deepspeed_zero3.json` – optional ZeRO-3 config.
- `scripts/`
  - `eval_baseline.sh`, `eval_baseline_quick.py` – baseline evaluation helpers.
  - `prep/` – data preparation utilities (ScanQA/SQA3D/ARKit).
- `train_stage1.sh`, `infer_stage1.sh` – canonical CLI wrappers.
- `docs/`
  - `index.md` – documentation landing page.
  - `model/architecture.md` – model details.
  - `dev/repo_structure.md` – repo layout and conventions.

## 7. Troubleshooting and tips

Common issues:

- **Missing `<image>` token in tokenizer**
  - Symptom: runtime error from `MultiViewCollator` complaining about missing `<image>` token; or debug guards raising `VGGT_DEBUG_INJECT` errors.
  - Fix: use the provided tokenizer builders (`build_tokenizer` functions in training/inference) and avoid manually altering special tokens.

- **Injection span overlapping answer tokens**
  - The Stage-1 collator explicitly inserts a reserved `[PAD]` span immediately after `<image>`, sized to `num_vis_tokens + geom_tokens`, with labels set to `-100`. This ensures the injection span never touches answer tokens. If you change `num_vis_tokens` or `geom_tokens`, keep `data.max_length` large enough (see `collate_multiview.py` for the safety check).

- **dtype or device mismatches**
  - On CPU, prefer running with `dtype="float32"` in the config.
  - On GPU, Stage 1 uses bf16 by default; ensure your hardware and drivers support it.

For deeper debugging and the history of injection-related fixes, see:

- `docs/dev/debug_history.md`
