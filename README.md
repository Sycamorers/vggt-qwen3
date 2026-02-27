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

Using conda:

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

## Inference / Evaluation (Stage 1)

Stage-1 QA inference uses:

- `vggt_qwen3.inference.qa_inference`

### Canonical inference command

```bash
python -m vggt_qwen3.inference.qa_inference   --config configs/stage1_3d.yaml   --checkpoint_dir ckpts/stage1_3d   --glob "data/processed/scanqa/test_split.jsonl"   --num_samples 200   --max_new_tokens 32   --output_jsonl outputs/qa/scanqa_predictions_test.jsonl
```

Or, via the wrapper:

```bash
./infer_stage1.sh
```

You can override:

- `CONFIG`, `CHECKPOINT_DIR`, `GLOB`, `NUM_SAMPLES`, `MAX_NEW_TOKENS`, `OUTPUT_JSONL`, `DEVICE`.

### Short-answer constraint

The QA inference script builds prompts using the Qwen3 chat template with a short-answer hint:

```text
{question}
<image>
Answer with a short phrase only.
```

Generation settings:

- `temperature = 0.0`
- `top_p = 1.0`
- `num_beams = 1`
- Small `max_new_tokens` (default 32)

This combination encourages deterministic, concise answers suitable for exact-match evaluation.

### Outputs and evaluation

The inference module writes a JSONL file with per-sample records:

- `question`, `prediction`, `reference`, `scene_id`, `task`, etc.

For quick metrics, you can use:

- `scripts/eval_baseline.sh` (shell wrapper).
- `scripts/eval_baseline_quick.py` (Python script).

These scripts compute exact-match and partial-match accuracy on ScanQA/SQA3D and optionally ARKit-style JSON outputs (Stage 2/3-style, currently for reference only).

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

- **Missing `<image>` token**
  - Symptom: `MultiViewCollator` or model raises an error about missing `<image>` or uses `unk` ID.
  - Fix: use the provided tokenizer builders in `train.stage1` and `qa_inference`; avoid custom tokenization that bypasses them.

- **Injection span overlapping answer tokens**
  - Stage-1 collator inserts a reserved padding span after `<image>` sized to `num_vis_tokens + geom_tokens` and labels it as `-100`. If you change `num_vis_tokens` or `geom_tokens`, keep `data.max_length` sufficiently large; `MultiViewCollator` enforces a sanity check and will raise if `max_length` is too small.

- **dtype or device mismatches**
  - On GPU, Stage 1 uses bf16 by default; ensure your hardware supports it.
  - On CPU, you may want to run with float32 (adjust `model.dtype` in the config or via the wrapper if needed).

For a deeper engineering/debugging history (including injection fixes), see:

- `docs/dev/debug_history.md`

## Citing

If you use this codebase or model in your research, please cite it using the metadata in `CITATION.cff`.

## License

- This repository is licensed under the **Apache License 2.0** (see `LICENSE`).
- VGGT, Qwen3, and any other third-party components in `third_party/` are subject to their respective licenses. Please review those files before redistribution or commercial use.
