## Stage‑1 QA Inference

The main entrypoint for ScanQA/SQA3D inference is:

- `src/vggt_qwen3/inference/qa_inference.py`
- Shell wrappers:
  - `infer_stage1.sh`
  - `src/scripts/infer_stage1_scanqa.sh`

### Quickstart: ScanQA

Assuming you have a trained Stage‑1 checkpoint (e.g. `ckpts/stage1_3d_debug`)
and processed ScanQA test data:

```bash
cd /path/to/vggt-qwen3-roomplan

CHECKPOINT_DIR=ckpts/stage1_3d_debug \
OUTPUT_JSONL=outputs/qa/scanqa/scanqa_predictions_test.jsonl \
src/scripts/infer_stage1_scanqa.sh
```

This will:

- Load Qwen3‑4B and the VGGT backbone.
- Convert the DeepSpeed ZeRO checkpoint to a single fp32 state dict.
- Run QA on all ScanQA test QA pairs.
- Write predictions to `outputs/qa/scanqa/scanqa_predictions_test.jsonl`.

You will also see a one‑line checkpoint summary:

```text
[model] base=Qwen/Qwen3-4B-Instruct-2507 | checkpoint=ckpts/stage1_3d_debug | missing_keys=0 unexpected_keys=0 | trainable_params=.../...
```

### Inference flags

Key CLI options (see `parse_args()` in `qa_inference.py`):

- `--config configs/stage1_3d.yaml` — model configuration.
- `--dataset {scanqa,sqa3d,scanqa+sqa3d}` — which dataset to run.
- `--glob` — override default JSONL glob.
- `--checkpoint_dir` — Stage‑1 checkpoint root (e.g. `ckpts/stage1_3d_debug`).
- `--allow_base_fallback` — allow running with base weights if checkpoint is missing.
- `--num_samples` — optional cap on the number of records to evaluate (0 = all).
- `--max_new_tokens` — generation length.
- `--output_jsonl` — where to write predictions.
- `--short_answer_only` — postprocess outputs to enforce short answers.
- `--do_sample`, `--temperature`, `--top_p`, `--top_k` — enable sampling.
- `--debug` — CPU debug mode with a tiny subset.
- `--debug_one INDEX` — detailed multimodal debug dump for a specific sample.
- `--verbose_samples N` — print at most `N` predictions to stdout.
- `--log_level {debug,info,warning,error}` — standard logging level.

Predictions JSONL schema:

- `index` — 0‑based index in the sampled subset.
- `task` — `"scanqa"` or `"sqa3d"`.
- `scene_id` — original scene identifier.
- `question_id` — unique question id from the dataset.
- `question` — input question text.
- `prediction` — model answer.
- `reference` — reference answer from the dataset (if available).
