## Stage‑1 ScanQA Results

This folder contains example inference and evaluation artifacts for the
Stage‑1 VGGT‑Qwen3 model on the ScanQA test split.

### Contents

- `predictions_test.sample.jsonl`  
  Small sample of predictions (first N examples) for quick inspection.

- `metrics.json`  
  Basic evaluation metrics produced by `scripts/eval_scanqa.py`:
  - `num_examples`
  - `num_non_empty_predictions`
  - `exact_match`

- `command.txt`  
  Exact command used to generate the predictions and metrics, plus the git
  commit hash and basic environment information.

### Regenerating results

From the repository root:

```bash
# 1) Run ScanQA inference
CHECKPOINT_DIR=ckpts/stage1_3d_debug \
OUTPUT_JSONL=outputs/qa/scanqa_predictions_test.jsonl \
scripts/infer_stage1_scanqa.sh

# 2) Evaluate and write metrics
scripts/eval_scanqa.sh \
  --predictions outputs/qa/scanqa_predictions_test.jsonl \
  --output_dir results/stage1_scanqa
```

You can adjust `CHECKPOINT_DIR` to point to any Stage‑1 checkpoint root.

