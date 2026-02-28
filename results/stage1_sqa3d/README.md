## Stage‑1 SQA3D Results

This folder contains example inference and evaluation artifacts for the
Stage‑1 VGGT‑Qwen3 model on the SQA3D test split.

### Contents

- `predictions_test.jsonl`  
  Full SQA3D predictions JSONL for the current Stage‑1 checkpoint.

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
# 1) Run SQA3D inference
CHECKPOINT_DIR=ckpts/stage1_3d_debug \
OUTPUT_JSONL=outputs/qa/sqa3d_predictions_test.jsonl \
DATASET=sqa3d \
./infer_stage1.sh

# 2) Evaluate and write metrics
scripts/eval_scanqa.sh \
  --predictions outputs/qa/sqa3d_predictions_test.jsonl \
  --output_dir results/stage1_sqa3d
```

You can adjust `CHECKPOINT_DIR` to point to any Stage‑1 checkpoint root.

