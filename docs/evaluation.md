## Evaluation

Stage‑1 currently provides a simple evaluation pipeline for **ScanQA**:

- `scripts/eval_scanqa.py` — Python evaluation script.
- `scripts/eval_scanqa.sh` — thin shell wrapper.

### Running evaluation

After running inference:

```bash
cd /path/to/vggt-qwen3-roomplan

# Evaluate ScanQA predictions
scripts/eval_scanqa.sh \
  --predictions outputs/qa/scanqa_predictions_test.jsonl \
  --output_dir results/stage1_scanqa
```

This writes:

- `results/stage1_scanqa/metrics.json`
  - `num_examples`
  - `num_non_empty_predictions`
  - `exact_match` (normalized string equality between `prediction` and `reference`)

Exact match uses a simple normalization:

- Lowercasing.
- Stripping leading/trailing spaces.
- Collapsing internal whitespace.

For richer evaluation (e.g., token‑level F1 scores, category‑wise analysis),
you can extend `scripts/eval_scanqa.py` or add new scripts under `scripts/`
and document them here.

