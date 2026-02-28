## Data Overview

Stage‑1 training and inference use two 3D QA datasets:

- **ScanQA** — questions and short answers about ScanNet scenes.
- **SQA3D** — situation‑aware 3D QA examples.

The repository expects preprocessed JSONL files under `data/processed`:

```text
data/processed/scanqa/train_split.jsonl
data/processed/scanqa/test_split.jsonl
data/processed/sqa3d/train_split.jsonl
data/processed/sqa3d/test_split.jsonl
```

Each JSONL record has at least:

- `images`: list of relative paths to rendered multi‑view RGB frames.
- `task`: `"scanqa"` or `"sqa3d"`.
- `question`: natural‑language question.
- `answer`: short textual answer.
- `scene_id`: dataset‑specific scene identifier.
- `question_id`: unique question identifier (used for evaluation and logging).
- Optional: `geom_token`, `situation`, etc.

### Preprocessing scripts

Use the `scripts/prep` utilities to build the processed datasets from raw
ARKitScenes/ScanQA sources:

- `scripts/prep/prepare_scanqa.py`
- `scripts/prep/prepare_arkit_from_3dod.py`
- `scripts/prep/split_train_test.py`

For a quick small‑scale run, you can use:

- `scripts/prep/make_toy_dataset.py` — creates a tiny toy dataset for smoke tests.

See `docs/FILE_GUIDE.md` and `docs/COMPLETE_TRAINING_GUIDE.md` for
step‑by‑step preprocessing instructions.

