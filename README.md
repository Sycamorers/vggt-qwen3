# VGGT-Qwen3 RoomPlan

This reposi## 3) Data

### 3.1 Current Dataset Organization (Post-Reorganization)

**Training Datasets** (with train/test splits):
- **ScanQA**: 21,161 train / 3,567 test samples
  - `data/processed/scanqa/train_split.jsonl` – ---

## 9) What's Done vs. To-Do

**Complete## 10) Extending## 11) Repository Map (active pieces)## 12) Quickstart Che## 13) ARKit / RoomPlan Evaluation (Stage 1 weights)

Once you have:
- Completed Stage ### 13.4 Evaluation behavior (current) and limitations

- Model repeats or elaborates on the instruction; does not emit the structured `action_json`.
- Exact-match metric is 0.0 because predictions are free-form text while references are JSON.
- Training support for `action_json` is absent in the current trainer (targets are dicts; no structured loss head).
- Baseline ARKit results (Section 7) show 0% match due to these limitations.

### 13.5 Future work (optional)

- Add a JSON-only target/head for `action_json`, or serialize targets to text for supervision.
- Constrain prompts to "respond only with JSON".
- Convert and load full fp32 checkpoints (all shards) when available for more complete evaluation.or are using a provided Stage 1 checkpoint), and
- Generated or reused ARKit synthetic data under `data/processed/arkit_synth/` (the repo already has `train.json`),

you can run ARKit evaluation to test the model's ability to understand RoomPlan-style spatial reasoning tasks.

**Note**: This is primarily for evaluation and testing purposes. The model generates free-form text responses rather than structured JSON actions (see limitations below).

### 13.1 Prepare or reuse ARKit synthetic data `conda activate roomplan` (after creating env)  
2) Place checkpoints: `third_party/vggt/vggt_1B_commercial.pt` and cache `Qwen/Qwen3-4B-Instruct-2507`  
3) Verify data exists: `ls data/processed/scanqa data/processed/sqa3d` (should see `train_split.jsonl` and `test_split.jsonl` in each)  
4) Run: `./train_fixed.sh debug 1` (sanity) → then `./train_fixed.sh full 4` (or Slurm via `run.sh`)  
5) Monitor: `tensorboard --logdir ckpts/stage1_3d/logs` or `python scripts/monitor_training.py --logdir ckpts/stage1_3d/logs/roomplan --watch`  
6) Evaluate: Run evaluation scripts on test splits after training

Stage 2 ARKit training is not supported yet. For ARKit evaluation, see Section 13.

---

## 13) ARKit / RoomPlan Evaluation (Stage 1 weights)/                # Stage configs (stage1_3d active), projector, DeepSpeed
data/processed/         # scanqa/{train,test}_split.jsonl, sqa3d/{train,test}_split.jsonl
data/raw/               # placeholders + ARKit README
outputs/qa/baseline_eval/ # Baseline evaluation results
src/dataio/             # dataset builder + multi-view collator
src/models/             # VGGT-Qwen3 wrapper, Perceiver projector
src/train/train_sft.py  # Accelerate/DeepSpeed trainer
src/eval/               # Evaluation scripts
train_fixed.sh          # Hardened launcher (defaults to stage1_3d)
run.sh                  # Slurm example invoking train_fixed.sh
logs/, ckpts/           # Training outputs
docs/                   # Guides (complete training, quickstart, monitoring, Slurm, file map)
scripts/prep/           # Data preparation scripts
scripts/eval_baseline.py # Baseline evaluation script
```
More file descriptions: `docs/FILE_GUIDE.md`.

---

## 12) Quickstart Checklistdd geometry + multi-view:** Rebuild ScanQA/SQA3D with poses/depth using `scripts/prep/prepare_scanqa.py --dataset scanqa|sqa3d --num-views 8 --output data/processed/{scanqa,sqa3d}/train_split.jsonl`. Remember to also generate corresponding test splits.
- **Generate RoomPlan data (for evaluation or future training):** Build `data/processed/arkit_synth/train.json` with `scripts/prep/prepare_arkit_from_3dod.py` (works with the current 3DOD layout). If you have full planes/cameras metadata, `scripts/prep/synth_roomplan_instructions.py` is also supported. Training with `configs/stage2_arkit.yaml` is not yet wired.
- **Implement Stage 2 training:** To support structured `action_json` targets, modify the trainer to handle dictionary outputs or serialize actions to text format.
- **Run full evaluation:** Evaluate trained models on complete test sets using evaluation scripts in `scripts/`.

Stage‑wise training strategies (Stage 1 → Stage 2) are described in more detail in `docs/COMPLETE_TRAINING_GUIDE.md`.

---

## 11) Repository Map (active pieces) Data reorganized with proper train/test splits
  - ScanQA: 21,161 train / 3,567 test samples
  - SQA3D: 22,959 train / 3,664 test samples
  - Total: ~44,120 train / ~7,231 test samples
- ✅ Image reference issues resolved (all samples properly reference SQA3D bird views)
- ✅ Baseline evaluation completed on all test sets
  - Results available in `outputs/qa/baseline_eval/`
  - Metrics: SQA3D (2% exact, 42% partial), ScanQA (4% exact, 12% partial)
- ✅ Code wired for Stage 1 multi-view QA training
- ✅ Distributed training supported (DeepSpeed ZeRO-3, 1-8 GPUs)
- ✅ Slurm templates and monitoring utilities in place
- ✅ Stage 2 (ARKit) demo data prepared for evaluation (`data/processed/arkit_synth/train.json`)
- ✅ Inference script available for ARKit evaluation

**In Progress / Planned:**
- 📋 Model retraining on new train splits (next immediate step)
  - Will use reorganized `train_split.jsonl` files
  - Expected to improve generalization on test sets
- 📋 Full evaluation on complete test sets (currently only 50 samples evaluated per dataset)

**Known Limitations:**
- ⚠️ Stage 2 ARKit training is **not implemented**: `action_json` targets are structured (dict), but the trainer assumes text labels; `loss_heads` in `configs/stage2_arkit*.yaml` are not consumed.
- ⚠️ Current data uses single bird-view images without geometry tokens; VGGT still runs but without explicit geometry conditioning (multi-view + geometry requires ScanNet RGB/depth/poses).
- ⚠️ ARKit evaluation has limited samples (9 synthetic samples); full RoomPlan training and JSON-style action prediction remain future work.

---

## 10) Extending the Project - `data/processed/scanqa/test_split.jsonl` – Test/evaluation set
- **SQA3D**: 22,959 train / 3,664 test samples
  - `data/processed/sqa3d/train_split.jsonl` – Training set
  - `data/processed/sqa3d/test_split.jsonl` – Test/evaluation set
- **Total**: ~44,120 training samples, ~7,231 test samples

**Evaluation Dataset**:
- **ARKit Synthetic**: 9 samples for inference/evaluation
  - `data/processed/arkit_synth/train.json` – ARKit synthetic data (inference only, no training head)

**Data characteristics**:
- All samples use single bird-view images from SQA3D (`data/processed/SQA3D/bird/<scene>_bird.png`)
- `geom_token: null` (geometry tokens not included in current data)
- Each record contains: `images`, `geom_token`, `task`, `question`, `answer`, `scene_id`, `question_id`, `object_ids`, `object_names`

**Default Stage 1 training mix**: **0.7 ScanQA / 0.3 SQA3D** from train splits only (see `configs/stage1_3d.yaml`).

**Note on data reorganization**: Previously, ScanQA and SQA3D were prepared entirely for training, with ARKit reserved for inference. The datasets have been reorganized into proper train/test splits to enable proper evaluation. Image reference issues that occurred during initial preparation (where training data referenced images from SQA3D) have been resolved.ins a VGGT + Qwen3 vision-language model to reason about indoor 3D scenes and (optionally) emit RoomPlan-style JSON actions.

---

## 1) What You’re Building
- **Goal:** Teach Qwen3-4B to answer 3D scene questions using VGGT multi-view features; later extend to ARKit RoomPlan actions.
- **Pipeline (code-backed):** VGGT aggregator (frozen) → Perceiver projector → optional geometry tokens → Qwen3-4B. See `src/models/vggt_qwen3_vlm.py` and `configs/perceiver_small.yaml`.
- **Current stage:** Stage 1 (3D QA) is ready to run with existing processed data. Stage 2 is **inference-only for plumbing checks** (9-sample ARKit synthetic JSON exists); Stage 2 training is **not wired** (see limitations below).
- **Distributed training:** Supported via `train_fixed.sh` (DeepSpeed ZeRO-3, 1–8 GPUs) and Slurm template `run.sh`.

For a deeper conceptual overview and a step‑by‑step walkthrough, see:
- `docs/COMPLETE_TRAINING_GUIDE.md` – detailed architecture, data, training, troubleshooting
- `docs/QUICK_START.md` – condensed “run it now” steps

---

## 2) Environment Setup (matches `env/environment.yml`)
1. Create the Conda env (PyTorch 2.4, CUDA 12.8, Python 3.10):
   ```bash
   conda env create -f env/environment.yml
   conda activate roomplan
   ```
2. Install VGGT (editable):
   ```bash
   pip install -e third_party/vggt
   ```
3. (Optional) Install Qwen3 editable if needed; otherwise Hugging Face cache is sufficient.
4. GPU requirements: NVIDIA GPUs with ≥20 GB each; bf16 training.
5. Cache guidance (already set in `train_fixed.sh`): use project-local `.cache/` for HF/torch/triton to avoid NFS issues.

---

## 3) Data

### 3.1 What’s already here (processed)
- `data/processed/scanqa/*.jsonl` – ScanQA, single bird-view, `geom_token: null`
- `data/processed/sqa3d/*.jsonl` – SQA3D, single bird-view, `geom_token: null`
- `data/processed/arkit_synth/train.json` – 9-sample ARKit synthetic for inference plumbing (no training head)

Default Stage 1 mix: **0.7 ScanQA / 0.3 SQA3D** (see `configs/stage1_3d.yaml`).

### 3.2 Raw downloads
- SQA3D: https://zenodo.org/records/7792397#.ZCkprfFBx3g
- ScanQA: https://drive.google.com/drive/folders/1-21A3TBE0QuofEwDg5oDz2z0HEdbVgL2

### 3.3 Preprocess options

**Current data** (already processed with train/test splits):
- The repository includes pre-split datasets in `data/processed/scanqa/` and `data/processed/sqa3d/`
- Each dataset has `train_split.jsonl` and `test_split.jsonl`
- Images reference bird-view PNGs from `data/processed/SQA3D/bird/`

**If regenerating from raw data:**

- **Option A (multi-view + geometry, needs ScanNet RGB/depth/poses):**
  ```bash
  # ScanQA
  python scripts/prep/prepare_scanqa.py \
    --dataset scanqa \
    --scan-root data/raw/scannet \
    --qa-file data/raw/scanqa/questions.json \
    --output data/processed/scanqa/train_split.jsonl \
    --num-views 8

  # SQA3D
  python scripts/prep/prepare_scanqa.py \
    --dataset sqa3d \
    --scan-root data/raw/scannet \
    --qa-file data/raw/sqa3d/questions.json \
    --output data/processed/sqa3d/train_split.jsonl \
    --num-views 8
  ```
  Emits `images` (up to `num_views`), `geom_token` (`R`, `t`, `K`, `depth_hist`), `question`, `answer`, `task`.

- **Option B (fast single-view rebuild, uses shipped bird views):**
  ```bash
  python scripts/prep/rebuild_scanqa_sqa3d.py
  ```
  Recreates train/test splits by pairing QA JSON with `data/processed/SQA3D/bird/<scene>_bird.png`; sets `geom_token: null`.

- **ARKit synthetic (evaluation/inference only):**
  ```bash
  python scripts/prep/prepare_arkit_from_3dod.py \
    --arkit-training-root data/processed/ARKit/Training \
    --output data/processed/arkit_synth/train.json \
    --num-views 10 \
    --max_scenes 10
  ```
  Note: ARKit data is for evaluation purposes only; Stage 2 training is not yet implemented.

### 3.4 How the dataloader/collator use the data
- `src/dataio/dataset_builder.py`: loads JSON/JSONL globs, normalizes to `{images, geom_token, question|instruction, answer|action_json, task}`, truncates to `num_views`, attempts image load (falls back to `data/raw/<path>`), and mixes datasets by ratio via `MultiSourceDataset`.
  - **Training**: loads from `train_split.jsonl` files
  - **Evaluation**: loads from `test_split.jsonl` files or evaluation-specific datasets
- `src/dataio/collate_multiview.py`: resizes/crops images, builds `{question}\n<image>\n` prompts, appends answers, pads to `max_length` (reserving space for vision/geom tokens), and stacks tensors. Geometry is bypassed when `geom_token` is null.

---

## 4) Model & Configs
- **Vision:** VGGT aggregator (`third_party/vggt/vggt_1B_commercial.pt`, frozen).
- **Projector:** Perceiver (`configs/perceiver_small.yaml`) with 128 latents, 6 layers, outputs Qwen3 hidden dim.
- **Language:** `Qwen/Qwen3-4B-Instruct-2507`.
- **Precision/optim:** bf16; AdamW with higher LR on projector/geom head; cosine schedule with warmup.
- **Active stage config:** `configs/stage1_3d.yaml`
  - Data: ScanQA (0.7) + SQA3D (0.3)
  - `num_views: 8`, `image_size: 448`, `max_length: 512`, `view_dropout: 0.3`
  - Train: `batch_size_per_gpu: 6`, `grad_accum: 32`, `max_steps: 30000`, `save_every_steps: 1500`
- **Stage 2 config:** `configs/stage2_arkit.yaml` documents intended ARKit/RoomPlan action training, but the current trainer does **not** support the structured `action_json` targets (no loss head, labels are non-text). Use only for reference; inference plumbing uses Stage 1 weights.

---

## 5) How to Train (Local or Slurm)
Primary launcher: `train_fixed.sh` (sets caches, NCCL envs, probes GPU/host memory, DeepSpeed ZeRO-3).

**Local / interactive**
```bash
# [mode]=full|debug, [num_gpus]=1-8
./train_fixed.sh full 4
./train_fixed.sh --safe debug 2   # 100 steps, conservative batch
```
Defaults: `CONFIG_FILE=configs/stage1_3d.yaml`, outputs to `ckpts/stage1_3d` (or `stage1_3d_debug` for debug mode).

**Slurm (HiPerGator template)**
```bash
sbatch run.sh   # calls train_fixed.sh --safe full 8 on partition hpg-b200
```
Adjust account/time/memory/partition in `run.sh` as needed. More Slurm templates: `scripts/slurm/`. Guide: `docs/SLURM_TRAINING_GUIDE.md`.

**Resuming**
- Accelerate state is saved every `save_every_steps` to `ckpts/stage1_3d/step_xxxxx/`.
- Resume by pointing `--output_dir` (or editing `train_fixed.sh`) to the existing folder; Accelerate will load state automatically.

**Training data**: By default, training uses only the train splits (`train_split.jsonl`) from ScanQA and SQA3D. Test splits are reserved for evaluation.

Detailed training workflow (including troubleshooting and tips) is documented in `docs/COMPLETE_TRAINING_GUIDE.md`.

---

## 6) Checkpoints & Weights Layout

This project uses DeepSpeed ZeRO-3 for multi-GPU training. Checkpoints are stored under `ckpts/` with the following recommended layout:

- Stage 1 (ScanQA/SQA3D 3D QA):
  - Training output root: `ckpts/stage1_3d/`
    - `pytorch_model/` – DeepSpeed ZeRO shards (per-rank model states and optimizer states).
    - `step_XXXX/` – periodic save_state folders used for resume.
    - `logs/` – TensorBoard logs.
- Stage 2 (ARKit / RoomPlan actions, future if training is added):
  - Training output root: `ckpts/stage2_arkit/` (or similar)
    - Same internal structure as Stage 1.

For inference and post-hoc analysis, it is often convenient to convert the ZeRO shards into a standard fp32 checkpoint:

1) Change into the Stage directory and run `zero_to_fp32.py`:
```bash
cd ckpts/stage1_3d

# Example: convert ZeRO shards in pytorch_model/ into sharded fp32 weights
python zero_to_fp32.py pytorch_model pytorch_model_fp32
```

2) This will create a directory:
- `ckpts/stage1_3d/pytorch_model_fp32/`
  - `pytorch_model-00001-of-00005.bin`
  - `pytorch_model-00002-of-00005.bin`
  - …

The inference code (`src/inference/arkit_inference.py`) is written to:
- Prefer the directory `pytorch_model_fp32/` inside the checkpoint directory.
- Still work with older runs where a directory named `pytorch_model_fp32.bin/` was created by mistake.
- Fall back to any `*.bin` / `*.safetensors` in the top-level checkpoint directory if no fp32 folder is present.

For a clean, professional layout going forward, use `pytorch_model_fp32/` as the folder name and avoid having directories that end in `.bin`.

---

## 7) Baseline Evaluation Results

Before retraining on the reorganized data, baseline evaluation was performed using the original model on all test sets. Results are stored in `outputs/qa/baseline_eval/`.

### 7.1 Performance Summary

The original model (trained before data reorganization) was evaluated on test samples from each dataset:

- **SQA3D** (50 test samples from `test_split.jsonl`):
  - Exact match: 1/50 (2.0%)
  - Partial match: 21/50 (42.0%)
  
- **ScanQA** (50 test samples from `test_split.jsonl`):
  - Exact match: 2/50 (4.0%)
  - Partial match: 6/50 (12.0%)
  
- **ARKit** (1 sample):
  - Exact match: 0/1 (0.0%)
  - Partial match: 0/1 (0.0%)

### 7.2 Evaluation Methodology

- **Exact match**: Model prediction exactly matches the ground truth answer (case-insensitive)
- **Partial match**: Model prediction contains the ground truth answer as a substring, or vice versa
- Evaluation scripts: `scripts/eval_baseline.py` or similar

### 7.3 Output Files

- `outputs/qa/baseline_eval/sqa3d_baseline.jsonl` – Per-sample predictions for SQA3D test set
- `outputs/qa/baseline_eval/scanqa_baseline.jsonl` – Per-sample predictions for ScanQA test set
- `outputs/qa/baseline_eval/arkit_baseline.jsonl` – Per-sample predictions for ARKit
- `outputs/qa/baseline_eval/baseline_summary.json` – Aggregate metrics

Each JSONL line contains: `question`, `prediction`, `reference`, `scene_id`, and match flags.

**Important**: These results represent the original model's performance and serve as a baseline. Retraining on the new train/test splits is planned to improve generalization.

---

## 8) Monitoring & Debugging
- **Console logging:** Enabled in `src/train/train_sft.py` (loss, LR, speed, ETA every `log_every_steps`).
- **TensorBoard:** `tensorboard --logdir ckpts/stage1_3d/logs --port 6006 --bind_all`
- **CLI monitor:** `python scripts/monitor_training.py --logdir ckpts/stage1_3d/logs/roomplan --watch`
- **Log tail:** `tail -f pytorchdist_*.out | grep "Step\\|Loss\\|ETA"` (Slurm) or `tail -f logs/stage2/*.log` if present.
- **Docs:** `docs/MONITORING_GUIDE.md`, `docs/TRAINING_MONITORING_SUMMARY.md`.

If you hit NCCL or cache issues, `train_fixed.sh` already sets conservative envs and local caches; see comments inside that script for overrides.

---

## 7) What’s Done vs. To-Do
- ✅ Code wired for Stage 1 multi-view QA; single-view ScanQA/SQA3D shards present and runnable.
- ✅ Distributed training supported (DeepSpeed ZeRO-3, Slurm templates).
- ✅ Monitoring utilities and guides are in place.
- ✅ Stage 2 (RoomPlan actions) demo data prepared for ARKit (`data/processed/arkit_synth/train.json`) and an inference script is wired.
- ⚠️ Stage 2 training is **not implemented**: `action_json` is structured (dict) and the trainer assumes text labels; `loss_heads` in `configs/stage2_arkit*.yaml` are not consumed.
- ⚠️ Only small-scale ARKit inference smoke tests were run with Stage 1 weights; full RoomPlan training and JSON-style action prediction remain future work.
- ⚠️ Current ScanQA/SQA3D shards lack geometry tokens and multi-view frames; VGGT still runs but without explicit geometry conditioning.

---

## 8) Extending the Project
- **Add geometry + multi-view:** Rebuild ScanQA/SQA3D with poses/depth using `scripts/prep/prepare_scanqa.py --dataset scanqa|sqa3d --num-views 8 --output data/processed/{scanqa,sqa3d}/train.jsonl`.
- **Generate RoomPlan data (for inference or future training):** Build `data/processed/arkit_synth/train.json` with `scripts/prep/prepare_arkit_from_3dod.py` (works with the current 3DOD layout). If you have full planes/cameras metadata, `scripts/prep/synth_roomplan_instructions.py` is also supported. Training with `configs/stage3_arkit.yaml` is not yet wired.
Stage‑wise training strategies (Stage 1 → Stage 2) are described in more detail in `docs/COMPLETE_TRAINING_GUIDE.md`.

---

## 9) Repository Map (active pieces)
```
configs/                # Stage configs (stage1_3d active), projector, DeepSpeed
data/processed/         # scanqa/*.jsonl, sqa3d/*.jsonl
data/raw/               # placeholders + ARKit README
src/dataio/             # dataset builder + multi-view collator
src/models/             # VGGT-Qwen3 wrapper, Perceiver projector
src/train/train_sft.py  # Accelerate/DeepSpeed trainer
train_fixed.sh          # Hardened launcher (defaults to stage1_3d)
run.sh                  # Slurm example invoking train_fixed.sh
logs/, ckpts/           # Training outputs
docs/                   # Guides (complete training, quickstart, monitoring, Slurm, file map)
```
More file descriptions: `docs/FILE_GUIDE.md`.

---

## 10) Quickstart Checklist
1) `conda activate roomplan` (after creating env)  
2) Place checkpoints: `third_party/vggt/vggt_1B_commercial.pt` and cache `Qwen/Qwen3-4B-Instruct-2507`  
3) Verify data exists: `ls data/processed/scanqa data/processed/sqa3d`  
4) Run: `./train_fixed.sh debug 1` (sanity) → then `./train_fixed.sh full 4` (or Slurm via `run.sh`)  
5) Monitor: `tensorboard --logdir ckpts/stage1_3d/logs` or `python scripts/monitor_training.py --logdir ckpts/stage1_3d/logs/roomplan --watch`  

Stage 2 training is not supported yet. To experiment with ARKit inference, use the instructions below with Stage 1 weights.

---

## 11) ARKit / RoomPlan Inference (Stage 1 weights)

Once you have:
- Completed Stage 1 training (or are using a provided Stage 1 checkpoint), and
- Generated or reused ARKit synthetic data under `data/processed/arkit_synth/` (the repo already has `train.json`),

you can run ARKit evaluation to test the model's ability to understand RoomPlan-style spatial reasoning tasks.

**Note**: This is primarily for evaluation and testing purposes. The model generates free-form text responses rather than structured JSON actions (see limitations below).

### 13.1 Prepare or reuse ARKit synthetic data

The included 9-sample file was generated with:
```bash
python scripts/prep/prepare_arkit_from_3dod.py \
  --arkit-training-root data/processed/ARKit/Training \
  --output data/processed/arkit_synth/train.json \
  --num-views 10 \
  --max_scenes 10
```
If you have full planes/cameras metadata, `scripts/prep/synth_roomplan_instructions.py` is also supported.

### 13.2 Single-GPU ARKit evaluation (Stage 1 weights)

```bash
python -m src.inference.arkit_inference \
  --config configs/stage1_3d.yaml \
  --arkit_glob "data/processed/arkit_synth/train.json" \
  --checkpoint_dir ckpts/stage1_3d \
  --num_scenes 9 \
  --max_new_tokens 256 \
  --device cuda:0 \
  --output_jsonl outputs/qa/arkit_eval/arkit_predictions_stage1.jsonl
```

- `--checkpoint_dir` should point to the Stage 1 directory (`ckpts/stage1_3d`) containing both the ZeRO shards and the `pytorch_model_fp32/` directory.
- The script prints per-sample logs and writes all results to the output JSONL file.

**Inputs and outputs**
- Input JSON (`data/processed/arkit_synth/train.json`):
  - Per-scene fields include `scene_id`, list of `images` (up to 10 rendered views), and a text `question`/instruction describing an ARKit-style action (e.g., “place an anchor on the nearest cabinet”).
  - Each record also carries a structured `action_json` reference with fields like `action`, `scene`, `center`, `normal`, and `extent` that describe the intended 3D RoomPlan action.
- Inference:
  - `src/inference/arkit_inference.py` rebuilds the VGGT-Qwen3 model from the stage config, loads Stage 2 weights if available, and for each sample:
    - Encodes the multi-view images via VGGT + Perceiver projector.
    - Injects the visual tokens into Qwen at the `<image>` position in the prompt `{question}\n<image>\n`.
    - Runs autoregressive generation with `max_new_tokens` to produce a free-form text `prediction`.
  - The script also computes a very rough exact-match metric by comparing the text output to the serialized reference when present.
- Output JSONL (`ckpts/arkit_infer/arkit_predictions_stage2_en.jsonl`):
  - Each line contains:
    - `index`: sequential sample index used in the run.
    - `scene_id`: copied from the ARKit synthetic sample.
    - `question`: the natural-language instruction given to the model.
    - `prediction`: model-generated text describing the inferred action.
    - `reference`: the ground-truth structured action (original `action_json`) used only for qualitative comparison/diagnostics.

### 13.3 Saved evaluation output format

Each line of the output JSONL is a JSON record:

```json
{
  "index": 0,
  "scene_id": "47333462",
  "question": "In scene 47333462, find an object belonging to the category 'cabinet' and place a virtual anchor at the center of that object.",
  "prediction": "... model-generated text ...",
  "reference": {
    "action": "place_anchor",
    "scene": "47333462",
    "center": [...],
    "normal": [0, 1, 0],
    "extent": [...]
  }
}
```

Inspect a few records via `head outputs/qa/arkit_eval/arkit_predictions_stage1.jsonl`.

### 13.4 Evaluation behavior (current) and limitations

- Model repeats or elaborates on the instruction; does not emit the structured `action_json`.
- Exact-match metric is 0.0 because predictions are free-form text while references are JSON.
- Only one fp32 shard was loaded for the plumbing test (~450 missing keys), so results are not indicative of final quality.
- Training support for `action_json` is absent in the current trainer (targets are dicts; no structured loss head).

### 11.5 Future work (optional)

- Add a JSON-only target/head for `action_json`, or serialize targets to text for supervision.
- Constrain prompts to “respond only with JSON”.
- Convert and load full fp32 checkpoints (all shards) when available.
