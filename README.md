# VGGT-Qwen3 RoomPlan

This repository trains a VGGT + Qwen3 vision-language model to reason about indoor 3D scenes and (optionally) emit RoomPlan-style JSON actions.

---

## 1) What You’re Building
- **Goal:** Teach Qwen3-4B to answer 3D scene questions using VGGT multi-view features; later extend to ARKit RoomPlan actions.
- **Pipeline (code-backed):** VGGT aggregator (frozen) → Perceiver projector → optional geometry tokens → Qwen3-4B. See `src/models/vggt_qwen3_vlm.py` and `configs/perceiver_small.yaml`.
- **Current stage:** Stage 2 (3D QA) is ready to run with existing processed data. Stage 3 is **inference-only for plumbing checks** (9-sample ARKit synthetic JSON exists); Stage 3 training is **not wired** (see limitations below).
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

Default Stage 2 mix: **0.7 ScanQA / 0.3 SQA3D** (see `configs/stage2_3d.yaml`).

### 3.2 Raw downloads
- SQA3D: https://zenodo.org/records/7792397#.ZCkprfFBx3g
- ScanQA: https://drive.google.com/drive/folders/1-21A3TBE0QuofEwDg5oDz2z0HEdbVgL2

### 3.3 Preprocess options
- **Option A (multi-view + geometry, needs ScanNet RGB/depth/poses):**
  ```bash
  # ScanQA
  python scripts/prep/prepare_scanqa.py \
    --dataset scanqa \
    --scan-root data/raw/scannet \
    --qa-file data/raw/scanqa/questions.json \
    --output data/processed/scanqa/train.jsonl \
    --num-views 8

  # SQA3D
  python scripts/prep/prepare_scanqa.py \
    --dataset sqa3d \
    --scan-root data/raw/scannet \
    --qa-file data/raw/sqa3d/questions.json \
    --output data/processed/sqa3d/train.jsonl \
    --num-views 8
  ```
  Emits `images` (up to `num_views`), `geom_token` (`R`, `t`, `K`, `depth_hist`), `question`, `answer`, `task`.

- **Option B (fast single-view rebuild, uses shipped bird views):**
  ```bash
  python scripts/prep/rebuild_scanqa_sqa3d.py
  ```
  Recreates `data/processed/{scanqa,sqa3d}/train.jsonl` by pairing QA JSON with `data/processed/SQA3D/bird/<scene>_bird.png`; sets `geom_token: null`.

- **ARKit synthetic (inference plumbing only):**
  ```bash
  python scripts/prep/prepare_arkit_from_3dod.py \
    --arkit-training-root data/processed/ARKit/Training \
    --output data/processed/arkit_synth/train.json \
    --num-views 10 \
    --max_scenes 10
  ```
  Stage 3 training remains unimplemented; use for inference smoke tests.

### 3.4 How the dataloader/collator use the data
- `src/dataio/dataset_builder.py`: loads JSON/JSONL globs, normalizes to `{images, geom_token, question|instruction, answer|action_json, task}`, truncates to `num_views`, attempts image load (falls back to `data/raw/<path>`), and mixes datasets by ratio via `MultiSourceDataset`.
- `src/dataio/collate_multiview.py`: resizes/crops images, builds `{question}\n<image>\n` prompts, appends answers, pads to `max_length` (reserving space for vision/geom tokens), and stacks tensors. Geometry is bypassed when `geom_token` is null.

---

## 4) Model & Configs
- **Vision:** VGGT aggregator (`third_party/vggt/vggt_1B_commercial.pt`, frozen).
- **Projector:** Perceiver (`configs/perceiver_small.yaml`) with 128 latents, 6 layers, outputs Qwen3 hidden dim.
- **Language:** `Qwen/Qwen3-4B-Instruct-2507`.
- **Precision/optim:** bf16; AdamW with higher LR on projector/geom head; cosine schedule with warmup.
- **Active stage config:** `configs/stage2_3d.yaml`
  - Data: ScanQA (0.7) + SQA3D (0.3)
  - `num_views: 8`, `image_size: 448`, `max_length: 512`, `view_dropout: 0.3`
  - Train: `batch_size_per_gpu: 6`, `grad_accum: 32`, `max_steps: 30000`, `save_every_steps: 1500`
- **Stage 3 config:** `configs/stage3_arkit.yaml` documents intended ARKit/RoomPlan action training, but the current trainer does **not** support the structured `action_json` targets (no loss head, labels are non-text). Use only for reference; inference plumbing uses Stage 2 weights.

---

## 5) How to Train (Local or Slurm)
Primary launcher: `train_fixed.sh` (sets caches, NCCL envs, probes GPU/host memory, DeepSpeed ZeRO-3).

**Local / interactive**
```bash
# [mode]=full|debug, [num_gpus]=1-8
./train_fixed.sh full 4
./train_fixed.sh --safe debug 2   # 100 steps, conservative batch
```
Defaults: `CONFIG_FILE=configs/stage2_3d.yaml`, outputs to `ckpts/stage2_3d` (or `_debug`).

**Slurm (HiPerGator template)**
```bash
sbatch run.sh   # calls train_fixed.sh --safe full 8 on partition hpg-b200
```
Adjust account/time/memory/partition in `run.sh` as needed. More Slurm templates: `scripts/slurm/`. Guide: `docs/SLURM_TRAINING_GUIDE.md`.

**Resuming**
- Accelerate state is saved every `save_every_steps` to `ckpts/stage2_3d/step_xxxxx/`.
- Resume by pointing `--output_dir` (or editing `train_fixed.sh`) to the existing folder; Accelerate will load state automatically.

Detailed training workflow (including troubleshooting and tips) is documented in `docs/COMPLETE_TRAINING_GUIDE.md`.

---

## 6) Checkpoints & Weights Layout

This project uses DeepSpeed ZeRO-3 for multi-GPU training. Checkpoints are stored under `ckpts/` with the following recommended layout:

- Stage 2 (ScanQA/SQA3D 3D QA):
  - Training output root: `ckpts/stage2_3d/`
    - `pytorch_model/` – DeepSpeed ZeRO shards (per-rank model states and optimizer states).
    - `step_XXXX/` – periodic save_state folders used for resume.
    - `logs/` – TensorBoard logs.
- Stage 3 (ARKit / RoomPlan actions, future if training is added):
  - Training output root: `ckpts/stage3_arkit/` (or similar)
    - Same internal structure as Stage 2.

For inference and post-hoc analysis, it is often convenient to convert the ZeRO shards into a standard fp32 checkpoint:

1) Change into the Stage directory and run `zero_to_fp32.py`:
```bash
cd ckpts/stage2_3d

# Example: convert ZeRO shards in pytorch_model/ into sharded fp32 weights
python zero_to_fp32.py pytorch_model pytorch_model_fp32
```

2) This will create a directory:
- `ckpts/stage2_3d/pytorch_model_fp32/`
  - `pytorch_model-00001-of-00005.bin`
  - `pytorch_model-00002-of-00005.bin`
  - …

The inference code (`src/inference/arkit_inference.py`) is written to:
- Prefer the directory `pytorch_model_fp32/` inside the checkpoint directory.
- Still work with older runs where a directory named `pytorch_model_fp32.bin/` was created by mistake.
- Fall back to any `*.bin` / `*.safetensors` in the top-level checkpoint directory if no fp32 folder is present.

For a clean, professional layout going forward, use `pytorch_model_fp32/` as the folder name and avoid having directories that end in `.bin`.

---

## 6) Monitoring & Debugging
- **Console logging:** Enabled in `src/train/train_sft.py` (loss, LR, speed, ETA every `log_every_steps`).
- **TensorBoard:** `tensorboard --logdir ckpts/stage2_3d/logs --port 6006 --bind_all`
- **CLI monitor:** `python scripts/monitor_training.py --logdir ckpts/stage2_3d/logs/roomplan --watch`
- **Log tail:** `tail -f pytorchdist_*.out | grep "Step\\|Loss\\|ETA"` (Slurm) or `tail -f logs/stage2/*.log` if present.
- **Docs:** `docs/MONITORING_GUIDE.md`, `docs/TRAINING_MONITORING_SUMMARY.md`.

If you hit NCCL or cache issues, `train_fixed.sh` already sets conservative envs and local caches; see comments inside that script for overrides.

---

## 7) What’s Done vs. To-Do
- ✅ Code wired for Stage 2 multi-view QA; single-view ScanQA/SQA3D shards present and runnable.
- ✅ Distributed training supported (DeepSpeed ZeRO-3, Slurm templates).
- ✅ Monitoring utilities and guides are in place.
- ✅ Stage 3 (RoomPlan actions) demo data prepared for ARKit (`data/processed/arkit_synth/train.json`) and an inference script is wired.
- ⚠️ Stage 3 training is **not implemented**: `action_json` is structured (dict) and the trainer assumes text labels; `loss_heads` in `configs/stage3_arkit*.yaml` are not consumed.
- ⚠️ Only small-scale ARKit inference smoke tests were run with Stage 2 weights; full RoomPlan training and JSON-style action prediction remain future work.
- ⚠️ Current ScanQA/SQA3D shards lack geometry tokens and multi-view frames; VGGT still runs but without explicit geometry conditioning.

---

## 8) Extending the Project
- **Add geometry + multi-view:** Rebuild ScanQA/SQA3D with poses/depth using `scripts/prep/prepare_scanqa.py --dataset scanqa|sqa3d --num-views 8 --output data/processed/{scanqa,sqa3d}/train.jsonl`.
- **Generate RoomPlan data (for inference or future training):** Build `data/processed/arkit_synth/train.json` with `scripts/prep/prepare_arkit_from_3dod.py` (works with the current 3DOD layout). If you have full planes/cameras metadata, `scripts/prep/synth_roomplan_instructions.py` is also supported. Training with `configs/stage3_arkit.yaml` is not yet wired.
Stage‑wise training strategies (Stage 1 → Stage 2 → Stage 3) are described in more detail in `docs/COMPLETE_TRAINING_GUIDE.md`.

---

## 9) Repository Map (active pieces)
```
configs/                # Stage configs (stage2_3d active), projector, DeepSpeed
data/processed/         # scanqa/*.jsonl, sqa3d/*.jsonl
data/raw/               # placeholders + ARKit README
src/dataio/             # dataset builder + multi-view collator
src/models/             # VGGT-Qwen3 wrapper, Perceiver projector
src/train/train_sft.py  # Accelerate/DeepSpeed trainer
train_fixed.sh          # Hardened launcher (defaults to stage2_3d)
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
5) Monitor: `tensorboard --logdir ckpts/stage2_3d/logs` or `python scripts/monitor_training.py --logdir ckpts/stage2_3d/logs/roomplan --watch`  

Stage 3 training is not supported yet. To experiment with ARKit inference, use the instructions below with Stage 2 weights.

---

## 11) ARKit / RoomPlan Inference (Stage 2 weights)

Once you have:
- Completed Stage 2 (or are using the provided Stage 2 checkpoint shards), and
- Generated or reused ARKit synthetic data under `data/processed/arkit_synth/` (the repo already has `train.json`),

you can run ARKit inference on the first 10 available scenes as follows.

### 11.1 Prepare or reuse ARKit synthetic data

The included 9-sample file was generated with:
```bash
python scripts/prep/prepare_arkit_from_3dod.py \
  --arkit-training-root data/processed/ARKit/Training \
  --output data/processed/arkit_synth/train.json \
  --num-views 10 \
  --max_scenes 10
```
If you have full planes/cameras metadata, `scripts/prep/synth_roomplan_instructions.py` is also supported.

### 11.2 Single-GPU ARKit inference (Stage 2 weights)

```bash
python -m src.inference.arkit_inference \
  --config configs/stage2_3d.yaml \
  --arkit_glob "data/processed/arkit_synth/train.json" \
  --checkpoint_dir ckpts/stage2_3d \
  --num_scenes 9 \
  --max_new_tokens 256 \
  --device cuda:0 \
  --output_jsonl ckpts/arkit_infer/arkit_predictions_stage2_en.jsonl
```

- `--checkpoint_dir` should point to the Stage 2 directory (`ckpts/stage2_3d`) containing both the ZeRO shards and the `pytorch_model_fp32/` directory.
- The script prints per-sample logs and writes all results to `ckpts/arkit_infer/arkit_predictions_stage2_en.jsonl`.

### 11.3 Saved inference output format

Each line of `ckpts/arkit_infer/arkit_predictions_stage2_en.jsonl` is a JSON record:

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

Inspect a few records via `head ckpts/arkit_infer/arkit_predictions_stage2_en.jsonl`.

### 11.4 Inference behavior (current) and limitations

- Model repeats or elaborates on the instruction; does not emit the structured `action_json`.
- Exact-match metric is 0.0 because predictions are free-form text while references are JSON.
- Only one fp32 shard was loaded for the plumbing test (~450 missing keys), so results are not indicative of final quality.
- Training support for `action_json` is absent in the current trainer (targets are dicts; no structured loss head).

### 11.5 Future work (optional)

- Add a JSON-only target/head for `action_json`, or serialize targets to text for supervision.
- Constrain prompts to “respond only with JSON”.
- Convert and load full fp32 checkpoints (all shards) when available.
