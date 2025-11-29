# VGGT-Qwen3 RoomPlan

This repository trains a VGGT + Qwen3 vision-language model to reason about indoor 3D scenes and (optionally) emit RoomPlan-style JSON actions.

---

## 1) What You’re Building
- **Goal:** Teach Qwen3-4B to answer 3D scene questions using VGGT multi-view features; later extend to ARKit RoomPlan actions.
- **Pipeline (code-backed):** VGGT aggregator (frozen) → Perceiver projector → optional geometry tokens → Qwen3-4B with LoRA. See `src/models/vggt_qwen3_vlm.py` and `configs/perceiver_small.yaml`.
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

### 3.1 Datasets currently available
The following processed datasets are already present in this repository (paths are relative to the project root):

| Dataset | Path glob | Format | Views/sample | Geometry tokens |
|---------|-----------|--------|--------------|-----------------|
| ScanQA | `data/processed/scanqa/*.jsonl` | JSONL | 1 (bird’s-eye) | `null` |
| SQA3D  | `data/processed/sqa3d/*.jsonl`  | JSONL | 1 (bird’s-eye) | `null` |
| ARKit synthetic (used for inference plumbing) | `data/processed/arkit_synth/train.json` | JSON | up to 10 | `null` |

Sample record (from `data/processed/sqa3d/train.jsonl`):
```json
{
  "images": ["data/processed/SQA3D/bird/scene0380_00_bird.png"],
  "geom_token": null,
  "question": "What color is the desk to my right?",
  "answer": "brown",
  "task": "sqa3d"
}
```

How the code consumes this:
- `src/dataio/dataset_builder.py` loads `.json` arrays or `.jsonl` lines and truncates to `num_views` (8 in Stage 2; single-view data is accepted).
- `src/dataio/collate_multiview.py` resizes to 448, builds text as `{question}\n<image>\n{answer}`, pads to ensure room for visual tokens, and skips geometry tokens when `geom_token` is null.
- Vision features are injected at the `<image>` position; geometry head is bypassed when absent.

If you regenerate data with poses/depth, keep keys `R`, `t`, `K`, `depth_hist` so `encode_geom` works unchanged.

### 3.2 Downloading / rebuilding datasets yourself

**ScanQA / SQA3D (Stage 2):**
- This repo already includes small, ready-to-use processed shards under `data/processed/scanqa/` and `data/processed/sqa3d/` for Stage 2 training and demos.
- If you want to rebuild them from the original datasets, follow the official dataset instructions to obtain the raw RGB + annotations, place them under `data/raw/`, and then use:
  ```bash
  # ScanQA
  python scripts/prep/prepare_scanqa.py \
    --dataset scanqa \
    --scan-root data/raw/<your_scan_root> \
    --qa-file data/raw/scanqa/questions.json \
    --output data/processed/scanqa/train.jsonl \
    --num-views 8

  # SQA3D
  python scripts/prep/prepare_scanqa.py \
    --dataset sqa3d \
    --scan-root data/raw/<your_scan_root> \
    --qa-file data/raw/sqa3d/questions.json \
    --output data/processed/sqa3d/train.jsonl \
    --num-views 8
  ```
  This will create multi‑view (up to 8 views), geometry‑aware samples compatible with the existing Stage 2 config.

**ARKitScenes / RoomPlan synthetic data (Stage 3 inference plumbing):**
- The repo already includes a small ARKit synthetic JSON (`data/processed/arkit_synth/train.json`, 9 samples) built from your partial ARKit 3DOD download under `data/processed/ARKit/Training/...` using:
  ```bash
  python scripts/prep/prepare_arkit_from_3dod.py \
    --arkit-training-root data/processed/ARKit/Training \
    --output data/processed/arkit_synth/train.json \
    --num-views 10 \
    --max_scenes 10
  ```
- If you have full ARKitScenes with `annotations/planes.json` and `cameras.json`, you can instead use the plane-based script:
  ```bash
  python scripts/prep/synth_roomplan_instructions.py \
    --arkit-root data/raw/arkitscenes \
    --output data/processed/arkit_synth/train.json \
    --num-views 10
  ```
  (This requires plane/camera metadata not present in the shipped subset.)

Stage 3 configs (`configs/stage3_arkit*.yaml`) target these files, but training is not implemented; the JSON is used only for inference smoke tests with Stage 2 weights.

For more discussion of data design and trade‑offs, see `docs/COMPLETE_TRAINING_GUIDE.md` (Data section).

### 3.3 ScanQA and SQA3D (Stage 2 datasets)

Stage 2 training (`configs/stage2_3d.yaml`) uses two 3D question–answering datasets:

- **ScanQA** (`data/processed/scanqa/*.jsonl`)
  - Task: free-form, open-vocabulary QA over indoor scenes (e.g., “What color is the cabinet to the left of the bed?”).
  - Processed format in this repo:
    - Each JSONL line has:
      - `images`: a list with a single canonical view image path.
      - `question`: a natural-language question about the scene.
      - `answer`: a short free-form answer.
      - `task`: `"scanqa"`.
  - Usage:
    - In Stage 2 configs, ScanQA contributes most of the training signal for general 3D reasoning.

- **SQA3D** (`data/processed/sqa3d/*.jsonl`)
  - Task: spatial and semantic QA about scenes, with a focus on relative location and object relationships.
  - Processed format matches ScanQA:
    - `images`: one bird’s-eye or canonical view.
    - `question`: localized question (e.g., “What is in front of the sofa?”).
    - `answer`: short answer.
    - `task`: `"sqa3d"`.
  - Usage:
    - Mixed with ScanQA during Stage 2 to improve robustness on spatial and relational questions.

In both cases:
- Images are single-view in the current processed shards, but the model and dataloader accept multi-view input seamlessly once such data is available.
- Geometry tokens are currently `null` and therefore bypassed; the model still uses VGGT’s visual tokens for 3D reasoning.

At training time (Stage 2):
- `configs/stage2_3d.yaml` mixes these datasets with a ratio of approximately **0.7 ScanQA / 0.3 SQA3D** via `MultiSourceDataset`:
  - ScanQA provides broad, free-form semantic supervision (object categories, attributes, colors, etc.).
  - SQA3D emphasizes spatial relationships and relative positions.
- This combination teaches the model to answer both generic semantic questions and more geometric, “where is X relative to Y” questions before moving on to ARKit/RoomPlan (Stage 3).

---

## 4) Model & Configs
- **Vision:** VGGT aggregator (`third_party/vggt/vggt_1B_commercial.pt`, frozen).
- **Projector:** Perceiver (`configs/perceiver_small.yaml`) with 128 latents, 6 layers, outputs Qwen3 hidden dim.
- **Language:** `Qwen/Qwen3-4B-Instruct-2507` + LoRA on q/k/v/o (see config).
- **Precision/optim:** bf16; AdamW with higher LR on projector/geom head; cosine schedule with warmup.
- **Active stage config:** `configs/stage2_3d.yaml`
  - Data: ScanQA (0.7) + SQA3D (0.3)
  - `num_views: 8`, `image_size: 448`, `max_length: 512`, `view_dropout: 0.3`
  - Train: `batch_size_per_gpu: 6`, `grad_accum: 32`, `max_steps: 30000`, `save_every_steps: 1500`
  - LoRA: rank 16 on q/k/v/o
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
- **Small smoke tests:** Use the mini configs in `configs/*_mini.yaml` and set VGGT to `mock` in `configs/local_*` if you need CPU-only checks.

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
