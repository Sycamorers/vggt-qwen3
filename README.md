# VGGT-Qwen3 RoomPlan – Complete Reproduction Guide

This repo trains a VGGT + Qwen3 vision-language model to reason about indoor 3D scenes and (optionally) emit RoomPlan-style JSON actions.

---

## 1) What You’re Building
- **Goal:** Teach Qwen3-4B to answer 3D scene questions using VGGT multi-view features; later extend to ARKit RoomPlan actions.
- **Pipeline (code-backed):** VGGT aggregator (frozen) → Perceiver projector → optional geometry tokens → Qwen3-4B with LoRA. See `src/models/vggt_qwen3_vlm.py` and `configs/perceiver_small.yaml`.
- **Current stage:** Stage 2 (3D QA) is ready to run with existing processed data; Stage 3 (RoomPlan actions) needs data generation.
- **Distributed training:** Supported via `train_fixed.sh` (DeepSpeed ZeRO-3, 1–8 GPUs) and Slurm template `run.sh`.

Further architectural details: `docs/COMPLETE_TRAINING_GUIDE.md` (theory) and inline code comments in `src/models/`.

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
3. (Optional) Install Qwen3 editable if needed; otherwise HF cache is sufficient.
4. GPU requirements: NVIDIA GPUs with ≥20 GB each; bf16 training.
5. Cache guidance (already set in `train_fixed.sh`): use project-local `.cache/` for HF/torch/triton to avoid NFS issues.

---

## 3) Data You Already Have (and How the Loader Uses It)
Active datasets on disk (all paths relative to repo root):

| Dataset | Path glob | Format | Views/sample | Geometry tokens |
|---------|-----------|--------|--------------|-----------------|
| ScanQA | `data/processed/scanqa/*.jsonl` | JSONL | 1 (bird’s-eye) | `null` |
| SQA3D  | `data/processed/sqa3d/*.jsonl`  | JSONL | 1 (bird’s-eye) | `null` |
| DocVQA (unused in current config) | `data/processed/DocVQA/llava_instruct_80k.json` | JSON | n/a | n/a |
| ARKit synthetic | **missing** (expected `data/processed/arkit_synth/*.json`) | — | — | — |

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

---

## 4) Model & Configs That Actually Run
- **Vision:** VGGT aggregator (`third_party/vggt/vggt_1B_commercial.pt`, frozen).
- **Projector:** Perceiver (`configs/perceiver_small.yaml`) with 128 latents, 6 layers, outputs Qwen3 hidden dim.
- **Language:** `Qwen/Qwen3-4B-Instruct-2507` + LoRA on q/k/v/o (see config).
- **Precision/optim:** bf16; AdamW with higher LR on projector/geom head; cosine schedule with warmup.
- **Active stage config:** `configs/stage2_3d.yaml`
  - Data: ScanQA (0.7) + SQA3D (0.3)
  - `num_views: 8`, `image_size: 448`, `max_length: 512`, `view_dropout: 0.3`
  - Train: `batch_size_per_gpu: 6`, `grad_accum: 32`, `max_steps: 30000`, `save_every_steps: 1500`
  - LoRA: rank 16 on q/k/v/o
- **Missing data for Stage 3:** `configs/stage3_arkit.yaml` expects `data/processed/arkit_synth/*.json`.

---

## 5) How to Train (Local or Slurm)
Primary launcher: `train_fixed.sh` (sets caches, NCCL envs, probes GPU/host mem, DeepSpeed ZeRO-3).

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
- ⚠️ Stage 1 instruction data not present (llava/sharegpt4v/docvqa/chartqa paths empty except DocVQA JSON).
- ⚠️ Stage 3 (RoomPlan actions) data missing (`data/processed/arkit_synth`).
- ⚠️ Current shards lack geometry tokens and multi-view frames; VGGT still runs but without geometry conditioning.

---

## 8) Extending the Project
- **Add geometry + multi-view:** Rebuild ScanQA/SQA3D with poses/depth using `scripts/prep/prepare_scanqa.py --dataset scanqa|sqa3d --num-views 8 --output data/processed/{scanqa,sqa3d}/train.jsonl`.
- **Generate RoomPlan data:** Run `scripts/prep/synth_roomplan_instructions.py --arkit-root data/raw/arkitscenes --output data/processed/arkit_synth/train.json --num-views 10`, then switch `CONFIG_FILE` to `configs/stage3_arkit.yaml`.
- **Small smoke tests:** Use the mini configs in `configs/*_mini.yaml` and set VGGT to `mock` in `configs/local_*` if you need CPU-only checks.

---

## 9) Repository Map (active pieces)
```
configs/                # Stage configs (stage2_3d active), projector, DeepSpeed
data/processed/         # scanqa/*.jsonl, sqa3d/*.jsonl, DocVQA JSON
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

Once Stage 2 converges and RoomPlan data is prepared, switch to `configs/stage3_arkit.yaml` and re-run `train_fixed.sh` with the new config.
