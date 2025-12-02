# VGGT-Qwen3 RoomPlan – Detailed Training Guide

This guide aligns with the current repository state: Qwen3-4B + VGGT, Stage 1 (3D QA) configured via `configs/stage1_3d.yaml`, single-view ScanQA/SQA3D shards in `data/processed/`, and the hardened launcher `train_fixed.sh`. Use this as a companion to the main README.

---

## 1. Project Overview
- **Goal:** Train a vision-language model that understands indoor 3D scenes and answers questions; later extend to RoomPlan-style JSON actions.
- **Pipeline:** VGGT aggregator (frozen) → Perceiver projector (trainable) → optional geometry tokens → Qwen3-4B with LoRA adapters.
- **Ready now:** Stage 1 multi-view QA using the existing processed ScanQA/SQA3D JSONL files (single-view per sample, `geom_token: null`).
- **Not ready yet:** Stage 2 RoomPlan training (structured `action_json` head not implemented; labels are non-text).

---

## 2. Environment & Dependencies
1. Create and activate the Conda env (PyTorch 2.4, CUDA 12.8):
   ```bash
   conda env create -f env/environment.yml
   conda activate roomplan
   ```
2. Install VGGT locally:
   ```bash
   pip install -e third_party/vggt
   ```
3. Checkpoints:
   - Qwen3: cache `Qwen/Qwen3-4B-Instruct-2507` (set `HF_HOME=$PWD/.cache/huggingface`).
   - VGGT: place `third_party/vggt/vggt_1B_commercial.pt`.
4. Hardware: NVIDIA GPUs ≥20 GB, bf16 training. For multi-GPU, DeepSpeed ZeRO-3 is configured by default in `train_fixed.sh`.
5. Caches/NCCL: `train_fixed.sh` sets project-local caches and conservative NCCL env vars automatically.

---

## 3. Data (Current and Expected)
Current processed datasets (all JSON/JSONL are relative to repo root):

| Dataset | Glob | Views/sample | Geometry | Notes |
|---------|------|--------------|----------|-------|
| ScanQA  | `data/processed/scanqa/*.jsonl` | 1 | `geom_token: null` | Uses bird’s-eye PNGs under `data/processed/SQA3D/bird/...`. |
| SQA3D   | `data/processed/sqa3d/*.jsonl`  | 1 | `geom_token: null` | Same image source; single-view. |
| DocVQA  | `data/processed/DocVQA/llava_instruct_80k.json` | n/a | n/a | Present but not wired to Stage 2. |
| ARKit synthetic | `data/processed/arkit_synth/train.json` | up to 10 | `geom_token: null` | 9-sample subset from 3DOD layout; used for inference plumbing only. |

Sample (from `data/processed/sqa3d/train.jsonl`):
```json
{
  "images": ["data/processed/SQA3D/bird/scene0380_00_bird.png"],
  "geom_token": null,
  "question": "What color is the desk to my right?",
  "answer": "brown",
  "task": "sqa3d"
}
```

How the loader uses it:
- `src/dataio/dataset_builder.py` reads `.json` arrays or `.jsonl` lines and truncates to `num_views` (8 in Stage 1; single-view is accepted).
- `src/dataio/collate_multiview.py` resizes to 448, builds `{question}\n<image>\n{answer}`, pads to fit vision tokens, and skips geometry when `geom_token` is null.

If you regenerate data with poses/depth, keep keys `R`, `t`, `K`, `depth_hist` so `encode_geom` works unchanged.

---

## 4. Model & Config (Stage 1)
- **Vision:** VGGT aggregator (frozen) from `third_party/vggt/vggt_1B_commercial.pt`.
- **Projector:** Perceiver (`configs/perceiver_small.yaml`) – 6 layers, 128 latents to Qwen3 hidden size.
- **Language:** `Qwen/Qwen3-4B-Instruct-2507` with LoRA on q/k/v/o.
- **Geometry head:** Enabled in code but bypassed when `geom_token` is null.
- **Active config:** `configs/stage1_3d.yaml`
  - Data mix: ScanQA 0.7, SQA3D 0.3
  - `num_views: 8`, `image_size: 448`, `max_length: 512`, `view_dropout: 0.3`
  - Train: `batch_size_per_gpu: 6`, `grad_accum: 32`, `lr: 5e-6`, `proj_lr: 1e-4`, `max_steps: 30000`, `save_every_steps: 1500`
  - LoRA: rank 16 on q/k/v/o

---

## 5. How to Train
**Primary launcher (recommended):** `train_fixed.sh` – sets caches, NCCL, batch-size probes, and DeepSpeed ZeRO-3.
```bash
# [mode]=full|debug, [num_gpus]=1-8
./train_fixed.sh full 4
./train_fixed.sh --safe debug 2   # 100 steps, conservative batch
```
Defaults: `CONFIG_FILE=configs/stage1_3d.yaml`, outputs to `ckpts/stage1_3d` (or `_debug`).

**Slurm example:** `run.sh` submits `train_fixed.sh --safe full 8` on HiPerGator B200. Edit account/partition/time as needed. More templates: `scripts/slurm/*.sbatch`.

**Direct torchrun (for experimentation):**
```bash
torchrun --standalone --nproc_per_node=4 -m src.train.train_sft \
  --config configs/stage1_3d.yaml \
  --deepspeed configs/deepspeed_zero3.json \
  --output_dir ckpts/stage1_3d \
  --max_steps 30000
```

**Resuming:** Point `--output_dir` (or update `train_fixed.sh`) to an existing `ckpts/stage2_3d/` with `step_xxxxx/` subfolders; Accelerate loads state automatically.

---

## 6. Monitoring
- Console logs: emitted every `log_every_steps` from `src/train/train_sft.py` (loss, LR, speed, ETA).
- TensorBoard: `tensorboard --logdir ckpts/stage1_3d/logs --port 6006 --bind_all`
- CLI monitor: `python scripts/monitor_training.py --logdir ckpts/stage1_3d/logs/roomplan --watch`
- Log tail: `tail -f logs/stage2/*.log` (if using Slurm templates) or `tail -f pytorchdist_*.out`.
- See `docs/MONITORING_GUIDE.md` for details.

---

## 7. Troubleshooting (common issues)
- **NCCL timeouts / cache stalls:** Use `train_fixed.sh` (already sets `NCCL_TIMEOUT`, local caches, disables IB if needed). Verify `NCCL_DEBUG=INFO` for diagnostics.
- **Out-of-memory (GPU/host):** Re-run with `--safe` or reduce `batch_size_per_gpu`/`grad_accum` in the stage config.
- **Missing weights:** Ensure VGGT checkpoint and Qwen3 cache are present; set `HF_HOME` to a writable local path.
- **Data path errors:** Confirm glob paths in `configs/stage2_3d.yaml` match files under `data/processed/`.

---

## 8. Roadmap / Next Steps
- **Better Stage 1 data:** Regenerate ScanQA/SQA3D with multi-view frames and geometry using `scripts/prep/prepare_scanqa.py --num-views 8 --dataset scanqa|sqa3d`.
- **Stage 2 actions (inference plumbing only):** Reuse or regenerate `data/processed/arkit_synth/train.json` with `scripts/prep/prepare_arkit_from_3dod.py`. Training with `configs/stage2_arkit.yaml` is not yet wired; use `src.inference.arkit_inference` with Stage 1 weights to run smoke tests.

---

## 9. File Pointers
- High-level README: `README.md`
- Configs: `configs/stage1_3d.yaml`, `configs/perceiver_small.yaml`, `configs/deepspeed_zero3.json`
- Launchers: `train_fixed.sh`, `run.sh` (Slurm), `scripts/slurm/*.sbatch`
- Data pipeline: `src/dataio/dataset_builder.py`, `src/dataio/collate_multiview.py`
- Model: `src/models/vggt_qwen3_vlm.py`, `src/models/projector_perceiver.py`
- Trainer: `src/train/train_sft.py`
- Monitoring: `docs/MONITORING_GUIDE.md`, `scripts/monitor_training.py`
