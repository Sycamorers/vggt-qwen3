# VGGT-Qwen3 RoomPlan (Local Training Stack)

End-to-end training stack that fuses VGGT multi-view perception with Qwen3 for indoor 3D reasoning. This README reflects the code paths and the data that are actually present in this workspace so a new contributor can start training immediately.

---

## What You Can Train Right Now
- **Stage available:** Multi-view 3D QA (Stage 2) using the processed `scanqa` and `sqa3d` JSONL shards that are already in `data/processed/`.
- **Launcher:** `train_fixed.sh` (supports 1–8 GPUs with DeepSpeed ZeRO-3; Slurm template in `run.sh`).
- **Model stack:** VGGT aggregator (frozen) → Perceiver projector (trainable) → Qwen/Qwen3-4B-Instruct-2507 with LoRA adapters.
- **Geometry tokens:** Configured, but the local datasets have `geom_token: null`, so geometry embeddings are not used unless you regenerate data with poses/depth.

---

## Current Data (Local Files)
The dataloader (`src/dataio/dataset_builder.py`) streams JSON/JSONL that point to local images. What exists now:

| Dataset | Path glob | Format | Views per sample | Geometry | Notes |
|---------|-----------|--------|------------------|----------|-------|
| ScanQA (processed) | `data/processed/scanqa/*.jsonl` | JSON Lines | 1 (top-down / bird’s-eye) | `geom_token: null` | Uses images under `data/processed/SQA3D/bird/…png`. |
| SQA3D (processed) | `data/processed/sqa3d/*.jsonl` | JSON Lines | 1 (top-down / bird’s-eye) | `geom_token: null` | Same image source as above. |
| DocVQA (unused by current config) | `data/processed/DocVQA/llava_instruct_80k.json` | JSON | n/a | n/a | Present but not wired to the active config. |
| ARKit synthetic (missing) | expected `data/processed/arkit_synth/*.json` | — | — | — | Not present; Stage 3 cannot run until generated. |

### Sample record (from `data/processed/sqa3d/train.jsonl`)
```json
{
  "images": ["data/processed/SQA3D/bird/scene0380_00_bird.png"],
  "geom_token": null,
  "question": "What color is the desk to my right?",
  "answer": "brown",
  "task": "sqa3d"
}
```
- The loader truncates/uses up to `num_views` from the config (8 for Stage 2). Because only 1 view exists, training runs as single-view while still exercising the multi-view pipeline.
- `geom_token` is optional. When null, the geometry head is skipped and only vision tokens are injected.

If you regenerate data with poses/depth, keep the same keys (`R`, `t`, `K`, `depth_hist`) so `encode_geom` can produce geometry tokens.

---

## Data Schema Expected by the Code
`src/dataio/dataset_builder.py` accepts both `.json` (arrays) and `.jsonl` (one object per line). Required/optional fields per sample:
```json
{
  "images": ["path1.png", "path2.png", "..."],    // required; truncated to num_views
  "geom_token": { "R": [...], "t": [...], "K": [...], "depth_hist": [...] } | null,
  "question": "...",                              // or "instruction" for action data
  "answer": "...",                                // or "action_json" for action data
  "task": "scanqa" | "sqa3d" | "arkit_synth" ...
}
```
The collator (`src/dataio/collate_multiview.py`):
- Resizes images to the configured `image_size` (448).
- Builds text as `{question}\n<image>\n{answer}` so the `<image>` token is where VGGT/Perceiver features are injected.
- Pads to at least `num_vis_tokens + geom_tokens + 64` tokens to leave space for visual embeddings.

---

## Model & Training Stack (matches code)
- **Vision encoder:** VGGT aggregator loaded from `third_party/vggt/vggt_1B_commercial.pt`, kept in eval/frozen mode (`freeze_vision: true`).
- **Projector:** Perceiver (`configs/perceiver_small.yaml`), 6 layers, 128 latents → maps VGGT features to Qwen3 hidden size.
- **Language model:** `Qwen/Qwen3-4B-Instruct-2507` with LoRA on q/k/v/o projections (see `configs/stage2_3d.yaml`).
- **Geometry head:** MLP over concatenated camera/depth stats; skipped when `geom_token` is missing.
- **Prompt injection:** `<image>` positions in `input_ids` are replaced with `[geom_tokens]+[vision_tokens]` before the forward pass (`src/models/vggt_qwen3_vlm.py`).
- **Precision:** bf16; optimizer AdamW with a higher LR for projector params (`proj_lr`).

---

## Ready-to-Run Training (Stage 2 QA)
`train_fixed.sh` is hardened for multi-GPU and cluster runs. By default it launches Stage 2 QA with DeepSpeed ZeRO-3.

### Prerequisites
- Download Qwen3 weights (or cache via HF): `Qwen/Qwen3-4B-Instruct-2507`.
- Place VGGT checkpoint at `third_party/vggt/vggt_1B_commercial.pt`.
- Conda env from `env/environment.yml` (PyTorch 2.4, CUDA 12.8) and install `third_party/vggt` editable.

### Local launch (single or multi-GPU)
```bash
# [mode] full|debug, [num_gpus] 1-8
./train_fixed.sh full 4
./train_fixed.sh --safe debug 2  # conservative settings, 100 steps
```
Defaults in `train_fixed.sh`:
- Config: `configs/stage2_3d.yaml`
- Output: `ckpts/stage2_3d` (or `_debug` in debug mode)
- NCCL/cache hardening and automatic batch-size probes are applied

### Slurm template
`run.sh` submits `train_fixed.sh` on HiPerGator B200 (`--gpus-per-node=8`). Adjust account/time/mem if needed.

---

## Stage 2 Config (what the trainer uses)
Key fields from `configs/stage2_3d.yaml`:
- Data: `scanqa` (0.7) + `sqa3d` (0.3), `num_views: 8`, `image_size: 448`, `max_length: 512`, `view_dropout: 0.3`.
- Train: `batch_size_per_gpu: 6`, `grad_accum: 32`, `lr: 5e-6`, `proj_lr: 1e-4`, `max_steps: 30000`, `save_every_steps: 1500`.
- LoRA: rank 16 on q/k/v/o projections.
Because the local shards are single-view and lack geometry, training effectively runs as single-view VGGT features with no geometry tokens. You can still keep `num_views=8`; the loader simply uses the available view count.

---

## Repository Map (trimmed to active pieces)
```
configs/                # Stage & projector configs (stage2_3d is active)
data/processed/         # scanqa/*.jsonl, sqa3d/*.jsonl, DocVQA JSON
data/raw/               # placeholders + ARKit README_download.txt
src/dataio/             # dataset builder + multi-view collator
src/models/             # VGGT-Qwen3 wrapper, Perceiver projector
src/train/train_sft.py  # Accelerate/DeepSpeed training entry
train_fixed.sh          # Hardened launcher (default stage2_3d)
run.sh                  # Slurm example invoking train_fixed.sh
logs/, ckpts/           # Training outputs
```

---

## Project Status & Next Steps
- **Status:** Ready to train Stage 2 (multi-view QA) with the existing single-view ScanQA/SQA3D shards; distributed training works via `train_fixed.sh`.
- **To unlock Stage 3 (RoomPlan actions):** Run `scripts/prep/synth_roomplan_instructions.py` to create `data/processed/arkit_synth/*.json`, then point `train_fixed.sh` to `configs/stage3_arkit.yaml`.
- **To improve Stage 2 data fidelity:** Regenerate ScanQA/SQA3D with multi-view frames and geometry (`scripts/prep/prepare_scanqa.py`) so `geom_tokens` and the VGGT multi-view aggregation are fully utilized.

