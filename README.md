# VGGT-Qwen3 RoomPlan Pipeline

End-to-end training stack that fuses VGGT’s multi-view perception with Qwen3 so the model can reason over indoor scans and emit ARKit/RoomPlan JSON action plans. Everything in this repo is oriented toward production training on multi-GPU nodes (UF HiPerGator B200), including curriculum configs, dataset builders, DeepSpeed/Accelerate launchers, and monitoring utilities.

---

## Project Overview
- **Goal**: Teach a VGGT + Qwen3 vision-language model to understand 3D indoor scenes from synchronized views and return either natural-language answers or executable RoomPlan instructions.
- **Input**: Up to 10 RGB views (448×448), optional camera/depth metadata, and a prompt.
- **Outputs**: Free-form answers (Stages 1–2) and structured RoomPlan JSON actions (Stage 3).
- **Training stack**: PyTorch 2.4 (CUDA 12.8) + Accelerate + DeepSpeed ZeRO-3. LoRA layers keep training feasible while VGGT stays frozen.

---

## Why This Project Is Different
1. **Multi-view grounding** – `src/models/vggt_qwen3_vlm.py` injects VGGT aggregated tokens directly into the Qwen3 embedding stream at each `<image>` marker, preserving spatial correspondences without flattening the data pipeline.
2. **Perceiver projector** – `src/models/projector_perceiver.py` resamples the VGGT feature cloud into 128 fixed tokens with 6 layers of latent cross-attention, providing controllable bandwidth between vision and language.
3. **Geometry tokens** – Camera intrinsics/extrinsics and depth histograms (generated in `scripts/prep/prepare_scanqa.py`) feed an MLP head that produces extra tokens to stabilize reasoning about pose.
4. **Stage-wise curriculum** – Configs under `configs/` move from instruction following → 3D QA → ARKit planners, gradually unfreezing more parameters and increasing view count.
5. **Action-aware losses** – Stage 3 introduces structured heads and loss weights (`configs/stage3_arkit.yaml`) so the RoomPlan JSON outputs learn alongside the language stream.
6. **Operational fixes baked in** – `train_fixed.sh` hardens runtime behavior (NCCL envs, cache placement, auto batch tuning) and Slurm recipes in `scripts/slurm/` mirror HiPerGator limits.

---

## Architecture in Brief
```
Multi-view images + camera tokens
        │
        ▼
VGGT aggregator (frozen, bfloat16) ──► 2048-d tokens / view
        │
        ▼
Perceiver projector (trainable) ──► 128 latents aligned to Qwen3 hidden size
        │                         (optionally prepended geometry tokens)
        ▼
Qwen3-4B Instruct + LoRA adapters ──► answers + action JSON logits
```
- VGGT weights live under `third_party/vggt/vggt_1B_commercial.pt`. Only the aggregator path is used, so you can keep the rest frozen and in eval mode.
- The Perceiver projector owns most trainable parameters besides LoRA. It is defined via `configs/perceiver_small.yaml`.
- Geometry tokens (`geom_head`) expand `R`, `t`, intrinsics `K`, and depth histograms into up to 8 prepended slots.
- Structured losses (`loss_heads` in Stage 3 config) balance language, JSON action, and geometry-consistency objectives.

---

## Stage Curriculum & Data Sources
| Stage | Config | Views | Datasets (expected under `data/processed/`) | Purpose |
|-------|--------|-------|---------------------------------------------|---------|
| 1. Instruction SFT | `configs/stage1_sft.yaml` | 1 view @448px | `llava/*.json`, `sharegpt4v/*.json`, `docvqa/*.json`, `chartqa/*.json` | Align Qwen3 with multimodal prompting while VGGT stays frozen and geom tokens disabled. |
| 2. Multi-view 3D QA | `configs/stage2_3d.yaml` | 8 views @448px | `scanqa/*.jsonl`, `sqa3d/*.jsonl` (scripts in `scripts/prep/`) | Teach geometric reasoning using multi-view RGB + pose/depth metadata; LoRA covers q/k/v/o projections. |
| 3. RoomPlan actions | `configs/stage3_arkit.yaml` | 10 views @448px | `arkit_synth/*.json` from `scripts/prep/synth_roomplan_instructions.py` | Jointly train language, JSON heads, and geometry consistency for ARKit-friendly action plans. |

Each stage writes checkpoints to `ckpts/<stage_name>/step_xxxxx/` via Accelerate’s `save_state`, so resuming simply points the trainer back to the latest folder.

---

## Repository Map
```
.
├── configs/               # Curriculum + projector + DeepSpeed configs
├── env/environment.yml    # Conda spec (PyTorch 2.4 + CUDA 12.8)
├── scripts/
│   ├── prep/              # Dataset download/conversion utilities
│   └── slurm/             # HiPerGator-ready sbatch templates
├── src/
│   ├── dataio/            # JSON dataloaders + multi-view collator
│   ├── models/            # VGGT-Qwen3 wrapper + Perceiver projector
│   └── train/             # Stage-wise Accelerate trainer
├── train_fixed.sh         # Production launcher with auto heuristics
├── run.sh                 # Example sbatch entry
├── third_party/           # Drop-in Qwen3 + VGGT repos (ignored by git)
├── data/{raw,processed}/  # Raw archives + JSON/JSONL shards
└── ckpts/, logs/          # Training artifacts
```
See `FILE_GUIDE.md` for a per-file description and links to the longer-form guides.

---

## Setup Checklist

### 1. Hardware & Access
- NVIDIA GPUs with ≥20 GB each (verified on 2–8 × H100 B200). Training uses bf16 everywhere.
- Slurm access on HiPerGator or equivalent multi-GPU node plus storage for datasets and checkpoints.
- Licenses for ScanQA, SQA3D, ARKitScenes, and other instruction-tuning corpora if you plan to rebuild Stage 1.

### 2. Clone the project and dependencies
```bash
git clone https://github.com/Sycamorers/vggt-qwen3-roomplan.git
cd vggt-qwen3-roomplan
git clone https://github.com/QwenLM/Qwen3 third_party/Qwen3
git clone https://github.com/Sycamorers/vggt third_party/vggt
```
`third_party/` is ignored to let you drop in private forks if needed.

### 3. Create the Conda env
```bash
conda env create -f env/environment.yml
conda activate roomplan
pip install -e third_party/vggt
# Optional: install Qwen3 editable if their setup.py supports it
```
The environment installs Transformers ≥4.51, Accelerate, DeepSpeed, PEFT, WandB, and CUDA 12.8 wheels.

### 4. Download pretrained checkpoints
| Component | Where to get it | Where to place it |
|-----------|-----------------|-------------------|
| **Qwen/Qwen3-4B-Instruct-2507** | `huggingface-cli download Qwen/Qwen3-4B-Instruct-2507 --token <HF_TOKEN>` | Keep in your Hugging Face cache (`HF_HOME=$PWD/.hf_cache`) or mirror under `third_party/Qwen3`. |
| **VGGT 1B commercial checkpoint** | Request from the official VGGT release (requires commercial license). | Save as `third_party/vggt/vggt_1B_commercial.pt`. The loader in `src/models/vggt_qwen3_vlm.py` looks for this exact filename. |
| **Stage checkpoints (optional)** | Any prior run in `ckpts/<stage>/step_xxxxx/` | Point `--output_dir` to the same folder to resume. |

Tips:
- Export `HF_HOME=$PWD/.hf_cache` before training to avoid NFS hits.
- If your cluster blocks outbound HF downloads, sync the cache to scratch and keep `transformers` in offline mode.

### 5. Configure cache & NCCL defaults
`train_fixed.sh` automatically sets `PYTORCH_ALLOC_CONF`, `NCCL_P2P_DISABLE`, and `NCCL_IB_DISABLE`. If running manually, export:
```bash
export HF_HOME=$PWD/.hf_cache
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256"
```

---

## Data Preparation Workflow

### Dataset schema
All JSON/JSONL shards consumed by `src/dataio/dataset_builder.py` share the fields:
```json
{
  "images": ["relative/or/abs/path.jpg", ...],   // truncated to num_views per stage
  "geom_token": { "R": [...], "t": [...], "K": [...], "depth_hist": [...] },
  "question": "...",                             // stage 1 & 2
  "answer": "...",                               // stage 1 & 2
  "instruction": "...",                          // stage 3
  "action_json": {...},                          // stage 3 output target
  "task": "scanqa" | "sqa3d" | "arkit_actions"…
}
```
The collator handles either `question`/`answer` or `instruction`/`action_json` pairs automatically.

### Stage 1 corpora (instruction SFT)
1. Obtain LLaVA-Instruct, ShareGPT4V, DocVQA, ChartQA, or compatible datasets.
2. Convert each corpus to JSON with the schema above and place them under `data/processed/{llava,sharegpt4v,...}/`.
3. Update the glob paths or mix ratios inside `configs/stage1_sft.yaml` if your filenames differ.

### Stage 2 (ScanQA & SQA3D multi-view QA)
```bash
# Example: build ScanQA from raw ScanNet frames
python scripts/prep/prepare_scanqa.py \
  --dataset scanqa \
  --scan-root data/raw/scannet \
  --qa-file data/raw/scanqa/questions.json \
  --output data/processed/scanqa/train.jsonl \
  --num-views 8

python scripts/prep/prepare_scanqa.py \
  --dataset sqa3d \
  --scan-root data/raw/scannet \
  --qa-file data/raw/sqa3d/questions.json \
  --output data/processed/sqa3d/train.jsonl \
  --num-views 8
```
- The script samples views per scene, reads poses/intrinsics, builds depth histograms, and emits JSON Lines so `MultiViewJsonDataset` can stream them lazily.
- Store validation splits alongside the train files so you can add eval hooks later.

### Stage 3 (ARKitScenes-driven action synthesis)
```bash
# Download ARKitScenes into data/raw/arkitscenes (EULA required)
bash scripts/prep/download_arkitscenes.sh --dest data/raw/arkitscenes

# Generate synthetic instruction/action pairs
python scripts/prep/synth_roomplan_instructions.py \
  --arkit-root data/raw/arkitscenes \
  --output data/processed/arkit_synth/train.json \
  --num-views 10
```
- The generator loops over ARKitScenes plane annotations, crafts Chinese/English hybrid prompts, and writes RoomPlan-compatible `action_json` blobs.

### Validation
After each conversion, spot-check examples with:
```bash
python - <<'PY'
import json, random, glob
sample = random.choice(glob.glob("data/processed/scanqa/*.jsonl"))
with open(sample) as f:
    print(f.readline())
PY
```
and ensure referenced image paths exist.

---

## Training Workflow

### Option A – Hardened launcher (`train_fixed.sh`)
Use this when running long jobs on shared clusters. It probes GPU/host memory, auto-tunes batch size and grad accumulation, and keeps caches local.
```bash
# [mode] = full | debug, [num_gpus] = 1-8
./train_fixed.sh full 4
./train_fixed.sh --safe debug 2   # conservative settings, 100 steps
```
Defaults:
- Config: `configs/stage2_3d.yaml` (edit the script to point at other stages).
- Output: `ckpts/stage2_3d` (or `_debug` when `debug` mode).
- Saves checkpoints every `save_every_steps` defined in the config (e.g., 1,500 for Stage 2).

### Option B – Direct Accelerate/DeepSpeed launch
Useful for experimentation or when you want to run different configs per stage.
```bash
torchrun --standalone --nproc_per_node=4 -m src.train.train_sft \
  --config configs/stage2_3d.yaml \
  --deepspeed configs/deepspeed_zero3.json \
  --output_dir ckpts/stage2_3d \
  --max_steps 30000
```
Notes:
- `src/train/train_sft.py` automatically builds the tokenizer, datasets, and model from the YAML file.
- The DeepSpeed plugin is optional; omit `--deepspeed` for pure DDP.
- Override `--max_steps` for partial runs (Stage 1 default 20k, Stage 3 default 10k).

### GPU scaling
All configs describe per-GPU batch and `grad_accum`. Total effective batch equals `batch_size_per_gpu × grad_accum × #GPUs`. Adjust these fields closer to your hardware if `train_fixed.sh` is not used.

---

## Slurm / HiPerGator Launch

1. Edit `scripts/slurm/stage*_*.sbatch` with your UFRC account, time limit, and mail info.
2. Submit each stage:
   ```bash
   sbatch scripts/slurm/stage1_sft_2xb200.sbatch
   sbatch scripts/slurm/stage2_3d_2xb200.sbatch
   sbatch scripts/slurm/stage3_arkit_2xb200.sbatch
   ```
3. Monitor:
   ```bash
   squeue -u $USER
   tail -f logs/stage2/*.log
   tensorboard --logdir ckpts/stage2_3d/logs --port 6006 --bind_all
   ```
4. Clean up with the maintenance scripts referenced in `SLURM_TRAINING_GUIDE.md` if scratch storage gets tight.

`run.sh` shows a template sbatch job that simply invokes `train_fixed.sh` with 8 A100/H100 GPUs.

---

## Outputs, Monitoring, and Evaluation
- **Checkpoints** – Stored under `ckpts/<stage>/step_xxxxx/`. Each directory contains the Accelerate state (optimizer, scheduler, RNG) plus DeepSpeed shards, so you can resume or convert to Hugging Face format later.
- **Logs** – TensorBoard events land in `ckpts/<stage>/logs/roomplan`. `accelerator.log()` already tracks loss and global step; extend it with custom scalars if needed.
- **Console metrics** – The trainer prints step, loss, LR, throughput, and ETA whenever `log_every_steps` is reached (see `src/train/train_sft.py`).
- **Monitoring helpers** – `MONITORING_GUIDE.md` and `TRAINING_MONITORING_SUMMARY.md` cover `tensorboard`, `scripts/monitor_training.py`, and Slurm-tail commands.
- **Evaluation hooks** – Add your metrics inside `src/eval/` and run them with saved checkpoints. Stage-specific validation JSON can be loaded via the same `MultiViewJsonDataset`.

---

## Supporting Documentation
- `COMPLETE_TRAINING_GUIDE.md` – Deep dive on architecture decisions, fixes, and troubleshooting.
- `SLURM_TRAINING_GUIDE.md` – Idiomatic HiPerGator workflow (sbatch usage, monitoring, cleanup).
- `MONITORING_GUIDE.md` – How to tail checkpoints, plot TensorBoard, and sanity-check dataloaders.
- `TRAINING_MONITORING_SUMMARY.md` – Condensed monitoring checklist.

These guides complement the README but everything required to reproduce training now lives above.

---

## License & Compliance
This project inherits the research-only licenses of VGGT and Qwen3. Respect the dataset terms (ARKitScenes EULA, ScanNet derivatives) and Apple’s RoomPlan usage policies before sharing checkpoints or deploying models commercially. Use of `vggt_1B_commercial.pt` requires the upstream license from the VGGT authors.

---

Happy training! Open issues or PRs if you discover improvements to the curriculum, dataset builders, or launcher scripts.
