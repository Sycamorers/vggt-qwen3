# VGGT-Qwen3 RoomPlan Pipeline

End-to-end training pipeline that injects VGGT's multi-view perception into Qwen3 so the model can emit ARKit/RoomPlan-friendly JSON action plans. The repo is structured for UF HiPerGator B200 nodes and includes Conda + Slurm workflows, dataset builders, and stage-wise training configs.

---

## ⚡ Quick Start

```bash
# 1. Setup environment
conda env create -f env/environment.yml
conda activate roomplan

# 2. Install VGGT
pip install -e third_party/vggt/

# 3. Download weights (see SETUP_WEIGHTS.md)
# Place vggt_1B_commercial.pt in third_party/vggt/

# 4. Prepare data (converts to JSONL format)
python scripts/prep/prepare_scanqa.py
python scripts/prep/prepare_sqa3d.py

# 5. Run training (supports 1-8 GPUs)
./train.sh debug        # Quick test with 2 GPUs
./train.sh full 4       # Full training with 4 GPUs
./train.sh debug 8      # Debug with 8 GPUs

# 6. Monitor with TensorBoard
tensorboard --logdir ckpts/stage2_3d_debug/logs/roomplan --port 6006
```

**Training Script Usage:**
```bash
./train.sh [mode] [num_gpus]

# mode:     debug | full (default: full)
# num_gpus: 1-8 (default: 2)

# Examples:
./train.sh                # Full training, 2 GPUs
./train.sh debug          # Debug mode, 2 GPUs  
./train.sh full 4         # Full training, 4 GPUs
./train.sh debug 8        # Debug mode, 8 GPUs
```

**📚 New to this repo?** Start with:
1. **[SETUP_WEIGHTS.md](SETUP_WEIGHTS.md)** - Download model weights & data
2. **[QUICK_START.md](QUICK_START.md)** - Step-by-step training guide  
3. **[TRAINING_FIXES.md](TRAINING_FIXES.md)** - Bug fixes & troubleshooting
4. **[MONITORING_GUIDE.md](MONITORING_GUIDE.md)** - Monitor training progress

---

## ✨ Recent Updates

### Training Now Works! (November 2025)

Fixed **8 critical bugs** that prevented training from starting:

✅ **Config file format** - Changed from `.json` to `.jsonl` patterns  
✅ **JSONL parsing** - Added support for JSON Lines format (one object per line)  
✅ **VGGT loading** - Load from local checkpoint instead of HuggingFace Hub  
✅ **Dtype handling** - Convert VGGT to bfloat16 to match training precision  
✅ **Dimension matching** - Fixed projector input dim (1024 → 2048)  
✅ **Shape handling** - Pass images directly to aggregator without reshaping  
✅ **Console output** - Added real-time progress monitoring  
✅ **Documentation** - Comprehensive guides for setup and troubleshooting  

See **[TRAINING_FIXES.md](TRAINING_FIXES.md)** for detailed technical information.

### Key Features

- **Real-time monitoring**: Progress bars with loss, LR, speed, and ETA
- **TensorBoard integration**: Automatic logging of metrics  
- **Comprehensive docs**: Setup guides, troubleshooting, and monitoring
- **Flexible GPU scaling**: Supports 1-8 GPUs with automatic configuration
- **Cache management**: All caches relocate to project directory (no NFS issues)
- **Verified working**: Successfully trains on multi-GPU setups with DeepSpeed ZeRO-3

---T-Qwen3 RoomPlan Pipeline

End-to-end training pipeline that injects VGGT’s multi-view perception into Qwen3 so the model can emit ARKit/RoomPlan-friendly JSON action plans. The repo is structured for UF HiPerGator B200 nodes and includes Conda + Slurm workflows, dataset builders, and stage-wise training configs.

---

## Highlights

- **Three-phase curriculum** – progressively teach instruction following, multi-view 3D reasoning, and ARKit-specific action heads (configs in `configs/`).
- **Flexible GPU scaling** – Training script supports 1-8 GPUs with automatic DeepSpeed ZeRO-3 configuration.
- **HiPerGator-ready** – Slurm templates target single-node multi-GPU runs with optimized cache management.
- **Deterministic data pipelines** – scripts in `scripts/prep/` transform raw ScanQA / SQA3D / ARKitScenes data into JSONL shards consumed by trainers.
- **Modular vision-text stack** – drop in your own `Qwen3` and `VGGT` repos under `third_party/`, swap projector configs, or extend LoRA heads without touching the core trainer.

---

## Repository Layout

```
vggt-qwen3-roomplan/
├── configs/
│   ├── stage{1,2,3}_*.yaml     # Curriculum configs
│   └── deepspeed_zero3.json    # Default DS strategy
├── data/
│   ├── raw/                    # Vendor datasets (not tracked)
│   └── processed/              # JSONL shards produced by prep scripts
├── env/environment.yml         # Conda environment definition
├── scripts/
│   ├── prep/                   # Download + preprocessing utilities
│   └── slurm/                  # Batch templates for HiPerGator
├── src/                        # Training / evaluation entrypoints
├── third_party/
│   ├── Qwen3/                  # Clone or symlink Qwen3 repo
│   └── vggt/                   # Clone or symlink VGGT repo
├── ckpts/                      # Saved checkpoints
├── logs/                       # Training & evaluation logs
└── README.md
```

> **Important**: `third_party/Qwen3` and `third_party/vggt` are intentionally empty. Clone or symlink the official repos so this project can reuse their code and pretrained weights.

---

## Prerequisites

- **Hardware**: NVIDIA GPUs with CUDA support (tested on H100s, supports 1-8 GPUs). Adjust batch sizes based on GPU memory.
- **Accounts & Access**:
  - UF Research Computing account with Slurm access and a project allocation.
  - Dataset licenses for ARKitScenes, ScanQA, SQA3D, ReferIt3D, ScanRefer, Scan2Cap, etc.
- **Software**:
  - Conda or Mamba (Miniconda ≥ 23).
  - Slurm client (`sbatch`, `srun`, `sacct`).
  - CUDA 12.8-compatible drivers on compute nodes.

---

## Pre-downloaded assets

- Hugging Face `Qwen/Qwen3-4B-Instruct-2507` is already mirrored under `.hf_cache/`. Set `HF_HOME=$PWD/.hf_cache` to reuse it locally or rsync that folder to HiPerGator scratch to avoid re-downloading on compute nodes.

---

## Environment Setup — HiPerGator

```bash
module purge
module load cuda/12.8 gcc/12.2  # adapt to your HiPerGator toolchain

cd vggt-qwen3-roomplan
conda env create -f env/environment.yml
conda activate roomplan

# Install editable dependencies once you bring in the third-party repos
pip install -e third_party/vggt
pip install -e third_party/Qwen3  # only if the upstream repo supports it
```

The Conda spec installs PyTorch 2.4 + CUDA 12.8 wheels plus the training stack (Transformers 4.51+, Accelerate, PEFT, DeepSpeed, WandB). Feel free to add extra logging/monitoring packages locally.

---

## Environment Setup — Local workstation

For CUDA 11.8-era desktop GPUs or CPU-only smoke tests:

```bash
cd vggt-qwen3-roomplan
conda env create -f env/environment_local.yml
conda activate roomplan-local

export HF_HOME=$PWD/.hf_cache  # reuse the pre-downloaded Qwen3 weights
pip install -e third_party/vggt
pip install -e third_party/Qwen3  # if editable install works for your platform
```

---

## HiPerGator Quickstart (B200)

1. **Clone + modules**
   ```bash
   module purge
   module load cuda/12.8 gcc/12.2
   git clone git@github.com:Sycamorers/vggt-qwen3-roomplan.git
   cd vggt-qwen3-roomplan
   ```
2. **Conda env**
   ```bash
   conda env create -f env/environment.yml
   conda activate roomplan
   ```
3. **Third-party repos**
   ```bash
   git clone https://github.com/QwenLM/Qwen3 third_party/Qwen3
   git clone https://github.com/Sycamorers/vggt third_party/vggt
   pip install -e third_party/vggt
   pip install -e third_party/Qwen3  # if supported
   ```
4. **Datasets**
   - Place raw data under `data/raw/<dataset>/`.
   - Use prep scripts (they write JSONL shards under `data/processed/`):
     ```bash
     bash scripts/prep/download_arkitscenes.sh --dest data/raw/arkitscenes   # example
     python scripts/prep/build_*.py  # run relevant builders per dataset
     ```
5. **Stage configs**
   - Adjust `configs/stage{1,2,3}_*.yaml` if needed (paths, batch sizes).
   - Optional: set `HF_HOME` to a node-local path to avoid shared FS thrash.
6. **Launch (Slurm)**
   ```bash
   sbatch scripts/slurm/stage1_sft_2xb200.sbatch
   sbatch scripts/slurm/stage2_3d_2xb200.sbatch
   sbatch scripts/slurm/stage3_arkit_2xb200.sbatch
   ```
   Tail logs: `tail -f logs/<stage>/*.log`; monitor: `squeue -u $USER`.

---

## Data Preparation

1. **Download source datasets** into `data/raw/<dataset_name>/`. Example:

   ```bash
   bash scripts/prep/download_arkitscenes.sh --dest data/raw/arkitscenes
   ```

   The script checks for the ARKitScenes EULA; for third-party datasets download manually and drop the archives in `data/raw/`.

2. **Convert to JSON packs**. Each script reads from `data/raw/` and writes `data/processed/<dataset>/*.json`.

   ```bash
   # Stage-2 datasets
   python scripts/prep/prepare_scanqa.py \
     --config configs/stage2_3d.yaml \
     --raw-root data/raw \
     --out-root data/processed

   # ARKit synthetic instructions (Stage-3)
   python scripts/prep/synth_roomplan_instructions.py \
     --arkit-root data/raw/arkitscenes \
     --out data/processed/arkit_synth
   ```

3. **Verify splits**. A quick sanity script (not tracked) can iterate over JSONL files and ensure fields `images`, `camera`, `instruction`, and `target_action` exist. Downstream trainers expect that schema.

---

## Training Pipeline

| Stage | Config | Views | Data Sources | Objective Highlights |
|-------|--------|-------|--------------|----------------------|
| Stage 1 – Instruction SFT | `configs/stage1_sft.yaml` | 1 view @ 448px | LLaVA-Instruct, ShareGPT4V, DocVQA, ChartQA | Teach base Qwen3 to follow structured prompts while VGGT remains frozen. LoRA rank 16 on Q/K/V/O. |
| Stage 2 – 3D Alignment | `configs/stage2_3d.yaml` | 8 views @ 448px | ScanQA, SQA3D, ScanRefer, ReferIt3D, Scan2Cap | Introduce geometric tokens, multi-view fusion, and freeze lower text layers for stability. |
| Stage 3 – RoomPlan Actions | `configs/stage3_arkit.yaml` | 10 views @ 448px | Synthetic ARKitScenes instructions | Jointly optimize language + JSON action heads + geometry consistency losses for executable RoomPlan outputs. |

**GPU Scaling:** The training script automatically adjusts for 1-8 GPUs:
- Effective batch size = `batch_per_gpu (6) × grad_accum (32) × num_gpus`
- Example: 4 GPUs → effective batch of 768
- DeepSpeed config generated dynamically based on GPU count

Shared practices:

- **Vision Backbone**: `third_party/vggt` is referenced directly; keep it frozen to prevent catastrophic forgetting unless you have compute for joint finetuning.
- **Projector**: `configs/perceiver_small.yaml` stitches VGGT tokens to Qwen3 hidden states. Modify to experiment with larger bottlenecks.
- **Optimization**: DeepSpeed ZeRO-3 config lives in `configs/deepspeed_zero3.json`. Override via CLI flag if you need custom bucket sizes.
- **Batching**: Stage configs set higher `grad_accum` so 2×GPU runs preserve the effective global batch used in the earlier 8×GPU recipes; lower it if wall-clock is a concern.

---

## Launching Jobs

### Interactive training (any GPU count)

Quick examples with the training script:

```bash
# Debug mode with 2 GPUs (default)
./train.sh debug

# Full training with 4 GPUs  
./train.sh full 4

# Debug with all 8 GPUs
./train.sh debug 8

# Use train_fixed.sh for cache management and NCCL fixes
./train_fixed.sh full 4
```

The script automatically:
- Validates GPU count (1-8)
- Generates appropriate accelerate config
- Calculates effective batch size
- Sets up DeepSpeed ZeRO-3

### Local / interactive dry run

Use torchrun to smoke-test configs on a smaller number of GPUs:

```bash
srun --account=<ACCOUNT> --partition=gpu --gpus=2 --cpus-per-task=24 \
     --mem=96G --time=02:00:00 --pty bash -i

conda activate roomplan
torchrun --standalone --nproc_per_node=2 -m src.train.train_sft \
  --config configs/stage1_sft.yaml \
  --deepspeed configs/deepspeed_zero3.json \
  --max_steps 1000 \
  --output_dir ckpts/dry_run_stage1
```

### Full-scale Slurm submits

Update the `SBATCH` directives inside `scripts/slurm/stage*_*.sbatch` (account, QoS, email). Then submit:

```bash
sbatch scripts/slurm/stage1_sft_2xb200.sbatch
sbatch scripts/slurm/stage2_3d_2xb200.sbatch
sbatch scripts/slurm/stage3_arkit_2xb200.sbatch
```

Each script:

- Loads the Conda env and third-party repos.
- Launches `torchrun` across 1 node × 2 GPUs.
- Logs to `logs/<stage>/<timestamp>` and checkpoints to `ckpts/<stage>/`.

Monitor with `squeue -u $USER` and `tail -f logs/<stage>/*.log`.

---

### CPU-only local smoke (no external deps)

Use the toy dataset + mock VGGT stub to sanity-check the training loop without GPUs or large checkpoints:

```bash
python scripts/prep/make_toy_dataset.py --output data/processed/toy/train.json
python -m src.train.train_sft \
  --config configs/local_smoke.yaml \
  --output_dir ckpts/local_smoke
```

For local single-GPU (RTX 3090/4090) runs with CUDA 11.8 drivers, create the `roomplan-local` env:

```bash
conda env create -f env/environment_local.yml
conda activate roomplan-local
```

`configs/local_smoke.yaml` swaps in a tiny GPT-2 stub (`sshleifer/tiny-gpt2`) and sets `vision_backbone: mock`, so everything runs offline once the small HF weights are downloaded.

### Local Qwen3 mini run (2×GPU, toy data)

This uses the Qwen/Qwen3-4B-Instruct-2507 checkpoint with the mock vision backbone to keep VRAM reasonable on 3090/4090:

```bash
conda activate roomplan-local
export HF_HOME=$PWD/.hf_cache  # optional cache path
MASTER_ADDR=127.0.0.1 MASTER_PORT=29515 \
torchrun --nproc_per_node=2 --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
  -m src.train.train_sft \
  --config configs/local_qwen3.yaml \
  --output_dir ckpts/local_qwen3_2gpu \
  --max_steps 40
```

Ensure the `Qwen/Qwen3-4B-Instruct-2507` weights/tokenizer are available in your HF cache or mirrored locally.

---

## Artifacts & Evaluation

- **Checkpoints**: Stored under `ckpts/<stage>/`. Stage-3 checkpoints include JSON action head weights; keep Stage-2 around for ablations.
- **WandB / TensorBoard**: Enable your preferred logger via environment variables (e.g., `WANDB_PROJECT=roomplan torchrun ...`).
- **Evaluation hooks**: `src/` contains evaluation entrypoints (add scripts for ScanQA accuracy or ARKit plan validity). Extend `scripts/slurm/` with eval templates to reproduce paper numbers.

---

## Customization Tips

- **Different base LLM**: Change `model.name_or_path` and `tokenizer_path` in configs plus update LoRA target modules.
- **Alternate projector**: Point `model.projector` to another YAML (e.g., a larger Perceiver) to change token folding capacity.
- **Unfreezing VGGT**: Set `freeze_vision: false` and lower the learning rate; watch for VRAM use.
- **Action schema changes**: Modify `loss_heads` in `configs/stage3_arkit.yaml` and align them with the trainer’s head definitions in `src/`.

---

## Contributing

1. Fork or branch off `main`.
2. Run `pre-commit` or your preferred formatter (not bundled) before pushing.
3. Open a PR with experiment notes, dataset versions, and hardware specs so results remain reproducible.

Please respect the licenses of all third-party datasets and models. ARKitScenes is restricted to research use; obtain explicit permission for commercial deployments.

---

## License

This repository inherits the licenses of the upstream Qwen3 and VGGT projects. Unless noted otherwise, all original code here is released under the same research-only terms. Ensure compliance with Apple’s ARKit license and UF Research Computing policies before redistributing checkpoints.
