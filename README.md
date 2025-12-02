# VGGT-Qwen3 RoomPlan

A multi-stage vision-language model for 3D indoor scene understanding, combining VGGT visual aggregation with Qwen3-4B language modeling. This project trains a model to answer questions about 3D scenes (ScanQA/SQA3D) and evaluate on RoomPlan-style spatial reasoning tasks.

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Environment Setup](#2-environment-setup)
3. [Data Preparation](#3-data-preparation)
4. [Model Architecture & Configuration](#4-model-architecture--configuration)
5. [Training](#5-training)
6. [Checkpoints & Weights](#6-checkpoints--weights)
7. [Baseline Evaluation Results](#7-baseline-evaluation-results)
8. [Monitoring & Debugging](#8-monitoring--debugging)
9. [ARKit / RoomPlan Evaluation](#9-arkit--roomplan-evaluation)
10. [Current Status & Known Limitations](#10-current-status--known-limitations)
11. [Extending the Project](#11-extending-the-project)
12. [Repository Structure](#12-repository-structure)
13. [Quick Start Checklist](#13-quick-start-checklist)

---

## 1) Project Overview

### What This Project Does

This repository trains a VGGT + Qwen3 vision-language model to:
- **Understand 3D indoor scenes**: Answer questions about objects, spatial relationships, and scene composition
- **Process multi-view imagery**: Aggregate visual information from multiple camera viewpoints
- **(Future) Generate RoomPlan actions**: Emit structured JSON for spatial tasks like anchor placement

### Architecture Pipeline

```
Multi-view Images → VGGT Aggregator (frozen) → Perceiver Projector → [Geometry Tokens] → Qwen3-4B LLM
```

- **VGGT**: Frozen vision backbone that aggregates multi-view features (`third_party/vggt/vggt_1B_commercial.pt`)
- **Perceiver**: Trainable projector (128 latents, 6 layers) that maps VGGT features to Qwen3's embedding space
- **Qwen3-4B**: Language model (`Qwen/Qwen3-4B-Instruct-2507`) that generates text responses conditioned on visual tokens
- **Geometry tokens** (optional): Camera pose/depth information for explicit 3D reasoning

See implementation in `src/models/vggt_qwen3_vlm.py` and `configs/perceiver_small.yaml`.

### Training Stages

- **Stage 1 (Current)**: 3D visual question answering on ScanQA + SQA3D datasets
  - Status: ✅ Fully implemented and ready to run
  - Data: 44,120 training samples, 7,231 test samples
  - Config: `configs/stage1_3d.yaml`
  - Output: `ckpts/stage1_3d/`

- **Stage 2 (Planned)**: RoomPlan action prediction on ARKit scenes
  - Status: ⚠️ Inference-only plumbing exists; training not yet implemented
  - Data: 9 synthetic ARKit samples available for evaluation
  - Config: `configs/stage2_arkit.yaml` (reference only)
  - Limitation: Trainer doesn't support structured `action_json` targets

### Key Features

- ✅ Distributed training via DeepSpeed ZeRO-3 (1-8 GPUs)
- ✅ Slurm integration for HPC clusters (HiPerGator templates included)
- ✅ Comprehensive monitoring and debugging tools
- ✅ Proper train/test splits with baseline evaluation results
- ✅ Memory-optimized training with automatic batch size adjustment
- ✅ NCCL timeout fixes and cache relocation for stability

### Documentation

- **`docs/COMPLETE_TRAINING_GUIDE.md`**: Detailed architecture, data pipeline, training workflow, and troubleshooting
- **`docs/QUICK_START.md`**: Condensed "run it now" steps for getting started quickly
- **`docs/SLURM_TRAINING_GUIDE.md`**: Slurm-specific configuration and job submission
- **`docs/MONITORING_GUIDE.md`**: Training monitoring, TensorBoard, and debugging techniques
- **`docs/FILE_GUIDE.md`**: Detailed file-by-file repository documentation

---

## 2) Environment Setup

### Requirements

- **Hardware**: NVIDIA GPUs with ≥20 GB VRAM each
- **Software**: Linux, CUDA 12.8, Python 3.10
- **Training precision**: bfloat16

### Installation Steps

1. **Create the Conda environment** (PyTorch 2.4, CUDA 12.8):
   ```bash
   conda env create -f env/environment.yml
   conda activate roomplan
   ```

2. **Install VGGT** (editable mode for development):
   ```bash
   pip install -e third_party/vggt
   ```

3. **(Optional) Install Qwen3** editable if needed; otherwise Hugging Face cache is sufficient.

4. **Verify installation**:
   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
   ```

### Cache Configuration

The training script (`train_fixed.sh`) automatically sets project-local caches to avoid NFS issues:
- `TRITON_CACHE_DIR`, `TORCH_HOME`, `HF_HOME`, `TRANSFORMERS_CACHE`, `HF_DATASETS_CACHE`
- `PYTHONPYCACHEPREFIX`, `TMPDIR`
- All caches stored in `$PROJECT_DIR/.cache/`

### Required Model Weights

Before training, ensure you have:
1. **VGGT checkpoint**: `third_party/vggt/vggt_1B_commercial.pt` (frozen vision backbone)
2. **Qwen3 model**: Automatically downloaded from Hugging Face to cache on first run
   - Model ID: `Qwen/Qwen3-4B-Instruct-2507`

---

## 3) Data Preparation

### Current Dataset Organization

The repository includes pre-processed datasets with proper train/test splits:

**Training Datasets**:
- **ScanQA**: 21,161 train / 3,567 test samples
  - `data/processed/scanqa/train_split.jsonl` – Training set
  - `data/processed/scanqa/test_split.jsonl` – Test/evaluation set
  
- **SQA3D**: 22,959 train / 3,664 test samples
  - `data/processed/sqa3d/train_split.jsonl` – Training set
  - `data/processed/sqa3d/test_split.jsonl` – Test/evaluation set
  
- **Total**: ~44,120 training samples, ~7,231 test samples

**Evaluation Dataset**:
- **ARKit Synthetic**: 9 samples for inference/evaluation
  - `data/processed/arkit_synth/train.json` – ARKit synthetic data (inference only, no training head)

**Data Characteristics**:
- All samples use single bird-view images from SQA3D (`data/processed/SQA3D/bird/<scene>_bird.png`)
- `geom_token: null` (geometry tokens not included in current data)
- Each record contains: `images`, `geom_token`, `task`, `question`, `answer`, `scene_id`, `question_id`, `object_ids`, `object_names`

**Default Stage 1 training mix**: **0.7 ScanQA / 0.3 SQA3D** from train splits only (see `configs/stage1_3d.yaml`).

### Raw Data Sources

If you need to regenerate data from scratch:

- **SQA3D**: https://zenodo.org/records/7792397#.ZCkprfFBx3g
- **ScanQA**: https://drive.google.com/drive/folders/1-21A3TBE0QuofEwDg5oDz2z0HEdbVgL2
- **ScanNet** (for multi-view + geometry): https://github.com/ScanNet/ScanNet

### Data Preprocessing Options

**Option A: Multi-view with geometry tokens** (requires ScanNet RGB/depth/poses):
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
Outputs: `images` (up to `num_views`), `geom_token` (R, t, K, depth_hist), `question`, `answer`, `task`.

**Option B: Fast single-view rebuild** (uses existing bird-view images):
```bash
python scripts/prep/rebuild_scanqa_sqa3d.py
```
Recreates train/test splits by pairing QA JSON with `data/processed/SQA3D/bird/<scene>_bird.png`; sets `geom_token: null`.

**Option C: ARKit synthetic data** (evaluation/inference only):
```bash
python scripts/prep/prepare_arkit_from_3dod.py \
  --arkit-training-root data/processed/ARKit/Training \
  --output data/processed/arkit_synth/train.json \
  --num-views 10 \
  --max_scenes 10
```
Note: ARKit data is for evaluation purposes only; Stage 2 training is not yet implemented.

### How the Dataloader Works

- **`src/dataio/dataset_builder.py`**: 
  - Loads JSON/JSONL files
  - Normalizes to `{images, geom_token, question|instruction, answer|action_json, task}`
  - Truncates to `num_views`, attempts image load with fallback
  - Mixes datasets by ratio via `MultiSourceDataset`
  
- **`src/dataio/collate_multiview.py`**: 
  - Resizes/crops images
  - Builds `{question}\n<image>\n` prompts
  - Appends answers, pads to `max_length`
  - Reserves space for vision/geometry tokens
  - Stacks tensors for batch processing

---

## 4) Model Architecture & Configuration

### Model Components

- **Vision Encoder**: VGGT aggregator (`third_party/vggt/vggt_1B_commercial.pt`)
  - Status: Frozen (not trained)
  - Purpose: Multi-view feature extraction and aggregation
  
- **Vision-Language Projector**: Perceiver architecture
  - Config: `configs/perceiver_small.yaml`
  - Parameters: 128 latents, 6 layers
  - Output dimension: Matches Qwen3 hidden size
  - Status: Trainable
  
- **Language Model**: Qwen3-4B-Instruct-2507
  - Status: Trainable (full fine-tuning)
  - Precision: bfloat16
  
- **Geometry Head** (optional): Projects camera parameters to tokens
  - Status: Trainable when `geom_token` is available

### Training Configuration

**Active Config**: `configs/stage1_3d.yaml`

**Data Settings**:
- Datasets: ScanQA (0.7 weight) + SQA3D (0.3 weight)
- `num_views: 8` – Maximum views per sample
- `image_size: 448` – Input image resolution
- `max_length: 512` – Maximum sequence length
- `view_dropout: 0.3` – Randomly drop views during training

**Training Hyperparameters**:
- `batch_size_per_gpu: 6` (auto-adjusted by `train_fixed.sh` based on available memory)
- `grad_accum: 32` – Gradient accumulation steps
- `max_steps: 30000` – Total training steps
- `save_every_steps: 1500` – Checkpoint save frequency
- `log_every_steps: 10` – Logging frequency

**Optimizer Settings**:
- Optimizer: AdamW
- Learning rates:
  - Projector: 2e-4 (higher LR for faster convergence)
  - Language model: 5e-6
  - Geometry head: 2e-4
- LR schedule: Cosine with warmup (10% warmup ratio)
- Weight decay: 0.01

**DeepSpeed Configuration**: `configs/deepspeed_zero3.json`
- Zero Optimization Stage 3
- Offload optimizer to CPU: enabled
- Gradient accumulation fusion: enabled
- Contiguous memory optimization: enabled

### Stage 2 Configuration (Reference Only)

**Config**: `configs/stage2_arkit.yaml`

⚠️ **Warning**: Stage 2 training is **not implemented**. This config is for reference and future development only.

**Known Issues**:
- `action_json` targets are structured (dict), but trainer assumes text labels
- `loss_heads` configuration is not consumed by current trainer
- Only inference plumbing is available with Stage 1 weights

---

## 5) Training

### Primary Training Script

The main launcher is `train_fixed.sh`, which includes:
- Automatic cache relocation to project directory
- NCCL timeout fixes and network optimization
- GPU/host memory probing and auto-adjustment
- DeepSpeed ZeRO-3 setup

### Training Modes

**Debug Mode** (100 steps, fast iteration):
```bash
./train_fixed.sh debug 1           # Single GPU
./train_fixed.sh debug 2           # 2 GPUs
```
Outputs to: `ckpts/stage1_3d_debug/`

**Full Training** (30,000 steps):
```bash
./train_fixed.sh full 4            # 4 GPUs
./train_fixed.sh full 8            # 8 GPUs
```
Outputs to: `ckpts/stage1_3d/` (or `ckpts/stage1_3d_test/` as currently configured)

**Safe Mode** (conservative settings for stability):
```bash
./train_fixed.sh --safe full 2     # Force BATCH_PER_GPU=1, GRAD_ACCUM=4
```

### Command-Line Arguments

```bash
./train_fixed.sh [--safe] [mode] [num_gpus]
```

- `--safe`: Force conservative memory settings (optional)
- `mode`: `debug` or `full` (default: `full`)
- `num_gpus`: 1-8 (default: 2)

### Slurm Training (HPC Clusters)

**Quick Start**:
```bash
sbatch run.sh   # Submits 8-GPU job on hpg-b200 partition
```

**Custom Slurm Jobs**:
```bash
# Edit run.sh to customize:
# - Account/partition
# - Time limit
# - Memory per GPU
# - Number of GPUs
# - Training mode (debug/full)

sbatch scripts/slurm/stage2_3d_2xb200.sbatch  # 2-node, 8 GPUs per node
```

**Slurm Configuration** (`run.sh`):
```bash
#SBATCH --job-name=vggt-qwen3
#SBATCH --partition=hpg-b200
#SBATCH --gpus=8
#SBATCH --mem-per-gpu=120gb
#SBATCH --time=48:00:00
```

See `docs/SLURM_TRAINING_GUIDE.md` for detailed Slurm usage.

### Resume Training

Checkpoints are saved every `save_every_steps` (default: 1500) to `ckpts/stage1_3d/checkpoint-*/`.

**To resume**:
```bash
# Training automatically resumes if output_dir contains checkpoints
./train_fixed.sh full 4

# Or manually specify checkpoint directory by editing train_fixed.sh
# Change OUTPUT_DIR to point to existing checkpoint folder
```

Accelerate automatically detects and loads the latest checkpoint state.

### Environment Variables Set by `train_fixed.sh`

**Cache Directories**:
- `TRITON_CACHE_DIR`, `TORCH_HOME`, `HF_HOME`, `TRANSFORMERS_CACHE`
- `HF_DATASETS_CACHE`, `PYTHONPYCACHEPREFIX`, `TMPDIR`

**NCCL Settings** (network optimization):
- `NCCL_TIMEOUT=3600` – 1 hour timeout
- `NCCL_DEBUG=INFO` – Enable debugging
- `NCCL_P2P_DISABLE=0` – Enable peer-to-peer transfers
- `NCCL_NET_MERGE_LEVEL=LOC` – Prevent NIC fusion issues
- `NCCL_IB_HCA` – InfiniBand adapter selection (HiPerGator default)

**PyTorch Settings**:
- `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256` – Memory allocation tuning
- `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` – Better error reporting

### Memory Management

The script automatically probes GPU and host memory, then adjusts:
- `BATCH_PER_GPU`: Reduced if GPU memory < 25 GB
- `GRAD_ACCUM`: Reduced if host memory is constrained
- `DATALOADER_NUM_WORKERS`: Set based on available resources

**Manual Override** (for expert users):
```bash
export BATCH_PER_GPU=4
export GRAD_ACCUM=16
./train_fixed.sh full 4
```

---

## 6) Checkpoints & Weights

### Checkpoint Directory Structure

**Stage 1 Output** (`ckpts/stage1_3d/`):
```
ckpts/stage1_3d/
├── checkpoint-1500/          # Periodic checkpoints for resume
├── checkpoint-3000/
├── ...
├── pytorch_model/            # DeepSpeed ZeRO-3 shards (per-rank states)
│   ├── mp_rank_00_model_states.pt
│   ├── mp_rank_01_model_states.pt
│   └── ...
├── pytorch_model_fp32/       # Converted fp32 weights (after conversion)
│   ├── pytorch_model-00001-of-00005.bin
│   ├── pytorch_model-00002-of-00005.bin
│   └── ...
├── logs/                     # TensorBoard logs
├── random_states_*.pkl       # Random number generator states
├── scheduler.bin             # LR scheduler state
└── zero_to_fp32.py           # Conversion script
```

### Converting ZeRO-3 Checkpoints to FP32

DeepSpeed ZeRO-3 shards model weights across GPUs. To use checkpoints for inference or analysis, convert to standard fp32:

```bash
cd ckpts/stage1_3d

# Convert ZeRO shards to fp32 weights
python zero_to_fp32.py pytorch_model pytorch_model_fp32
```

**Output**: `pytorch_model_fp32/` directory with standard PyTorch weights
- Compatible with `torch.load()` and Hugging Face `from_pretrained()`
- Required for inference scripts

### Inference Checkpoint Loading

`src/inference/arkit_inference.py` loads checkpoints with fallback logic:
1. First tries: `checkpoint_dir/pytorch_model_fp32/`
2. Falls back to: `checkpoint_dir/pytorch_model_fp32.bin/` (legacy naming)
3. Falls back to: Any `*.bin` or `*.safetensors` in `checkpoint_dir/`

**Recommended naming**: Use `pytorch_model_fp32/` for clean organization.

---

## 7) Baseline Evaluation Results

Before retraining on the reorganized data, baseline evaluation was performed using the original model on test sets.

### Performance Summary

**SQA3D** (50 test samples from `test_split.jsonl`):
- Exact match: 1/50 (2.0%)
- Partial match: 21/50 (42.0%)

**ScanQA** (50 test samples from `test_split.jsonl`):
- Exact match: 2/50 (4.0%)
- Partial match: 6/50 (12.0%)

**ARKit** (1 sample):
- Exact match: 0/9 (0.0%)
- Partial match: 0/9 (0.0%)

### Evaluation Metrics

- **Exact match**: Model prediction exactly matches ground truth (case-insensitive)
- **Partial match**: Prediction contains ground truth as substring, or vice versa

### Output Files

Results stored in `outputs/qa/baseline_eval/`:
- `sqa3d_baseline.jsonl` – Per-sample predictions for SQA3D test set
- `scanqa_baseline.jsonl` – Per-sample predictions for ScanQA test set
- `arkit_baseline.jsonl` – Per-sample predictions for ARKit
- `baseline_summary.json` – Aggregate metrics

Each JSONL line contains:
```json
{
  "question": "...",
  "prediction": "...",
  "reference": "...",
  "scene_id": "...",
  "exact_match": false,
  "partial_match": true
}
```

### Interpretation

These results represent the **original model's performance** and serve as a baseline. Retraining on the new train/test splits is expected to improve generalization.

**Next Steps**:
- Retrain model on reorganized `train_split.jsonl` files
- Evaluate on complete test sets (currently only 50 samples per dataset)
- Compare new results against this baseline

---

## 8) Monitoring & Debugging

### Console Logging

Training progress is logged to console every `log_every_steps` (default: 10):
- Loss values
- Learning rate
- Training speed (samples/sec)
- ETA (estimated time to completion)

Implemented in `src/train/train_sft.py`.

### TensorBoard

**Start TensorBoard**:
```bash
tensorboard --logdir ckpts/stage1_3d/logs --port 6006 --bind_all
```

**Access**: Open browser to `http://localhost:6006` (or server IP for remote access)

**Metrics Available**:
- Training loss
- Learning rate curves
- Gradient norms
- Memory usage

### CLI Monitoring Tool

**Interactive monitoring**:
```bash
python scripts/monitor_training.py \
  --logdir ckpts/stage1_3d/logs/roomplan \
  --watch
```

**Features**:
- Real-time training progress
- Loss curve visualization
- Step timing statistics
- Memory usage tracking

### Log Files

**Slurm Output**:
```bash
# View Slurm job output
tail -f pytorchdist_*.out

# Filter for key metrics
tail -f pytorchdist_*.out | grep -E "Step|Loss|ETA"
```

**Training Logs**:
```bash
# View training logs (if separate log files are created)
tail -f logs/stage1_3d/*.log
```

### Common Issues & Solutions

**NCCL Timeout**:
- Already addressed in `train_fixed.sh` with `NCCL_TIMEOUT=3600`
- Check network connectivity between nodes
- Verify `NCCL_IB_HCA` settings match your cluster

**OOM (Out of Memory)**:
- Use `--safe` flag: `./train_fixed.sh --safe full 2`
- Manually reduce `BATCH_PER_GPU` or increase `GRAD_ACCUM`
- Script auto-adjusts based on available GPU memory

**Data Loading Errors**:
- Verify data files exist: `ls data/processed/scanqa/ data/processed/sqa3d/`
- Check file naming: Should be `train_split.jsonl`, not `train.jsonl`
- Ensure images are accessible in `data/processed/SQA3D/bird/`

**Cache/Disk Space Issues**:
- Already addressed: All caches relocated to `.cache/` in project directory
- Clean old caches: `rm -rf .cache/triton/* .cache/python/*`
- Check disk usage: `df -h`

### Documentation

- **`docs/MONITORING_GUIDE.md`**: Detailed monitoring techniques
- **`docs/TRAINING_MONITORING_SUMMARY.md`**: Quick reference for monitoring
- **`docs/COMPLETE_TRAINING_GUIDE.md`**: Comprehensive troubleshooting guide

---

## 9) ARKit / RoomPlan Evaluation

Stage 1 weights can be evaluated on ARKit synthetic data for spatial reasoning capabilities, though the model generates free-form text rather than structured JSON.

### Prerequisites

1. **Stage 1 checkpoint**: Complete Stage 1 training or use provided checkpoint
2. **ARKit synthetic data**: Already included in `data/processed/arkit_synth/train.json` (9 samples)
3. **Converted fp32 weights**: Run `zero_to_fp32.py` as described in Section 6

### Generating ARKit Synthetic Data (Optional)

If you want to regenerate or expand the ARKit evaluation set:

```bash
python scripts/prep/prepare_arkit_from_3dod.py \
  --arkit-training-root data/processed/ARKit/Training \
  --output data/processed/arkit_synth/train.json \
  --num-views 10 \
  --max_scenes 10
```

Alternative (if you have full planes/cameras metadata):
```bash
python scripts/prep/synth_roomplan_instructions.py \
  --input data/processed/ARKit/Training \
  --output data/processed/arkit_synth/train.json
```

### Running ARKit Inference

**Single-GPU evaluation**:
```bash
python -m src.inference.arkit_inference \
  --config configs/stage1_3d.yaml \
  --arkit_glob "data/processed/arkit_synth/test.json" \
  --checkpoint_dir ckpts/stage1_3d \
  --num_scenes 9 \
  --max_new_tokens 256 \
  --device cuda:0 \
  --output_jsonl outputs/qa/arkit_eval/arkit_predictions_stage1.jsonl
```

**Parameters**:
- `--config`: Stage configuration file (use Stage 1 config)
- `--arkit_glob`: Path to ARKit JSON data
- `--checkpoint_dir`: Must contain `pytorch_model_fp32/` directory
- `--num_scenes`: Number of scenes to evaluate
- `--max_new_tokens`: Maximum tokens to generate
- `--output_jsonl`: Where to save predictions

### Input Data Format

ARKit JSON format (`data/processed/arkit_synth/train.json`):
```json
{
  "scene_id": "47333462",
  "images": ["view_0.png", "view_1.png", ...],
  "question": "In scene 47333462, find an object belonging to the category 'cabinet' and place a virtual anchor at the center of that object.",
  "action_json": {
    "action": "place_anchor",
    "scene": "47333462",
    "center": [1.5, 0.8, 2.3],
    "normal": [0, 1, 0],
    "extent": [0.6, 0.4, 0.8]
  }
}
```

### Output Format

Each line in the output JSONL (`outputs/qa/arkit_eval/arkit_predictions_stage1.jsonl`):
```json
{
  "index": 0,
  "scene_id": "47333462",
  "question": "In scene 47333462, find an object belonging to the category 'cabinet' and place a virtual anchor at the center of that object.",
  "prediction": "To place a virtual anchor at the center of the cabinet in scene 47333462, you would need to...",
  "reference": {
    "action": "place_anchor",
    "scene": "47333462",
    "center": [1.5, 0.8, 2.3],
    "normal": [0, 1, 0],
    "extent": [0.6, 0.4, 0.8]
  }
}
```

### Current Behavior & Limitations

**What happens during inference**:
1. Multi-view images are encoded via VGGT + Perceiver projector
2. Visual tokens are injected into Qwen3 at `<image>` position in prompt
3. Model generates free-form text response (not structured JSON)

**Known Limitations**:
- ❌ Model does **not** generate structured `action_json` output
- ❌ Exact match metric is 0.0% (predictions are text, references are JSON)
- ❌ Model tends to repeat or elaborate on instruction rather than answer directly
- ⚠️ Only 9 synthetic samples available for evaluation
- ⚠️ Training for structured JSON actions is not implemented

**Why this happens**:
- Current trainer only supports text targets, not structured dictionaries
- No loss head for JSON generation
- Model was trained on QA tasks, not action prediction

### Future Work

To enable proper ARKit/RoomPlan action prediction:
1. **Add JSON target support** in trainer:
   - Serialize `action_json` to text format for supervision, or
   - Implement dedicated output head for structured predictions
   
2. **Constrain generation**:
   - Modify prompts to request JSON-only responses
   - Add JSON schema validation during generation
   
3. **Training data**:
   - Generate larger ARKit synthetic dataset
   - Implement Stage 2 training pipeline
   
4. **Evaluation metrics**:
   - Add JSON-specific metrics (structure match, key presence)
   - Measure spatial accuracy (center/normal/extent errors)

See `configs/stage2_arkit.yaml` for reference configuration (not currently functional).

---

## 10) Current Status & Known Limitations

### Completed Features

- ✅ **Data reorganization**: Proper train/test splits for ScanQA and SQA3D
  - ScanQA: 21,161 train / 3,567 test samples
  - SQA3D: 22,959 train / 3,664 test samples
  - Total: ~44,120 train / ~7,231 test samples

- ✅ **Image reference fixes**: All samples properly reference SQA3D bird views

- ✅ **Baseline evaluation**: Completed on test sets
  - Results in `outputs/qa/baseline_eval/`
  - Metrics: SQA3D (2% exact, 42% partial), ScanQA (4% exact, 12% partial)

- ✅ **Stage 1 training pipeline**: Fully implemented and tested
  - Multi-view QA training on ScanQA + SQA3D
  - DeepSpeed ZeRO-3 distributed training
  - Automatic memory management

- ✅ **Distributed training infrastructure**:
  - Support for 1-8 GPUs
  - DeepSpeed ZeRO-3 optimization
  - Slurm templates for HPC clusters

- ✅ **Monitoring and debugging tools**:
  - TensorBoard integration
  - CLI monitoring script
  - Comprehensive logging

- ✅ **ARKit evaluation infrastructure**:
  - 9 synthetic samples prepared
  - Inference script implemented
  - Baseline results available

### In Progress

- 📋 **Model retraining**: Retrain on new train splits (next immediate step)
  - Will use reorganized `train_split.jsonl` files
  - Expected to improve generalization on test sets

- 📋 **Full test set evaluation**: Currently only 50 samples evaluated per dataset
  - Plan to evaluate on complete test sets (~7,231 samples total)

### Known Limitations

- ⚠️ **Stage 2 ARKit training not implemented**:
  - `action_json` targets are structured (dict), but trainer assumes text labels
  - `loss_heads` in `configs/stage2_arkit*.yaml` are not consumed
  - Only inference plumbing exists with Stage 1 weights

- ⚠️ **Current data uses single bird-view images**:
  - No geometry tokens (`geom_token: null`)
  - VGGT runs but without explicit geometry conditioning
  - Multi-view + geometry requires ScanNet RGB/depth/poses

- ⚠️ **ARKit evaluation has limited samples**:
  - Only 9 synthetic samples available
  - Full RoomPlan training remains future work
  - JSON-style action prediction not supported

- ⚠️ **Model output for ARKit**:
  - Generates free-form text, not structured JSON
  - Tends to repeat/elaborate on instruction
  - No spatial action grounding

### Roadmap

**Short-term** (Current Focus):
1. Complete Stage 1 retraining on reorganized data
2. Run full evaluation on complete test sets
3. Compare performance against baseline
4. Document performance improvements

**Medium-term**:
1. Implement multi-view data preprocessing with geometry tokens
2. Generate larger ARKit synthetic dataset
3. Improve data augmentation and mixing strategies

**Long-term**:
1. Implement Stage 2 training for structured action prediction
2. Add JSON-specific loss heads and constraints
3. Full RoomPlan integration and evaluation
4. Production deployment and API

---

## 11) Extending the Project

### Adding Multi-view + Geometry Tokens

Rebuild ScanQA/SQA3D with poses/depth from ScanNet:

```bash
# Download ScanNet data first (requires ScanNet access)

# Process ScanQA with 8 views
python scripts/prep/prepare_scanqa.py \
  --dataset scanqa \
  --scan-root data/raw/scannet \
  --qa-file data/raw/scanqa/questions.json \
  --output data/processed/scanqa/train_split.jsonl \
  --num-views 8

# Process SQA3D with 8 views
python scripts/prep/prepare_scanqa.py \
  --dataset sqa3d \
  --scan-root data/raw/scannet \
  --qa-file data/raw/sqa3d/questions.json \
  --output data/processed/sqa3d/train_split.jsonl \
  --num-views 8

# Also generate test splits with same parameters
```

**Benefits**:
- Explicit 3D geometry information
- Better spatial reasoning
- Multi-view feature aggregation

### Generating More ARKit Data

**Option A**: Use existing 3DOD layout:
```bash
python scripts/prep/prepare_arkit_from_3dod.py \
  --arkit-training-root data/processed/ARKit/Training \
  --output data/processed/arkit_synth/train_large.json \
  --num-views 10 \
  --max_scenes 100  # Generate more scenes
```

**Option B**: Use full planes/cameras metadata:
```bash
python scripts/prep/synth_roomplan_instructions.py \
  --input data/processed/ARKit/Training \
  --output data/processed/arkit_synth/train_full.json \
  --num-instructions-per-scene 5  # Multiple instructions per scene
```

### Implementing Stage 2 Training

To enable structured action prediction:

**1. Modify data format** (`src/dataio/dataset_builder.py`):
```python
# Serialize action_json to text format
def serialize_action(action_dict):
    return json.dumps(action_dict, sort_keys=True)

# Or keep as dict and add special handling in collator
```

**2. Update trainer** (`src/train/train_sft.py`):
```python
# Add JSON-specific loss head
# Implement structured output parsing
# Add validation for generated JSON
```

**3. Modify model** (`src/models/vggt_qwen3_vlm.py`):
```python
# Add action_head for structured prediction
# Or use constrained generation for JSON
```

**4. Update config** (`configs/stage2_arkit.yaml`):
```yaml
# Enable loss_heads configuration
# Add JSON-specific generation constraints
```

### Adding New Datasets

**1. Prepare data in standard format**:
```python
{
  "images": ["path/to/view1.png", "path/to/view2.png"],
  "geom_token": {"R": [...], "t": [...], "K": [...], "depth_hist": [...]} or null,
  "question": "What color is the chair?",
  "answer": "The chair is red",
  "task": "color_recognition",
  "scene_id": "scene_0001",
  "question_id": "q_0001"
}
```

**2. Add dataset to config**:
```yaml
# configs/stage1_3d.yaml
data:
  sources:
    - path: "data/processed/my_dataset/*.jsonl"
      weight: 0.2  # Mix ratio
      task: "my_task"
```

**3. Update mixing weights** to balance with existing datasets.

### Fine-tuning for Specific Tasks

**Domain adaptation**:
```bash
# Start from Stage 1 checkpoint
./train_fixed.sh full 4

# Modify config to use only target domain data
# Reduce learning rate for fine-tuning
```

**Task-specific heads**:
- Add custom output heads in model definition
- Implement task-specific loss functions
- Update trainer to handle multiple tasks

### Evaluation on Custom Data

**1. Create evaluation script**:
```bash
# scripts/eval_custom.py
python -m src.inference.arkit_inference \
  --config configs/stage1_3d.yaml \
  --arkit_glob "data/processed/my_data/*.json" \
  --checkpoint_dir ckpts/stage1_3d \
  --output_jsonl outputs/my_eval/predictions.jsonl
```

**2. Implement custom metrics**:
- Exact match, partial match
- Task-specific accuracy
- Spatial grounding metrics

---

## 12) Repository Structure

```
vggt-qwen3-roomplan/
├── configs/                          # Training configurations
│   ├── stage1_3d.yaml               # Stage 1 QA training (ACTIVE)
│   ├── stage2_arkit.yaml            # Stage 2 action training (REFERENCE ONLY)
│   ├── perceiver_small.yaml         # Perceiver projector config
│   ├── deepspeed_zero3.json         # DeepSpeed optimization config
│   └── accelerate_*gpu.yaml         # Accelerate multi-GPU configs
│
├── data/                            # Training and evaluation data
│   ├── processed/                   # Ready-to-use processed datasets
│   │   ├── scanqa/                  # ScanQA dataset
│   │   │   ├── train_split.jsonl   # 21,161 training samples
│   │   │   └── test_split.jsonl    # 3,567 test samples
│   │   ├── sqa3d/                   # SQA3D dataset
│   │   │   ├── train_split.jsonl   # 22,959 training samples
│   │   │   └── test_split.jsonl    # 3,664 test samples
│   │   ├── SQA3D/bird/              # Bird-view images
│   │   └── arkit_synth/             # ARKit synthetic data
│   │       └── train.json           # 9 evaluation samples
│   └── raw/                         # Raw downloaded datasets
│       ├── scanqa/                  # Raw ScanQA
│       ├── sqa3d/                   # Raw SQA3D
│       └── arkitscenes/             # ARKit scenes
│
├── src/                             # Source code
│   ├── dataio/                      # Data loading and processing
│   │   ├── dataset_builder.py      # Multi-source dataset builder
│   │   └── collate_multiview.py    # Multi-view batch collation
│   ├── models/                      # Model definitions
│   │   ├── vggt_qwen3_vlm.py       # VGGT-Qwen3 wrapper
│   │   └── perceiver.py            # Perceiver projector
│   ├── train/                       # Training scripts
│   │   └── train_sft.py            # Main training script
│   ├── inference/                   # Inference scripts
│   │   └── arkit_inference.py      # ARKit evaluation
│   └── eval/                        # Evaluation utilities
│
├── scripts/                         # Utility scripts
│   ├── prep/                        # Data preparation
│   │   ├── prepare_scanqa.py       # Multi-view data prep
│   │   ├── rebuild_scanqa_sqa3d.py # Fast single-view rebuild
│   │   ├── prepare_arkit_from_3dod.py  # ARKit synthetic gen
│   │   └── synth_roomplan_instructions.py  # RoomPlan data gen
│   ├── slurm/                       # Slurm job templates
│   │   ├── stage2_3d_2xb200.sbatch
│   │   └── stage3_arkit_2xb200.sbatch
│   ├── eval_baseline.py            # Baseline evaluation script
│   ├── monitor_training.py         # Training monitoring tool
│   └── test_dataloader.py          # Data loading tests
│
├── ckpts/                           # Training outputs
│   └── stage1_3d/                   # Stage 1 checkpoints
│       ├── checkpoint-*/            # Periodic checkpoints
│       ├── pytorch_model/           # DeepSpeed ZeRO-3 shards
│       ├── pytorch_model_fp32/      # Converted fp32 weights
│       ├── logs/                    # TensorBoard logs
│       └── zero_to_fp32.py         # Conversion script
│
├── outputs/                         # Evaluation outputs
│   └── qa/
│       ├── baseline_eval/           # Baseline evaluation results
│       └── arkit_eval/              # ARKit evaluation results
│
├── docs/                            # Documentation
│   ├── COMPLETE_TRAINING_GUIDE.md  # Comprehensive training guide
│   ├── QUICK_START.md              # Quick start guide
│   ├── SLURM_TRAINING_GUIDE.md     # Slurm-specific guide
│   ├── MONITORING_GUIDE.md         # Monitoring and debugging
│   ├── TRAINING_MONITORING_SUMMARY.md  # Quick monitoring reference
│   └── FILE_GUIDE.md               # File-by-file documentation
│
├── third_party/                     # Third-party dependencies
│   ├── vggt/                        # VGGT vision backbone
│   │   └── vggt_1B_commercial.pt   # Frozen vision weights
│   └── Qwen3/                       # Qwen3 (optional editable install)
│
├── env/                             # Environment files
│   ├── environment.yml              # Conda environment spec
│   └── environment_local.yml        # Local dev environment
│
├── train_fixed.sh                   # Main training launcher
├── run.sh                           # Slurm job submission script
├── README.md                        # This file
└── .cache/                          # Local caches (auto-created)
    ├── triton/                      # Triton kernel cache
    ├── torch/                       # PyTorch cache
    ├── huggingface/                 # HuggingFace model cache
    └── ...
```

### Key Files

**Training Entry Points**:
- `train_fixed.sh`: Main training script with automatic configuration
- `run.sh`: Slurm submission wrapper
- `src/train/train_sft.py`: Core training logic

**Model Definition**:
- `src/models/vggt_qwen3_vlm.py`: VGGT-Qwen3 integration
- `src/models/perceiver.py`: Perceiver projector implementation

**Data Pipeline**:
- `src/dataio/dataset_builder.py`: Dataset loading and mixing
- `src/dataio/collate_multiview.py`: Batch preparation

**Configuration**:
- `configs/stage1_3d.yaml`: Stage 1 training settings
- `configs/perceiver_small.yaml`: Projector architecture
- `configs/deepspeed_zero3.json`: Distributed training optimization

For detailed file descriptions, see `docs/FILE_GUIDE.md`.

---

## 13) Quick Start Checklist

### Initial Setup

- [ ] **Step 1**: Create conda environment
  ```bash
  conda env create -f env/environment.yml
  conda activate roomplan
  ```

- [ ] **Step 2**: Install VGGT
  ```bash
  pip install -e third_party/vggt
  ```

- [ ] **Step 3**: Verify environment
  ```bash
  python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
  ```

### Before Training

- [ ] **Step 4**: Place VGGT checkpoint
  - Ensure `third_party/vggt/vggt_1B_commercial.pt` exists

- [ ] **Step 5**: Verify data files
  ```bash
  ls data/processed/scanqa/train_split.jsonl
  ls data/processed/sqa3d/train_split.jsonl
  ```
  - Should see `train_split.jsonl` and `test_split.jsonl` in each directory

- [ ] **Step 6**: (Optional) Check Qwen3 model cache
  - Will auto-download on first run to `.cache/huggingface/`

### Training

- [ ] **Step 7**: Test with debug mode (100 steps)
  ```bash
  ./train_fixed.sh debug 1
  ```

- [ ] **Step 8**: Run full training
  ```bash
  ./train_fixed.sh full 4              # Local: 4 GPUs
  # OR
  sbatch run.sh                         # Slurm: 8 GPUs
  ```

### Monitoring

- [ ] **Step 9**: Start TensorBoard
  ```bash
  tensorboard --logdir ckpts/stage1_3d/logs --port 6006 --bind_all
  ```

- [ ] **Step 10**: Monitor training progress
  ```bash
  python scripts/monitor_training.py --logdir ckpts/stage1_3d/logs/roomplan --watch
  ```

### After Training

- [ ] **Step 11**: Convert checkpoint to fp32
  ```bash
  cd ckpts/stage1_3d
  python zero_to_fp32.py pytorch_model pytorch_model_fp32
  ```

- [ ] **Step 12**: Evaluate on test sets
  ```bash
  # Add your evaluation commands here
  # See Section 7 for baseline evaluation examples
  ```

- [ ] **Step 13**: (Optional) Run ARKit evaluation
  ```bash
  python -m src.inference.arkit_inference \
    --config configs/stage1_3d.yaml \
    --checkpoint_dir ckpts/stage1_3d \
    --arkit_glob "data/processed/arkit_synth/train.json" \
    --output_jsonl outputs/qa/arkit_eval/predictions.jsonl
  ```

### Troubleshooting

If you encounter issues:

1. **Check logs**: `tail -f pytorchdist_*.out` (Slurm) or console output
2. **Verify data**: `python scripts/test_dataloader.py`
3. **Monitor GPU**: `nvidia-smi -l 1`
4. **Review docs**: See `docs/COMPLETE_TRAINING_GUIDE.md` for detailed troubleshooting

---

## Additional Resources

### Documentation

- **Complete Training Guide**: `docs/COMPLETE_TRAINING_GUIDE.md`
  - Detailed architecture explanation
  - Step-by-step training workflow
  - Comprehensive troubleshooting

- **Quick Start Guide**: `docs/QUICK_START.md`
  - Condensed setup and training steps
  - Minimal configuration examples

- **Slurm Training Guide**: `docs/SLURM_TRAINING_GUIDE.md`
  - HPC cluster configuration
  - Job submission and monitoring
  - Multi-node training

- **Monitoring Guide**: `docs/MONITORING_GUIDE.md`
  - TensorBoard usage
  - CLI monitoring tools
  - Debugging techniques

- **File Guide**: `docs/FILE_GUIDE.md`
  - Detailed file-by-file documentation
  - Code organization

### External Links

- **Qwen3 Model**: https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507
- **ScanQA Dataset**: https://drive.google.com/drive/folders/1-21A3TBE0QuofEwDg5oDz2z0HEdbVgL2
- **SQA3D Dataset**: https://zenodo.org/records/7792397
- **ARKitScenes**: https://github.com/apple/ARKitScenes
- **DeepSpeed**: https://www.deepspeed.ai/

---

## License

[Add your license information here]

## Contact

[Add contact information or contribution guidelines here]

---

**Last Updated**: December 2, 2025
