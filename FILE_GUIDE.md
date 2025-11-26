# 📁 Project File Structure & Documentation

## 🎯 **START HERE**
- **`QUICK_START.md`** - Step-by-step training guide
- **`SETUP_WEIGHTS.md`** - Download model weights & data
- **`TRAINING_FIXES.md`** - Technical bug fixes documentation
- **`train.sh`** - Main training script

## 📚 Documentation Files

| File | Description |
|------|-------------|
| `README.md` | Project overview with quick start |
| `QUICK_START.md` | Step-by-step training guide |
| `SETUP_WEIGHTS.md` | **Weight download guide** - VGGT, Qwen3, datasets |
| `TRAINING_FIXES.md` | **Bug fix documentation** - all 8 fixes explained |
| `MONITORING_GUIDE.md` | **Training monitoring** - TensorBoard, progress tracking |
| `GITHUB_PUSH_GUIDE.md` | **Git/GitHub guide** - how to push without large files |
| `FILE_GUIDE.md` | This file - project structure reference |
| `COMPLETE_TRAINING_GUIDE.md` | Legacy comprehensive guide |

## 🔧 Configuration Files

| File | Purpose |
|------|---------|
| `configs/stage2_3d.yaml` | **Main training config** - hyperparameters, data paths |
| `configs/accelerate_config.yaml` | Distributed training (DeepSpeed, 2 GPUs) |
| `configs/deepspeed_zero3.json` | DeepSpeed ZeRO-3 settings |
| `configs/perceiver_small.yaml` | Perceiver projector architecture |
| `configs/local_qwen3.yaml` | Local testing config |

## 🐍 Source Code

### Core Model Files
| File | Description |
|------|-------------|
| `src/models/vggt_qwen3_vlm.py` | **Main VLM model** - integrates vision + projector + LLM |
| `src/models/projector_perceiver.py` | Perceiver resampler (vision→text projection) |
| `src/train/train_sft.py` | **Training script** - main entry point |
| `src/train/losses.py` | Loss functions |

### Data Processing
| File | Description |
|------|-------------|
| `src/dataio/dataset_builder.py` | Loads JSONL datasets (ScanQA, SQA3D) |
| `src/dataio/collate_multiview.py` | Batching, tokenization, padding |

### Evaluation
| File | Description |
|------|-------------|
| `src/eval/eval_3dqa.py` | Evaluate on 3D VQA tasks |
| `src/eval/eval_ref3d.py` | Evaluate on 3D grounding tasks |

## 📊 Data

```
data/
├── processed/           # Ready-to-use JSONL datasets
│   ├── scanqa/
│   │   ├── train.jsonl  (~41K samples)
│   │   └── val.jsonl
│   └── sqa3d/
│       ├── train.jsonl  (~18K samples)
│       └── val.jsonl
└── raw/                 # Original datasets (not used directly)
```

## 🚀 Scripts

| File | Purpose |
|------|---------|
| `train.sh` | **Main training script** - run this to train |
| `check_training.sh` | **Quick status checker** - checkpoints, GPU, disk space |
| `scripts/monitor_training.py` | **Detailed metrics monitor** - loss trends, speed, progress |
| `scripts/run_debug_training.sh` | Debug training (100 steps) |
| `scripts/check_init.py` | Verify model initialization |
| `scripts/test_dataloader.py` | Test data loading |
| `scripts/validate_data.py` | Validate dataset format |
| `scripts/slurm/stage2_3d_2xb200.sbatch` | SLURM job script |

## 📦 Third-Party Code

```
third_party/
├── vggt/               # VGGT vision encoder
└── Qwen3/              # Qwen3 language model resources
```

## 💾 Output (Generated During Training)

```
ckpts/
└── stage2_3d/
    ├── checkpoint-1500/
    ├── checkpoint-3000/
    ├── ...
    ├── checkpoint-30000/  # Final model
    └── logs/
        └── roomplan/      # TensorBoard logs
```

## 🔑 Key Files Modified to Fix NaN Loss

### Critical Fixes Applied:

1. **`src/models/vggt_qwen3_vlm.py`** (3 fixes)
   - ✅ Removed `@torch.no_grad()` decorator → gradient flow fix
   - ✅ Added `images.to(dtype=self.model_dtype)` → dtype fix
   - ✅ Added `self.proj_norm` → output normalization
   - ✅ Added label shifting logic → preserve answer labels

2. **`src/models/projector_perceiver.py`** (1 fix)
   - ✅ Changed `torch.randn(...)` → `torch.randn(...) * 0.02` → initialization fix

3. **`src/dataio/collate_multiview.py`** (2 fixes)
   - ✅ Added `min_text_length` → ensure room for visual tokens
   - ✅ Changed prompt format: `<image>\n{q}\n` → `{q}\n<image>\n` → fix label positions

4. **`src/train/train_sft.py`** (1 fix)
   - ✅ Pass `num_vis_tokens` and `geom_tokens` to collator

5. **`configs/stage2_3d.yaml`** (1 fix)
   - ✅ Set `max_length: 512` (was 4096, unnecessarily large)

---

## 🎓 Understanding the Fixes

**Why NaN loss occurred:**
1. **No gradients for projector** → couldn't learn
2. **Dtype mismatch** → runtime error
3. **Poor initialization** → unstable training
4. **Large output values** → numerical overflow
5. **All labels set to -100** → no supervision signal ← **MAIN ISSUE**

**How we fixed it:**
- See `COMPLETE_TRAINING_GUIDE.md` Section: "Critical Fixes Applied"
- Each fix addresses a specific failure mode
- Fix #5 (label shifting) was the most subtle and critical

---

## 📖 How to Use This Project

### For Quick Training:
1. Read: `QUICK_START.md`
2. Run: `./train.sh`
3. Monitor: `./check_training.sh` or `python scripts/monitor_training.py --logdir ckpts/stage2_3d/logs/roomplan --watch`

### For Understanding:
1. Read: `COMPLETE_TRAINING_GUIDE.md` (comprehensive)
2. Check: `src/models/vggt_qwen3_vlm.py` (model architecture)
3. Review: `configs/stage2_3d.yaml` (hyperparameters)

### For Monitoring:
1. Quick check: `./check_training.sh`
2. Detailed metrics: `python scripts/monitor_training.py --logdir ckpts/stage2_3d/logs/roomplan`
3. Full visualization: `tensorboard --logdir ckpts/stage2_3d/logs/roomplan --port 6006`
4. See: `MONITORING_GUIDE.md` for all options

### For Debugging:
1. Check: `ALL_FIXES_APPLIED.md` (what was fixed)
2. Run: `./train.sh debug` (100 steps test)
3. View: `scripts/check_init.py` (verify setup)

---

## 📝 Command Cheat Sheet

```bash
# Training
./train.sh              # Full training
./train.sh debug        # Debug mode

# Quick Monitoring
./check_training.sh     # Status overview

# Detailed Monitoring
python scripts/monitor_training.py --logdir ckpts/stage2_3d/logs/roomplan          # One-time
python scripts/monitor_training.py --logdir ckpts/stage2_3d/logs/roomplan --watch  # Auto-refresh

# TensorBoard
tensorboard --logdir ckpts/stage2_3d/logs/roomplan --port 6006

# Verification
python scripts/check_init.py
python scripts/test_dataloader.py

# Evaluation
python src/eval/eval_3dqa.py --checkpoint ckpts/stage2_3d/checkpoint-30000
```

---

**Last Updated**: November 25, 2025  
**Status**: ✅ All bugs fixed, monitoring tools added, ready for production training
