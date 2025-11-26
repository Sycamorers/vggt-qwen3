# 📊 TRAINING VISUALIZATION SUMMARY

## ✅ What Was Added

I've enhanced your training setup with **comprehensive visualization and monitoring tools**. You now have 4 different ways to track training progress!

---

## 🚀 Quick Start - Monitoring Commands

### 1. Quick Status Check (Fastest)
```bash
./check_training.sh
```
**Shows**: Checkpoints saved, GPU usage, process status, disk space  
**Use when**: You want a quick overview

### 2. Detailed Metrics (Rich Info)
```bash
python scripts/monitor_training.py --logdir ckpts/stage2_3d/logs/roomplan
```
**Shows**: Loss trends, LR, speed, ASCII plot  
**Use when**: You want detailed metrics snapshot

### 3. Auto-Refresh Monitoring (Recommended!)
```bash
python scripts/monitor_training.py --logdir ckpts/stage2_3d/logs/roomplan --watch
```
**Shows**: Same as #2, but auto-refreshes every 30s  
**Use when**: You want continuous monitoring in terminal

### 4. TensorBoard (Full Visualization)
```bash
tensorboard --logdir ckpts/stage2_3d/logs/roomplan --port 6006
# Open http://localhost:6006
```
**Shows**: Interactive web UI with publication-quality plots  
**Use when**: You want detailed analysis and export

---

## 🎨 Enhanced Training Output

The training script now prints beautiful, informative output:

```
================================================================================
🚀 TRAINING STARTING
================================================================================
📊 Configuration:
   Max steps: 30,000
   Batch size per GPU: 6
   Gradient accumulation: 32
   Effective batch size: 384
   Learning rate: 5.00e-06
   Projector LR: 1.00e-04
   Total samples: 59,207
   Precision: bfloat16
   Num GPUs: 2
...
================================================================================

================================================================================
Step 1,500/30,000 [ 5.0%] [██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]
────────────────────────────────────────────────────────────────────────────────
📉 Loss:        5.2341
📚 LR (base):   4.95e-06  |  LR (proj): 9.90e-05
⏱️  Speed:       1.23 steps/s  |  0.81s per step
⏰ Elapsed:     0.34h  |  ETA: 6.45h
💾 Checkpoint:  Next at step 3,000 (in 1,500 steps)
================================================================================

🔖 SAVING CHECKPOINT
────────────────────────────────────────────────────────────────────────────────
   Step: 3,000/30,000
   Progress: 10.0%
   Location: ckpts/stage2_3d/step_3000
   Size: 15.23 GB
✅                    Checkpoint saved successfully
```

**Features:**
- Progress bars with percentage
- Loss, learning rates, speed metrics
- Time elapsed and ETA
- Checkpoint countdown
- Emoji indicators for easy scanning

---

## 📊 Monitor Training Script Features

When you run the monitoring script, you get:

```
================================================================================
                         📊 TRAINING MONITOR
================================================================================
🕐 Updated: 2025-11-25 14:35:42

────────────────────────────────────────────────────────────────────────────────
📈 Current Progress:
────────────────────────────────────────────────────────────────────────────────
   Step: 7,500
   Progress: [████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 25.0%

────────────────────────────────────────────────────────────────────────────────
📉 Loss:
────────────────────────────────────────────────────────────────────────────────
   Current: 3.8234
   Recent avg (last 10): 3.8521
   Min: 3.2134  |  Max: 5.8234

────────────────────────────────────────────────────────────────────────────────
📊 Loss Trend (last 20 steps):
────────────────────────────────────────────────────────────────────────────────
   █    █  █
   ██  ████
   ████████ █
   ████████ ██
   ████████ ███████████ ██   ██████
   3.75                                4.12
   └──────────────────────────────────────┘
```

**Features:**
- Loss statistics (current, avg, min, max)
- Learning rate tracking (separate for base and projector)
- Training speed metrics
- **ASCII line plot** showing loss trend
- Auto-refresh option

---

## 🔍 Status Checker Features

Quick system health check:

```
📦 Checkpoints Saved: 5

📁 Checkpoint List:
   • step_1500 - 15.2G
   • step_3000 - 15.2G
   • step_4500 - 15.2G

✅ Training Process: RUNNING
   PID: 12345
   Runtime: 02:15:30
   CPU: 125.3% | Memory: 8.2%

🎮 GPU Status:
   GPU 0: NVIDIA A100-SXM4-40GB
      Utilization: 98% | Memory: 38521MB / 40960MB | Temp: 65°C
   GPU 1: NVIDIA A100-SXM4-40GB
      Utilization: 97% | Memory: 38234MB / 40960MB | Temp: 64°C

💾 Disk Space:
   Checkpoint directory size: 76G
   Free space on disk: 2.3T
```

---

## 📁 New Files Created

### Scripts:
1. **`check_training.sh`** - Quick status checker
2. **`scripts/monitor_training.py`** - Detailed metrics monitor
3. **`demo_monitoring.sh`** - Interactive tutorial for all monitoring tools

### Documentation:
1. **`MONITORING_GUIDE.md`** - Complete monitoring guide (comprehensive)
2. **`VISUALIZATION_ADDED.md`** - Summary of what was added
3. **`TRAINING_MONITORING_SUMMARY.md`** - This file (quick reference)

### Updated:
1. **`src/train/train_sft.py`** - Enhanced logging
2. **`QUICK_START.md`** - Added monitoring section
3. **`FILE_GUIDE.md`** - Updated with monitoring tools

---

## 🎓 Tutorial Mode

Try the interactive tutorial:
```bash
./demo_monitoring.sh
```

This will guide you through all monitoring options step-by-step!

---

## 📖 Documentation

| Document | Purpose |
|----------|---------|
| `MONITORING_GUIDE.md` | **Complete guide** - all monitoring tools, troubleshooting, tips |
| `QUICK_START.md` | Quick start with monitoring commands |
| `TRAINING_MONITORING_SUMMARY.md` | This file - quick reference |
| `VISUALIZATION_ADDED.md` | Details on what was added |

---

## 💡 Recommended Workflow

### Starting Training:
```bash
# Terminal 1: Start training
./train.sh

# Terminal 2: Monitor continuously
python scripts/monitor_training.py --logdir ckpts/stage2_3d/logs/roomplan --watch
```

### Checking Progress:
```bash
# Quick check
./check_training.sh

# Or view in browser
tensorboard --logdir ckpts/stage2_3d/logs/roomplan --port 6006
# Open http://localhost:6006
```

---

## 🎯 What to Monitor

### ✅ Healthy Training Signs:
- Loss gradually decreasing
- Speed: ~1-2 steps/sec (on 2× A100)
- GPU utilization: 90-100%
- No NaN warnings

### ⚠️ Warning Signs:
- Loss increasing or stuck
- Slow speed (< 0.5 steps/sec)
- GPU < 50% utilization
- Frequent NaN warnings

---

## 📊 Metrics Logged to TensorBoard

All these metrics are automatically logged:
- `train/loss` - Training loss
- `train/learning_rate_base` - Base model LR
- `train/learning_rate_proj` - Projector LR
- `train/steps_per_sec` - Training throughput
- `train/progress_pct` - Progress percentage
- `train/step_time` - Time per step

---

## 🆘 Quick Help

**Monitor script won't run?**
```bash
pip install tensorboard
```

**Can't see TensorBoard on remote server?**
```bash
# On local machine:
ssh -L 6006:localhost:6006 user@remote-server
# Then open http://localhost:6006
```

**Want to save console output?**
```bash
./train.sh 2>&1 | tee training_log.txt
```

---

## ✨ Summary

You now have a **production-grade monitoring setup** with:

✅ **4 monitoring tools** (quick check, detailed metrics, auto-refresh, TensorBoard)  
✅ **Enhanced console output** (progress bars, emojis, formatted metrics)  
✅ **Comprehensive documentation** (guides, tutorials, troubleshooting)  
✅ **Auto-logging to TensorBoard** (all key metrics tracked)  
✅ **Interactive tutorial** (demo_monitoring.sh)

**Start training and monitoring:**
```bash
# Terminal 1
./train.sh

# Terminal 2
python scripts/monitor_training.py --logdir ckpts/stage2_3d/logs/roomplan --watch
```

**Happy training! 🚀**

---

**Last Updated**: November 25, 2025  
**Status**: ✅ All monitoring tools ready and documented
