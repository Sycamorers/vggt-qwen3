# 📊 Training Monitoring & Visualization Guide

This document describes all the tools available for monitoring your training progress.

---

## 🎯 Quick Reference

| Tool | Command | Purpose | Update Frequency |
|------|---------|---------|------------------|
| **Quick Status** | `./check_training.sh` | Overview of training state | Manual |
| **Detailed Monitor** | `python scripts/monitor_training.py --logdir ckpts/stage2_3d/logs/roomplan` | Metrics + loss plot | Manual |
| **Auto-Refresh** | `python scripts/monitor_training.py --logdir ckpts/stage2_3d/logs/roomplan --watch` | Same as above | Every 30s |
| **TensorBoard** | `tensorboard --logdir ckpts/stage2_3d/logs/roomplan --port 6006` | Full visualization | Real-time |
| **Training Logs** | Training outputs to console | Progress bars, metrics | Every 20 steps |

---

## 1. 📺 Training Console Output (Built-in)

**What it shows:**
- Real-time progress during training
- Formatted with emojis and progress bars
- Most immediate way to monitor

**Example output:**
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

📝 Logging & Checkpoints:
   Logging interval: every 20 steps
   Checkpoint interval: every 1500 steps
   Output directory: ckpts/stage2_3d
   TensorBoard logs: ckpts/stage2_3d/logs

🔬 Model Configuration:
   Visual tokens: 128
   Geometry tokens: 8
   Max sequence length: 512
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

**Logged Metrics:**
- Loss value
- Learning rates (base and projector)
- Training speed (steps/sec and sec/step)
- Time elapsed and ETA
- Next checkpoint countdown
- Progress bar and percentage

---

## 2. ✅ Quick Status Checker (`check_training.sh`)

**When to use:**
- Quick health check
- See how many checkpoints saved
- Check if training is still running
- Monitor GPU usage
- Check disk space

**Command:**
```bash
./check_training.sh

# Or specify custom checkpoint directory
./check_training.sh ckpts/my_experiment
```

**What it shows:**
```
================================================================================
                    🔍 TRAINING STATUS CHECKER
================================================================================

📦 Checkpoints Saved: 5

📁 Checkpoint List:
   • step_1500 - 15.2G
   • step_3000 - 15.2G
   • step_4500 - 15.2G
   • step_6000 - 15.2G
   • step_7500 - 15.2G

────────────────────────────────────────────────────────────────────────────────
📊 TensorBoard Event Files: 1
   Last updated: 2025-11-25 14:32:15

────────────────────────────────────────────────────────────────────────────────
✅ Training Process: RUNNING
   PID: 12345
   Runtime: 02:15:30
   CPU: 125.3% | Memory: 8.2%

────────────────────────────────────────────────────────────────────────────────
🎮 GPU Status:
   GPU 0: NVIDIA A100-SXM4-40GB
      Utilization: 98% | Memory: 38521MB / 40960MB | Temp: 65°C
   GPU 1: NVIDIA A100-SXM4-40GB
      Utilization: 97% | Memory: 38234MB / 40960MB | Temp: 64°C

────────────────────────────────────────────────────────────────────────────────
💾 Disk Space:
   Checkpoint directory size: 76G
   Free space on disk: 2.3T

================================================================================
                           📖 MONITORING COMMANDS
================================================================================
[... helpful commands ...]
```

---

## 3. 📈 Detailed Metrics Monitor (`monitor_training.py`)

**When to use:**
- Detailed metrics analysis
- See loss trends
- Check training stability
- Verify learning rates

**Commands:**
```bash
# One-time check
python scripts/monitor_training.py --logdir ckpts/stage2_3d/logs/roomplan

# Auto-refresh every 30 seconds
python scripts/monitor_training.py --logdir ckpts/stage2_3d/logs/roomplan --watch

# Custom refresh interval (e.g., every 60 seconds)
python scripts/monitor_training.py --logdir ckpts/stage2_3d/logs/roomplan --watch --interval 60
```

**What it shows:**
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
📚 Learning Rates:
────────────────────────────────────────────────────────────────────────────────
   Base LR: 4.85e-06
   Projector LR: 9.70e-05

────────────────────────────────────────────────────────────────────────────────
⏱️  Speed:
────────────────────────────────────────────────────────────────────────────────
   Current: 1.23 steps/sec
   Average (last 10): 1.21 steps/sec

────────────────────────────────────────────────────────────────────────────────
📊 Loss Trend (last 20 steps):
────────────────────────────────────────────────────────────────────────────────
   █    █  █
   ██  ████
   ████████ █
   ████████ █
   ████████ ██
   ████████ ██  █
   ████████ ██  █
   ████████ ██ ██ █
   ████████ ██ ████
   ████████ ███████████ ██   ██████
   3.75                                4.12
   └──────────────────────────────────────┘

================================================================================
💡 Tips:
   • Loss should generally decrease over time
   • If loss is NaN or increasing rapidly, training may have issues
   • Run with --watch to auto-refresh every 30 seconds
================================================================================
```

**Features:**
- Current step and progress percentage
- Loss statistics (current, recent average, min/max)
- Learning rate tracking (separate for base and projector)
- Training speed metrics
- ASCII plot of loss trend (last 20 steps)
- Auto-refresh mode for continuous monitoring

---

## 4. 📊 TensorBoard (Full Visualization)

**When to use:**
- Publication-quality plots
- Long-term trend analysis
- Compare multiple runs
- Export metrics

**Command:**
```bash
# Start TensorBoard
tensorboard --logdir ckpts/stage2_3d/logs/roomplan --port 6006

# If port 6006 is in use, try another port
tensorboard --logdir ckpts/stage2_3d/logs/roomplan --port 6007
```

**Access:**
- Open browser: http://localhost:6006
- If on remote server, set up SSH tunnel:
  ```bash
  # On your local machine
  ssh -L 6006:localhost:6006 user@remote-server
  ```

**Available Metrics:**
| Metric | Description |
|--------|-------------|
| `train/loss` | Training loss value |
| `train/learning_rate_base` | Base model learning rate |
| `train/learning_rate_proj` | Projector learning rate |
| `train/steps_per_sec` | Training throughput |
| `train/progress_pct` | Training progress percentage |
| `train/step_time` | Time per training step |

**TensorBoard Features:**
- Interactive line plots
- Smoothing controls
- Download data as CSV/JSON
- Compare multiple experiments
- Scalar, image, and histogram visualization

---

## 5. 📁 Direct Log Files

**View raw logs:**
```bash
# Find log files
find ckpts/stage2_3d/logs -name "*.log"

# Tail the latest log
tail -f $(find ckpts/stage2_3d/logs -name "*.log" -type f | tail -1)
```

---

## 📋 Monitoring Workflow Recommendations

### During Training Start (First 100 steps)
1. Watch console output for initialization
2. Check for NaN warnings
3. Verify GPU utilization: `./check_training.sh`

### During Active Training
1. Use `monitor_training.py --watch` in a separate terminal
2. Check status every few hours: `./check_training.sh`
3. Review TensorBoard daily for trends

### Before Stopping Training
1. Check latest checkpoint: `./check_training.sh`
2. Verify loss is stable/decreasing: `python scripts/monitor_training.py --logdir ...`
3. Review TensorBoard for anomalies

---

## 🔍 What to Look For

### ✅ Healthy Training Indicators
- **Loss**: Gradually decreasing (may have plateaus)
- **Speed**: Consistent (1-2 steps/sec on 2× A100)
- **GPU**: 90-100% utilization
- **Memory**: Stable, not growing
- **Learning Rate**: Following cosine schedule

### ⚠️ Warning Signs
- **Loss NaN or Inf**: Training failure (should stop automatically)
- **Loss increasing**: Learning rate too high or data issue
- **Loss stuck**: Learning rate too low or model converged
- **Slow speed**: GPU not fully utilized or I/O bottleneck
- **GPU < 50%**: Check data loading or model parallelism

### 🚨 Critical Issues
- **NaN count > 10**: Training stops automatically
- **OOM errors**: Reduce batch size
- **Process killed**: Out of memory (system, not GPU)

---

## 💡 Tips & Tricks

### Save Terminal Output
```bash
./train.sh 2>&1 | tee training_log.txt
```

### Monitor From Another Machine
```bash
# SSH tunnel for TensorBoard
ssh -L 6006:localhost:6006 user@training-server

# Then open http://localhost:6006 on your local machine
```

### Check Training at Specific Intervals
```bash
# Every hour, check status and save to log
while true; do
    echo "=== $(date) ===" >> status_log.txt
    ./check_training.sh >> status_log.txt
    sleep 3600
done
```

### Compare Multiple Runs
```bash
tensorboard --logdir_spec=run1:ckpts/exp1/logs,run2:ckpts/exp2/logs --port 6006
```

---

## 🆘 Troubleshooting

**"No event files found yet"**
- Training hasn't started writing logs yet (wait ~1 minute)
- Check TensorBoard log directory path

**TensorBoard shows old data**
- Refresh the browser (Ctrl+R or Cmd+R)
- Click the refresh icon in TensorBoard UI

**Can't access TensorBoard on remote server**
- Set up SSH tunnel (see above)
- Check firewall settings
- Verify port is not in use: `lsof -i :6006`

**Monitor script crashes**
- Install TensorBoard: `pip install tensorboard`
- Check log directory exists
- Verify event files are not corrupted

---

## 📚 Related Documentation

- **Training Guide**: `COMPLETE_TRAINING_GUIDE.md`
- **Quick Start**: `QUICK_START.md`
- **File Structure**: `FILE_GUIDE.md`

---

**Last Updated**: November 25, 2025  
**Status**: All monitoring tools ready for use
