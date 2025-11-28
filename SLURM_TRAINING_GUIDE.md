# 🚀 Slurm Training Guide

## Quick Start for Production Training

### 1. Submit Training Job

```bash
# For Stage 2 (3D understanding) - Full training with 5 GPUs
sbatch scripts/slurm/stage2_3d_2xb200.sbatch

# Check job status
squeue -u $USER

# Check job details
scontrol show job JOBID
```

### 2. Monitor Training Progress

#### Option A: TensorBoard (Recommended for visualization)

```bash
# Start TensorBoard on login node
tensorboard --logdir ckpts/stage2_3d/logs --port 6006 --bind_all &

# Note your login node
hostname  # e.g., login9.ufhpc

# On your local machine, create SSH tunnel:
ssh -L 6006:login9.ufhpc:6006 YOUR_USERNAME@hpg.rc.ufl.edu

# Open browser: http://localhost:6006
```

#### Option B: Watch Log File

```bash
# Find your job's output file
ls -lt pytorchdist_*.out | head -1

# Tail the most recent output
tail -f pytorchdist_*.out | grep --line-buffered "Step\|Loss\|ETA"

# Or watch for progress every 10 seconds
watch -n 10 'tail -50 pytorchdist_*.out | grep "Step.*/"'
```

#### Option C: Check Saved Checkpoints

```bash
# List checkpoints
ls -lth ckpts/stage2_3d/step_*/

# Checkpoints are saved every 1500 steps
```

### 3. Disk Space Management

#### Clean Old Training Artifacts

```bash
# Run the cleanup script (preserves model weights)
./cleanup_old_runs.sh
```

This removes:
- ❌ Core dumps (often 10+ GB!)
- ❌ Old Slurm .out files
- ❌ Old .log files  
- ❌ TensorBoard event files (regenerated)
- ❌ Python bytecode cache
- ❌ Triton kernel cache (regenerated)

This preserves:
- ✅ Model weights (.cache/huggingface/, .cache/transformers/)
- ✅ VGGT weights (third_party/vggt/vggt_1B_commercial.pt)
- ✅ Training data (data/)
- ✅ Most recent .log and .out files

### 4. Job Management

```bash
# Cancel a job
scancel JOBID

# Cancel all your jobs
scancel -u $USER

# Check job queue
squeue -u $USER

# Check job efficiency after completion
seff JOBID
```

### 5. Training Configuration Files

- **Slurm scripts**: `scripts/slurm/`
  - `stage2_3d_2xb200.sbatch` - Stage 2 training (2 nodes, 200GB mem each)
  
- **Training configs**: `configs/`
  - `stage2_3d.yaml` - Full Stage 2 config
  - `stage2_3d_mini.yaml` - Debug config (fewer steps)

- **Accelerate configs**: Auto-generated per job
  - `configs/accelerate_5gpu.yaml` - Generated for 5 GPU runs

### 6. Expected Training Times

Based on recent runs with dataloader workers:

- **With dataloader workers (NEW)**: ~2-4 steps/sec
  - 30,000 steps ≈ **2-4 hours**
  
- **Without workers (OLD)**: ~0.3 steps/sec
  - 30,000 steps ≈ **20-30 hours** ❌ Too slow!

The updated `train_fixed.sh` now uses dataloader workers for much faster training.

### 7. Troubleshooting

#### Job fails immediately
```bash
# Check Slurm output file for errors
cat pytorchdist_JOBID.out

# Check node allocation
scontrol show job JOBID
```

#### Out of memory errors
```bash
# Edit sbatch script to request more memory
# or reduce batch size in config
vim scripts/slurm/stage2_3d_2xb200.sbatch
```

#### NCCL timeout errors
```bash
# Check if all GPUs are visible
srun --jobid=JOBID nvidia-smi

# The training script already sets NCCL_TIMEOUT=3600
```

#### Training stuck
```bash
# Check if processes are running
srun --jobid=JOBID ps aux | grep python

# Check GPU utilization
srun --jobid=JOBID nvidia-smi
```

### 8. After Training Completes

```bash
# Check final checkpoint
ls -lh ckpts/stage2_3d/step_30000/

# Review TensorBoard logs
tensorboard --logdir ckpts/stage2_3d/logs

# Clean up old files
./cleanup_old_runs.sh
```

---

## 📋 Quick Reference Commands

```bash
# Submit job
sbatch scripts/slurm/stage2_3d_2xb200.sbatch

# Monitor with TensorBoard
tensorboard --logdir ckpts/stage2_3d/logs --port 6006 --bind_all &

# Watch progress in terminal
tail -f pytorchdist_*.out | grep "Step"

# Cancel job if needed
scancel JOBID

# Clean up afterward
./cleanup_old_runs.sh
```

---

**Ready for your sbatch submission!** 🚀
