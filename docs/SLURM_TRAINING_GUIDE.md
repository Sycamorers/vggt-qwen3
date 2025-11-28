# Slurm Training Guide (HiPerGator B200)

This matches the current launcher (`train_fixed.sh`, Stage 2 config by default).

---

## Submit a Job
```bash
sbatch run.sh
```
- `run.sh` calls `train_fixed.sh --safe full 8` on partition `hpg-b200`. Edit partition/time/memory/mail/user in `run.sh` as needed.
- Alternative templates: `scripts/slurm/stage{1,2,3}_*.sbatch` (Stage 2 is ready with existing data).

---

## Monitor a Running Job
- Queue: `squeue -u $USER`
- Job details: `scontrol show job <JOBID>`
- Logs: `tail -f pytorchdist_*.out | grep --line-buffered "Step\\|Loss\\|ETA"`
- TensorBoard:
  ```bash
  tensorboard --logdir ckpts/stage2_3d/logs --port 6006 --bind_all &
  # SSH tunnel from local: ssh -L 6006:<login_host>:6006 <user>@hpg.rc.ufl.edu
  ```
- CLI monitor (from login node):
  ```bash
  python scripts/monitor_training.py --logdir ckpts/stage2_3d/logs/roomplan --watch --interval 30
  ```

---

## Checkpoints & Output
- Saved every `save_every_steps` (1500 in Stage 2) to `ckpts/stage2_3d/step_xxxxx/`.
- TensorBoard events: `ckpts/stage2_3d/logs/roomplan`.

---

## Common Tweaks
- Change GPU count: edit `--gpus-per-node` in `run.sh` and pass the same number to `train_fixed.sh`.
- Batch/grad accum: adjust in `configs/stage2_3d.yaml` or use `--safe` to auto-downscale.
- Environment caches: already set to project-local paths inside `train_fixed.sh` to avoid NFS stalls.

---

## Troubleshooting
- **OOM:** Re-run with `--safe` or lower `batch_size_per_gpu`/`grad_accum` in the config.
- **NCCL timeout:** Ensure all GPUs are visible (`srun --jobid <JOBID> nvidia-smi`); `train_fixed.sh` sets `NCCL_TIMEOUT=3600` and local caches.
- **Slow data loading:** Increase `DATALOADER_NUM_WORKERS` in the environment before launch; `train_fixed.sh` sets a default based on host memory.

For deeper guidance, see `docs/COMPLETE_TRAINING_GUIDE.md` (Troubleshooting) and `docs/MONITORING_GUIDE.md`.
