# Training Monitoring Summary (Current Stack)

Use these options while running `train_fixed.sh` (Stage 2 default).

1) **TensorBoard (full UI)**
```bash
tensorboard --logdir ckpts/stage2_3d/logs --port 6006 --bind_all
```

2) **CLI monitor (snapshot)**
```bash
python scripts/monitor_training.py --logdir ckpts/stage2_3d/logs/roomplan
```

3) **CLI monitor (auto-refresh)**
```bash
python scripts/monitor_training.py --logdir ckpts/stage2_3d/logs/roomplan --watch --interval 30
```

4) **Tail logs**
- Slurm: `tail -f pytorchdist_*.out | grep --line-buffered "Step\|Loss\|ETA"`
- Local logs: `tail -f logs/stage2/*.log` (if present)

5) **GPU/host sanity**
```bash
watch -n 5 nvidia-smi
df -h .
```

If metrics are missing, confirm the logdir (`ckpts/stage2_3d/logs/roomplan`) matches your run and that training has progressed past the first `log_every_steps`.
