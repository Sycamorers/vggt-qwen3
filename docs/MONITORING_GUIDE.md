# Training Monitoring & Visualization Guide

Applies to the current Stage 1 setup (`train_fixed.sh` + `configs/stage1_3d.yaml`). All commands assume repo root as CWD.

---

## Quick Reference
| Tool | Command | Purpose |
|------|---------|---------|
| TensorBoard | `tensorboard --logdir ckpts/stage1_3d/logs --port 6006 --bind_all` | Full visualization of loss/LR/speed |
| CLI Monitor (snapshot) | `python scripts/monitor_training.py --logdir ckpts/stage1_3d/logs/roomplan` | One-time metrics, ASCII plot |
| CLI Monitor (watch) | `python scripts/monitor_training.py --logdir ckpts/stage1_3d/logs/roomplan --watch --interval 30` | Auto-refreshing metrics |
| Log tail | `tail -f pytorchdist_*.out` (Slurm) or `tail -f logs/stage2/*.log` | Live console output |
| GPU check | `watch -n 5 nvidia-smi` | Utilization/memory sanity |

---

## 1) Console Output (built-in)
`src/train/train_sft.py` logs every `log_every_steps`:
- step, loss, base LR / projector LR
- throughput (steps/s) and ETA
- checkpoint cadence (every `save_every_steps`)

If launched via Slurm, capture stdout/stderr in `pytorchdist_*.out`; otherwise view directly in the terminal.

---

## 2) TensorBoard
```bash
tensorboard --logdir ckpts/stage1_3d/logs --port 6006 --bind_all
```
Open `http://localhost:6006` (or tunnel if on a cluster). Scalars include loss, LR, and speed from `accelerator.log`.

---

## 3) CLI Monitor (`scripts/monitor_training.py`)
One-time snapshot:
```bash
python scripts/monitor_training.py --logdir ckpts/stage1_3d/logs/roomplan
```
Auto-refresh (recommended while training):
```bash
python scripts/monitor_training.py --logdir ckpts/stage1_3d/logs/roomplan --watch --interval 30
```
Shows: current/avg/min/max loss, LRs, step progress bar, speed, ASCII trend.

---

## 4) Log Tail & System Checks
- Slurm logs: `tail -f pytorchdist_*.out | grep --line-buffered "Step\|Loss\|ETA"`
- Local logs (if any): `tail -f logs/stage2/*.log`
- GPUs: `watch -n 5 nvidia-smi`
- Disk space: `du -sh ckpts/stage1_3d` and `df -h .`

---

## 5) Troubleshooting Monitoring
- **No events in TensorBoard:** Ensure `ckpts/stage1_3d/logs/roomplan` exists; training must reach first `log_every_steps`.
- **Monitor script shows 0 steps:** Confirm the `--logdir` matches the `project_dir` used by `train_fixed.sh` (defaults under `ckpts/stage1_3d/logs/roomplan`).
- **Slow terminal refresh:** Increase `--interval` (e.g., `--interval 60`) for `monitor_training.py`.

For end-to-end training issues, see `docs/COMPLETE_TRAINING_GUIDE.md` (Troubleshooting section).
