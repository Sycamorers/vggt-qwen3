# Quick Start (Stage 2 – Ready to Run)

1) **Environment**
```bash
conda env create -f env/environment.yml
conda activate roomplan
pip install -e third_party/vggt
```

2) **Weights**
- Qwen3: cache `Qwen/Qwen3-4B-Instruct-2507` (set `HF_HOME=$PWD/.cache/huggingface`).
- VGGT: place `third_party/vggt/vggt_1B_commercial.pt`.

3) **Data Check**
```bash
ls data/processed/scanqa data/processed/sqa3d   # expect *.jsonl
```
Current shards are single-view with `geom_token: null`; Stage 2 still runs.

4) **Train**
```bash
# [mode]=full|debug, [num_gpus]=1-8
./train_fixed.sh full 4
./train_fixed.sh --safe debug 1   # 100-step smoke test
```
Defaults: `configs/stage2_3d.yaml`, outputs `ckpts/stage2_3d`.

5) **Monitor**
```bash
tensorboard --logdir ckpts/stage2_3d/logs --port 6006 --bind_all
python scripts/monitor_training.py --logdir ckpts/stage2_3d/logs/roomplan --watch
```

6) **Next Steps**
- For RoomPlan actions: generate `data/processed/arkit_synth/*.json` with `scripts/prep/synth_roomplan_instructions.py` and switch to `configs/stage3_arkit.yaml`.
- For multi-view geometry: rebuild ScanQA/SQA3D with `scripts/prep/prepare_scanqa.py --num-views 8`.
