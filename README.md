# VGGT-Qwen3 RoomPlan

VGGT multi-view encoder + Qwen3 finetuning stack for 3D QA and ARKit/RoomPlan action planning. Targets UF HiPerGator B200 nodes but also supports local smoke runs. Includes stage configs, data prep utilities, and Slurm launchers.

## Repository Layout
- `configs/`: stage1/2/3 YAMLs, mini/local variants, DeepSpeed ZeRO-3 JSON, accelerate templates.
- `src/`: training entrypoint (`src/train/train_sft.py`), datasets (`src/dataio/*`), and model glue (`src/models/*`).
- `scripts/prep/`: ScanQA/SQA3D builder (`prepare_scanqa.py`), ARKit synthetic instructions (`synth_roomplan_instructions.py`), ARKitScenes downloader.
- `scripts/slurm/`: single-node templates for stage1/2/3.
- `train_fixed.sh`: guarded multi-GPU launcher with cache relocation and NCCL tweaks; `run.sh` is a full-node B200 example.
- Data: processed ScanQA/SQA3D JSONL shards already under `data/processed/scanqa` and `data/processed/sqa3d`.

## Prerequisites
- 1–8 CUDA GPUs (tested on H100/B200). Adjust batch size/grad accumulation to your memory budget.
- Conda/Mamba. Main spec: `env/environment.yml` (PyTorch 2.4 + CUDA 12.8). Local/older driver smoke tests: `env/environment_local.yml`.
- VGGT code + weights in `third_party/vggt` with `vggt_1B_commercial.pt`.
- Access to `Qwen/Qwen3-4B-Instruct-2507` (set `HF_HOME=$PWD/.cache/huggingface` to reuse cache).

## Setup
```bash
# VGGT code + checkpoint
git clone https://github.com/Sycamorers/vggt third_party/vggt
cp /path/to/vggt_1B_commercial.pt third_party/vggt/

# Env
conda env create -f env/environment.yml
conda activate roomplan
pip install -e third_party/vggt
# optional if vendored: pip install -e third_party/Qwen3
export HF_HOME=$PWD/.cache/huggingface
```

## Data layout & schema
- Expected processed splits (consumed by `src/dataio/dataset_builder.py`, which supports JSON arrays or JSONL):
  - `data/processed/scanqa/{train,val}.jsonl`
  - `data/processed/sqa3d/{train,val}.jsonl`
  - Optional ARKit synth: `data/processed/arkit_synth/*.json`
- Sample record:
  ```json
  {
    "images": ["data/raw/scannet/scene0000/color/000100.jpg", "..."],   // up to V views
    "geom_token": {"R":[...], "t":[...], "K":[...], "depth_hist":[...]}, // optional
    "question": "...",   // or "instruction"
    "answer": "...",     // or "action_json"
    "task": "scanqa"
  }
  ```

## Preparing data
Adjust paths to your raw dumps.
```bash
# ScanQA / SQA3D (JSON array output; keep .json or change config globs)
python scripts/prep/prepare_scanqa.py \
  --dataset scanqa \
  --scan-root data/raw/scannet \
  --qa-file data/raw/scanqa/questions.json \
  --output data/processed/scanqa/train.json \
  --num-views 8

python scripts/prep/prepare_scanqa.py \
  --dataset sqa3d \
  --scan-root data/raw/scannet \
  --qa-file data/raw/sqa3d/questions.json \
  --output data/processed/sqa3d/train.json \
  --num-views 8

# ARKit synthetic instructions (Stage 3)
bash scripts/prep/download_arkitscenes.sh --dest data/raw/arkitscenes
python scripts/prep/synth_roomplan_instructions.py \
  --arkit-root data/raw/arkitscenes \
  --output data/processed/arkit_synth/train.json
```
If you keep `.json` outputs, update `configs/stage*_*.yaml` globs to match; the loader accepts both `.json` and `.jsonl`.

## Training
- Recommended: `train_fixed.sh` (defaults to `configs/stage2_3d.yaml`, outputs to `ckpts/stage2_3d/`, uses 100-step debug mode when `mode=debug`).
  ```bash
  ./train_fixed.sh --safe debug 2   # quick smoke
  ./train_fixed.sh --safe full 4    # full Stage 2 on 4 GPUs
  ./train_fixed.sh --safe full 8    # full node (matches run.sh)
  ```
  The script relocates caches to `.cache/`, probes GPU/host memory to downscale batch/grad_accum, and auto-writes `configs/accelerate_<N>gpu.yaml`. Edit `CONFIG_FILE` / `OUTPUT_DIR` at the top to target Stage 1 or Stage 3.

- Direct torchrun (matches Slurm templates):
  ```bash
  torchrun --standalone --nproc_per_node=2 \
    -m src.train.train_sft \
    --config configs/stage2_3d.yaml \
    --deepspeed configs/deepspeed_zero3.json \
    --output_dir ckpts/stage2_3d
  ```
  Local smoke options: `configs/local_smoke.yaml` (CPU/GPT-2 stub) or `configs/local_qwen3.yaml` (2×GPU, mock vision).

- Stage configs:
  - `configs/stage1_sft.yaml`: 1-view instruction SFT (llava/sharegpt4v/docvqa/chartqa).
  - `configs/stage2_3d.yaml`: 8-view 3D QA on ScanQA/SQA3D + geometry tokens.
  - `configs/stage3_arkit.yaml`: 10-view ARKit synthetic action heads.

## Slurm (HiPerGator)
- Templates: `scripts/slurm/stage{1,2,3}_2xb200.sbatch` (edit `--account`, modules, env name).
- Full-node example: `run.sh` submits `train_fixed.sh --safe full 8` on `hpg-b200`.
- Submit/monitor: `sbatch scripts/slurm/stage2_3d_2xb200.sbatch`, `squeue -u $USER`, tail logs in `logs/`.

## Monitoring & outputs
- Logs + TensorBoard: `<output_dir>/logs/` (e.g., `ckpts/stage2_3d/logs/roomplan`); view with `tensorboard --logdir ckpts/stage2_3d/logs --port 6006`.
- Checkpoints: `<output_dir>/step_*` (from `train_sft`) or `checkpoint-*` (Accelerate save_state from `train_fixed.sh`).
- Caches: `.cache/` when using `train_fixed.sh`; keep on local/scratch storage to avoid NFS contention.
