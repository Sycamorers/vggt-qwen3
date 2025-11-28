# VGGT-Qwen3 RoomPlan

This repository trains a VGGT + Qwen3 vision-language model to reason about indoor 3D scenes and (optionally) emit RoomPlan-style JSON actions.

---

## 1) What You’re Building
- **Goal:** Teach Qwen3-4B to answer 3D scene questions using VGGT multi-view features; later extend to ARKit RoomPlan actions.
- **Pipeline (code-backed):** VGGT aggregator (frozen) → Perceiver projector → optional geometry tokens → Qwen3-4B with LoRA. See `src/models/vggt_qwen3_vlm.py` and `configs/perceiver_small.yaml`.
- **Current stage:** Stage 2 (3D QA) is ready to run with existing processed data; Stage 3 (RoomPlan actions) needs data generation.
- **Distributed training:** Supported via `train_fixed.sh` (DeepSpeed ZeRO-3, 1–8 GPUs) and Slurm template `run.sh`.

For a deeper conceptual overview and a step‑by‑step walkthrough, see:
- `docs/COMPLETE_TRAINING_GUIDE.md` – detailed architecture, data, training, troubleshooting
- `docs/QUICK_START.md` – condensed “run it now” steps

---

## 2) Environment Setup (matches `env/environment.yml`)
1. Create the Conda env (PyTorch 2.4, CUDA 12.8, Python 3.10):
   ```bash
   conda env create -f env/environment.yml
   conda activate roomplan
   ```
2. Install VGGT (editable):
   ```bash
   pip install -e third_party/vggt
   ```
3. (Optional) Install Qwen3 editable if needed; otherwise Hugging Face cache is sufficient.
4. GPU requirements: NVIDIA GPUs with ≥20 GB each; bf16 training.
5. Cache guidance (already set in `train_fixed.sh`): use project-local `.cache/` for HF/torch/triton to avoid NFS issues.

---

## 3) Data

### 3.1 Datasets currently available
The following processed datasets are already present in this repository (paths are relative to the project root):

| Dataset | Path glob | Format | Views/sample | Geometry tokens |
|---------|-----------|--------|--------------|-----------------|
| ScanQA | `data/processed/scanqa/*.jsonl` | JSONL | 1 (bird’s-eye) | `null` |
| SQA3D  | `data/processed/sqa3d/*.jsonl`  | JSONL | 1 (bird’s-eye) | `null` |
| DocVQA (unused in current config) | `data/processed/DocVQA/llava_instruct_80k.json` | JSON | n/a | n/a |
| ARKit synthetic | `data/processed/arkit_synth/*.json` (downloaded) | JSON | up to 10 | optional |

Sample record (from `data/processed/sqa3d/train.jsonl`):
```json
{
  "images": ["data/processed/SQA3D/bird/scene0380_00_bird.png"],
  "geom_token": null,
  "question": "What color is the desk to my right?",
  "answer": "brown",
  "task": "sqa3d"
}
```

How the code consumes this:
- `src/dataio/dataset_builder.py` loads `.json` arrays or `.jsonl` lines and truncates to `num_views` (8 in Stage 2; single-view data is accepted).
- `src/dataio/collate_multiview.py` resizes to 448, builds text as `{question}\n<image>\n{answer}`, pads to ensure room for visual tokens, and skips geometry tokens when `geom_token` is null.
- Vision features are injected at the `<image>` position; geometry head is bypassed when absent.

If you regenerate data with poses/depth, keep keys `R`, `t`, `K`, `depth_hist` so `encode_geom` works unchanged.

### 3.2 Downloading / rebuilding datasets yourself

**ScanQA / SQA3D (Stage 2):**
- Obtain ScanNet, ScanQA, and SQA3D according to their licenses.
- Place raw data under `data/raw/` (e.g., `data/raw/scannet`, question files under `data/raw/scanqa`, `data/raw/sqa3d`).
- Build processed JSONL shards:
  ```bash
  # ScanQA
  python scripts/prep/prepare_scanqa.py \
    --dataset scanqa \
    --scan-root data/raw/scannet \
    --qa-file data/raw/scanqa/questions.json \
    --output data/processed/scanqa/train.jsonl \
    --num-views 8

  # SQA3D
  python scripts/prep/prepare_scanqa.py \
    --dataset sqa3d \
    --scan-root data/raw/scannet \
    --qa-file data/raw/sqa3d/questions.json \
    --output data/processed/sqa3d/train.jsonl \
    --num-views 8
  ```
  This will create multi‑view, geometry‑aware samples compatible with the existing Stage 2 config.

**ARKitScenes / RoomPlan synthetic data (Stage 3):**
- Download ARKitScenes into `data/raw/arkitscenes` (already done in your setup for a subset of scenes):
  ```bash
  bash scripts/prep/download_arkitscenes.sh --dest data/raw/arkitscenes
  ```
- Generate synthetic instruction/action pairs (first pass can be partial; only the first 10 scenes are needed for initial inference):
  ```bash
  python scripts/prep/synth_roomplan_instructions.py \
    --arkit-root data/raw/arkitscenes \
    --output data/processed/arkit_synth/train.json \
    --num-views 10
  ```
  Stage 3 configs (`configs/stage3_arkit*.yaml`) expect these files under `data/processed/arkit_synth/`. For the current ARKit inference task, only the first 10 available scenes from this JSON will be used.

For more discussion of data design and trade‑offs, see `docs/COMPLETE_TRAINING_GUIDE.md` (Data section).

### 3.3 ScanQA and SQA3D (Stage 2 datasets)

Stage 2 training (`configs/stage2_3d.yaml`) uses two 3D question–answering datasets:

- **ScanQA** (`data/processed/scanqa/*.jsonl`)
  - Task: free-form, open-vocabulary QA over indoor scenes (e.g., “What color is the cabinet to the left of the bed?”).
  - Processed format in this repo:
    - Each JSONL line has:
      - `images`: a list with a single canonical view image path.
      - `question`: a natural-language question about the scene.
      - `answer`: a short free-form answer.
      - `task`: `"scanqa"`.
  - Usage:
    - In Stage 2 configs, ScanQA contributes most of the training signal for general 3D reasoning.

- **SQA3D** (`data/processed/sqa3d/*.jsonl`)
  - Task: spatial and semantic QA about scenes, with a focus on relative location and object relationships.
  - Processed format matches ScanQA:
    - `images`: one bird’s-eye or canonical view.
    - `question`: localized question (e.g., “What is in front of the sofa?”).
    - `answer`: short answer.
    - `task`: `"sqa3d"`.
  - Usage:
    - Mixed with ScanQA during Stage 2 to improve robustness on spatial and relational questions.

In both cases:
- Images are single-view in the current processed shards, but the model and dataloader accept multi-view input seamlessly once such data is available.
- Geometry tokens are currently `null` and therefore bypassed; the model still uses VGGT’s visual tokens for 3D reasoning.

---

## 4) Model & Configs
- **Vision:** VGGT aggregator (`third_party/vggt/vggt_1B_commercial.pt`, frozen).
- **Projector:** Perceiver (`configs/perceiver_small.yaml`) with 128 latents, 6 layers, outputs Qwen3 hidden dim.
- **Language:** `Qwen/Qwen3-4B-Instruct-2507` + LoRA on q/k/v/o (see config).
- **Precision/optim:** bf16; AdamW with higher LR on projector/geom head; cosine schedule with warmup.
- **Active stage config:** `configs/stage2_3d.yaml`
  - Data: ScanQA (0.7) + SQA3D (0.3)
  - `num_views: 8`, `image_size: 448`, `max_length: 512`, `view_dropout: 0.3`
  - Train: `batch_size_per_gpu: 6`, `grad_accum: 32`, `max_steps: 30000`, `save_every_steps: 1500`
  - LoRA: rank 16 on q/k/v/o
- **Missing data for Stage 3:** `configs/stage3_arkit.yaml` expects `data/processed/arkit_synth/*.json`.

---

## 5) How to Train (Local or Slurm)
Primary launcher: `train_fixed.sh` (sets caches, NCCL envs, probes GPU/host memory, DeepSpeed ZeRO-3).

**Local / interactive**
```bash
# [mode]=full|debug, [num_gpus]=1-8
./train_fixed.sh full 4
./train_fixed.sh --safe debug 2   # 100 steps, conservative batch
```
Defaults: `CONFIG_FILE=configs/stage2_3d.yaml`, outputs to `ckpts/stage2_3d` (or `_debug`).

**Slurm (HiPerGator template)**
```bash
sbatch run.sh   # calls train_fixed.sh --safe full 8 on partition hpg-b200
```
Adjust account/time/memory/partition in `run.sh` as needed. More Slurm templates: `scripts/slurm/`. Guide: `docs/SLURM_TRAINING_GUIDE.md`.

**Resuming**
- Accelerate state is saved every `save_every_steps` to `ckpts/stage2_3d/step_xxxxx/`.
- Resume by pointing `--output_dir` (or editing `train_fixed.sh`) to the existing folder; Accelerate will load state automatically.

Detailed training workflow (including troubleshooting and tips) is documented in `docs/COMPLETE_TRAINING_GUIDE.md`.

---

## 6) Checkpoints & Weights Layout

This project uses DeepSpeed ZeRO-3 for multi-GPU training. Checkpoints are stored under `ckpts/` with the following recommended layout:

- Stage 2 (ScanQA/SQA3D 3D QA):
  - Training output root: `ckpts/stage2_3d/`
    - `pytorch_model/` – DeepSpeed ZeRO shards (per-rank model states and optimizer states).
    - `step_XXXX/` – periodic save_state folders used for resume.
    - `logs/` – TensorBoard logs.
- Stage 3 (ARKit / RoomPlan actions, when trained):
  - Training output root: `ckpts/stage3_arkit/` (or similar)
    - Same internal structure as Stage 2.

For inference and post-hoc analysis, it is often convenient to convert the ZeRO shards into a standard fp32 checkpoint:

1) Change into the Stage directory and run `zero_to_fp32.py`:
```bash
cd ckpts/stage2_3d

# Example: convert ZeRO shards in pytorch_model/ into sharded fp32 weights
python zero_to_fp32.py pytorch_model pytorch_model_fp32
```

2) This will create a directory:
- `ckpts/stage2_3d/pytorch_model_fp32/`
  - `pytorch_model-00001-of-00005.bin`
  - `pytorch_model-00002-of-00005.bin`
  - …

The inference code (`src/inference/arkit_inference.py`) is written to:
- Prefer the directory `pytorch_model_fp32/` inside the checkpoint directory.
- Still work with older runs where a directory named `pytorch_model_fp32.bin/` was created by mistake.
- Fall back to any `*.bin` / `*.safetensors` in the top-level checkpoint directory if no fp32 folder is present.

For a clean, professional layout going forward, use `pytorch_model_fp32/` as the folder name and avoid having directories that end in `.bin`.

---

## 6) Monitoring & Debugging
- **Console logging:** Enabled in `src/train/train_sft.py` (loss, LR, speed, ETA every `log_every_steps`).
- **TensorBoard:** `tensorboard --logdir ckpts/stage2_3d/logs --port 6006 --bind_all`
- **CLI monitor:** `python scripts/monitor_training.py --logdir ckpts/stage2_3d/logs/roomplan --watch`
- **Log tail:** `tail -f pytorchdist_*.out | grep "Step\\|Loss\\|ETA"` (Slurm) or `tail -f logs/stage2/*.log` if present.
- **Docs:** `docs/MONITORING_GUIDE.md`, `docs/TRAINING_MONITORING_SUMMARY.md`.

If you hit NCCL or cache issues, `train_fixed.sh` already sets conservative envs and local caches; see comments inside that script for overrides.

---

## 7) What’s Done vs. To-Do
- ✅ Code wired for Stage 2 multi-view QA; single-view ScanQA/SQA3D shards present and runnable.
- ✅ Distributed training supported (DeepSpeed ZeRO-3, Slurm templates).
- ✅ Monitoring utilities and guides are in place.
- ⚠️ Stage 1 instruction data not present (llava/sharegpt4v/docvqa/chartqa paths empty except DocVQA JSON).
- ⚠️ Stage 3 (RoomPlan actions) data missing (`data/processed/arkit_synth`).
- ⚠️ Current shards lack geometry tokens and multi-view frames; VGGT still runs but without geometry conditioning.

---

## 8) Extending the Project
- **Add geometry + multi-view:** Rebuild ScanQA/SQA3D with poses/depth using `scripts/prep/prepare_scanqa.py --dataset scanqa|sqa3d --num-views 8 --output data/processed/{scanqa,sqa3d}/train.jsonl`.
- **Generate RoomPlan data:** Run `scripts/prep/synth_roomplan_instructions.py --arkit-root data/raw/arkitscenes --output data/processed/arkit_synth/train.json --num-views 10`, then switch `CONFIG_FILE` to `configs/stage3_arkit.yaml`.
- **Small smoke tests:** Use the mini configs in `configs/*_mini.yaml` and set VGGT to `mock` in `configs/local_*` if you need CPU-only checks.

Stage‑wise training strategies (Stage 1 → Stage 2 → Stage 3) are described in more detail in `docs/COMPLETE_TRAINING_GUIDE.md`.

---

## 9) Repository Map (active pieces)
```
configs/                # Stage configs (stage2_3d active), projector, DeepSpeed
data/processed/         # scanqa/*.jsonl, sqa3d/*.jsonl, DocVQA JSON
data/raw/               # placeholders + ARKit README
src/dataio/             # dataset builder + multi-view collator
src/models/             # VGGT-Qwen3 wrapper, Perceiver projector
src/train/train_sft.py  # Accelerate/DeepSpeed trainer
train_fixed.sh          # Hardened launcher (defaults to stage2_3d)
run.sh                  # Slurm example invoking train_fixed.sh
logs/, ckpts/           # Training outputs
docs/                   # Guides (complete training, quickstart, monitoring, Slurm, file map)
```
More file descriptions: `docs/FILE_GUIDE.md`.

---

## 10) Quickstart Checklist
1) `conda activate roomplan` (after creating env)  
2) Place checkpoints: `third_party/vggt/vggt_1B_commercial.pt` and cache `Qwen/Qwen3-4B-Instruct-2507`  
3) Verify data exists: `ls data/processed/scanqa data/processed/sqa3d`  
4) Run: `./train_fixed.sh debug 1` (sanity) → then `./train_fixed.sh full 4` (or Slurm via `run.sh`)  
5) Monitor: `tensorboard --logdir ckpts/stage2_3d/logs` or `python scripts/monitor_training.py --logdir ckpts/stage2_3d/logs/roomplan --watch`  

Once Stage 2 converges and RoomPlan data is prepared, switch to `configs/stage3_arkit.yaml` and re-run `train_fixed.sh` with the new config.

---

## 11) ARKit / RoomPlan Inference

Once you have:
- Completed Stage 2 (or Stage 3 for action JSON, when trained), and
- Generated ARKit synthetic data under `data/processed/arkit_synth/` (even partially),

you can run ARKit inference on the first 10 available scenes as follows.

### 11.1 Prepare ARKit synthetic data (current setup)

With the available ARKit 3DOD-style data in this repo, you can build a small synthetic ARKit instruction dataset as follows:

```bash
conda activate roomplan

python scripts/prep/prepare_arkit_from_3dod.py \
  --arkit-training-root data/processed/ARKit/Training \
  --output data/processed/arkit_synth/train.json \
  --num-views 10 \
  --max_scenes 10
```

This:
- Scans `data/processed/ARKit/Training/<scene_id>/` for 3D object annotations and RGB frames.
- Produces `data/processed/arkit_synth/train.json` (array of ~10 samples for your current partial download).
- Uses **English** instructions of the form:
  - `"In scene 47333462, find an object belonging to the category 'cabinet' and place a virtual anchor at the center of that object."`
- Associates each instruction with a target `action_json`:
  - `{"action": "place_anchor", "scene": "47333462", "center": [...], "normal": [0,1,0], "extent": [...]}`.

### 11.2 Single-GPU ARKit inference (Stage 2 weights)

After Stage 2 training and the fp32 conversion described in section 6, you can run a small ARKit inference sweep using the Stage 2 checkpoint:

```bash
conda activate roomplan

python -m src.inference.arkit_inference \
  --config configs/stage2_3d.yaml \
  --arkit_glob "data/processed/arkit_synth/train.json" \
  --checkpoint_dir ckpts/stage2_3d \
  --num_scenes 9 \
  --max_new_tokens 256 \
  --device cuda:0 \
  --output_jsonl ckpts/arkit_infer/arkit_predictions_stage2_en.jsonl
```

- `--checkpoint_dir` should point to the Stage 2 directory (`ckpts/stage2_3d`) containing both the ZeRO shards and the `pytorch_model_fp32/` directory.
- `--arkit_glob` can be a single JSON file (`train.json`) or a glob like `arkit_synth/*.json`.
- The script:
  - Reconstructs the VGGT-Qwen3 model from the config.
  - Loads fp32 weights from `pytorch_model_fp32/` when available.
  - Runs inference on the first `--num_scenes` samples (9 in the current generated dataset).
  - Prints per-sample logs (instruction, prediction, reference) and writes all results to `ckpts/arkit_infer/arkit_predictions_stage2_en.jsonl`.

### 11.3 Saved inference output format

Each line of `ckpts/arkit_infer/arkit_predictions_stage2_en.jsonl` is a JSON record:

```json
{
  "index": 0,
  "scene_id": "47333462",
  "question": "In scene 47333462, find an object belonging to the category 'cabinet' and place a virtual anchor at the center of that object.",
  "prediction": "... model-generated text ...",
  "reference": {
    "action": "place_anchor",
    "scene": "47333462",
    "center": [...],
    "normal": [0, 1, 0],
    "extent": [...]
  }
}
```

You can inspect a few records via:

```bash
head ckpts/arkit_infer/arkit_predictions_stage2_en.jsonl
```

or with a small Python snippet.

---

## 12) ARKit Inference Results & Analysis (Current Status)

This section summarizes the current ARKit inference behavior based on running Stage 2 weights on the synthetic ARKit dataset described above.

### 12.1 What we did

- **Data**:
  - Built `data/processed/arkit_synth/train.json` from `data/processed/ARKit/Training/47333462/...` using `scripts/prep/prepare_arkit_from_3dod.py`.
  - Each sample is a multi-view scene + English instruction + reference `action_json`.
  - For your current download, this yields 9 samples from scene `47333462`.

- **Model & weights**:
  - Used `configs/stage2_3d.yaml` (ScanQA/SQA3D Stage 2 config).
  - Loaded VGGT weights from `third_party/vggt/vggt_1B_commercial.pt`.
  - Loaded Stage 2 fp32 weights from `ckpts/stage2_3d/pytorch_model_fp32/` (created via `zero_to_fp32.py`).
  - Note: only one fp32 shard is currently used (with ~450 missing keys), so the loaded state is effectively “Stage 2 + some base Qwen3 weights”. This is acceptable for wiring tests, but not ideal for final quality.

- **Inference**:
  - Ran:

    ```bash
    python -m src.inference.arkit_inference \
      --config configs/stage2_3d.yaml \
      --arkit_glob "data/processed/arkit_synth/train.json" \
      --checkpoint_dir ckpts/stage2_3d \
      --num_scenes 9 \
      --max_new_tokens 256 \
      --device cuda:0 \
      --output_jsonl ckpts/arkit_infer/arkit_predictions_stage2_en.jsonl
    ```

  - Inference completed without runtime errors (dtype mismatch and sequence-length issues were fixed in the code).

### 12.2 Observed behavior

From the saved predictions:

- For each instruction like:

  > In scene 47333462, find an object belonging to the category 'cabinet' and place a virtual anchor at the center of that object.

  the model typically:
  - Repeats the instruction verbatim.
  - Produces long, generic explanations such as:
    - “We are given a scene ID… This is a hypothetical or simulation-based instruction…”
    - Step-by-step reasoning about how to locate the object in Unity/Unreal and compute its center.
  - Occasionally fabricates concrete coordinates or orientations (e.g., “coordinates (12.45, 18.32, 2.15)”), but these are **not** the ground-truth `center` from the reference JSON; they are hallucinations.

- For some categories (`chair`, `tv_monitor`), the model:
  - Loops on the same sentence multiple times (“In scene 47333462, find an object belonging to the category 'chair'…”).
  - Or asserts that the scene does not contain such an object (which is unrelated to the synthetic reference).

- The `reference` field in each record is a structured dict derived from the ARKit annotation:

  ```json
  {
    "action": "place_anchor",
    "scene": "47333462",
    "center": [...],
    "normal": [0, 1, 0],
    "extent": [...]
  }
  ```

- The script computes an **exact-match** metric between `prediction` and `reference`:
  - Since `prediction` is long-form English text and `reference` is JSON, the exact-match score is:

    ```text
    Summary over 9 samples with reference: exact_match = 0.000
    ```

  - This is expected under the current setup.

### 12.3 Why the results look like this

There are three main reasons:

1. **Model behavior (instruction-tuned Qwen3)**:
   - The base model `Qwen/Qwen3-4B-Instruct-2507` is trained as a general-purpose assistant.
   - Given an instruction, it tends to:
     - Explain context and limitations.
     - Provide step-by-step reasoning.
     - Apologize or simulate actions when it cannot access a real scene.
   - It is **not** natively trained to output concise JSON actions for ARKit scenes.

2. **Training mismatch (Stage 2 vs. ARKit actions)**:
   - The Stage 2 checkpoint (`configs/stage2_3d.yaml`) was trained on ScanQA/SQA3D question answering, not on ARKit `action_json`.
   - The ARKit synthetic data (instructions + `action_json`) is new to the model at inference time.
   - As a result, the model behaves like a generic instruction follower rather than a RoomPlan action predictor.

3. **Prompt format & target mismatch**:
   - Current prompts are natural-language instructions followed by `<image>`, with no explicit constraint such as “respond only with JSON” or “output the action_json”.
   - The supervised target for ARKit has not yet been integrated into training (Stage 3 is not trained/loaded), so the model has no strong prior to emit `action_json` instead of free-form text.

Additionally:
- We currently load only one fp32 shard (`pytorch_model-00001-of-00005.bin`), resulting in ~450 missing keys when applying the state dict. This is acceptable for plumbing tests but means we are not using the full Stage 2 finetuned model.

### 12.4 Recommended next steps

To move from “plumbing works” to a useful ARKit / RoomPlan model:

1. **Train a dedicated Stage 3 ARKit model**:
   - Use `configs/stage3_arkit.yaml` (or a variant) and `data/processed/arkit_synth/train.json` as the main dataset.
   - Supervise the model to output `action_json` (or a JSON-only textual representation) conditioned on the instruction and images.
   - Save the Stage 3 checkpoint under `ckpts/stage3_arkit/` and convert it to `pytorch_model_fp32/` via `zero_to_fp32.py`.

2. **Constrain the output format**:
   - In data prep and/or prompts, explicitly tell the model to respond with JSON only, e.g.:

     ```text
     In scene 47333462, find an object belonging to the category 'cabinet' and place a virtual anchor at the center of that object.
     Respond ONLY with a JSON object in the following format:
     {"action": "place_anchor", "scene": "<scene_id>", "center": [x, y, z], "normal": [nx, ny, nz], "extent": [ex, ey, ez]}
     <image>
     ```

   - Use this as the supervised target for Stage 3 so the model learns to output structured actions instead of long explanations.

3. **Load the complete fp32 state**:
   - When converting ZeRO checkpoints, prefer a single `pytorch_model.bin` (or full set of shards) and load that, to avoid the “Missing keys” situation and ensure you are using all trained parameters.

4. **Evaluation**:
   - Replace strict string exact-match with metrics that compare parsed JSON (e.g., center L2 distance, category correctness, etc.) once the model starts emitting JSON.

With these steps, the current code and naming conventions (including the cleaned-up checkpoint folder names) provide a solid, professional baseline for continued ARKit / RoomPlan experimentation.

---

## 11) ARKit / RoomPlan Inference

Once you have:
- Completed Stage 3 (or at least have a RoomPlan-capable checkpoint under `ckpts/`), and
- Generated ARKit synthetic data under `data/processed/arkit_synth/` (even partially),

you can run ARKit inference on the first 10 available scenes as follows.

### 11.1 Single-GPU inference

```bash
conda activate roomplan

# Example: use the final Stage 3 checkpoint
python -m src.inference.arkit_inference \
  --config configs/stage3_arkit.yaml \
  --arkit_glob "data/processed/arkit_synth/*.json" \
  --checkpoint_dir ckpts/stage3_arkit/step_10000 \
  --num_scenes 10 \
  --max_new_tokens 256 \
  --device cuda:0
```

Notes:
- `--num_scenes 10` ensures only the first 10 scenes are processed, which is useful if ARKit data has only been partially downloaded.
- `--device` can be set to `cpu` for a quick sanity check (slow) or `cuda` / `cuda:N` for GPU.
- If `--checkpoint_dir` is omitted, the script will run using the base Qwen3 + VGGT weights defined in the config.

### 11.2 Multi-GPU inference (optional)

The inference script itself runs on a single GPU. If you want to parallelize over multiple GPUs, you can launch multiple processes manually, each with a different `CUDA_VISIBLE_DEVICES` and a different range of scenes (for example, `--num_scenes` + an additional index filter in the JSON). For the current task, single-GPU inference on the first 10 scenes is sufficient.
