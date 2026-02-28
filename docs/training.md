## Stage‑1 Training

Stage‑1 trains a VGGT‑Qwen3 vision‑language model on ScanQA + SQA3D using
supervised next‑token prediction.

Key entrypoints:

- `train_fixed.sh` — Slurm‑friendly launcher with safety probes and caching fixes.
- `scripts/train_stage1.sh` — thin wrapper that calls `train_fixed.sh`.
- `src/vggt_qwen3/train/stage1.py` — actual training loop (Accelerate + DeepSpeed).
- `configs/stage1_3d.yaml` — model + optimizer + schedule configuration.

### Model components

- **VGGT** (`third_party/vggt`)  
  - Frozen multi‑view 3D vision backbone.  
  - Takes stacked RGB views `[B, V, 3, H, W]`.  
  - Produces aggregated visual tokens `[B, S, num_vis_tokens, D_vggt]`.

- **Perceiver projector** (`vggt_qwen3.models.projector_perceiver`)  
  - Perceiver‑style cross‑attention and MLP.  
  - Maps VGGT tokens from `D_vggt` to Qwen3 hidden size `D_text`.  
  - Output shape: `[B, num_vis_tokens, D_text]`.

- **Geometry head** (optional, Stage‑1 enabled by default)  
  - Input features: rotation, translation, intrinsics, depth histogram.  
  - Concatenated dimension `geom_feature_dim = 37`.  
  - Two‑layer MLP → `geom_tokens` of shape `[B, G, D_text]`.

- **Text model (Qwen3‑4B‑Instruct)**  
  - HuggingFace `AutoModelForCausalLM`.  
  - LoRA fine‑tuned on select attention and MLP projection modules.  
  - Extra `<image>` token added to tokenizer.

### Token flow and loss

1. Collator builds prompts with a `<image>` token as a placeholder in the
   question context.
2. During training (`stage1.py`):
   - Images → VGGT → `vis_tokens` `[B, T_vis, D_text]`.
   - Optional `geom_tokens` `[B, T_geom, D_text]`.
   - Concatenate: `features = [geom_tokens, vis_tokens]` or just `vis_tokens`.
   - Token embeddings from the text model are computed for `input_ids`.
   - Positions of `<image>` in `input_ids` are located, and the corresponding
     span in `inputs_embeds` is replaced with `features`.
3. Labels are standard CausalLM labels:
   - Question + context tokens use label `-100` (ignored).
   - Answer tokens use label indices of the target text.
4. Loss is standard **next‑token cross‑entropy** over the answer span.

### Optimization and schedule

Configured in `configs/stage1_3d.yaml`:

- Optimizer: `AdamW` on:
  - Base text + LoRA parameters (learning rate `lr`).
  - Projector + geometry head (learning rate `proj_lr`).
- Precision: `bf16` (with `Accelerate` mixed precision and DeepSpeed ZeRO‑3).
- Batch size: `batch_size_per_gpu` × `grad_accum` × `num_gpus`.
- LR schedule: cosine with warmup:
  - `warmup_ratio` fraction of `max_steps`.
- Regularization:
  - Weight decay `0.1`.
  - Gradient clipping `1.0`.
- LoRA config:
  - Rank `16`, alpha `32`, dropout `0.05`.
  - Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`.

### Distributed training

- `train_fixed.sh` configures:
  - `Accelerate` with DeepSpeed ZeRO‑3 (`configs/deepspeed_zero3.json`).
  - Safe defaults for NCCL / InfiniBand and cache directories.
  - GPU and host memory probes to adjust `BATCH_PER_GPU` and `GRAD_ACCUM`.
- Stage‑1 is typically run on a single node with 1–8 GPUs.

### Design rationale

- **Frozen VGGT + projector + LoRA** keeps the heavy vision backbone and most
  of Qwen3 stable while adapting a relatively small parameter subset.
- This is well‑suited for:
  - Rapid iteration on dataloaders and prompts.
  - Training under ZeRO‑3 with constrained GPU memory.
  - Sharing and re‑using the base Qwen3 weights.
- Stage‑1 primarily proves that:
  - VGGT‑based multi‑view tokens are usable as conditioning for Qwen3.
  - Simple short‑answer QA prompts over 3D scenes can produce grounded,
    non‑hallucinatory answers after fine‑tuning.

