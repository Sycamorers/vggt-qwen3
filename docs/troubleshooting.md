## Troubleshooting

### Checkpoint loading errors

- **Symptom**: `Checkpoint directory '...' does not exist and --allow_base_fallback is not set.`
  - Fix: Double‑check `--checkpoint_dir` and point it to the Stage‑1 training
    root (e.g. `ckpts/stage1_3d_debug`), not an internal subfolder.
  - If you truly want to run with base Qwen3 weights only, add
    `--allow_base_fallback` (or `ALLOW_BASE_FALLBACK=1` for `infer_stage1.sh`).

- **Symptom**: Error mentioning DeepSpeed ZeRO checkpoint and missing `deepspeed`.
  - Fix: Install `deepspeed` in your environment, or run the offline converter:
    ```bash
    cd ckpts/stage1_3d_debug
    python zero_to_fp32.py . pytorch_model_fp32 --safe_serialization
    ```
    Then point `--checkpoint_dir` to `ckpts/stage1_3d_debug`.

### Unstable multi‑GPU training

- Use `train_fixed.sh` (or `scripts/train_stage1.sh`) instead of calling
  `accelerate launch` directly.
- The script:
  - Probes GPU and host memory and dynamically shrinks `BATCH_PER_GPU` and
    `GRAD_ACCUM`.
  - Relocates caches (`HF_HOME`, `TRITON_CACHE_DIR`, etc.) to a local folder.
  - Configures NCCL / InfiniBand defaults for better stability.

Common issues:

- **OOM or SIGKILL during model load**:
  - Reduce `NUM_GPUS` and/or use `--safe` mode on `train_fixed.sh`.
  - Ensure your Slurm job requests enough host memory.

- **NCCL timeouts**:
  - Verify that you are running on a single node when using
    `compute_environment: LOCAL_MACHINE`.
  - Make sure `NCCL_P2P_DISABLE` and `NCCL_IB_HCA` are set as in `train_fixed.sh`.

### Weird or empty predictions

- Run inference with:
  ```bash
  python -m vggt_qwen3.inference.qa_inference \
    --config configs/stage1_3d.yaml \
    --checkpoint_dir ckpts/stage1_3d_debug \
    --dataset scanqa \
    --short_answer_only \
    --debug_one 0 \
    --verbose_samples 5
  ```
- This prints:
  - Prompt previews.
  - `<image>` token statistics.
  - Shapes and stats of VGGT conditioning tensors.
  - A few decoded predictions.

If `vis_tokens` are empty or all zeros, the code will raise an error instead
of silently returning ungrounded text.

