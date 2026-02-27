# Stage-1 Debug Log (ScanQA + SQA3D)

This log tracks all debugging and hardening work for **Stage 1** of the VGGT-Qwen3 pipeline on ScanQA and SQA3D. Stage 2 is explicitly out of scope.

## Context

**Ground-truth Stage-1 pipeline**
- Multi-view images → VGGT aggregator (frozen) → Perceiver projector
- Geometry tokens are **bypassed in Stage 1**
- Projected visual tokens → Qwen3-4B (LoRA) via token-level injection
- Causal LM objective with teacher forcing: labels are `-100` for prompt and padding; loss is computed only on answer tokens.

**Key injection mechanism**
- Text inputs are tokenized with a dedicated image placeholder token (e.g., `<image>`).
- The model finds positions in `input_ids` where `input_ids == image_token_id`.
- At those positions, the embedding span is overwritten by a continuous visual feature span:
  - `vis_tokens` from VGGT/Perceiver
  - optional `geom_tokens` (not used in Stage 1)
- No loss is applied on the injected visual span (labels `-100` there as part of the prompt).

**High-priority suspected bug**
- Stage-1 collator may not be inserting the image placeholder token into the prompt.
- If true:
  - `image_positions` becomes empty.
  - Injection never happens.
  - Model degenerates into a blind text-only LLM during training and inference.

This document is the single source of truth for:
- What was inspected and why.
- All code changes related to Stage-1 injection, collator behavior, and inference/eval formatting.
- Exact commands to reproduce and validate each change (CPU-friendly where possible).

## Checklist

- [x] Locate special image placeholder token and `image_token_id` definition.
- [x] Document the Stage-1 injection contract (span length, how positions are found, collator assumptions).
- [x] Prove whether visual injection is currently happening in Stage 1.
- [x] Ensure collator **always** inserts the image placeholder token and reserves padding for overwrite.
- [x] Add CPU-only injection validation script (`scripts/validate_injection.py`).
- [x] Add CPU-friendly smoke forward script (`scripts/smoke_forward_cpu.py`).
- [x] Add optional debug-time guardrails in the injection code path.
- [x] Harden inference/eval for short-answer formatting (reduce “format mismatch” failures).
- [x] Define a HiPerGator preflight checklist (CPU preflight, GPU minimal training, short-answer inference).

## HiPerGator Preflight Checklist (Stage 1 only)

**Environment**
- [ ] Load environment and deps  
  - `module load cuda` (if needed on your partition)  
  - `conda activate roomplan`  
  - Ensure `torch`, `transformers`, `accelerate`, and `vggt` are installed (see `docs/QUICK_START.md`).
- [ ] Cache HF + VGGT weights (once)  
  - Qwen3: `HF_HOME=$PWD/.cache/huggingface` and run any small HF load.  
  - VGGT: place `third_party/vggt/vggt_1B_commercial.pt` (only needed for full vision, not for `--use-mock-vision` tests).

**CPU-only preflight (no GPU required)**
- [ ] Validate collator + labels + injection span  
  - `python scripts/validate_injection.py --config configs/stage1_3d.yaml --num-samples 4`  
  - Expect: non-empty `<image>` positions, injection span fully inside `labels == -100`, no overlap with answer tokens.
- [ ] Run single-batch forward with mock VGGT backbone  
  - `export VGGT_DEBUG_INJECT=1`  
  - `python scripts/smoke_forward_cpu.py --config configs/stage1_3d.yaml --device cpu --batch-size 1 --use-mock-vision`  
  - Expect: one scalar loss printed; no runtime errors from injection guardrails.

**GPU-terminal minimal sanity (short, cheap runs)**
- [ ] 20–50 step Stage-1 training smoke test  
  - From a GPU-enabled terminal (e.g., in a SLURM job):  
    - `export CUDA_VISIBLE_DEVICES=0` (or as assigned)  
    - `accelerate launch --config_file configs/accelerate_single_gpu.yaml src/train/train_sft.py --config configs/stage1_3d.yaml --output_dir ckpts/stage1_3d_debug --max_steps 50`  
  - Monitor that:  
    - Loss decreases slightly over the 50 steps.  
    - No `VGGT_DEBUG_INJECT` errors if you keep it enabled.
- [ ] Short-answer QA inference sanity check  
  - With a Stage-1 checkpoint (or base weights for a coarse check):  
    - `python src/inference/qa_inference.py --config configs/stage1_3d.yaml --glob 'data/processed/scanqa/*.jsonl' --checkpoint_dir ckpts/stage1_3d_debug --num_samples 20 --max_new_tokens 32 --output_jsonl ckpts/stage1_3d_debug/qa_preds.jsonl`  
  - Inspect a few lines of `qa_preds.jsonl` to confirm predictions are short phrases, not full sentences.
- [ ] Short-answer format smoke evaluation  
  - `python scripts/smoke_eval_format.py --config configs/stage1_3d.yaml --glob 'data/processed/scanqa/*.jsonl' --checkpoint_dir ckpts/stage1_3d_debug --num_samples 5`  
  - Expect:  
    - Reasonable normalized EM (not necessarily high early in training).  
    - Clear logging of any systematic format issues (e.g., extra boilerplate around answers).

## Stage-1 Injection Contract (current implementation)

- **Special token**
  - Literal placeholder string: `<image>`.
  - Added to tokenizer vocab in:
    - `src/train/train_sft.py::build_tokenizer`
    - `src/models/vggt_qwen3_vlm.py::__init__`
    - `src/inference/qa_inference.py::build_tokenizer`
    - `src/inference/arkit_inference.py::build_tokenizer`
  - `image_token_id = tokenizer.convert_tokens_to_ids("<image>")`.

- **Injection span length**
  - Model config: `num_vis_tokens` and `geom_tokens` in `configs/stage1_3d.yaml`.
  - Vision-language wrapper (`VGGTQwen3VLM`) encodes:
    - `vis_tokens = encode_images(...)  # [B, num_vis_tokens, H]`
    - `geom_tokens = encode_geom(...)   # [B, geom_tokens, H]` when both:
      - `geom_token` batch is not `None`, **and**
      - `self.geom_tokens > 0`.
  - Combined feature span:
    - If `geom_tokens is None`: `features.shape[1] == num_vis_tokens`.
    - Else: `features.shape[1] == geom_tokens + num_vis_tokens`.
  - In current ScanQA/SQA3D data, `geom_token` is `null`, so **Stage 1 effectively injects `num_vis_tokens` only** (no geometry tokens).

- **How injection positions are found (training forward pass)**
  - In `VGGTQwen3VLM.forward`:
    - `inputs_embeds = text_model.get_input_embeddings()(input_ids)`.
    - `image_id = tokenizer.convert_tokens_to_ids("<image>")`.
    - `image_positions = (input_ids == image_id).nonzero(as_tuple=False)  # [K, 2]`.
    - For each `(batch_idx, pos)` in `image_positions`:
      - `span = features[batch_idx]  # [span_len, H]`.
      - `inputs_embeds[batch_idx, pos : pos + span.size(0), :] = span`.
  - This **overwrites** the existing text embeddings in-place; sequence length is unchanged.

- **Collator contract / assumptions**
  - Collator: `src/dataio/collate_multiview.py::MultiViewCollator`.
  - Prompt construction (Stage-1 path):
    - `question` from dataset: `sample.get("question") or sample.get("instruction")`.
    - Prompt string: `f"{question}\n<image>\n"`.
    - Answers are serialized to text and concatenated after the prompt.
  - Tokenization & labels:
    - `prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]`.
    - `answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]`.
    - `ids = (prompt_ids + answer_ids)[:max_length]`.
    - `labels = ([-100] * len(prompt_ids) + answer_ids)[:max_length]`.
    - Padding up to `max_len` with:
      - `pad_id` tokens in `input_ids`.
      - `-100` in `labels`.
  - Reserved length:
    - `min_text_length = num_vis_tokens + geom_tokens + 64`.
    - `max_len = max(max_len, min_text_length)`, then pad sequences to `max_len`.
  - **Important current behavior (potential issue)**:
    - No explicit reserved padding span is inserted immediately after `<image>`.
    - The overwrite span `[pos, pos + span_len)` may extend from prompt into answer tokens if:
      - `pos + span_len > len(prompt_ids)`.
    - Labels are `-100` on the prompt region (including `<image>`), then non-`-100` on answer tokens, then `-100` again on end padding.
    - This means the current contract **relies on the prompt being long enough** so that the overwrite span stays inside the prompt region; there is no hard guarantee.

## Changelog

- 2026-02-27 10:05 UTC — Initialize Stage-1 debug log  
  - Files: `docs/STAGE1_DEBUG_LOG.md`  
  - Reason: Create living documentation for Stage-1 injection debugging and validation.  
  - Validation: `cat docs/STAGE1_DEBUG_LOG.md`

- 2026-02-27 10:20 UTC — Document Stage-1 image token and injection contract  
  - Files: `docs/STAGE1_DEBUG_LOG.md`  
  - Reason: Record where `<image>` is defined, how `image_token_id` is computed, and how the training forward pass currently overwrites embeddings based on collator outputs.  
  - Validation: `sed -n '1,260p' docs/STAGE1_DEBUG_LOG.md`

- 2026-02-27 10:30 UTC — Add CPU-side injection validation script  
  - Files: `scripts/validate_injection.py`, `docs/STAGE1_DEBUG_LOG.md`  
  - Reason: Provide a fast, CPU-only preflight that checks `<image>` presence, sequence shapes, and label masking vs. the expected injection span implied by `num_vis_tokens` and `geom_tokens`, without loading Qwen3/VGGT.  
  - Validation:  
    - Ensure `torch` and other deps are installed (e.g., `conda activate roomplan`).  
    - Run: `python scripts/validate_injection.py --config configs/stage1_3d.yaml --num-samples 4`

- 2026-02-27 10:45 UTC — Harden collator and injection debug path  
  - Files: `src/dataio/collate_multiview.py`, `src/models/vggt_qwen3_vlm.py`, `docs/STAGE1_DEBUG_LOG.md`  
  - Reason:  
    - Proved that the original collator placed `<image>` in the prompt but did **not** reserve a dedicated padding span; the injection window `[pos, pos + span_len)` therefore overwrote some answer tokens (labels != -100) for typical ScanQA/SQA3D prompts.  
    - Updated `MultiViewCollator` to:  
      - Assert the tokenizer contains a valid `<image>` token.  
      - Enforce `max_length` is sufficiently large relative to `num_vis_tokens + geom_tokens`.  
      - Insert an explicit `[PAD]` reserved span of length `num_vis_tokens + geom_tokens` immediately after the `<image>` token for every sample.  
      - Keep labels `-100` over the entire prompt + reserved span, so the model never accumulates loss on injected visual embeddings.  
    - Added `VGGT_DEBUG_INJECT` guardrails in `VGGTQwen3VLM.forward` that, when enabled, raise clear errors if:  
      - No `<image>` token is present.  
      - The injection span would go out of bounds.  
      - The injection span would overlap any labels ≠ -100 (answer tokens).  
  - Validation:  
    - Static: `python -m py_compile src/dataio/collate_multiview.py src/models/vggt_qwen3_vlm.py`  
    - Dynamic (after env setup):  
      - `python scripts/validate_injection.py --config configs/stage1_3d.yaml --num-samples 4`  
      - `VGGT_DEBUG_INJECT=1 python scripts/smoke_forward_cpu.py --config configs/stage1_3d.yaml --device cpu --batch-size 1 --use-mock-vision`

- 2026-02-27 11:00 UTC — Add smoke forward + short-answer eval helpers  
  - Files: `scripts/smoke_forward_cpu.py`, `scripts/smoke_eval_format.py`, `src/inference/qa_inference.py`, `docs/STAGE1_DEBUG_LOG.md`  
  - Reason:  
    - Provide a minimal single-batch forward sanity check that exercises the full Stage-1 training path (collator → injection → loss) with a mock VGGT backbone and CPU-friendly dtype.  
    - Tighten Stage-1 QA inference to better respect the short-answer constraint by:  
      - Appending “Answer with a short phrase only.” to the user prompt in `qa_inference.run_inference`.  
      - Using `max_new_tokens=32`, `temperature=0.0`, and `top_p=1.0` for deterministic, concise generations.  
    - Add `scripts/smoke_eval_format.py` to run a 5-example QA probe and report normalized exact-match (lowercased, punctuation-stripped, whitespace-collapsed) to quickly diagnose format mismatches.  
  - Validation (after env + checkpoint setup):  
    - `python scripts/smoke_forward_cpu.py --config configs/stage1_3d.yaml --device cpu --batch-size 1 --use-mock-vision`  
    - `python scripts/smoke_eval_format.py --config configs/stage1_3d.yaml --glob 'data/processed/scanqa/*.jsonl' --checkpoint_dir ckpts/stage1_3d`
