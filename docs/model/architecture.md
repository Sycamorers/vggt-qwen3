# VGGT-Qwen3 RoomPlan – Stage-1 Architecture

This document describes the Stage-1 model used for multi-view 3D question answering on ScanQA and SQA3D, and how it integrates VGGT with Qwen3-4B via token-level visual injection.

Stage 2 (RoomPlan action JSON prediction) is intentionally **out of scope** here and is treated as future work.

## 1. High-level pipeline

Conceptual flow for a single sample:

```text
Multi-view RGB images  ─▶  VGGT aggregator (frozen)
                            │
                            ▼
                       Perceiver projector
                            │
                            ▼
                Fixed-length visual token sequence
                            │
                            │  (token-level injection at `<image>`)
                            ▼
                 Qwen3-4B (full fine-tune in current Stage-1 code)
                            │
                            ▼
                       Textual answer
```

- **Inputs**
  - A small set of RGB views (e.g., 1 “bird-view” crop per scene for current Stage-1 shards).
  - A text question about the scene.
- **Outputs**
  - A short textual answer (one word or short phrase) suitable for exact-match metrics.

## 2. VGGT aggregator (frozen)

Implementation: `vggt_qwen3.models.vggt_qwen3_vlm.VGGTQwen3VLM._load_vggt`

- We reuse the VGGT aggregator from the external VGGT repo (under `third_party/vggt`).
- Input shape: `[B, V, C, H, W]` where `V` is the number of views.
- The aggregator produces a sequence of tokens with embedding dimension `embed_dim` (internally 1024, often concatenated to 2048 in the repo).
- Stage-1 training **freezes** VGGT:
  - `freeze_vision: true` in `configs/stage1_3d.yaml`.
  - All VGGT parameters have `requires_grad=False`.
- For fast CPU sanity checks, the wrapper supports a `"mock"` VGGT backbone that emits dummy tokens; full training uses the real checkpoint.

Rationale:

- VGGT is a heavy 3D-aware vision backbone; freezing it keeps Stage-1 training tractable and isolates learning to the projector + language side.

## 3. Perceiver projector

Implementation: `vggt_qwen3.models.projector_perceiver.PerceiverProjector` and `PerceiverConfig`.

Purpose:

- Map variable-length VGGT token sequences into a fixed-length set of **visual tokens** in the Qwen3 hidden dimension.

Key points:

- Input: token sequence from VGGT aggregator:
  - Shape: `[B, S, embed_dim]` or `[B, V, T, embed_dim]` depending on VGGT flavor.
- Projector:
  - Uses a Perceiver-style cross-attention stack over a set of latent queries.
  - Parameters controlled by `configs/perceiver_small.yaml` (latent dim, num latents, layers, heads, etc.).
- Output:
  - Shape: `[B, num_vis_tokens, hidden_size]`
  - `num_vis_tokens` is configured in `configs/stage1_3d.yaml`.
  - `hidden_size` matches Qwen3-4B’s embedding dimension.

By default, VGGT is frozen while projector/geom modules and the full Qwen3 text model are trained in Stage 1.

## 4. Qwen3-4B integration (current implementation)

Implementation: `vggt_qwen3.models.vggt_qwen3_vlm.VGGTQwen3VLM`

- Text model: `Qwen/Qwen3-4B-Instruct-2507` by default.
- Tokenizer: loaded from the same identifier; we ensure:
  - `pad_token` is set (falls back to `eos_token` if missing).
  - A dedicated `<image>` token is present (added if missing).
- Current training behavior:
  - Stage-1 config includes a `lora` block and `freeze_text_layers` list.
  - In the current revision, `src/vggt_qwen3/train/stage1.py` does not read/apply those fields.
  - Qwen3 parameters therefore remain trainable and are optimized in the base parameter group (full-parameter fine-tuning).

The key integration point is the **embedding swap**: visual tokens overwrite the text embeddings at the `<image>` placeholder position in the input sequence.

## 5. Token-level injection mechanism

### 5.1 Where the `<image>` placeholder appears

- Data collation: `vggt_qwen3.dataio.collate_multiview.MultiViewCollator`
  - Builds prompt strings of the form:

    ```text
    "{question}\n<image>\n"
    ```

  - Answers are concatenated after this prompt.
  - The tokenizer sees `<image>` as a single special token.

- Tokenizer setup:
  - Training and inference builders:
    - `build_tokenizer` in `vggt_qwen3.train.stage1`
    - `build_tokenizer` in `vggt_qwen3.inference.qa_inference`
  - Both ensure `<image>` exists in the vocabulary, adding it if necessary.

### 5.2 Reserved injection span

To avoid overwriting answer tokens, the collator reserves a dedicated padding span immediately after `<image>`:

- Let:
  - `num_vis_tokens` – number of visual tokens.
  - `geom_tokens` – number of geometry tokens (Stage 1 typically uses `geom_token: null`, so this is effectively 0).
  - `span_len = num_vis_tokens + geom_tokens`.

- For each prompt:
  1. Tokenize the prompt (with `<image>` included).
  2. Locate the single `<image>` token index `img_pos`.
  3. Insert `span_len` `[PAD]` tokens right after `img_pos`.
  4. Construct labels:
     - All prompt tokens (including `<image>` and the reserved `[PAD]` span) receive label `-100`.
     - Answer tokens receive their actual token IDs as labels.
     - Trailing padding is also labeled `-100`.

This guarantees that:

- The injection window `[img_pos, img_pos + span_len)` always lies within a region where `labels == -100`.
- The answer region (labels != -100) starts strictly after the reserved span.

### 5.3 How embeddings are overwritten

In `VGGTQwen3VLM.forward`:

1. Encode images:
   - `vis_tokens = encode_images(images)` → `[B, num_vis_tokens, hidden_size]`.
   - Optionally encode geometry: `geom_tokens = encode_geom(geom_token)` → `[B, geom_tokens, hidden_size]` (unused in Stage 1 if `geom_token` is null).
   - Concatenate into `features`:
     - If `geom_tokens is None`: `features.shape[1] == num_vis_tokens`.
     - Else: `features.shape[1] == num_vis_tokens + geom_tokens`.

2. Build text embeddings:
   - `inputs_embeds = text_model.get_input_embeddings()(input_ids)`.

3. Find injection positions:
   - `image_id = tokenizer.convert_tokens_to_ids("<image>")`.
   - `image_positions = (input_ids == image_id).nonzero(as_tuple=False)` → shape `[K, 2]` of `(batch_idx, pos)`.

4. For each `(batch_idx, pos)`:
   - Let `span = features[batch_idx]` (`[span_len, hidden_size]`).
   - Overwrite:

     ```python
     inputs_embeds[batch_idx, pos : pos + span.size(0), :] = span
     ```

5. Call Qwen3 with `inputs_embeds`, `attention_mask`, and `labels`.

### 5.4 Label masking and loss

- Loss function: standard causal LM cross-entropy with teacher forcing.
- `labels == -100` for:
  - The entire prompt (including `<image>`).
  - The reserved `[PAD]` injection span after `<image>`.
  - Any trailing padding tokens.
- `labels != -100` only for answer tokens.

This ensures:

- No loss is applied on the injected visual span.
- The model is trained only to predict the answer tokens given:
  - The question.
  - The presence of injected visual features at the `<image>` position.

## 6. Training objective and optimizer

### 6.1 Objective

- Causal language modeling loss:
  - Implemented implicitly via Hugging Face’s `AutoModelForCausalLM` with `labels` argument.
  - Masked with `-100` for non-answer tokens.

### 6.2 Parameter groups

In `vggt_qwen3.train.stage1`:

- Parameters are split into:

  - **Base parameters** (Qwen3 + any unfrozen layers):
    - Learning rate: `train.lr`.
  - **Projector and geometry head parameters**:
    - Modules whose names include `"projector"` or `"geom_head"`.
    - Learning rate: `train.proj_lr` (can be higher than the base LR).

- Optimizer: AdamW.
- Scheduler: cosine with warmup (`get_cosine_schedule_with_warmup`).
- Mixed precision and distributed training handled by `accelerate.Accelerator`, with optional DeepSpeed ZeRO-3.

## 7. Stage-1 limitations and assumptions

Current Stage-1 setup has the following limitations:

- **View configuration**
  - Many current shards use a single “bird-view” crop per scene (`num_views=1`).
  - The architecture supports multi-view (`num_views > 1`), but data preprocessing must be updated accordingly.

- **Geometry tokens**
  - `geom_token` field is `null` in the current ScanQA/SQA3D shards; geometry encoding is effectively bypassed.
  - The model still has a `geom_head` and `geom_tokens` configuration; enabling geometry requires non-null `geom_token` structures in the dataset.

- **Short answers**
  - Stage 1 is optimized for short answers (single word or phrase).
  - Inference defaults (prompt suffix and generation settings) are tuned for exact-match metrics, not for long-form explanation.

- **Frozen VGGT and full Qwen3 fine-tune (current implementation)**
  - Stage-1 training does not fine-tune VGGT by default (`model.freeze_vision: true`).
  - Qwen3 is currently fine-tuned end-to-end (no LoRA/PEFT adapters applied).

## 8. Configuration reference (Stage 1)

Key config fields in `configs/stage1_3d.yaml`:

- `model.name_or_path`
  - HF identifier for the base Qwen3 model.
- `model.tokenizer_path`
  - Optional separate tokenizer path; usually same as `name_or_path`.
- `model.vision_backbone`
  - Path to VGGT repo / checkpoint (e.g., `third_party/vggt`).
- `model.num_vis_tokens`
  - Number of visual tokens produced by the Perceiver projector and injected at `<image>`.
- `model.geom_tokens`
  - Number of geometry tokens reserved in the injection span (0 for current Stage 1).
- `model.projector`
  - Path to Perceiver config YAML (latent dim, layers, etc.).
- `model.freeze_vision`
  - If `true`, VGGT weights are frozen.
- `model.freeze_text_layers`
  - Present in `configs/stage1_3d.yaml`, but currently unused by `src/vggt_qwen3/train/stage1.py`.
- `data.datasets`
  - JSONL globs for ScanQA and SQA3D.
- `data.num_views`
  - Number of image views per sample.
- `data.image_size`
  - Image resolution used in preprocessing and the VGGT pipeline.
- `data.max_length`
  - Text sequence length (must accommodate prompt + reserved injection span + answer tokens).
- `train.batch_size_per_gpu`
  - Per-device batch size.
- `train.grad_accum`
  - Gradient accumulation steps (effective batch size = `batch_size_per_gpu * num_gpus * grad_accum`).
- `train.max_steps`
  - Total number of optimization steps.
- `train.lr`, `train.proj_lr`
  - Learning rates for base parameters and projector/geom head.
- `lora.*`
  - Present in `configs/stage1_3d.yaml`, but currently unused by `src/vggt_qwen3/train/stage1.py`.

For detailed debugging history (e.g., injection span bugs and fixes), see:

- `docs/dev/debug_history.md`
