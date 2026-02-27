# Stage-1 Debug / Engineering History

This document summarizes key debugging and engineering changes for **Stage 1** of the VGGT-Qwen3 RoomPlan pipeline. It is intended for maintainers and contributors, not first-time users.

For user-facing instructions, see:
- `README.md`
- `docs/stage1_quickstart.md`
- `docs/model/architecture.md`

## 1. `<image>` placeholder and injection pitfalls

### 1.1 Initial design

- The Stage-1 model uses a dedicated `<image>` token in the text prompt as a placeholder for visual features.
- During the forward pass, positions where `input_ids == image_token_id` are located, and the corresponding text embeddings are overwritten with a continuous span of visual tokens (and optional geometry tokens).
- Training is standard causal LM with teacher forcing:
  - `labels == -100` for prompt and padding.
  - `labels != -100` only for answer tokens.

### 1.2 Original collator behavior

The original `MultiViewCollator`:

- Built prompts as:

  ```text
  "{question}\n<image>\n"
  ```

- Tokenized `prompt` and `answer`, concatenated them, and padded to a minimum length (`min_text_length = num_vis_tokens + geom_tokens + 64`).
- Labeled:
  - `labels = [-100] * len(prompt_ids) + answer_ids`, padded with `-100`s at the end.

This ensured that `<image>` existed in the prompt but **did not explicitly reserve** a padding span after `<image>` for the injection window. As a result:

- The injection window `[pos, pos + span_len)` (where `span_len = num_vis_tokens + geom_tokens`) could extend from the prompt region into the answer region.
- Some answer tokens had:
  - Correct labels (non-`-100`), but
  - Embeddings overwritten by visual features.

This created a mismatch between the intended objective (predict answer tokens from text+vision) and the actual behavior (some answer positions used visual embeddings instead of text embeddings).

### 1.3 Fix: reserved padding span after `<image>`

To fix this, the collator was updated to:

- Enforce the presence and uniqueness of the `<image>` token in each prompt:
  - If the tokenizer does not have a real `<image>` token (or maps it to `unk`), raise a clear error.
  - For each prompt, locate exactly one `<image>` token index `img_pos` in `prompt_ids`; error out if missing or duplicated.
- Insert a reserved `[PAD]` span of length `num_vis_tokens + geom_tokens` immediately **after** `img_pos`.
- Construct labels so that:
  - All prompt tokens, including `<image>` and the reserved `[PAD]` span, receive label `-100`.
  - Answer tokens follow the reserved span and receive their actual token IDs as labels.

Effect:

- The injection window `[img_pos, img_pos + span_len)` now always lies fully inside a `labels == -100` region (prompt + reserved padding).
- The answer region (labels != -100) is strictly disjoint from the injection span.

### 1.4 Model-side guardrails

To catch any future violations of the injection contract, the model wrapper (`VGGTQwen3VLM`) includes a debug mode controlled by the environment variable `VGGT_DEBUG_INJECT`:

- If `VGGT_DEBUG_INJECT=1`:
  - Raise an error if no `<image>` token is found in `input_ids`.
  - Check that the injection span `[pos, pos + span_len)` is within sequence bounds.
  - If `labels` are provided, verify that all labels in the injection window are `-100` (i.e., no overlap with answer tokens); otherwise, raise an error.

These checks run only in debug mode and are safe to disable for production training.

## 2. Collator and tokenizer assumptions

Key assumptions enforced in Stage 1:

- The tokenizer:
  - Must have a dedicated `<image>` token (or one that is added programmatically).
  - Must have a valid `pad_token_id` (defaults to `eos_token` if missing).
  - Uses right-padding for training (`padding_side = "right"`).
- Each sample’s prompt:
  - Contains exactly one `<image>` token.
  - Places `<image>` after the question and before the answer.
  - Has a reserved padding span sized to `num_vis_tokens + geom_tokens` immediately after `<image>`.

These assumptions are documented in `docs/model/architecture.md` and enforced in code via assertions and guardrails.

## 3. Inference and “short answer” formatting

### 3.1 Initial behavior

- Early inference scripts sometimes yielded verbose, sentence-level answers or repeated the question text, which hurt exact-match metrics on ScanQA/SQA3D.

### 3.2 Prompt and generation updates

To make Stage-1 QA inference more consistent with short-answer metrics:

- The user message in `qa_inference` was updated to:

  ```text
  "{question}\n<image>\nAnswer with a short phrase only."
  ```

- Generation settings were set to:
  - `temperature = 0.0`
  - `top_p = 1.0`
  - `num_beams = 1`
  - `max_new_tokens` defaulting to a small value (32)

This combination encourages deterministic, concise predictions (single words or short phrases) without changing the training objective.

## 4. Repository and packaging clean-up

As part of turning this into a maintainable open-source project:

- The Python code under `src/` was migrated into a proper package:
  - `src/vggt_qwen3/` with subpackages:
    - `dataio`, `models`, `train`, `inference`, `eval`.
- All imports and scripts now refer to the `vggt_qwen3` namespace instead of `src.*`.
- A `pyproject.toml` file was added to support `pip install -e .` and make the package importable in other projects.
- Stage-1 training and inference entry points were standardized:
  - `python -m vggt_qwen3.train.stage1`
  - `python -m vggt_qwen3.inference.qa_inference`

## 5. Future work (Stage 2 and beyond)

Stage 2 (RoomPlan action JSON prediction) and additional tasks (e.g., ARKit integration, multi-view geometry) remain under active development and are outside the scope of the current Stage-1–focused docs.

For now:

- Stage-2 configs and scripts exist but are **not** part of the canonical user workflows.
- When Stage 2 is revisited, the same principles should be applied:
  - Clear separation of user-facing vs. maintainer-facing documentation.
  - Strong guardrails around token injection and label masking.
  - Canonical, well-documented entry points for training and inference.

