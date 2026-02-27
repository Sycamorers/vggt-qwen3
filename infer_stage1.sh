#!/usr/bin/env bash
# Canonical entry point for Stage-1 QA inference on ScanQA/SQA3D.

set -euo pipefail

CONFIG="${CONFIG:-configs/stage1_3d.yaml}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-ckpts/stage1_3d}"
GLOB="${GLOB:-data/processed/scanqa/test_split.jsonl}"
NUM_SAMPLES="${NUM_SAMPLES:-200}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
OUTPUT_JSONL="${OUTPUT_JSONL:-outputs/qa/scanqa_predictions_test.jsonl}"
DEVICE="${DEVICE:-cuda:0}"

mkdir -p "$(dirname "${OUTPUT_JSONL}")"

echo "Config:          ${CONFIG}"
echo "Checkpoint dir:  ${CHECKPOINT_DIR}"
echo "Data glob:       ${GLOB}"
echo "Num samples:     ${NUM_SAMPLES}"
echo "Max new tokens:  ${MAX_NEW_TOKENS}"
echo "Output JSONL:    ${OUTPUT_JSONL}"
echo "Device:          ${DEVICE}"

python -m vggt_qwen3.inference.qa_inference \
  --config "${CONFIG}" \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --glob "${GLOB}" \
  --num_samples "${NUM_SAMPLES}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --output_jsonl "${OUTPUT_JSONL}" \
  --device "${DEVICE}"

