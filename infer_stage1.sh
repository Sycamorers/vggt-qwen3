#!/usr/bin/env bash
# Canonical entry point for Stage-1 QA inference on ScanQA/SQA3D.

set -euo pipefail

CONFIG="${CONFIG:-configs/stage1_3d.yaml}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-ckpts/stage1_3d}"
DATASET="${DATASET:-scanqa}"   # scanqa | sqa3d | scanqa+sqa3d
NUM_SAMPLES="${NUM_SAMPLES:-200}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
DEVICE="${DEVICE:-cuda:0}"
ALLOW_BASE_FALLBACK="${ALLOW_BASE_FALLBACK:-0}"
SHORT_ANSWER_ONLY="${SHORT_ANSWER_ONLY:-1}"

# Default output location if not provided by the caller.
if [[ -z "${OUTPUT_JSONL:-}" ]]; then
  case "${DATASET}" in
    scanqa)
      OUTPUT_JSONL="outputs/qa/scanqa_predictions_test.jsonl"
      ;;
    sqa3d)
      OUTPUT_JSONL="outputs/qa/sqa3d_predictions_test.jsonl"
      ;;
    *)
      OUTPUT_JSONL="outputs/qa/qa_predictions_test.jsonl"
      ;;
  esac
fi

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_ROOT="${SCRIPT_DIR}/logs"
LOG_DIR="${LOG_ROOT}/inference"
mkdir -p "${LOG_DIR}"

TIMESTAMP="$(date +'%Y%m%d_%H%M%S')"
OUTPUT_BASENAME="$(basename "${OUTPUT_JSONL}")"
LOG_FILE="${LOG_DIR}/${OUTPUT_BASENAME%.jsonl}_${TIMESTAMP}.log"

exec > >(tee -a "${LOG_FILE}") 2>&1
echo "📓 Logging to: ${LOG_FILE}"

echo "Config:          ${CONFIG}"
echo "Checkpoint dir:  ${CHECKPOINT_DIR}"
echo "Dataset:         ${DATASET}"
echo "Num samples:     ${NUM_SAMPLES}"
echo "Max new tokens:  ${MAX_NEW_TOKENS}"
echo "Device:          ${DEVICE}"
echo "Output JSONL:    ${OUTPUT_JSONL}"

mkdir -p "$(dirname "${OUTPUT_JSONL}")"

BASE_FALLBACK_FLAG=()
if [[ "${ALLOW_BASE_FALLBACK}" == "1" ]]; then
  BASE_FALLBACK_FLAG=(--allow_base_fallback)
fi

SHORT_FLAG=()
if [[ "${SHORT_ANSWER_ONLY}" == "1" ]]; then
  SHORT_FLAG=(--short_answer_only)
fi

python -m vggt_qwen3.inference.qa_inference \
  --config "${CONFIG}" \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --dataset "${DATASET}" \
  --num_samples "${NUM_SAMPLES}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --output_jsonl "${OUTPUT_JSONL}" \
  --device "${DEVICE}" \
  "${BASE_FALLBACK_FLAG[@]}" \
  "${SHORT_FLAG[@]}"
