#!/usr/bin/env bash
#
# Convenience wrapper for Stage-1 ScanQA inference.
#
# Example:
#   CHECKPOINT_DIR=ckpts/stage1_3d_debug \
#   OUTPUT_JSONL=outputs/qa/scanqa_predictions_test.jsonl \
#   scripts/infer_stage1_scanqa.sh
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/.."

cd "${REPO_ROOT}"

: "${CHECKPOINT_DIR:=ckpts/stage1_3d_debug}"
: "${OUTPUT_JSONL:=outputs/qa/scanqa_predictions_test.jsonl}"

CHECKPOINT_DIR="${CHECKPOINT_DIR}" \
OUTPUT_JSONL="${OUTPUT_JSONL}" \
DATASET="scanqa" \
./infer_stage1.sh "$@"

