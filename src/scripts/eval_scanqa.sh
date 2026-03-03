#!/usr/bin/env bash
#
# Convenience wrapper to evaluate Stage-1 ScanQA predictions.
#
# Example:
#   src/scripts/eval_scanqa.sh \
#     --predictions outputs/qa/scanqa/scanqa_predictions_test.jsonl
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/.."

cd "${REPO_ROOT}"

python -m src.scripts.eval_scanqa "$@"
