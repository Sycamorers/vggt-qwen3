#!/usr/bin/env bash
#
# Thin wrapper around the canonical Stage-1 training script.
# Usage:
#   scripts/train_stage1.sh [train_fixed.sh args...]
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/.."

cd "${REPO_ROOT}"
exec ./train_fixed.sh "$@"

