#!/usr/bin/env bash
# Canonical entry point for Stage-1 training (ScanQA + SQA3D).

set -euo pipefail

CONFIG="${CONFIG:-configs/stage1_3d.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-ckpts/stage1_3d}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-configs/accelerate_single_gpu.yaml}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-}"

echo "Using config:         ${CONFIG}"
echo "Output directory:     ${OUTPUT_DIR}"
echo "Accelerate config:    ${ACCELERATE_CONFIG}"
if [[ -n "${DEEPSPEED_CONFIG}" ]]; then
  echo "DeepSpeed config:     ${DEEPSPEED_CONFIG}"
fi

mkdir -p "${OUTPUT_DIR}"

cmd=(accelerate launch --config_file "${ACCELERATE_CONFIG}" -m vggt_qwen3.train.stage1 --config "${CONFIG}" --output_dir "${OUTPUT_DIR}")
if [[ -n "${DEEPSPEED_CONFIG}" ]]; then
  cmd+=(--deepspeed "${DEEPSPEED_CONFIG}")
fi

echo "Running: ${cmd[*]}"
"${cmd[@]}"

