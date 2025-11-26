#!/bin/bash
#
# VGGT-Qwen3 Training Script
# Run this script to start training the multi-view 3D VQA model
#
# Usage:
#   ./train.sh              # Full training (30K steps, ~15-20 hours)
#   ./train.sh debug        # Debug mode (100 steps, ~10 minutes)
#

set -e  # Exit on error

# Configuration
PROJECT_DIR="/blue/hmedeiros/qinruoyao/roomplan/vggt-qwen3-roomplan"
CONFIG_FILE="configs/stage2_3d.yaml"
ACCELERATE_CONFIG="configs/accelerate_config.yaml"
OUTPUT_DIR="ckpts/stage2_3d"

# Parse arguments
MODE="${1:-full}"

if [ "$MODE" = "debug" ]; then
    echo "🐛 Running in DEBUG mode (100 steps)"
    MAX_STEPS=100
    OUTPUT_DIR="ckpts/stage2_3d_debug"
else
    echo "🚀 Running FULL training (30,000 steps)"
    MAX_STEPS=30000
fi

# Print configuration
echo "================================================================================"
echo "VGGT-Qwen3 Training Configuration"
echo "================================================================================"
echo "Project Directory: $PROJECT_DIR"
echo "Config File:       $CONFIG_FILE"
echo "Output Directory:  $OUTPUT_DIR"
echo "Max Steps:         $MAX_STEPS"
echo "GPUs:              2 (DeepSpeed ZeRO-3)"
echo "Effective Batch:   384 (6 per GPU × 32 grad accum × 2 GPUs)"
echo "================================================================================"
echo ""

# Change to project directory
cd "$PROJECT_DIR"

# Verify environment
echo "📋 Verifying environment..."
if ! command -v accelerate &> /dev/null; then
    echo "❌ Error: accelerate not found. Please activate the conda environment:"
    echo "   conda activate vggt_new"
    exit 1
fi

# Check CUDA
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "❌ Error: CUDA not available. Please check your PyTorch installation."
    exit 1
fi
echo "✅ CUDA available"

# Verify data
if [ ! -f "data/processed/scanqa/train.jsonl" ] || [ ! -f "data/processed/sqa3d/train.jsonl" ]; then
    echo "❌ Error: Training data not found in data/processed/"
    echo "   Expected:"
    echo "     - data/processed/scanqa/train.jsonl"
    echo "     - data/processed/sqa3d/train.jsonl"
    exit 1
fi
echo "✅ Training data found"

# Create output directory
mkdir -p "$OUTPUT_DIR"
echo "✅ Output directory ready: $OUTPUT_DIR"
echo ""

# Start training
echo "🎯 Starting training..."
echo "   Logs will be saved to: $OUTPUT_DIR/logs/"
echo "   Checkpoints saved every 1500 steps to: $OUTPUT_DIR/checkpoint-*/"
echo ""
echo "   Monitor progress:"
echo "     tensorboard --logdir $OUTPUT_DIR/logs"
echo ""
echo "================================================================================"
echo ""

# Run training
accelerate launch \
    --config_file "$ACCELERATE_CONFIG" \
    src/train/train_sft.py \
    --config "$CONFIG_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --max_steps "$MAX_STEPS"

# Training completed
EXIT_CODE=$?

echo ""
echo "================================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully!"
    echo ""
    echo "📊 View training curves:"
    echo "   tensorboard --logdir $OUTPUT_DIR/logs"
    echo ""
    echo "💾 Final checkpoint:"
    echo "   $OUTPUT_DIR/checkpoint-$MAX_STEPS/"
else
    echo "❌ Training failed with exit code $EXIT_CODE"
    echo ""
    echo "🔍 Check logs for errors:"
    echo "   tail -100 $OUTPUT_DIR/logs/*/events.out.tfevents.*"
fi
echo "================================================================================"

exit $EXIT_CODE
