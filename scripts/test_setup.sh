#!/bin/bash
# Quick test to verify training setup before running full training

set -e  # Exit on error

echo "=========================================="
echo "VGGT-Qwen3 Training Setup Test"
echo "=========================================="
echo ""

# Check GPU availability
echo "1. Checking GPUs..."
GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo "0")
echo "   ✓ Found $GPU_COUNT GPU(s)"
if [ "$GPU_COUNT" -eq "0" ]; then
    echo "   ⚠️  WARNING: No GPUs detected! Make sure you're in a SLURM session."
fi
echo ""

# Check SLURM environment
echo "2. Checking SLURM environment..."
echo "   Job ID: ${SLURM_JOB_ID:-Not in SLURM}"
echo "   GPUs allocated: ${SLURM_GPUS:-N/A}"
echo ""

# Check conda environment
echo "3. Checking conda environment..."
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "   ✓ Active environment: $CONDA_DEFAULT_ENV"
else
    echo "   ⚠️  No conda environment active"
fi
echo ""

# Check data
echo "4. Checking datasets..."
SCANQA_TRAIN=$(wc -l < data/processed/scanqa/train.jsonl 2>/dev/null || echo "0")
SCANQA_VAL=$(wc -l < data/processed/scanqa/val.jsonl 2>/dev/null || echo "0")
SQA3D_TRAIN=$(wc -l < data/processed/sqa3d/train.jsonl 2>/dev/null || echo "0")
SQA3D_VAL=$(wc -l < data/processed/sqa3d/val.jsonl 2>/dev/null || echo "0")

echo "   scanqa/train.jsonl: $SCANQA_TRAIN samples"
echo "   scanqa/val.jsonl: $SCANQA_VAL samples"
echo "   sqa3d/train.jsonl: $SQA3D_TRAIN samples"
echo "   sqa3d/val.jsonl: $SQA3D_VAL samples"

TOTAL=$((SCANQA_TRAIN + SCANQA_VAL + SQA3D_TRAIN + SQA3D_VAL))
echo "   ✓ Total: $TOTAL samples"
echo ""

# Check config files
echo "5. Checking configuration files..."
for config in configs/accelerate_config.yaml configs/stage2_3d.yaml configs/deepspeed_zero3.json; do
    if [ -f "$config" ]; then
        echo "   ✓ $config"
    else
        echo "   ✗ $config (MISSING)"
    fi
done
echo ""

# Check Python packages
echo "6. Checking Python packages..."
python3 -c "import torch; print(f'   ✓ PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})')" 2>/dev/null || echo "   ✗ PyTorch not found"
python3 -c "import transformers; print(f'   ✓ Transformers {transformers.__version__}')" 2>/dev/null || echo "   ✗ Transformers not found"
python3 -c "import accelerate; print(f'   ✓ Accelerate {accelerate.__version__}')" 2>/dev/null || echo "   ✗ Accelerate not found"
python3 -c "import deepspeed; print(f'   ✓ DeepSpeed {deepspeed.__version__}')" 2>/dev/null || echo "   ✗ DeepSpeed not found"
echo ""

# Run quick validation
echo "7. Running data validation..."
python3 scripts/validate_data.py > /tmp/validation.log 2>&1
if [ $? -eq 0 ]; then
    echo "   ✓ Data validation passed"
else
    echo "   ✗ Data validation failed (check /tmp/validation.log)"
fi
echo ""

# Test training for 1 step
echo "8. Testing training (1 step, dry run)..."
echo "   This will take a few minutes to load the model..."
timeout 300 accelerate launch --config_file configs/accelerate_single_gpu.yaml \
    src/train/train_sft.py \
    --config configs/stage2_3d.yaml \
    --output_dir /tmp/test_training \
    --max_steps 1 > /tmp/test_training.log 2>&1

if [ $? -eq 0 ]; then
    echo "   ✓ Test training completed successfully!"
else
    echo "   ⚠️  Test training encountered an issue (check /tmp/test_training.log)"
    echo "   This might be normal if model download is required"
fi
echo ""

echo "=========================================="
echo "Setup Test Complete!"
echo "=========================================="
echo ""
echo "If all checks passed, you can start training with:"
echo ""
echo "  accelerate launch --config_file configs/accelerate_config.yaml \\"
echo "      src/train/train_sft.py \\"
echo "      --config configs/stage2_3d.yaml \\"
echo "      --output_dir ckpts/stage2_3d \\"
echo "      --max_steps 30000"
echo ""
