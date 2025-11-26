#!/bin/bash
# Quick debug training run

cd /blue/hmedeiros/qinruoyao/roomplan/vggt-qwen3-roomplan

# Remove old checkpoint
rm -rf ckpts/stage2_3d_debug

# Run with debug output
accelerate launch --config_file configs/accelerate_config.yaml \
    src/train/train_sft.py \
    --config configs/stage2_3d.yaml \
    --output_dir ckpts/stage2_3d_debug \
    --max_steps 10 2>&1 | tee debug_run.log

echo "==== Debug log saved to debug_run.log ===="
echo "Search for '❌' to find where NaN first appears"
