#!/bin/bash
# Quick training status checker
# Usage: ./check_training.sh [checkpoint_dir]

CHECKPOINT_DIR="${1:-ckpts/stage2_3d}"
LOG_DIR="${CHECKPOINT_DIR}/logs/roomplan"

echo "================================================================================"
echo "                    🔍 TRAINING STATUS CHECKER"
echo "================================================================================"
echo ""

# Check if checkpoint directory exists
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "❌ Checkpoint directory not found: $CHECKPOINT_DIR"
    echo ""
    echo "💡 Training may not have started yet, or the path is incorrect."
    exit 1
fi

# Count checkpoints
CHECKPOINT_COUNT=$(find "$CHECKPOINT_DIR" -maxdepth 1 -type d -name "step_*" 2>/dev/null | wc -l)
echo "📦 Checkpoints Saved: $CHECKPOINT_COUNT"

if [ $CHECKPOINT_COUNT -gt 0 ]; then
    echo ""
    echo "📁 Checkpoint List:"
    find "$CHECKPOINT_DIR" -maxdepth 1 -type d -name "step_*" -exec basename {} \; | sort -V | while read ckpt; do
        STEP=$(echo "$ckpt" | sed 's/step_//')
        SIZE=$(du -sh "$CHECKPOINT_DIR/$ckpt" 2>/dev/null | cut -f1)
        echo "   • $ckpt - $SIZE"
    done
fi

echo ""
echo "────────────────────────────────────────────────────────────────────────────────"

# Check for log files
if [ -d "$LOG_DIR" ]; then
    EVENT_FILES=$(find "$LOG_DIR" -name "events.out.tfevents.*" 2>/dev/null | wc -l)
    echo "📊 TensorBoard Event Files: $EVENT_FILES"
    
    if [ $EVENT_FILES -gt 0 ]; then
        LATEST_EVENT=$(find "$LOG_DIR" -name "events.out.tfevents.*" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
        LAST_MODIFIED=$(stat -c %y "$LATEST_EVENT" | cut -d'.' -f1)
        echo "   Last updated: $LAST_MODIFIED"
    fi
else
    echo "⚠️  No log directory found: $LOG_DIR"
fi

echo ""
echo "────────────────────────────────────────────────────────────────────────────────"

# Check if training is running
TRAINING_PROC=$(ps aux | grep "[p]ython.*train_sft.py" | head -1)
if [ -n "$TRAINING_PROC" ]; then
    echo "✅ Training Process: RUNNING"
    PID=$(echo "$TRAINING_PROC" | awk '{print $2}')
    CPU=$(echo "$TRAINING_PROC" | awk '{print $3}')
    MEM=$(echo "$TRAINING_PROC" | awk '{print $4}')
    RUNTIME=$(ps -p $PID -o etime= | xargs)
    echo "   PID: $PID"
    echo "   Runtime: $RUNTIME"
    echo "   CPU: ${CPU}% | Memory: ${MEM}%"
else
    echo "⏸️  Training Process: NOT RUNNING"
fi

echo ""
echo "────────────────────────────────────────────────────────────────────────────────"

# Check GPU usage
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 GPU Status:"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | while IFS=',' read -r idx name util mem_used mem_total temp; do
        echo "   GPU $idx: $name"
        echo "      Utilization: ${util}% | Memory: ${mem_used}MB / ${mem_total}MB | Temp: ${temp}°C"
    done
else
    echo "⚠️  nvidia-smi not available (GPU status unknown)"
fi

echo ""
echo "────────────────────────────────────────────────────────────────────────────────"

# Disk space
echo "💾 Disk Space:"
DISK_USAGE=$(du -sh "$CHECKPOINT_DIR" 2>/dev/null | cut -f1)
echo "   Checkpoint directory size: $DISK_USAGE"
FREE_SPACE=$(df -h "$CHECKPOINT_DIR" | tail -1 | awk '{print $4}')
echo "   Free space on disk: $FREE_SPACE"

echo ""
echo "================================================================================"
echo "                           📖 MONITORING COMMANDS"
echo "================================================================================"
echo ""
echo "🔍 View detailed metrics:"
echo "   python scripts/monitor_training.py --logdir $LOG_DIR"
echo ""
echo "♻️  Auto-refresh monitoring (every 30s):"
echo "   python scripts/monitor_training.py --logdir $LOG_DIR --watch"
echo ""
echo "📈 Launch TensorBoard:"
echo "   tensorboard --logdir $LOG_DIR --port 6006"
echo ""
echo "📋 View recent logs:"
echo "   tail -f \$(find $CHECKPOINT_DIR -name '*.log' -type f | tail -1)"
echo ""
echo "================================================================================"
