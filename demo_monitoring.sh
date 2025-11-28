#!/bin/bash
# 📊 Training Monitoring Demo
# This script shows all the monitoring options available

echo "================================================================================"
echo "                  🎓 TRAINING MONITORING TUTORIAL"
echo "================================================================================"
echo ""
echo "This tutorial shows you the monitoring options while training is running."
echo ""

# Check if training is running
TRAINING_RUNNING=$(ps aux | grep -E "[p]ython.*train_sft.py" | wc -l)

if [ $TRAINING_RUNNING -eq 0 ]; then
    echo "⚠️  Training is not currently running."
    echo ""
    echo "To start training, run:"
    echo "   ./train_fixed.sh full 4      # Full training"
    echo "   ./train_fixed.sh --safe debug 1  # Quick smoke test"
    echo ""
    echo "After starting training, run this script again to see monitoring options."
    echo ""
    exit 0
fi

echo "✅ Training is running! Here are your monitoring options:"
echo ""

echo "================================================================================"
echo "Option 1: Detailed Metrics (One-time)"
echo "================================================================================"
echo "Command:  python scripts/monitor_training.py --logdir ckpts/stage2_3d/logs/roomplan"
echo "Shows:    Loss trends, learning rates, speed, progress bar, ASCII plot"
echo "Updates:  Manual (run whenever you want)"
echo ""
read -p "Press ENTER to view detailed metrics (or Ctrl+C to skip)..."
python scripts/monitor_training.py --logdir ckpts/stage2_3d/logs/roomplan
echo ""
read -p "Press ENTER to continue to next option..."
echo ""

echo "================================================================================"
echo "Option 2: Auto-Refresh Monitoring (Recommended)"
echo "================================================================================"
echo "Command:  python scripts/monitor_training.py --logdir ckpts/stage2_3d/logs/roomplan --watch"
echo "Shows:    Same as Option 2, but auto-refreshes every 30 seconds"
echo "Updates:  Automatic (press Ctrl+C to stop)"
echo ""
echo "💡 TIP: Run this in a separate terminal window to monitor continuously"
echo ""
read -p "Press ENTER to start auto-refresh mode (Ctrl+C to stop, then ENTER to continue tutorial)..."
python scripts/monitor_training.py --logdir ckpts/stage2_3d/logs/roomplan --watch --interval 10
echo ""

echo "================================================================================"
echo "Option 3: TensorBoard (Full Visualization)"
echo "================================================================================"
echo "Command:  tensorboard --logdir ckpts/stage2_3d/logs/roomplan --port 6006"
echo "Shows:    Interactive web UI with plots, scalars, distributions"
echo "Updates:  Real-time"
echo "Access:   http://localhost:6006"
echo ""
echo "💡 TIP: Run TensorBoard in the background:"
echo "   tensorboard --logdir ckpts/stage2_3d/logs/roomplan --port 6006 &"
echo ""
echo "   Then open http://localhost:6006 in your browser"
echo ""
read -p "Do you want to start TensorBoard? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Starting TensorBoard..."
    tensorboard --logdir ckpts/stage2_3d/logs/roomplan --port 6006 &
    TB_PID=$!
    echo "✅ TensorBoard started (PID: $TB_PID)"
    echo "   Open http://localhost:6006 in your browser"
    echo ""
    echo "   To stop TensorBoard later: kill $TB_PID"
fi
echo ""

echo "================================================================================"
echo "                           📚 SUMMARY"
echo "================================================================================"
echo ""
echo "You now have 4 ways to monitor training:"
echo ""
echo "1. Detailed Metrics:  python scripts/monitor_training.py --logdir ckpts/stage2_3d/logs/roomplan"
echo "2. Auto-Refresh:      python scripts/monitor_training.py --logdir ckpts/stage2_3d/logs/roomplan --watch"
echo "3. TensorBoard:       tensorboard --logdir ckpts/stage2_3d/logs/roomplan --port 6006"
echo "4. Log tail (manual): tail -f pytorchdist_*.out | grep \"Step\\|Loss\\|ETA\""
echo ""
echo "📖 For complete guide, see: docs/MONITORING_GUIDE.md"
echo ""
echo "================================================================================"
echo "                        ✅ TUTORIAL COMPLETE"
echo "================================================================================"
