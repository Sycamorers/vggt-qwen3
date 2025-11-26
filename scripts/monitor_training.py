#!/usr/bin/env python3
"""
Real-time training monitoring script.

Usage:
    python scripts/monitor_training.py --logdir ckpts/stage2_3d/logs
    python scripts/monitor_training.py --logdir ckpts/stage2_3d/logs --watch
"""

import argparse
import os
import sys
import time
from pathlib import Path
from datetime import datetime

try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError:
    print("❌ TensorBoard not found. Install with: pip install tensorboard")
    sys.exit(1)


def format_time(seconds):
    """Convert seconds to human-readable format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def get_latest_event_file(logdir):
    """Find the most recent TensorBoard event file."""
    event_files = list(Path(logdir).rglob("events.out.tfevents.*"))
    if not event_files:
        return None
    return max(event_files, key=lambda p: p.stat().st_mtime)


def load_training_metrics(event_file):
    """Load metrics from TensorBoard event file."""
    ea = event_accumulator.EventAccumulator(str(event_file))
    ea.Reload()
    
    metrics = {}
    for tag in ea.Tags()['scalars']:
        try:
            events = ea.Scalars(tag)
            metrics[tag] = [(e.step, e.value) for e in events]
        except Exception as e:
            print(f"⚠️  Could not load {tag}: {e}")
    
    return metrics


def print_training_status(metrics, start_time=None):
    """Print formatted training status."""
    os.system('clear' if os.name != 'nt' else 'cls')
    
    print("\n" + "="*80)
    print("📊 TRAINING MONITOR".center(80))
    print("="*80)
    print(f"🕐 Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if not metrics:
        print("\n⚠️  No metrics found yet. Training may just be starting...")
        return
    
    # Get latest values
    loss_data = metrics.get('train/loss', metrics.get('loss', []))
    lr_base_data = metrics.get('train/learning_rate_base', metrics.get('train/learning_rate', []))
    lr_proj_data = metrics.get('train/learning_rate_proj', [])
    speed_data = metrics.get('train/steps_per_sec', [])
    progress_data = metrics.get('train/progress_pct', [])
    
    if not loss_data:
        print("\n⚠️  No loss data found yet...")
        return
    
    latest_step, latest_loss = loss_data[-1]
    
    print(f"\n{'─'*80}")
    print(f"📈 Current Progress:")
    print(f"{'─'*80}")
    print(f"   Step: {latest_step:,}")
    
    if progress_data:
        progress_pct = progress_data[-1][1]
        bar_width = 50
        filled = int(bar_width * progress_pct / 100)
        bar = '█' * filled + '░' * (bar_width - filled)
        print(f"   Progress: [{bar}] {progress_pct:.1f}%")
    
    print(f"\n{'─'*80}")
    print(f"📉 Loss:")
    print(f"{'─'*80}")
    print(f"   Current: {latest_loss:.4f}")
    
    if len(loss_data) >= 10:
        recent_losses = [v for _, v in loss_data[-10:]]
        avg_loss = sum(recent_losses) / len(recent_losses)
        min_loss = min(v for _, v in loss_data)
        max_loss = max(v for _, v in loss_data)
        print(f"   Recent avg (last 10): {avg_loss:.4f}")
        print(f"   Min: {min_loss:.4f}  |  Max: {max_loss:.4f}")
    
    # Learning rates
    if lr_base_data:
        print(f"\n{'─'*80}")
        print(f"📚 Learning Rates:")
        print(f"{'─'*80}")
        base_lr = lr_base_data[-1][1]
        print(f"   Base LR: {base_lr:.2e}")
        if lr_proj_data:
            proj_lr = lr_proj_data[-1][1]
            print(f"   Projector LR: {proj_lr:.2e}")
    
    # Speed metrics
    if speed_data:
        print(f"\n{'─'*80}")
        print(f"⏱️  Speed:")
        print(f"{'─'*80}")
        current_speed = speed_data[-1][1]
        print(f"   Current: {current_speed:.2f} steps/sec")
        
        if len(speed_data) >= 10:
            recent_speeds = [v for _, v in speed_data[-10:]]
            avg_speed = sum(recent_speeds) / len(recent_speeds)
            print(f"   Average (last 10): {avg_speed:.2f} steps/sec")
    
    # Loss trend (last 20 steps)
    if len(loss_data) >= 20:
        print(f"\n{'─'*80}")
        print(f"📊 Loss Trend (last 20 steps):")
        print(f"{'─'*80}")
        recent_loss_values = [v for _, v in loss_data[-20:]]
        min_val = min(recent_loss_values)
        max_val = max(recent_loss_values)
        
        # Simple ASCII plot
        height = 10
        width = 40
        
        for i in range(height, 0, -1):
            threshold = min_val + (max_val - min_val) * i / height
            line = "   "
            for val in recent_loss_values:
                line += "█" if val >= threshold else " "
            print(line)
        
        print(f"   {min_val:.2f}" + " "*(width-10) + f"{max_val:.2f}")
        print(f"   └" + "─"*(width-2) + "┘")
    
    print("\n" + "="*80)
    print("💡 Tips:")
    print("   • Loss should generally decrease over time")
    print("   • If loss is NaN or increasing rapidly, training may have issues")
    print("   • Run with --watch to auto-refresh every 30 seconds")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Monitor training progress")
    parser.add_argument("--logdir", required=True, help="Path to TensorBoard log directory")
    parser.add_argument("--watch", action="store_true", help="Continuously monitor (refresh every 30s)")
    parser.add_argument("--interval", type=int, default=30, help="Refresh interval in seconds (default: 30)")
    args = parser.parse_args()
    
    logdir = Path(args.logdir)
    if not logdir.exists():
        print(f"❌ Log directory not found: {logdir}")
        print(f"\n💡 Make sure training has started and check the path.")
        sys.exit(1)
    
    print(f"🔍 Monitoring: {logdir}")
    
    try:
        while True:
            event_file = get_latest_event_file(logdir)
            
            if event_file is None:
                print("\n⚠️  No event files found yet. Waiting for training to start...")
                if not args.watch:
                    break
                time.sleep(args.interval)
                continue
            
            print(f"📂 Reading: {event_file.name}")
            metrics = load_training_metrics(event_file)
            print_training_status(metrics)
            
            if not args.watch:
                break
            
            print(f"♻️  Refreshing in {args.interval} seconds... (Press Ctrl+C to stop)")
            time.sleep(args.interval)
            
    except KeyboardInterrupt:
        print("\n\n👋 Monitoring stopped by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()
