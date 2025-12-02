#!/bin/bash
# Quick evaluation script for ScanQA and SQA3D test sets
# Uses the current checkpoint to get baseline results before retraining

set -e

# Configuration
CONFIG="configs/stage1_3d.yaml"
CHECKPOINT_DIR="ckpts/stage1_3d"
DEVICE="cuda:0"
NUM_SAMPLES=50  # Adjust as needed
MAX_TOKENS=32

# Output directory
OUTPUT_DIR="outputs/qa/baseline_eval"
mkdir -p "$OUTPUT_DIR"

echo "================================================================================"
echo "Baseline Evaluation - Current Model on Test Sets"
echo "================================================================================"
echo "Config: $CONFIG"
echo "Checkpoint: $CHECKPOINT_DIR"
echo "Test samples per dataset: $NUM_SAMPLES"
echo "================================================================================"

# Evaluate SQA3D test set
echo ""
echo "📊 Evaluating SQA3D test set..."
echo "--------------------------------------------------------------------------------"
python -m src.inference.qa_inference \
  --config "$CONFIG" \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --glob "data/processed/sqa3d/test_split.jsonl" \
  --num_samples $NUM_SAMPLES \
  --max_new_tokens $MAX_TOKENS \
  --output_jsonl "$OUTPUT_DIR/sqa3d_baseline.jsonl" \
  --device "$DEVICE" 2>&1 | grep -E "^\[|prediction|reference|scene_id"

# Evaluate ScanQA test set
echo ""
echo "📊 Evaluating ScanQA test set..."
echo "--------------------------------------------------------------------------------"
python -m src.inference.qa_inference \
  --config "$CONFIG" \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --glob "data/processed/scanqa/test_split.jsonl" \
  --num_samples $NUM_SAMPLES \
  --max_new_tokens $MAX_TOKENS \
  --output_jsonl "$OUTPUT_DIR/scanqa_baseline.jsonl" \
  --device "$DEVICE" 2>&1 | grep -E "^\[|prediction|reference|scene_id"

# Evaluate ARKit test set (all samples)
echo ""
echo "📊 Evaluating ARKit test set (all samples)..."
echo "--------------------------------------------------------------------------------"
python -m src.inference.qa_inference \
  --config "$CONFIG" \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --glob "data/processed/arkit_synth/test.json" \
  --num_samples 10000 \
  --max_new_tokens $MAX_TOKENS \
  --output_jsonl "$OUTPUT_DIR/arkit_baseline.jsonl" \
  --device "$DEVICE" 2>&1 | grep -E "^\[|prediction|reference|scene_id"

# Compute quick accuracy metrics
echo ""
echo "================================================================================"
echo "📈 Computing Accuracy Metrics"
echo "================================================================================"

python3 << 'EOF'
import json
from pathlib import Path

def compute_metrics(jsonl_path, dataset_name):
    samples = []
    with open(jsonl_path) as f:
        for line in f:
            samples.append(json.loads(line))
    
    exact_match = 0
    total = len(samples)
    
    for sample in samples:
        pred = sample['prediction'].lower().strip()
        ref = sample['reference'].lower().strip()
        if pred == ref:
            exact_match += 1
    
    accuracy = exact_match / total * 100 if total > 0 else 0
    
    print(f"\n{dataset_name} Results:")
    print(f"  Total samples: {total}")
    print(f"  Exact matches: {exact_match}")
    print(f"  Accuracy: {accuracy:.1f}%")
    
    # Show some examples
    print(f"\n  Example predictions:")
    for i, sample in enumerate(samples[:3]):
        match_icon = "✅" if sample['prediction'].lower().strip() == sample['reference'].lower().strip() else "❌"
        print(f"    {match_icon} Q: {sample['question'][:60]}...")
        print(f"       Pred: {sample['prediction'][:50]}")
        print(f"       Ref:  {sample['reference'][:50]}")

output_dir = Path("outputs/qa/baseline_eval")
compute_metrics(output_dir / "sqa3d_baseline.jsonl", "SQA3D Test Set")
compute_metrics(output_dir / "scanqa_baseline.jsonl", "ScanQA Test Set")
compute_metrics(output_dir / "arkit_baseline.jsonl", "ARKit Test Set")

print("\n" + "="*80)
print("✅ Baseline evaluation complete!")
print("="*80)
print("\nResults saved to:")
print(f"  - {output_dir}/sqa3d_baseline.jsonl")
print(f"  - {output_dir}/scanqa_baseline.jsonl")
print(f"  - {output_dir}/arkit_baseline.jsonl")
print("\nUse these as baselines to compare against retrained model.")
EOF

echo ""
echo "================================================================================"
echo "Done! Results saved to $OUTPUT_DIR/"
echo "================================================================================"
