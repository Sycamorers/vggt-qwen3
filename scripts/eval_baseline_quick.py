#!/usr/bin/env python3
"""Quick baseline evaluation on ScanQA and SQA3D test sets.

This script runs inference on the test splits using the current checkpoint
to establish baseline metrics before retraining.

Usage:
    python scripts/eval_baseline_quick.py
    python scripts/eval_baseline_quick.py --num_samples 100
"""

import argparse
import json
import subprocess
from pathlib import Path


def run_inference(config, checkpoint_dir, glob_pattern, output_path, num_samples, max_tokens, device):
    """Run inference using the qa_inference module."""
    cmd = [
        "python", "-m", "src.inference.qa_inference",
        "--config", config,
        "--checkpoint_dir", checkpoint_dir,
        "--glob", glob_pattern,
        "--num_samples", str(num_samples),
        "--max_new_tokens", str(max_tokens),
        "--output_jsonl", str(output_path),
        "--device", device
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    return result.returncode == 0


def compute_metrics(jsonl_path, dataset_name):
    """Compute accuracy metrics from predictions."""
    samples = []
    with open(jsonl_path) as f:
        for line in f:
            samples.append(json.loads(line))
    
    exact_match = 0
    partial_match = 0
    total = len(samples)
    
    for sample in samples:
        pred = sample['prediction']
        ref = sample['reference']
        
        # Handle different reference types (string vs dict/JSON)
        if isinstance(ref, dict):
            # For ARKit-style JSON references, compare as JSON strings
            ref_str = json.dumps(ref, sort_keys=True).lower()
            pred_lower = pred.lower().strip()
            
            # Check if prediction contains key elements
            if 'action' in ref and ref['action'] in pred_lower:
                partial_match += 1
            # Try to parse prediction as JSON for exact match
            try:
                pred_json = json.loads(pred)
                if pred_json == ref:
                    exact_match += 1
            except:
                pass
        else:
            # For string references (ScanQA, SQA3D)
            pred = pred.lower().strip()
            ref = str(ref).lower().strip()
            
            if pred == ref:
                exact_match += 1
            elif ref in pred or pred in ref:
                partial_match += 1
    
    accuracy = exact_match / total * 100 if total > 0 else 0
    partial_acc = (exact_match + partial_match) / total * 100 if total > 0 else 0
    
    print(f"\n{'='*80}")
    print(f"{dataset_name} Results")
    print(f"{'='*80}")
    print(f"Total samples:     {total}")
    print(f"Exact matches:     {exact_match} ({accuracy:.1f}%)")
    print(f"Partial matches:   {partial_match}")
    print(f"Partial accuracy:  {partial_acc:.1f}%")
    
    # Show some examples
    print(f"\nExample predictions:")
    for i, sample in enumerate(samples[:5]):
        pred = sample['prediction']
        ref = sample['reference']
        
        # Determine match status
        if isinstance(ref, dict):
            ref_str = json.dumps(ref, sort_keys=True)
            try:
                pred_json = json.loads(pred)
                match = pred_json == ref
            except:
                match = False
            
            if match:
                match_icon = "✅"
            elif isinstance(ref, dict) and 'action' in ref and ref['action'] in pred.lower():
                match_icon = "⚠️"
            else:
                match_icon = "❌"
            
            ref_display = ref_str[:60] + "..." if len(ref_str) > 60 else ref_str
        else:
            ref_str = str(ref)
            pred_lower = pred.lower().strip()
            ref_lower = ref_str.lower().strip()
            
            if pred_lower == ref_lower:
                match_icon = "✅"
            elif ref_lower in pred_lower or pred_lower in ref_lower:
                match_icon = "⚠️"
            else:
                match_icon = "❌"
            
            ref_display = ref_str[:60]
        
        print(f"\n  {match_icon} [{i+1}] {sample['question'][:70]}...")
        print(f"      Pred: '{pred[:60]}'")
        print(f"      Ref:  '{ref_display}'")
    
    return {
        'total': total,
        'exact_match': exact_match,
        'partial_match': partial_match,
        'accuracy': accuracy,
        'partial_accuracy': partial_acc
    }


def main():
    parser = argparse.ArgumentParser(description="Quick baseline evaluation")
    parser.add_argument("--config", default="configs/stage1_3d.yaml")
    parser.add_argument("--checkpoint_dir", default="ckpts/stage1_3d")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples per dataset")
    parser.add_argument("--max_tokens", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", default="outputs/qa/baseline_eval")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("Baseline Evaluation - Current Model on Test Sets")
    print("="*80)
    print(f"Config:       {args.config}")
    print(f"Checkpoint:   {args.checkpoint_dir}")
    print(f"Samples:      {args.num_samples} per dataset")
    print(f"Device:       {args.device}")
    print("="*80)
    
    results = {}
    
    # Evaluate SQA3D
    print("\n📊 Evaluating SQA3D test set...")
    print("-"*80)
    sqa3d_output = output_dir / "sqa3d_baseline.jsonl"
    if run_inference(
        args.config,
        args.checkpoint_dir,
        "data/processed/sqa3d/test_split.jsonl",
        sqa3d_output,
        args.num_samples,
        args.max_tokens,
        args.device
    ):
        results['sqa3d'] = compute_metrics(sqa3d_output, "SQA3D Test Set")
    
    # Evaluate ScanQA
    print("\n📊 Evaluating ScanQA test set...")
    print("-"*80)
    scanqa_output = output_dir / "scanqa_baseline.jsonl"
    if run_inference(
        args.config,
        args.checkpoint_dir,
        "data/processed/scanqa/test_split.jsonl",
        scanqa_output,
        args.num_samples,
        args.max_tokens,
        args.device
    ):
        results['scanqa'] = compute_metrics(scanqa_output, "ScanQA Test Set")
    
    # Evaluate ARKit (use all samples - no limit)
    print("\n📊 Evaluating ARKit test set...")
    print("-"*80)
    arkit_output = output_dir / "arkit_baseline.jsonl"
    # For ARKit, run on all samples (set to large number)
    if run_inference(
        args.config,
        args.checkpoint_dir,
        "data/processed/arkit_synth/test.json",
        arkit_output,
        10000,  # Large number to get all samples
        args.max_tokens,
        args.device
    ):
        results['arkit'] = compute_metrics(arkit_output, "ARKit Test Set")
    
    # Save summary
    summary_path = output_dir / "baseline_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    if results:
        for dataset, metrics in results.items():
            print(f"{dataset.upper()}: {metrics['accuracy']:.1f}% exact match ({metrics['exact_match']}/{metrics['total']})")
    
    print("\n✅ Baseline evaluation complete!")
    print("="*80)
    print(f"\nResults saved to:")
    print(f"  - {sqa3d_output}")
    print(f"  - {scanqa_output}")
    print(f"  - {arkit_output}")
    print(f"  - {summary_path}")
    print(f"\nUse these as baselines to compare against retrained model.")


if __name__ == "__main__":
    main()
