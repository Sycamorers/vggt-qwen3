#!/usr/bin/env python3
"""Split ScanQA and SQA3D datasets into train and test sets.

This script:
1. Loads the processed JSONL files
2. Splits by scene_id to ensure no scene leakage between train/test
3. Creates train.jsonl and test.jsonl for each dataset
4. Maintains the original JSONL format

Usage:
    python scripts/prep/split_train_test.py --dataset sqa3d --test_ratio 0.1
    python scripts/prep/split_train_test.py --dataset scanqa --test_ratio 0.1
    python scripts/prep/split_train_test.py --all --test_ratio 0.1
"""

import argparse
import json
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple


def load_jsonl(path: Path) -> List[Dict]:
    """Load samples from a JSONL file."""
    samples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def save_jsonl(samples: List[Dict], path: Path) -> None:
    """Save samples to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f"  Saved {len(samples)} samples to {path}")


def split_by_scenes(
    samples: List[Dict],
    test_ratio: float,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """Split samples by scene_id to avoid data leakage.
    
    Args:
        samples: List of samples, each must have a 'scene_id' field
        test_ratio: Fraction of scenes to use for testing (0.0 to 1.0)
        seed: Random seed for reproducibility
    
    Returns:
        (train_samples, test_samples)
    """
    rng = random.Random(seed)
    
    # Group samples by scene_id
    scenes_to_samples = defaultdict(list)
    null_scene_samples = []
    
    for sample in samples:
        scene_id = sample.get('scene_id')
        if scene_id is None:
            null_scene_samples.append(sample)
        else:
            scenes_to_samples[scene_id].append(sample)
    
    # Get all unique scenes
    all_scenes = sorted(scenes_to_samples.keys())
    
    # Shuffle and split scenes
    rng.shuffle(all_scenes)
    n_test_scenes = max(1, int(len(all_scenes) * test_ratio))
    test_scenes = set(all_scenes[:n_test_scenes])
    train_scenes = set(all_scenes[n_test_scenes:])
    
    # Collect samples
    train_samples = []
    test_samples = []
    
    for scene_id in train_scenes:
        train_samples.extend(scenes_to_samples[scene_id])
    
    for scene_id in test_scenes:
        test_samples.extend(scenes_to_samples[scene_id])
    
    # Add null scene samples to training (if any)
    if null_scene_samples:
        print(f"  ⚠️  Warning: {len(null_scene_samples)} samples with null scene_id added to training")
        train_samples.extend(null_scene_samples)
    
    return train_samples, test_samples


def split_dataset(dataset_name: str, test_ratio: float, seed: int = 42) -> None:
    """Split a dataset into train and test sets."""
    print(f"\n{'='*80}")
    print(f"Splitting {dataset_name.upper()} dataset")
    print(f"{'='*80}")
    
    # Paths
    input_path = Path(f"data/processed/{dataset_name}/train.jsonl")
    train_path = Path(f"data/processed/{dataset_name}/train_split.jsonl")
    test_path = Path(f"data/processed/{dataset_name}/test_split.jsonl")
    
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return
    
    # Load samples
    print(f"Loading samples from {input_path}")
    samples = load_jsonl(input_path)
    print(f"  Total samples: {len(samples)}")
    
    # Count unique scenes
    unique_scenes = set(s.get('scene_id') for s in samples if s.get('scene_id') is not None)
    null_scenes = sum(1 for s in samples if s.get('scene_id') is None)
    print(f"  Unique scenes: {len(unique_scenes)}")
    if null_scenes > 0:
        print(f"  Samples with null scene_id: {null_scenes}")
    
    # Split by scenes
    train_samples, test_samples = split_by_scenes(samples, test_ratio, seed)
    
    # Get scene counts
    train_scenes = set(s.get('scene_id') for s in train_samples if s.get('scene_id') is not None)
    test_scenes = set(s.get('scene_id') for s in test_samples if s.get('scene_id') is not None)
    
    print(f"\nSplit summary:")
    print(f"  Train: {len(train_samples)} samples from {len(train_scenes)} scenes")
    print(f"  Test:  {len(test_samples)} samples from {len(test_scenes)} scenes")
    print(f"  Test ratio: {len(test_samples)/len(samples):.1%} of samples, {len(test_scenes)/len(unique_scenes):.1%} of scenes")
    
    # Verify no scene leakage
    overlap = train_scenes & test_scenes
    if overlap:
        print(f"  ❌ ERROR: {len(overlap)} scenes appear in both train and test!")
        print(f"     Overlapping scenes: {sorted(overlap)[:5]}...")
    else:
        print(f"  ✅ No scene leakage - train and test scenes are disjoint")
    
    # Save splits
    print(f"\nSaving splits:")
    save_jsonl(train_samples, train_path)
    save_jsonl(test_samples, test_path)
    
    # Create backup of original
    backup_path = input_path.with_suffix('.jsonl.original')
    if not backup_path.exists():
        import shutil
        shutil.copy2(input_path, backup_path)
        print(f"  Backup of original: {backup_path}")
    
    print(f"\n✅ {dataset_name.upper()} split complete!")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split datasets into train/test sets")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["scanqa", "sqa3d"],
        help="Dataset to split (scanqa or sqa3d)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Split all datasets (scanqa and sqa3d)"
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Fraction of scenes to use for testing (default: 0.1 = 10%%)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    if not args.dataset and not args.all:
        print("❌ Error: Must specify --dataset or --all")
        return
    
    print("="*80)
    print("Dataset Train/Test Splitter")
    print("="*80)
    print(f"Test ratio: {args.test_ratio:.1%} of scenes")
    print(f"Random seed: {args.seed}")
    
    datasets = []
    if args.all:
        datasets = ["scanqa", "sqa3d"]
    else:
        datasets = [args.dataset]
    
    for dataset_name in datasets:
        split_dataset(dataset_name, args.test_ratio, args.seed)
    
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print("\n✅ Dataset splitting complete!")
    print("\nNext steps:")
    print("  1. Review the split statistics above")
    print("  2. Use train_split.jsonl for training")
    print("  3. Use test_split.jsonl for evaluation")
    print("  4. Original files backed up as *.jsonl.original")
    print("\nFiles created:")
    for dataset_name in datasets:
        print(f"  - data/processed/{dataset_name}/train_split.jsonl")
        print(f"  - data/processed/{dataset_name}/test_split.jsonl")


if __name__ == "__main__":
    main()
