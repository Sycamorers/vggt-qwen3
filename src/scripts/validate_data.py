#!/usr/bin/env python3
"""Validate that all datasets in data/processed can be loaded correctly."""

import json
import sys
from pathlib import Path
from collections import defaultdict

import yaml
from PIL import Image


def check_jsonl_file(jsonl_path: Path):
    """Check a JSONL file for valid format and accessible images."""
    issues = []
    samples = []
    
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    sample = json.loads(line)
                    samples.append(sample)
                    
                    # Check required fields
                    if 'images' not in sample:
                        issues.append(f"Line {line_num}: Missing 'images' field")
                    else:
                        # Check image paths
                        for img_path in sample['images']:
                            full_path = Path(img_path)
                            if not full_path.exists():
                                issues.append(f"Line {line_num}: Image not found: {img_path}")
                    
                    if 'question' not in sample and 'instruction' not in sample:
                        issues.append(f"Line {line_num}: Missing 'question' or 'instruction' field")
                    
                    if 'answer' not in sample and 'action_json' not in sample:
                        issues.append(f"Line {line_num}: Missing 'answer' or 'action_json' field")
                    
                except json.JSONDecodeError as e:
                    issues.append(f"Line {line_num}: Invalid JSON - {e}")
    except Exception as e:
        issues.append(f"Error reading file: {e}")
    
    return samples, issues


def check_dataset_directory(dataset_dir: Path):
    """Check all JSONL files in a dataset directory."""
    print(f"\n{'='*80}")
    print(f"Checking: {dataset_dir.name}")
    print(f"{'='*80}")
    
    jsonl_files = list(dataset_dir.glob("*.jsonl"))
    
    if not jsonl_files:
        print(f"⚠️  No JSONL files found in {dataset_dir}")
        return False
    
    all_valid = True
    total_samples = 0
    
    for jsonl_file in sorted(jsonl_files):
        print(f"\n📄 {jsonl_file.name}")
        samples, issues = check_jsonl_file(jsonl_file)
        total_samples += len(samples)
        
        if issues:
            all_valid = False
            print(f"  ❌ Found {len(issues)} issue(s):")
            for issue in issues[:10]:  # Show first 10 issues
                print(f"     - {issue}")
            if len(issues) > 10:
                print(f"     ... and {len(issues) - 10} more issues")
        else:
            print(f"  ✅ Valid - {len(samples)} samples")
            
            # Show sample structure
            if samples:
                sample = samples[0]
                print(f"     Sample structure:")
                print(f"       - images: {len(sample.get('images', []))} image(s)")
                print(f"       - task: {sample.get('task', 'N/A')}")
                print(f"       - question: {'✓' if 'question' in sample or 'instruction' in sample else '✗'}")
                print(f"       - answer: {'✓' if 'answer' in sample or 'action_json' in sample else '✗'}")
                print(f"       - geom_token: {sample.get('geom_token', 'null')}")
    
    print(f"\n📊 Total samples in {dataset_dir.name}: {total_samples}")
    return all_valid


def check_config_datasets(config_path: Path):
    """Check datasets referenced in a config file."""
    print(f"\n{'='*80}")
    print(f"Checking config: {config_path.name}")
    print(f"{'='*80}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    datasets = config.get('data', {}).get('datasets', {})
    
    for name, glob_pattern in datasets.items():
        print(f"\n🔍 Dataset '{name}': {glob_pattern}")
        matches = list(Path().glob(glob_pattern))
        
        if not matches:
            print(f"   ❌ No files match pattern: {glob_pattern}")
        else:
            print(f"   ✅ Found {len(matches)} file(s)")
            for match in matches:
                print(f"      - {match}")


def main():
    """Main validation routine."""
    print("="*80)
    print("VGGT-Qwen3 Data Validation")
    print("="*80)
    
    processed_dir = Path("data/processed")
    
    if not processed_dir.exists():
        print(f"❌ Directory not found: {processed_dir}")
        sys.exit(1)
    
    # Check lowercase directories (processed JSONL files)
    compatible_datasets = ['scanqa', 'sqa3d']
    incompatible_datasets = ['ARKit', 'DocVQA', 'ScanQA', 'SQA3D']
    
    print("\n" + "="*80)
    print("COMPATIBLE DATASETS (JSONL format)")
    print("="*80)
    
    all_valid = True
    for dataset_name in compatible_datasets:
        dataset_dir = processed_dir / dataset_name
        if dataset_dir.exists():
            valid = check_dataset_directory(dataset_dir)
            all_valid = all_valid and valid
        else:
            print(f"\n⚠️  Dataset directory not found: {dataset_name}")
            all_valid = False
    
    print("\n" + "="*80)
    print("INCOMPATIBLE/RAW DATASETS (excluded from pipeline)")
    print("="*80)
    
    for dataset_name in incompatible_datasets:
        dataset_dir = processed_dir / dataset_name
        if dataset_dir.exists():
            print(f"\n📁 {dataset_name}/")
            if dataset_name == 'ARKit':
                print("   ⚠️  Raw ARKit data - needs processing with synth_roomplan_instructions.py")
                training_dir = dataset_dir / "Training"
                if training_dir.exists():
                    scenes = list(training_dir.iterdir())
                    print(f"   Found {len(scenes)} scenes in Training/")
            elif dataset_name == 'DocVQA':
                json_files = list(dataset_dir.glob("*.json"))
                print(f"   ⚠️  LLaVA conversation format - incompatible with multi-view 3D pipeline")
                print(f"   Found {len(json_files)} JSON file(s)")
            elif dataset_name in ['ScanQA', 'SQA3D']:
                print(f"   ℹ️  Raw data directory - use lowercase version ({dataset_name.lower()}/) instead")
    
    # Check configuration files
    print("\n" + "="*80)
    print("CONFIGURATION FILES")
    print("="*80)
    
    config_files = [
        Path("configs/stage1_sft.yaml"),
        Path("configs/stage1_3d.yaml"),
        Path("configs/stage2_arkit.yaml"),
    ]
    
    for config_file in config_files:
        if config_file.exists():
            check_config_datasets(config_file)
        else:
            print(f"\n⚠️  Config file not found: {config_file}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if all_valid:
        print("✅ All compatible datasets are valid and ready to use!")
        print("\n📝 Recommendations:")
        print("   1. Use 'scanqa' and 'sqa3d' for Stage 2 training (3D QA)")
        print("   2. Process ARKit data for Stage 3 training (RoomPlan)")
        print("   3. Exclude DocVQA, ScanQA/, and SQA3D/ from pipeline")
    else:
        print("❌ Some datasets have issues - please review the output above")
        sys.exit(1)


if __name__ == "__main__":
    main()
