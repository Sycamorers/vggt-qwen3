#!/usr/bin/env python3
"""Quick test to verify datasets can be loaded by the DataLoader."""

import sys
from pathlib import Path

import yaml
from transformers import AutoTokenizer

from src.dataio.dataset_builder import DatasetConfig, MultiViewJsonlDataset, MultiSourceDataset
from src.dataio.collate_multiview import MultiViewCollator


def test_dataset_loading(config_path: str):
    """Test loading a dataset from a config file."""
    print(f"\n{'='*80}")
    print(f"Testing: {config_path}")
    print(f"{'='*80}\n")
    
    # Load config
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    data_cfg = cfg['data']
    
    # Build tokenizer
    print("📝 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg['model']['name_or_path'],
        use_fast=False,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Try to add <image> token if not present
    img_id = tokenizer.convert_tokens_to_ids("<image>")
    if img_id is None or (tokenizer.unk_token_id is not None and img_id == tokenizer.unk_token_id):
        tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})
    tokenizer.padding_side = "right"
    print(f"   ✅ Tokenizer loaded: {cfg['model']['name_or_path']}")
    
    # Build datasets
    print("\n📂 Loading datasets...")
    datasets = {}
    missing = []
    
    for name, glob_path in data_cfg['datasets'].items():
        print(f"\n   Loading '{name}' from {glob_path}")
        ds_cfg = DatasetConfig(
            path_glob=glob_path,
            num_views=data_cfg['num_views'],
            image_size=data_cfg['image_size'],
            task=name
        )
        try:
            dataset = MultiViewJsonlDataset(ds_cfg)
            datasets[name] = dataset
            print(f"   ✅ Loaded {len(dataset)} samples from '{name}'")
        except FileNotFoundError as e:
            missing.append((name, glob_path))
            print(f"   ❌ Failed to load '{name}': {e}")
    
    if missing:
        print(f"\n⚠️  Skipped {len(missing)} dataset(s) with no files:")
        for name, glob_path in missing:
            print(f"   - {name}: {glob_path}")
    
    if not datasets:
        print("\n❌ No datasets could be loaded!")
        return False
    
    # Build multi-source dataset
    print("\n🔀 Creating multi-source dataset...")
    mix_ratio = {k: v for k, v in data_cfg['mix_ratio'].items() if k in datasets}
    multi_dataset = MultiSourceDataset(datasets, mix_ratio)
    print(f"   ✅ Combined dataset: {len(multi_dataset)} samples")
    print(f"   Mix ratio: {mix_ratio}")
    
    # Test loading a sample
    print("\n🧪 Testing sample loading...")
    try:
        sample = multi_dataset[0]
        print(f"   ✅ Sample loaded successfully:")
        print(f"      - Images: {len(sample['images'])} PIL.Image objects")
        print(f"      - Question: {sample['question'][:80]}...")
        print(f"      - Answer: {sample['answer'][:80]}...")
        print(f"      - Task: {sample['task']}")
        print(f"      - Geom token: {sample['geom_token']}")
    except Exception as e:
        print(f"   ❌ Failed to load sample: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test collator
    print("\n📦 Testing collator...")
    try:
        collator = MultiViewCollator(
            data_cfg['image_size'],
            tokenizer,
            data_cfg['max_length']
        )
        batch = collator([multi_dataset[i] for i in range(min(2, len(multi_dataset)))])
        print(f"   ✅ Batch created successfully:")
        print(f"      - pixel_values shape: {batch['pixel_values'].shape}")
        print(f"      - input_ids shape: {batch['input_ids'].shape}")
        print(f"      - attention_mask shape: {batch['attention_mask'].shape}")
        print(f"      - labels shape: {batch['labels'].shape}")
    except Exception as e:
        print(f"   ❌ Failed to create batch: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\n{'='*80}")
    print("✅ All tests passed!")
    print(f"{'='*80}\n")
    return True


def main():
    """Run tests on all config files."""
    configs = [
        "configs/stage1_3d.yaml",
        # "configs/stage3_arkit.yaml",  # Skip - needs processing first
    ]
    
    print("="*80)
    print("VGGT-Qwen3 Dataset Loading Test")
    print("="*80)
    
    all_passed = True
    for config_path in configs:
        if Path(config_path).exists():
            passed = test_dataset_loading(config_path)
            all_passed = all_passed and passed
        else:
            print(f"\n⚠️  Config not found: {config_path}")
            all_passed = False
    
    if all_passed:
        print("\n🎉 All dataset tests passed!")
        print("\n✅ Your data pipeline is ready for training!")
    else:
        print("\n❌ Some tests failed - please review the output above")
        sys.exit(1)


if __name__ == "__main__":
    main()
