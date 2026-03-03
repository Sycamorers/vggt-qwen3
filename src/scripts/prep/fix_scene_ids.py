#!/usr/bin/env python3
"""Fix missing scene_ids in processed ScanQA and SQA3D JSONL files.

This script adds back the scene_id field that was lost during preprocessing.
It creates backup files before modifying.

Usage:
    python scripts/prep/fix_scene_ids.py
"""

import json
from pathlib import Path
import shutil


def fix_scanqa():
    """Add scene_ids back to ScanQA processed data."""
    print("=" * 80)
    print("Fixing ScanQA scene_ids")
    print("=" * 80)
    
    # Load original data
    original_path = Path("data/processed/ScanQA/ScanQA_v1.0_train.json")
    processed_path = Path("data/processed/scanqa/train.jsonl")
    
    if not original_path.exists():
        print(f"❌ Original file not found: {original_path}")
        return False
    
    if not processed_path.exists():
        print(f"❌ Processed file not found: {processed_path}")
        return False
    
    # Load original to create scene_id mapping
    with open(original_path) as f:
        original_data = json.load(f)
    
    # Create mapping: (question, answer) -> scene_id
    # This is imperfect but better than nothing
    qa_to_scene = {}
    for item in original_data:
        key = (item['question'], item['answers'][0])
        qa_to_scene[key] = {
            'scene_id': item['scene_id'],
            'question_id': item['question_id'],
            'object_ids': item.get('object_ids', []),
            'object_names': item.get('object_names', [])
        }
    
    print(f"✓ Loaded {len(original_data)} original samples")
    print(f"✓ Created mapping for {len(qa_to_scene)} Q&A pairs")
    
    # Backup processed file
    backup_path = processed_path.with_suffix('.jsonl.backup')
    shutil.copy2(processed_path, backup_path)
    print(f"✓ Created backup: {backup_path}")
    
    # Read processed data, add scene_ids, write back
    fixed_samples = []
    matched = 0
    unmatched = 0
    
    with open(processed_path, 'r') as f:
        for line in f:
            sample = json.loads(line)
            key = (sample['question'], sample['answer'])
            
            if key in qa_to_scene:
                metadata = qa_to_scene[key]
                sample['scene_id'] = metadata['scene_id']
                sample['question_id'] = metadata['question_id']
                sample['object_ids'] = metadata['object_ids']
                sample['object_names'] = metadata['object_names']
                matched += 1
            else:
                sample['scene_id'] = None
                unmatched += 1
            
            fixed_samples.append(sample)
    
    # Write fixed data
    with open(processed_path, 'w') as f:
        for sample in fixed_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"✓ Fixed {matched} samples with scene_ids")
    if unmatched > 0:
        print(f"⚠️  {unmatched} samples could not be matched (scene_id set to null)")
    
    return True


def fix_sqa3d():
    """Add scene_ids back to SQA3D processed data."""
    print("\n" + "=" * 80)
    print("Fixing SQA3D scene_ids")
    print("=" * 80)
    
    # Load original data
    questions_path = Path("data/processed/SQA3D/sqa_task/balanced/v1_balanced_questions_train_scannetv2.json")
    annotations_path = Path("data/processed/SQA3D/sqa_task/balanced/v1_balanced_sqa_annotations_train_scannetv2.json")
    processed_path = Path("data/processed/sqa3d/train.jsonl")
    
    if not questions_path.exists():
        print(f"❌ Questions file not found: {questions_path}")
        return False
    
    if not annotations_path.exists():
        print(f"❌ Annotations file not found: {annotations_path}")
        return False
    
    if not processed_path.exists():
        print(f"❌ Processed file not found: {processed_path}")
        return False
    
    # Load original data
    with open(questions_path) as f:
        questions_data = json.load(f)
    
    with open(annotations_path) as f:
        annotations_data = json.load(f)
    
    # Create mappings
    question_map = {q['question_id']: q for q in questions_data['questions']}
    answer_map = {a['question_id']: a['answers'][0]['answer'] for a in annotations_data['annotations']}
    
    # Create (question, answer) -> metadata mapping
    qa_to_metadata = {}
    for qid, q_data in question_map.items():
        if qid in answer_map:
            key = (q_data['question'], answer_map[qid])
            qa_to_metadata[key] = {
                'scene_id': q_data['scene_id'],
                'question_id': qid,
                'situation': q_data.get('situation', ''),
            }
    
    print(f"✓ Loaded {len(questions_data['questions'])} questions")
    print(f"✓ Loaded {len(annotations_data['annotations'])} annotations")
    print(f"✓ Created mapping for {len(qa_to_metadata)} Q&A pairs")
    
    # Backup processed file
    backup_path = processed_path.with_suffix('.jsonl.backup')
    shutil.copy2(processed_path, backup_path)
    print(f"✓ Created backup: {backup_path}")
    
    # Read processed data, add scene_ids, write back
    fixed_samples = []
    matched = 0
    unmatched = 0
    
    with open(processed_path, 'r') as f:
        for line in f:
            sample = json.loads(line)
            key = (sample['question'], sample['answer'])
            
            if key in qa_to_metadata:
                metadata = qa_to_metadata[key]
                sample['scene_id'] = metadata['scene_id']
                sample['question_id'] = metadata['question_id']
                sample['situation'] = metadata['situation']
                matched += 1
            else:
                sample['scene_id'] = None
                unmatched += 1
            
            fixed_samples.append(sample)
    
    # Write fixed data
    with open(processed_path, 'w') as f:
        for sample in fixed_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"✓ Fixed {matched} samples with scene_ids")
    if unmatched > 0:
        print(f"⚠️  {unmatched} samples could not be matched (scene_id set to null)")
    
    return True


def verify_fixes():
    """Verify that the fixes were applied correctly."""
    print("\n" + "=" * 80)
    print("Verification")
    print("=" * 80)
    
    for dataset_name, path in [('ScanQA', 'data/processed/scanqa/train.jsonl'),
                                ('SQA3D', 'data/processed/sqa3d/train.jsonl')]:
        print(f"\n{dataset_name}:")
        with open(path) as f:
            sample = json.loads(f.readline())
            print(f"  Sample keys: {list(sample.keys())}")
            print(f"  scene_id: {sample.get('scene_id')}")
            if 'question_id' in sample:
                print(f"  question_id: {sample.get('question_id')}")


def main():
    """Main entry point."""
    print("\n" + "=" * 80)
    print("ScanQA & SQA3D Scene ID Fixer")
    print("=" * 80)
    print("\nThis script will:")
    print("  1. Add scene_id back to processed JSONL files")
    print("  2. Add additional metadata (question_id, object info, etc.)")
    print("  3. Create backup files (.jsonl.backup)")
    print("\n" + "=" * 80)
    
    input("Press Enter to continue (Ctrl+C to cancel)...")
    
    success = True
    success = fix_scanqa() and success
    success = fix_sqa3d() and success
    
    if success:
        verify_fixes()
        print("\n" + "=" * 80)
        print("✅ All fixes completed successfully!")
        print("=" * 80)
        print("\nBackup files created:")
        print("  - data/processed/scanqa/train.jsonl.backup")
        print("  - data/processed/sqa3d/train.jsonl.backup")
        print("\nYou can now run inference with scene deduplication.")
    else:
        print("\n" + "=" * 80)
        print("❌ Some fixes failed. Please check the output above.")
        print("=" * 80)


if __name__ == "__main__":
    main()
