# 🎯 VGGT-Qwen3 RoomPlan Training Guide

**Complete guide for training a multi-view 3D visual question answering model**

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Model Architecture](#model-architecture)
3. [Data](#data)
4. [Training Pipeline](#training-pipeline)
5. [Critical Fixes Applied](#critical-fixes-applied)
6. [How to Run Training](#how-to-run-training)
7. [Troubleshooting](#troubleshooting)

---

## Project Overview

### Goal
Train a vision-language model (VLM) that can:
- **Understand 3D indoor scenes** from multiple camera viewpoints
- **Answer questions** about objects, spatial relationships, and scene understanding
- **Process 8 multi-view images** simultaneously to build coherent 3D spatial reasoning

### Use Case
**3D Visual Question Answering (3D-VQA)** for indoor scenes:
- Input: 8 camera views of a room + a question
- Output: Natural language answer about the 3D scene

**Example:**
```
Question: "What is in the right corner of room by curtains?"
Answer: "brown cabinet with tv sitting in it"
```

### Model Name
**VGGT-Qwen3-VLM** (Vision Geometry Grounding Transformer + Qwen3 Language Model)

---

## Model Architecture

### Components

```
┌─────────────────────────────────────────────────────────────┐
│                    VGGT-Qwen3-VLM                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Multi-view Images (8 views, 448×448)                     │
│           ↓                                                 │
│  ┌──────────────────────────────────┐                     │
│  │  VGGT Vision Encoder (frozen)    │                     │
│  │  - Processes multi-view images   │                     │
│  │  - Outputs: [B, V*tokens, 1024]  │                     │
│  └──────────────────────────────────┘                     │
│           ↓                                                 │
│  ┌──────────────────────────────────┐                     │
│  │  Perceiver Projector (trainable) │                     │
│  │  - Resamples to 128 tokens       │                     │
│  │  - Projects to Qwen3 dim (4096)  │                     │
│  │  - 6 layers, 8 heads             │                     │
│  └──────────────────────────────────┘                     │
│           ↓                                                 │
│  ┌──────────────────────────────────┐                     │
│  │  LayerNorm (stabilization)       │                     │
│  └──────────────────────────────────┘                     │
│           ↓                                                 │
│  ┌──────────────────────────────────┐                     │
│  │  Qwen3-8B LLM (LoRA fine-tuned) │                     │
│  │  - Processes vision + text       │                     │
│  │  - Generates answers             │                     │
│  └──────────────────────────────────┘                     │
│           ↓                                                 │
│  Natural Language Answer                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Detailed Component Specifications

#### 1. **VGGT Vision Encoder** (Frozen)
- **Purpose**: Extract visual features from multi-view images
- **Input**: 8 views × 448×448 RGB images
- **Output**: Visual tokens with 3D spatial understanding
- **Parameters**: ~400M (frozen, not trained)
- **Key Feature**: Captures geometric relationships across views

#### 2. **Perceiver Projector** (Trainable)
- **Purpose**: Bridge vision and language modalities
- **Architecture**:
  - Learnable latent queries: 128 tokens
  - Cross-attention to vision features
  - 6 Perceiver layers
  - 8 attention heads, dim 128 per head
  - FFN dimension: 16384
  - Dropout: 0.1
- **Input**: Variable-length VGGT features
- **Output**: Fixed 128 tokens × 4096 dimensions
- **Parameters**: ~50M (trainable)

#### 3. **LayerNorm** (Added for Stability)
- **Purpose**: Normalize projector outputs to prevent numerical overflow
- **Critical**: Prevents NaN loss in text model

#### 4. **Qwen3-8B Language Model** (LoRA Fine-tuned)
- **Base Model**: Qwen3-8B (8 billion parameters)
- **Training Method**: LoRA (Low-Rank Adaptation)
  - Rank: 16
  - Alpha: 32
  - Dropout: 0.05
  - Target modules: q_proj, k_proj, v_proj, o_proj
- **Trainable Parameters**: ~67M LoRA parameters
- **Frozen Parameters**: Base model (8B)

#### 5. **Geometry Head** (Optional, Trainable)
- **Purpose**: Encode camera parameters (R, t, K) and depth
- **Input**: 37-dim vector (R[9] + t[3] + K[9] + depth_hist[16])
- **Output**: 8 geometry tokens × 4096 dimensions
- **Architecture**: MLP with SiLU activation

### Total Parameters
- **Total**: ~8.5B parameters
- **Trainable**: ~117M parameters (1.4% of total)
  - Perceiver Projector: ~50M
  - LoRA adapters: ~67M
  - Geometry Head: <1M

---

## Data

### Datasets Used

#### 1. **ScanQA** (Primary Dataset)
- **Source**: `data/processed/scanqa/*.jsonl`
- **Format**: Multi-view 3D question answering
- **Samples**: ~41,000 training samples
- **Scene Type**: Indoor 3D scans (ScanNet)
- **Questions**: Object-centric, spatial reasoning

**Sample Entry:**
```json
{
  "images": ["path/to/view1.png", ..., "path/to/view8.png"],
  "geom_token": null,
  "question": "What is in the right corner of room by curtains?",
  "answer": "brown cabinet with tv sitting in it",
  "task": "scanqa"
}
```

#### 2. **SQA3D** (Situational Question Answering)
- **Source**: `data/processed/sqa3d/*.jsonl`
- **Format**: Multi-view 3D question answering
- **Samples**: ~18,000 training samples
- **Scene Type**: Indoor 3D scans
- **Questions**: Situational reasoning, spatial relationships

**Mix Ratio**: 50% ScanQA, 50% SQA3D

### Data Statistics
- **Total Training Samples**: 59,207
- **Number of Views per Sample**: 8
- **Image Resolution**: 448×448 pixels
- **Answer Length**: Typically 5-20 tokens

### Excluded Datasets
The following datasets were in the original config but excluded due to incompatible formats:
- ~~ScanRefer~~ (not in JSONL format)
- ~~Referit3D~~ (not in JSONL format)
- ~~Scan2Cap~~ (not in JSONL format)
- ~~DocVQA~~ (2D images, not multi-view)
- ~~ARKit~~ (not in compatible JSONL format)

---

## Training Pipeline

### Input Processing Flow

```
1. Data Loading
   ├─ MultiViewJsonlDataset reads JSONL files
   ├─ Loads 8 images per sample
   └─ Extracts question & answer

2. Collation (MultiViewCollator)
   ├─ Resize images to 448×448
   ├─ Convert to tensors (stays in float32)
   ├─ Create prompt: "{question}\n<image>\n"
   ├─ Tokenize: prompt_ids + answer_ids
   ├─ Labels: [-100] * len(prompt) + answer_ids
   ├─ Pad to minimum length (200 tokens)
   └─ Output: pixel_values, input_ids, labels, attention_mask

3. Model Forward Pass
   ├─ Vision Encoding (VGGT, frozen, no_grad)
   │  ├─ Input: [B, 8, 3, 448, 448]
   │  └─ Output: [B*8, tokens, 1024]
   │
   ├─ Projection (Perceiver, trainable)
   │  ├─ Resample to 128 tokens per sample
   │  ├─ Project to 4096 dimensions
   │  └─ LayerNorm for stability
   │
   ├─ Text Embedding Preparation
   │  ├─ Convert input_ids to embeddings
   │  ├─ Find <image> token positions
   │  ├─ **SHIFT labels** to make room for 128 visual tokens
   │  ├─ Replace <image> embedding with 128 visual tokens
   │  └─ Set visual token labels to -100 (ignore)
   │
   └─ Language Model Forward (Qwen3-8B + LoRA)
      ├─ Process combined vision+text embeddings
      ├─ Generate logits
      └─ Compute cross-entropy loss on answer tokens only

4. Optimization
   ├─ DeepSpeed ZeRO-3 for memory efficiency
   ├─ Gradient accumulation (32 steps)
   ├─ Mixed precision (bfloat16)
   ├─ Gradient clipping (max_norm=1.0)
   └─ AdamW optimizer with different LRs for projector vs base
```

### Training Configuration

**File**: `configs/stage2_3d.yaml`

```yaml
model:
  name_or_path: Qwen/Qwen3-8B
  vision_backbone: third_party/vggt
  num_vis_tokens: 128
  geom_tokens: 8
  projector: configs/perceiver_small.yaml
  freeze_vision: true

data:
  datasets:
    scanqa: data/processed/scanqa/*.jsonl
    sqa3d: data/processed/sqa3d/*.jsonl
  mix_ratio:
    scanqa: 0.5
    sqa3d: 0.5
  num_views: 8
  image_size: 448
  max_length: 512  # Enough for 128 vis + 8 geom + text
  view_dropout: 0.3

train:
  precision: bf16
  optimizer: adamw
  lr: 5.0e-6           # Base learning rate
  proj_lr: 1.0e-4      # Projector learning rate (higher)
  weight_decay: 0.01
  warmup_ratio: 0.1
  batch_size_per_gpu: 6
  grad_accum: 32       # Effective batch = 6 * 32 * 2 GPUs = 384
  max_steps: 30000
  save_every_steps: 1500
  log_every_steps: 20
  gradient_clip: 1.0

lora:
  enable: true
  rank: 16
  alpha: 32
  dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj]
```

### Distributed Training Setup

**File**: `configs/accelerate_config.yaml`

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: DEEPSPEED
deepspeed_config:
  deepspeed_config_file: configs/deepspeed_zero3.json
  zero3_init_flag: true
num_processes: 2  # 2 GPUs
machine_rank: 0
num_machines: 1
main_process_ip: null
main_process_port: null
main_training_function: main
mixed_precision: bf16
use_cpu: false
```

---

## Critical Fixes Applied

During development, we encountered and fixed **5 critical bugs** that caused NaN loss:

### Fix 1: Gradient Flow Bug ⚡
**Problem**: `@torch.no_grad()` decorator on `encode_images()` disabled gradients for the projector.

**File**: `src/models/vggt_qwen3_vlm.py`

**Before** (BROKEN):
```python
@torch.no_grad()  # ❌ Disabled gradients for projector!
def encode_images(self, images):
    agg = self.vision_model.aggregator(images)
    return self.projector(agg)  # Can't learn!
```

**After** (FIXED):
```python
def encode_images(self, images):
    with torch.no_grad():  # ✅ Only for vision encoder
        agg = self.vision_model.aggregator(images)
    proj = self.projector(agg)  # ✅ Can learn!
    return self.proj_norm(proj)
```

---

### Fix 2: Dtype Mismatch 🔧
**Problem**: Input images were float32, but model weights were bfloat16.

**File**: `src/models/vggt_qwen3_vlm.py`

**Before** (BROKEN):
```python
images = images.float()  # ❌ float32
```

**After** (FIXED):
```python
images = images.to(dtype=self.model_dtype)  # ✅ bfloat16
```

---

### Fix 3: Projector Initialization 🎲
**Problem**: Random latent initialization had too high variance (std=1.0).

**File**: `src/models/projector_perceiver.py`

**Before** (BROKEN):
```python
self.latents = nn.Parameter(torch.randn(...))  # ❌ std=1.0
```

**After** (FIXED):
```python
self.latents = nn.Parameter(torch.randn(...) * 0.02)  # ✅ std=0.02
```

---

### Fix 4: Output Normalization 📊
**Problem**: Projector outputs had large magnitude, causing overflow in text model.

**File**: `src/models/vggt_qwen3_vlm.py`

**Added**:
```python
self.proj_norm = nn.LayerNorm(self.text_model.config.hidden_size)

# In encode_images():
proj = self.projector(agg)
proj = self.proj_norm(proj)  # ✅ Stabilize output
```

---

### Fix 5: Label Shifting (MOST CRITICAL) 🎯
**Problem**: All answer labels were being overwritten to -100 when inserting visual tokens.

**Root Cause**:
1. Prompt format: `"{question}\n<image>\n"` followed by answer
2. Labels: `[-100, ..., -100, <image_pos>, answer_token_1, ..., answer_token_N]`
3. When inserting 128 visual tokens at `<image>` position, they overwrote answer tokens
4. Result: No valid labels → NaN loss

**Files Modified**:
- `src/dataio/collate_multiview.py` - Changed prompt format
- `src/models/vggt_qwen3_vlm.py` - Added label shifting logic

**Before** (BROKEN):
```python
# Prompt
prompt = f"<image>\n{question}\n"

# Forward pass
labels[batch_idx, pos:end] = -100  # ❌ Overwrites all answer labels!
```

**After** (FIXED):
```python
# Prompt - image AFTER question
prompt = f"{question}\n<image>\n"

# Forward pass - shift labels to preserve answer tokens
if end < labels.size(1):
    labels[batch_idx, end:] = labels[batch_idx, pos+1:-span_len+1].clone()
labels[batch_idx, pos:end] = -100  # ✅ Only visual tokens
```

**Why this works**:
- Question tokens: positions 0-N (labels = -100, ignored)
- Image token: position N+1
- Answer tokens: positions N+2 onwards (labels = actual token IDs)
- When inserting 128 visual tokens at N+1:
  - Shift answer tokens from N+2 to N+129
  - Set positions N+1 to N+128 as -100
  - Answer labels preserved at new positions!

---

### Fix 6: Minimum Sequence Length 📏
**Problem**: Collator created sequences too short to hold visual tokens + text.

**File**: `src/dataio/collate_multiview.py`

**Added**:
```python
class MultiViewCollator:
    def __init__(self, ..., num_vis_tokens=128, geom_tokens=8):
        self.min_text_length = num_vis_tokens + geom_tokens + 64
    
    def __call__(self, batch):
        # ... tokenization ...
        max_len = max(max_len, self.min_text_length)  # ✅ Ensure room
```

---

## How to Run Training

### Prerequisites

1. **Environment Setup**:
```bash
conda activate vggt_new
```

2. **Verify PyTorch with CUDA**:
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
# Should print: CUDA available: True
```

3. **Data Verification**:
```bash
ls data/processed/scanqa/*.jsonl
ls data/processed/sqa3d/*.jsonl
# Should show train.jsonl and val.jsonl for each
```

### Training Commands

#### **Single Training Run** (Recommended)

```bash
# Stage 2: Multi-view 3D Visual Question Answering
accelerate launch \
    --config_file configs/accelerate_config.yaml \
    src/train/train_sft.py \
    --config configs/stage2_3d.yaml \
    --output_dir ckpts/stage2_3d \
    --max_steps 30000
```

#### **Debug Training** (Short run to verify setup)

```bash
accelerate launch \
    --config_file configs/accelerate_config.yaml \
    src/train/train_sft.py \
    --config configs/stage2_3d.yaml \
    --output_dir ckpts/stage2_3d_debug \
    --max_steps 100
```

#### **Using SLURM** (For HPC clusters)

```bash
sbatch scripts/slurm/stage2_3d_2xb200.sbatch
```

### Training Progress

**Expected Output**:
```
================================================================================
Starting training - Max steps: 30000
Batch size per GPU: 6
Gradient accumulation: 32
Effective batch size: 384
Total samples: 59207
================================================================================

[Collator DEBUG] Created batch: seq_len=200, valid_labels=10/1200
DEBUG: Labels before visual insertion: 10 valid tokens
DEBUG: Labels after visual insertion: 10 valid tokens (preserved!)
✓ inputs_embeds OK: min=-4.50, max=4.50, mean=0.00
✓ Loss OK: 5.2341

[Step    0/30000] Loss: 5.2341 | LR: 5.00e-07 | ETA: 15.5h
[Step   20/30000] Loss: 4.8123 | LR: 6.67e-06 | ETA: 15.2h
[Step  100/30000] Loss: 4.1234 | LR: 5.00e-06 | ETA: 14.8h
...
[Step 1500/30000] 💾 Saved checkpoint to ckpts/stage2_3d/checkpoint-1500
...
```

**Key Metrics to Watch**:
- ✅ **Loss should start around 5-8** (not NaN!)
- ✅ **Loss should gradually decrease**
- ✅ **Valid labels count > 0** in debug output
- ✅ **No NaN/Inf warnings**

### Training Time Estimates

- **Hardware**: 2× NVIDIA A100 GPUs (80GB each)
- **Total Steps**: 30,000
- **Batch Size**: 384 (6 per GPU × 32 grad accum × 2 GPUs)
- **Expected Duration**: ~15-20 hours

---

## Output Files

### Checkpoints
```
ckpts/stage2_3d/
├── checkpoint-1500/     # Every 1500 steps
│   ├── pytorch_model.bin
│   ├── config.json
│   └── ...
├── checkpoint-3000/
├── ...
└── checkpoint-30000/    # Final model
```

### Logs
```
ckpts/stage2_3d/logs/
└── roomplan/
    └── events.out.tfevents.*  # TensorBoard logs
```

### View Training Curves
```bash
tensorboard --logdir ckpts/stage2_3d/logs
```

---

## Troubleshooting

### Issue 1: NaN Loss
**Symptoms**: Loss shows `nan` immediately

**Diagnosis**:
```bash
# Check debug output
grep "❌" ckpts/stage2_3d/logs/*.log
```

**Solutions**:
- ✅ All fixes already applied in current code
- If still occurs: Check data for corrupted images
- Reduce learning rate further (lr: 1e-6)

---

### Issue 2: CUDA Out of Memory
**Symptoms**: `RuntimeError: CUDA out of memory`

**Solutions**:
```yaml
# In configs/stage2_3d.yaml, reduce:
train:
  batch_size_per_gpu: 4  # Was 6
  grad_accum: 48         # Increase to keep effective batch size
```

---

### Issue 3: Slow Data Loading
**Symptoms**: Long pauses between steps

**Solutions**:
- Use faster storage (move data to local SSD)
- Increase DataLoader workers (not currently configurable)
- Preload images to RAM

---

### Issue 4: Checkpoints Too Large
**Symptoms**: Each checkpoint is >30GB

**Explanation**: DeepSpeed ZeRO-3 saves full model state

**Solutions**:
- Save only every 3000 steps instead of 1500
- Use `save_zero_checkpoint=false` (saves only trainable params)

---

## Validation & Evaluation

### During Training
Evaluation happens every 3000 steps (if validation data exists).

### After Training
```bash
# Evaluate final model on test set
python src/eval/eval_3dqa.py \
    --checkpoint ckpts/stage2_3d/checkpoint-30000 \
    --data data/processed/scanqa/val.jsonl \
    --output results/scanqa_val.json
```

---

## Model Usage (After Training)

### Inference Example
```python
from src.models.vggt_qwen3_vlm import VGGTQwen3VLM, VisionLanguageConfig
from src.models.projector_perceiver import PerceiverConfig
import torch

# Load model
config = VisionLanguageConfig(
    text_model_name="Qwen/Qwen3-8B",
    vision_ckpt_dir="third_party/vggt",
    num_vis_tokens=128,
    geom_tokens=8,
    projector_cfg=PerceiverConfig(
        latent_dim=4096,
        num_latents=128,
        num_heads=8,
        num_layers=6,
        ffn_dim=16384,
        dropout=0.1,
    ),
    freeze_vision=True,
    dtype="bfloat16",
)

model = VGGTQwen3VLM(config)
model.load_state_dict(torch.load("ckpts/stage2_3d/checkpoint-30000/pytorch_model.bin"))
model = model.to("cuda").eval()

# Inference
question = "What is on the table?"
# images: [8, 3, 448, 448] tensor
# ... (prepare images) ...

with torch.no_grad():
    answer = model.generate(images, question)
print(f"Answer: {answer}")
```

---

## Citation

If you use this code or model, please cite:

```bibtex
@software{vggt_qwen3_roomplan,
  title={VGGT-Qwen3 for Multi-View 3D Visual Question Answering},
  author={Your Name},
  year={2025},
  url={https://github.com/Sycamorers/vggt-qwen3-roomplan}
}
```

---

## Additional Resources

- **VGGT Paper**: See `third_party/vggt/README.md`
- **Qwen3 Technical Report**: See `third_party/Qwen3/Qwen3_Technical_Report.pdf`
- **DeepSpeed Documentation**: https://www.deepspeed.ai/
- **Accelerate Documentation**: https://huggingface.co/docs/accelerate

---

## License

See LICENSE file for details.

---

**Last Updated**: November 25, 2025
**Status**: ✅ All critical bugs fixed, ready for training
