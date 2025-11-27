#!/bin/bash
#
# VGGT-Qwen3 Training Script (FIXED)
# Includes fixes for NCCL timeout, disk space, and cache issues
#
# Usage:
#   ./train_fixed.sh [mode] [num_gpus]
#
#   mode:      debug | full (default: full)
#   num_gpus:  1-8 (default: 2)
#
# Examples:
#   ./train_fixed.sh                  # Full training with 2 GPUs
#   ./train_fixed.sh debug            # Debug mode with 2 GPUs
#   ./train_fixed.sh full 4           # Full training with 4 GPUs
#   ./train_fixed.sh debug 8          # Debug mode with 8 GPUs
#

set -e

# Configuration
PROJECT_DIR="/blue/hmedeiros/qinruoyao/roomplan/vggt-qwen3-roomplan"
CONFIG_FILE="configs/stage2_3d.yaml"
OUTPUT_DIR="ckpts/stage2_3d"

# Parse arguments
# Support optional --safe flag: usage: ./train_fixed.sh [--safe] [mode] [num_gpus]
SAFE_MODE=0
if [ "${1:-}" = "--safe" ]; then
    SAFE_MODE=1
    shift
fi

MODE="${1:-full}"
NUM_GPUS="${2:-2}"

# Validate NUM_GPUS
if ! [[ "$NUM_GPUS" =~ ^[1-8]$ ]]; then
    echo "❌ Error: NUM_GPUS must be between 1 and 8"
    echo "   Usage: ./train_fixed.sh [mode] [num_gpus]"
    echo "   Example: ./train_fixed.sh full 4"
    exit 1
fi

# Set mode-specific parameters
if [ "$MODE" = "debug" ]; then
    echo "🐛 Running in DEBUG mode (100 steps)"
    MAX_STEPS=100
    OUTPUT_DIR="ckpts/stage2_3d_debug"
else
    echo "🚀 Running FULL training (30,000 steps)"
    MAX_STEPS=30000
fi

if [ "$SAFE_MODE" -eq 1 ]; then
    echo "🔒 SAFE MODE enabled: forcing conservative settings"
    BATCH_PER_GPU=1
    GRAD_ACCUM=4
    DATALOADER_NUM_WORKERS=0
    EFFECTIVE_BATCH=$((BATCH_PER_GPU * GRAD_ACCUM * NUM_GPUS))
    echo "   -> BATCH_PER_GPU=$BATCH_PER_GPU, GRAD_ACCUM=$GRAD_ACCUM, DATALOADER_NUM_WORKERS=$DATALOADER_NUM_WORKERS, EFFECTIVE_BATCH=$EFFECTIVE_BATCH"
fi

# Calculate effective batch size
# Default training sizes (may be auto-adjusted below if GPUs are constrained)
# Allow environment overrides for expert users
BATCH_PER_GPU=${BATCH_PER_GPU:-6}
GRAD_ACCUM=${GRAD_ACCUM:-32}

# Auto-adjust BATCH_PER_GPU based on available GPU memory to avoid OOM/killed
function probe_and_adjust_batch() {
    # Get smallest free memory across the first NUM_GPUS devices (MiB)
    local min_free=999999
        # Determine device indices to probe. Respect CUDA_VISIBLE_DEVICES if set.
        if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
            dev_list=$(echo $CUDA_VISIBLE_DEVICES | tr ',' ' ' | awk '{for(i=1;i<=NF;i++) print $i}')
        else
            # Default to 0..NUM_GPUS-1
            dev_list=$(seq 0 $((NUM_GPUS-1)))
        fi
        for i in $dev_list; do
            free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $i 2>/dev/null | tr -d '\r' | tail -n1)
        if [ -z "$free_mb" ]; then
            free_mb=0
        fi
        if [ "$free_mb" -lt "$min_free" ]; then
            min_free=$free_mb
        fi
    done

    echo "🔎 Minimum free GPU memory across $NUM_GPUS GPUs: ${min_free} MiB"

    # Conservative heuristics: require ~20GB per GPU for BATCH_PER_GPU=6 on H100-like devices
    # Scale down BATCH_PER_GPU if free memory is low
    if [ "$min_free" -lt 25000 ]; then
        echo "⚠️  Low GPU memory detected. Reducing batch per GPU to 1"
        BATCH_PER_GPU=1
    elif [ "$min_free" -lt 40000 ]; then
        echo "⚠️  Moderate GPU memory detected. Reducing batch per GPU to 2"
        BATCH_PER_GPU=2
    elif [ "$min_free" -lt 80000 ]; then
        echo "⚠️  Ample GPU memory detected. Reducing batch per GPU to 4"
        BATCH_PER_GPU=4
    else
        echo "✅ Using default BATCH_PER_GPU=$BATCH_PER_GPU"
    fi
}

probe_and_adjust_batch

EFFECTIVE_BATCH=$((BATCH_PER_GPU * GRAD_ACCUM * NUM_GPUS))

    # Probe host available memory (MiB) and reduce grad accum if system memory is constrained
    HOST_FREE_MEM_MB=$(awk '/MemAvailable/ {printf "%d", $2/1024}' /proc/meminfo 2>/dev/null || echo 0)
    echo "🔎 Host available memory: ${HOST_FREE_MEM_MB} MiB"

    # Early exit if host memory is critically low (<8GB)
    if [ "$HOST_FREE_MEM_MB" -lt 8000 ]; then
        echo "⛔ Host available memory < 8GB. Aborting to avoid repeated OOM kills."
        exit 2
    fi

    if [ "$HOST_FREE_MEM_MB" -lt 16000 ]; then
        echo "⚠️  Low host memory detected (<16GB). Lowering grad_accum to reduce peak host memory usage."
        GRAD_ACCUM=8
    elif [ "$HOST_FREE_MEM_MB" -lt 32000 ]; then
        echo "⚠️  Moderate host memory detected (<32GB). Lowering grad_accum to 16."
        GRAD_ACCUM=16
    else
        echo "✅ Host memory ok; using GRAD_ACCUM=$GRAD_ACCUM"
    fi

    # Reduce data loader workers under low host memory to lower RSS usage
    if [ "$HOST_FREE_MEM_MB" -lt 48000 ]; then
        export DATALOADER_NUM_WORKERS=0
        echo "⚠️  Low host memory: setting DATALOADER_NUM_WORKERS=$DATALOADER_NUM_WORKERS"
    else
        export DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-8}
        echo "✅ DATALOADER_NUM_WORKERS=$DATALOADER_NUM_WORKERS"
    fi

    # If running under Slurm, further constrain defaults when cgroup mem limit is present
    if [ -n "$SLURM_JOB_ID" ]; then
        echo "🔎 Detected Slurm job: $SLURM_JOB_ID"
        CGROUP_MEM_LIMIT_RAW=""
        # Try cgroup v2 memory.max under the job's cgroup
        for path in /sys/fs/cgroup/*/memory.max /sys/fs/cgroup/memory.max; do
            if [ -f "$path" ]; then
                val=$(cat "$path" 2>/dev/null || true)
                # numeric check
                if [[ "$val" =~ ^[0-9]+$ ]]; then
                    CGROUP_MEM_LIMIT_RAW=$val
                    break
                fi
            fi
        done
        # Fallback: try reading memory limit via /proc/self/cgroup mapping
        if [ -z "$CGROUP_MEM_LIMIT_RAW" ]; then
            grp=$(awk -F: '$3 ~ /memory/ {print $3; exit}' /proc/self/cgroup 2>/dev/null || true)
            if [ -n "$grp" ]; then
                path="/sys/fs/cgroup/$grp/memory.max"
                if [ -f "$path" ]; then
                    val=$(cat "$path" 2>/dev/null || true)
                    if [[ "$val" =~ ^[0-9]+$ ]]; then
                        CGROUP_MEM_LIMIT_RAW=$val
                    fi
                fi
            fi
        fi

        if [[ "$CGROUP_MEM_LIMIT_RAW" =~ ^[0-9]+$ ]]; then
            CGROUP_MEM_MB=$((CGROUP_MEM_LIMIT_RAW/1024/1024))
            echo "🔎 Slurm cgroup memory limit: ${CGROUP_MEM_MB} MiB"
            if [ "$CGROUP_MEM_MB" -lt 65536 ]; then
                echo "⚠️  Slurm memory limit is <64GB: enforcing conservative training settings"
                BATCH_PER_GPU=1
                GRAD_ACCUM=4
                DATALOADER_NUM_WORKERS=0
                echo "   -> BATCH_PER_GPU=$BATCH_PER_GPU, GRAD_ACCUM=$GRAD_ACCUM, DATALOADER_NUM_WORKERS=$DATALOADER_NUM_WORKERS"
            fi
        else
            echo "ℹ️  No numeric cgroup memory limit discovered; proceeding with probe-based heuristics"
        fi
    fi

    # Set PYTORCH_ALLOC_CONF which replaces deprecated PYTORCH_CUDA_ALLOC_CONF
    export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-"max_split_size_mb:256"}
    echo "🔧 PYTORCH_ALLOC_CONF set: $PYTORCH_ALLOC_CONF"

    # Recompute effective batch size after adjustments
    EFFECTIVE_BATCH=$((BATCH_PER_GPU * GRAD_ACCUM * NUM_GPUS))
    echo "🔢 Adjusted settings -> BATCH_PER_GPU=$BATCH_PER_GPU, GRAD_ACCUM=$GRAD_ACCUM, EFFECTIVE_BATCH=$EFFECTIVE_BATCH"

    # Log probes for post-mortem
    mkdir -p "$PROJECT_DIR/.cache/logs"
    probe_log="$PROJECT_DIR/.cache/logs/train_probe.log"
    echo "$(date -u) GPU_FREE_MIN_MiB=$min_free HOST_FREE_MiB=$HOST_FREE_MEM_MB BATCH_PER_GPU=$BATCH_PER_GPU GRAD_ACCUM=$GRAD_ACCUM DATALOADER_NUM_WORKERS=$DATALOADER_NUM_WORKERS PYTORCH_ALLOC_CONF=$PYTORCH_ALLOC_CONF" >> "$probe_log"
    echo "Probe log appended to $probe_log"

# Generate accelerate config for this run
ACCELERATE_CONFIG="configs/accelerate_${NUM_GPUS}gpu.yaml"

# Print configuration
echo "================================================================================"
echo "VGGT-Qwen3 Training Configuration (FIXED)"
echo "================================================================================"
echo "Project Directory: $PROJECT_DIR"
echo "Config File:       $CONFIG_FILE"
echo "Output Directory:  $OUTPUT_DIR"
echo "Max Steps:         $MAX_STEPS"
echo "GPUs:              $NUM_GPUS (DeepSpeed ZeRO-3)"
echo "Batch per GPU:     $BATCH_PER_GPU"
echo "Grad Accumulation: $GRAD_ACCUM"
echo "Effective Batch:   $EFFECTIVE_BATCH"
echo "================================================================================"
echo ""

# Change to project directory
cd "$PROJECT_DIR"

# =============================================================================
# CRITICAL FIXES FOR NCCL/CACHE ISSUES
# =============================================================================

echo "🔧 Applying fixes..."

# Fix 1: Relocate ALL caches to project directory (non-NFS storage)
export TRITON_CACHE_DIR="$PROJECT_DIR/.cache/triton"
export TORCH_HOME="$PROJECT_DIR/.cache/torch"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export TRANSFORMERS_CACHE="$PROJECT_DIR/.cache/transformers"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache/datasets"
export PYTHONPYCACHEPREFIX="$PROJECT_DIR/.cache/python"
export PIP_CACHE_DIR="$PROJECT_DIR/.cache/pip"
export TMPDIR="$PROJECT_DIR/.cache/tmp"
export TEMP="$PROJECT_DIR/.cache/tmp"
export TMP="$PROJECT_DIR/.cache/tmp"

# Create temp directory
mkdir -p "$PROJECT_DIR/.cache/tmp"

# Fix 2: NCCL timeout and debugging settings
export NCCL_TIMEOUT=3600                    # Increase timeout to 1 hour
export NCCL_DEBUG=INFO                      # Enable NCCL debugging
export NCCL_DEBUG_SUBSYS=ALL                # Debug all subsystems
export NCCL_IB_DISABLE=0                    # Enable InfiniBand if available
export NCCL_NET_GDR_LEVEL=2                 # GPU Direct RDMA
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1    # Better error reporting
export TORCH_NCCL_TRACE_BUFFER_SIZE=10000   # Enable flight recorder

# Fix 3: DeepSpeed settings
export CUDA_LAUNCH_BLOCKING=0               # Async kernel launches
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"  # Memory allocation

# Fix 4: OMP threads (suppress warning)
export OMP_NUM_THREADS=1

# Fix 5: Prevent hanging on exit
export NCCL_ASYNC_ERROR_HANDLING=1

echo "✅ Environment variables set:"
echo "   TRITON_CACHE_DIR=$TRITON_CACHE_DIR"
echo "   TORCH_HOME=$TORCH_HOME"
echo "   HF_HOME=$HF_HOME"
echo "   TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"
echo "   HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
echo "   PYTHONPYCACHEPREFIX=$PYTHONPYCACHEPREFIX"
echo "   TMPDIR=$TMPDIR"
echo "   NCCL_TIMEOUT=$NCCL_TIMEOUT"
echo "   NCCL_DEBUG=$NCCL_DEBUG"
echo ""

# Verify environment
echo "📋 Verifying environment..."
if ! command -v accelerate &> /dev/null; then
    echo "❌ Error: accelerate not found. Please activate the conda environment:"
    echo "   conda activate vggt_new"
    exit 1
fi

# Check CUDA
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "❌ Error: CUDA not available. Please check your PyTorch installation."
    exit 1
fi
echo "✅ CUDA available"

# Check GPU count
AVAILABLE_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
if [ "$AVAILABLE_GPUS" -lt "$NUM_GPUS" ]; then
    echo "❌ Error: Requested $NUM_GPUS GPUs but only $AVAILABLE_GPUS available"
    exit 1
fi
echo "✅ $AVAILABLE_GPUS GPUs available (using $NUM_GPUS)"

# Verify data
if [ ! -f "data/processed/scanqa/train.jsonl" ] || [ ! -f "data/processed/sqa3d/train.jsonl" ]; then
    echo "❌ Error: Training data not found in data/processed/"
    exit 1
fi
echo "✅ Training data found"

# Create output directory
mkdir -p "$OUTPUT_DIR"
echo "✅ Output directory ready: $OUTPUT_DIR"
echo ""

# Create accelerate config for this GPU count
echo "📝 Creating accelerate config for $NUM_GPUS GPUs..."
cat > "$ACCELERATE_CONFIG" << EOF
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: DEEPSPEED
downcast_bf16: 'no'
enable_cpu_affinity: false
machine_rank: 0
main_training_function: main
num_machines: 1
num_processes: $NUM_GPUS
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
deepspeed_config:
  deepspeed_config_file: configs/deepspeed_zero3.json
EOF
echo "✅ Accelerate config created: $ACCELERATE_CONFIG"
echo ""

# Start training
echo "🎯 Starting training..."
echo "   Checkpoints saved every 1500 steps to: $OUTPUT_DIR/checkpoint-*/"
echo ""
echo "================================================================================"
echo ""

# Run training with better error handling
set +e  # Don't exit on error yet
accelerate launch \
    --config_file "$ACCELERATE_CONFIG" \
    src/train/train_sft.py \
    --config "$CONFIG_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --max_steps "$MAX_STEPS"

EXIT_CODE=$?
set -e

# Training completed or failed
echo ""
echo "================================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully!"
    echo ""
    echo "💾 Final checkpoint: $OUTPUT_DIR/checkpoint-$MAX_STEPS/"
else
    echo "❌ Training failed with exit code $EXIT_CODE"
    echo ""
    echo "🔍 Check the output above for errors"
    echo ""
    echo "💡 Common issues:"
    echo "   - NCCL timeout: Check network/GPU communication"
    echo "   - OOM: Reduce batch_size_per_gpu or grad_accum"
    echo "   - Data loading: Check data files exist"
fi
echo "================================================================================"

exit $EXIT_CODE
