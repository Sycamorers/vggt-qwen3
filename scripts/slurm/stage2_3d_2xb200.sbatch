#!/bin/bash
#SBATCH --job-name=roomplan-stage1
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
##SBATCH --account=<your_slurm_account>

set -euo pipefail

module purge
conda activate roomplan

export OMP_NUM_THREADS=8
export NCCL_DEBUG=warn
export NCCL_IB_DISABLE=0
export TORCH_DISTRIBUTED_DEBUG=DETAIL

torchrun --standalone --nproc_per_node=2 \
  -m src.train.train_sft \
  --config configs/stage1_3d.yaml \
  --deepspeed configs/deepspeed_zero3.json \
  --output_dir ckpts/stage1_3d
