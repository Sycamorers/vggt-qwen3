#!/bin/bash
#SBATCH --job-name=roomplan          # Job name
#SBATCH --mail-type=ALL              # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=qinruoyao@ufl.edu  # Where to send mail
#SBATCH --partition=hpg-b200             # Partition (b200 = 1 GPU per node)
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --gpus-per-node=8            # GPUs per node
#SBATCH --cpus-per-gpu=4            # CPUs per GPU
#SBATCH --mem=128GB                  # Total memory requested
#SBATCH --time=24:00:00              # Time limit hrs:min:sec
#SBATCH --output=pytorchdist_%j.out  # Standard output and error log

#===============================
# Environment setup
#===============================
module purge
module load conda
module load cuda/12.8.1
conda activate vggt_new

#===============================
# Run script
#===============================
/blue/hmedeiros/qinruoyao/roomplan/vggt-qwen3-roomplan/train_fixed.sh --safe full 8
#===============================
# Diagnostic prints
#===============================
date; pwd; which python
export HOST=$(hostname -s)
NODES=$(scontrol show hostnames | grep -v $HOST | tr '\n' ' ')
echo "Host: $HOST"
echo "Other nodes: $NODES"
