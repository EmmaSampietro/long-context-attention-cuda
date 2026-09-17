#!/bin/bash
#SBATCH --job-name=attn_multi
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurms/logs/multi_%A.out
#SBATCH --error=slurms/logs/multi_%A.err

# Head-parallel MPI driver — kernel-agnostic. Pass KERNEL=dense|windowed|blocksparse.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurms/logs report/data

module load cuda/12.2 2>/dev/null || true
module load openmpi/4.1.5 2>/dev/null || true
module load gcc/11.3.0 2>/dev/null || true

KERNEL=${KERNEL:-dense}

echo "=== Host: $(hostname)  Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)  Kernel: $KERNEL ==="
nvidia-smi || true

# TODO: scale P via --ntasks once the head-parallel binary lands.
echo "[todo] mpirun -np $SLURM_NTASKS build/attn_headpar --kernel=$KERNEL --config configs/bench_matrix.yaml --scope strong_scaling_headpar"
