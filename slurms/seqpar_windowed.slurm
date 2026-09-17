#!/bin/bash
#SBATCH --job-name=attn_seqpar
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurms/logs/seqpar_%A.out
#SBATCH --error=slurms/logs/seqpar_%A.err

# Sequence-parallel windowed MPI: each rank owns a sequence chunk and exchanges
# K/V halos of width w with its two neighbors via MPI_Isend/Irecv.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurms/logs report/data

module load cuda/12.2 2>/dev/null || true
module load openmpi/4.1.5 2>/dev/null || true
module load gcc/11.3.0 2>/dev/null || true

W=${W:-256}
N=${N:-8192}

echo "=== Host: $(hostname)  Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)  N=$N w=$W ==="
nvidia-smi || true

# TODO: replace with the actual sequence-parallel binary.
echo "[todo] mpirun -np $SLURM_NTASKS build/attn_seqpar --kernel=windowed --N=$N --w=$W --config configs/bench_matrix.yaml --scope seqpar_vs_headpar_windowed"
