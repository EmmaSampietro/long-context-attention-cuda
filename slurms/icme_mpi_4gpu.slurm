#!/bin/bash
#SBATCH --job-name=attn_mpi
#SBATCH --partition=gpu-turing
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=slurms/logs/icme_mpi_%A.out
#SBATCH --error=slurms/logs/icme_mpi_%A.err

# Head-parallel MPI run on Stanford ICME gpu-turing partition.
# Defaults to 4 ranks (one per GPU on the node). Override with
#   sbatch --export=ALL,RANKS=2,N=4096,HV=16 slurms/icme_mpi_4gpu.slurm
# Run a sweep over rank counts by submitting once with RANKS=1, 2, 4.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurms/logs report/data build

module load course/cme213/nvhpc/24.1

RANKS=${RANKS:-4}
KERNEL=${KERNEL:-dense}
N=${N:-4096}
D=${D:-64}
HV=${HV:-16}                # heads — must be divisible by RANKS
W=${W:-256}
G=${G:-8}
CAUSAL=${CAUSAL:-0}
SM_ARCH=${SM_ARCH:-sm_75}
JOB_TAG=$(hostname)_${SLURM_JOB_ID}

echo "=== Host: $(hostname)  Date: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "=== RANKS=$RANKS KERNEL=$KERNEL N=$N D=$D H=$HV W=$W G=$G CAUSAL=$CAUSAL ==="
nvidia-smi --query-gpu=index,name --format=csv,noheader

# Build the MPI binary (incremental: only rebuilds if sources changed).
make mpi-headpar SM_ARCH=$SM_ARCH

if [[ "$KERNEL" == "windowed" ]]; then
    ARGS="--kernel=windowed --N=$N --d=$D --H=$HV --w=$W --G=$G --causal=$CAUSAL --warmup=5 --iters=10"
else
    ARGS="--kernel=$KERNEL --N=$N --d=$D --H=$HV --causal=$CAUSAL --warmup=5 --iters=10"
fi

# Map 1 MPI rank to 1 GPU on this node. With --gres=gpu:4 each rank sees all 4
# GPUs initially, then the driver does cudaSetDevice(rank % num_devices).
echo "=== mpirun -np $RANKS ==="
mpirun -np $RANKS build/attn_mpi_headpar $ARGS \
    | tee report/data/icme_mpi_headpar_${KERNEL}_R${RANKS}_${JOB_TAG}.txt

echo "=== Done ==="
