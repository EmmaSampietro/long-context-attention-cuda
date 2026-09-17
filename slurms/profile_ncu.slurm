#!/bin/bash
#SBATCH --job-name=attn_ncu
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurms/logs/ncu_%A.out
#SBATCH --error=slurms/logs/ncu_%A.err

# Nsight Compute profile of the dense kernel.
# Submit with: sbatch --export=ALL,KERNEL=dense,N=2048,D=64,H=16 slurms/profile_ncu.slurm

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurms/logs analysis/profiling/runs build

module load cuda/13.1.0 2>/dev/null || module load cuda/13.0.1 2>/dev/null || \
  module load cuda/12.9.1 2>/dev/null || module load cuda 2>/dev/null || true
module load gcc 2>/dev/null || true

KERNEL=${KERNEL:-dense}
N=${N:-2048}
D=${D:-64}
H=${H:-1}
COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo dev)

if [[ -z "${SM_ARCH:-}" ]]; then
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "")
  case "$GPU_NAME" in
    *H100*|*H200*) SM_ARCH=sm_90 ;;
    *L40*)         SM_ARCH=sm_89 ;;
    *A100*)        SM_ARCH=sm_80 ;;
    *V100*)        SM_ARCH=sm_70 ;;
    *)             SM_ARCH=sm_80 ;;
  esac
fi
echo "=== SM_ARCH=$SM_ARCH ==="

[[ -x build/attn ]] || make all SM_ARCH=$SM_ARCH

OUT="analysis/profiling/runs/${KERNEL}_N${N}_d${D}_H${H}_${COMMIT}"
echo "=== Profiling $KERNEL (N=$N d=$D H=$H) → ${OUT}.ncu-rep ==="
ncu --set full --import-source yes -o "$OUT" \
    build/attn --kernel=$KERNEL --N=$N --d=$D --H=$H --warmup=2 --iters=3
