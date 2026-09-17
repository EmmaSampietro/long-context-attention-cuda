#!/bin/bash
#SBATCH --job-name=attn_nsys
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=slurms/logs/nsys_%A.out
#SBATCH --error=slurms/logs/nsys_%A.err

# Nsight Systems timeline of the dense kernel.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurms/logs analysis/profiling/runs build

module load cuda/13.1.0 2>/dev/null || module load cuda/13.0.1 2>/dev/null || \
  module load cuda/12.9.1 2>/dev/null || module load cuda 2>/dev/null || true
module load gcc 2>/dev/null || true

KERNEL=${KERNEL:-dense}
N=${N:-4096}
D=${D:-64}
H=${H:-16}
COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo dev)

if [[ -z "${SM_ARCH:-}" ]]; then
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "")
  case "$GPU_NAME" in
    *H100*|*H200*) SM_ARCH=sm_90 ;;
    *L40*)         SM_ARCH=sm_89 ;;
    *A100*)        SM_ARCH=sm_80 ;;
    *)             SM_ARCH=sm_80 ;;
  esac
fi

[[ -x build/attn ]] || make all SM_ARCH=$SM_ARCH

OUT="analysis/profiling/runs/${KERNEL}_N${N}_d${D}_H${H}_${COMMIT}"
echo "=== Nsight Systems → ${OUT}.nsys-rep ==="
nsys profile -t cuda,nvtx -o "$OUT" --force-overwrite=true \
    build/attn --kernel=$KERNEL --N=$N --d=$D --H=$H --warmup=2 --iters=5
