#!/bin/bash
#SBATCH --job-name=attn_single
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurms/logs/single_%A.out
#SBATCH --error=slurms/logs/single_%A.err

# Single-GPU smoke + dense kernel sweep on MIT Engaging.
# Submit with: sbatch slurms/single_gpu_bench.slurm
# Override SM arch on H100/H200: sbatch --export=ALL,SM_ARCH=sm_90 ...

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurms/logs report/data build

# --- Modules. Cluster naming may drift; fall back gracefully. ---
module load cuda/13.1.0 2>/dev/null || module load cuda/13.0.1 2>/dev/null || \
  module load cuda/12.9.1 2>/dev/null || module load cuda 2>/dev/null || true
module load gcc 2>/dev/null || true

echo "=== Host: $(hostname)  Date: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
nvidia-smi || true
which nvcc && nvcc --version || true

# --- Detect SM arch if not provided ---
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

make clean
make all SM_ARCH=$SM_ARCH

echo "--- smoke test ---"
make test SM_ARCH=$SM_ARCH

echo "--- numerical validation vs PyTorch ---"
python bench/validate.py --kernel dense || true

echo "--- dense sweep ---"
python bench/sweep.py --kernel dense \
    --N 512,1024,2048,4096 --d 64 --H 16 \
    --out report/data/dense_sweep_$(hostname)_$SLURM_JOB_ID.csv
