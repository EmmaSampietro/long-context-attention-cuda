#!/bin/bash
# Convenience wrapper: load CUDA module, detect GPU arch, build everything.
# Idempotent — safe to re-run.
set -euo pipefail

cd "$(dirname "$0")/.."

module load cuda/13.1.0 2>/dev/null || module load cuda/13.0.1 2>/dev/null || \
  module load cuda/12.9.1 2>/dev/null || module load cuda 2>/dev/null || true
module load gcc 2>/dev/null || true

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

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'none')"
echo "SM_ARCH=$SM_ARCH"
echo "nvcc: $(which nvcc) -> $(nvcc --version 2>/dev/null | tail -1 || echo 'not found')"

make all SM_ARCH="$SM_ARCH" "$@"
