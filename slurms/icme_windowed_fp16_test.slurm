#!/bin/bash
#SBATCH --job-name=attn_wfp16_test
#SBATCH --partition=gpu-turing
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=slurms/logs/icme_windowed_fp16_test_%A.out
#SBATCH --error=slurms/logs/icme_windowed_fp16_test_%A.err

# Day 3: build the new WMMA windowed kernel (B2) and verify it runs.
#   1) Clean rebuild.
#   2) Correctness check at small N: windowed (FP32) vs windowed_fp16 (WMMA)
#      with identical seed; compute max abs + rel l2 error.
#   3) Quick wall-time at N=4096, H=16, w=256 for both kernels.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurms/logs report/data build

module load course/cme213/nvhpc/24.1

SM_ARCH=${SM_ARCH:-sm_75}

echo "=== Host: $(hostname)  Date: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
nvidia-smi --query-gpu=index,name --format=csv,noheader

# 1) Clean rebuild
echo ""
echo "=== make clean && make all ==="
make clean SM_ARCH=$SM_ARCH || true
make all SM_ARCH=$SM_ARCH

# 2) Small-N correctness check
TMP=$(mktemp -d)
trap "rm -rf $TMP" EXIT

echo ""
echo "=== Correctness: windowed (FP32) vs windowed_fp16 (WMMA), N=256, H=1, d=64, w=64 ==="
build/attn --kernel=windowed       --N=256 --H=1 --d=64 --w=64 --G=0 \
    --warmup=0 --iters=1 --seed=42 \
    --out="$TMP/out_fp32.bin"
build/attn --kernel=windowed_fp16  --N=256 --H=1 --d=64 --w=64 --G=0 \
    --warmup=0 --iters=1 --seed=42 \
    --out="$TMP/out_fp16.bin"

python3 - <<EOF
import numpy as np, sys
a = np.fromfile("$TMP/out_fp32.bin", dtype=np.float32)
b = np.fromfile("$TMP/out_fp16.bin", dtype=np.float32)
if a.shape != b.shape:
    sys.exit(f"shape mismatch: fp32 {a.shape} vs fp16 {b.shape}")
diff = a - b
max_abs = float(np.max(np.abs(diff)))
rel_l2  = float(np.linalg.norm(diff) / max(np.linalg.norm(a), 1e-12))
print(f"  fp32 shape={a.shape}  fp16 shape={b.shape}")
print(f"  max_abs={max_abs:.3e}  rel_l2={rel_l2:.3e}")
if not np.isfinite(max_abs) or rel_l2 > 0.05:
    sys.exit(f"windowed_fp16 correctness check FAILED: rel_l2={rel_l2}")
print("  windowed_fp16 correctness OK")
EOF

# 3) Wall-time at N=4096, H=16, w=256
echo ""
echo "=== Wall time: N=4096, H=16, d=64, w=256 ==="
echo "--- windowed (FP32 chunked) ---"
build/attn --kernel=windowed      --N=4096 --H=16 --d=64 --w=256 --G=0 \
    --warmup=5 --iters=10 --seed=42
echo "--- windowed_fp16 (WMMA) ---"
build/attn --kernel=windowed_fp16 --N=4096 --H=16 --d=64 --w=256 --G=0 \
    --warmup=5 --iters=10 --seed=42

echo ""
echo "=== Done ==="
