#!/bin/bash
#SBATCH --job-name=attn_wmma_test
#SBATCH --partition=gpu-turing
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=slurms/logs/icme_wmma_test_%A.out
#SBATCH --error=slurms/logs/icme_wmma_test_%A.err

# Day 2: build the new FP16+WMMA dense kernel and verify it runs.
# This SLURM is a "smoke test" only - it:
#   1) Forces a clean rebuild so the new dense_fp16.cu actually links.
#   2) Runs a small N comparison: dense vs dense_fp16 with identical seed,
#      writes both outputs to disk, computes max abs error.
#   3) Runs a quick wall-time at N=4096 for a sense of the FP16 speedup.
# Heavier benchmarking + final crossover sweep can run later once the
# kernel is confirmed correct.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurms/logs report/data build

module load course/cme213/nvhpc/24.1

SM_ARCH=${SM_ARCH:-sm_75}
JOB_TAG=$(hostname)_${SLURM_JOB_ID}

echo "=== Host: $(hostname)  Date: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
nvidia-smi --query-gpu=index,name --format=csv,noheader

# 1) Clean rebuild. Surface any compile errors loudly.
echo ""
echo "=== make clean && make all ==="
make clean SM_ARCH=$SM_ARCH || true
make all SM_ARCH=$SM_ARCH

# 2) Small-N correctness check. Same seed for both runs so the random
#    Q/K/V are identical; diff the output blobs.
TMP=$(mktemp -d)
trap "rm -rf $TMP" EXIT

echo ""
echo "=== Correctness: dense (FP32) vs dense_fp16 (WMMA), N=256, H=1, d=64 ==="
build/attn --kernel=dense       --N=256 --H=1 --d=64 \
    --warmup=0 --iters=1 --seed=42 \
    --out="$TMP/out_fp32.bin"
build/attn --kernel=dense_fp16  --N=256 --H=1 --d=64 \
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
# FP16 mixed-precision attention typically lands at ~1e-3 / ~1e-4 rel.
# Hard fail if it's way off (NaN, all zeros, completely wrong).
if not np.isfinite(max_abs) or rel_l2 > 0.05:
    sys.exit(f"WMMA correctness check FAILED: rel_l2={rel_l2}")
print("  WMMA correctness OK")
EOF

# 3) Quick wall-time at N=4096, H=16 to see what FP16 does for us.
echo ""
echo "=== Wall time: N=4096, H=16, d=64 ==="
echo "--- dense (FP32) ---"
build/attn --kernel=dense      --N=4096 --H=16 --d=64 \
    --warmup=5 --iters=10 --seed=42
echo "--- dense_fp16 (WMMA) ---"
build/attn --kernel=dense_fp16 --N=4096 --H=16 --d=64 \
    --warmup=5 --iters=10 --seed=42

echo ""
echo "=== Done ==="
