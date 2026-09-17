#!/bin/bash
#SBATCH --job-name=attn_d128_full
#SBATCH --partition=gpu-turing
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=slurms/logs/icme_d128_full_%A.out
#SBATCH --error=slurms/logs/icme_d128_full_%A.err

# Phase 4.2 validation: d=128 for all 6 kernels (FP32 dense/windowed/bsp +
# FP16 WMMA dense/windowed/bsp). Each FP16 kernel is validated against its
# FP32 counterpart at the same shape (expect FP16 envelope: rel_l2 ~ 1e-4).

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurms/logs report/data build

module load course/cme213/nvhpc/24.1

SM_ARCH=${SM_ARCH:-sm_75}

echo "=== Host: $(hostname)  Date: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
nvidia-smi --query-gpu=index,name --format=csv,noheader

echo ""
echo "=== make clean && make all ==="
make clean SM_ARCH=$SM_ARCH || true
make all SM_ARCH=$SM_ARCH

TMP=$(mktemp -d)
trap "rm -rf $TMP" EXIT

# Helper: run kernel pair at d=128 and compare outputs.
compare_pair() {
    local TAG="$1"; shift
    local FP32_KERNEL="$1"; shift
    local FP16_KERNEL="$1"; shift
    local EXTRA="$1"; shift

    echo ""
    echo "=== $TAG  d=128, N=256, H=1 ==="
    build/attn --kernel=$FP32_KERNEL --N=256 --H=1 --d=128 $EXTRA \
        --warmup=0 --iters=1 --seed=42 --out="$TMP/${TAG}_fp32.bin"
    build/attn --kernel=$FP16_KERNEL --N=256 --H=1 --d=128 $EXTRA \
        --warmup=0 --iters=1 --seed=42 --out="$TMP/${TAG}_fp16.bin"

    python3 - <<EOF
import numpy as np, sys
a = np.fromfile("$TMP/${TAG}_fp32.bin", dtype=np.float32)
b = np.fromfile("$TMP/${TAG}_fp16.bin", dtype=np.float32)
diff = a - b
m = float(np.max(np.abs(diff)))
r = float(np.linalg.norm(diff)/max(np.linalg.norm(a),1e-12))
print(f"  max_abs={m:.3e}  rel_l2={r:.3e}")
if r > 0.05: sys.exit(f"  $TAG FAILED: rel_l2={r}")
print(f"  $TAG OK (FP16 envelope)")
EOF
}

compare_pair "4.2a-dense"        dense        dense_fp16        ""
compare_pair "4.2b-windowed"     windowed     windowed_fp16     "--w=64 --G=0"
compare_pair "4.2c-blocksparse"  blocksparse  blocksparse_fp16  "--rho=0.25 --B=32"

# Wall-time sweep at N=4096 H=16 d=128 for the 3 FP16 kernels (FP32 numbers
# from the prior d=128 FP32 SLURM are already in the log: dense ~187 ms,
# windowed ~25 ms, block-sparse ~19 ms).
echo ""
echo "=== Wall time at N=4096, H=16, d=128 (FP16 WMMA path) ==="
echo "--- dense_fp16 ---"
build/attn --kernel=dense_fp16        --N=4096 --H=16 --d=128 \
    --warmup=3 --iters=10 --seed=42
echo "--- windowed_fp16 (w=256, G=0) ---"
build/attn --kernel=windowed_fp16     --N=4096 --H=16 --d=128 --w=256 --G=0 \
    --warmup=3 --iters=10 --seed=42
echo "--- blocksparse_fp16 (rho=0.10, B=32) ---"
build/attn --kernel=blocksparse_fp16  --N=4096 --H=16 --d=128 --rho=0.10 --B=32 \
    --warmup=3 --iters=10 --seed=42

echo ""
echo "=== Done ==="
