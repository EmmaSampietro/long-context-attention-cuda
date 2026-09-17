#!/bin/bash
#SBATCH --job-name=attn_d128_fp32
#SBATCH --partition=gpu-turing
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=slurms/logs/icme_d128_fp32_smoke_%A.out
#SBATCH --error=slurms/logs/icme_d128_fp32_smoke_%A.err

# Phase 4.1: validate d=128 for the three FP32 kernels by comparing against
# the existing NumPy validation harness, then a quick wall-time check at
# moderate N. If any kernel fails correctness, the test continues to the
# next so we collect all diagnostics in one run.

set -uo pipefail   # NOTE: no -e: a failure in one check should not stop others
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

# Correctness helper: generate the same inputs at d=128 via Python (NumPy
# matches the LCG used by the binary at the same seed), then run the kernel
# and diff the output. We reuse the binary's --in / --out plumbing.

run_check() {
    local TAG="$1"; shift
    local KERNEL="$1"; shift
    local EXTRA="$1"; shift   # e.g. "--w=64 --G=0"  or  "--rho=0.25 --B=32"

    echo ""
    echo "=== $TAG: kernel=$KERNEL d=128 N=256 H=1 ==="
    local OUT_FP32="$TMP/${TAG}_d128.bin"
    build/attn --kernel=$KERNEL --N=256 --H=1 --d=128 $EXTRA \
        --warmup=0 --iters=1 --seed=42 --out="$OUT_FP32"

    # Compare against an independent NumPy reference at d=128
    python3 - <<EOF
import numpy as np, sys, subprocess, os, tempfile

# Reproduce the binary's LCG fill to get identical Q, K, V at d=128.
def lcg_fill(seed, n):
    s = (seed * 6364136223846793005) + 1442695040888963407
    s &= (1<<64)-1
    arr = np.empty(n, dtype=np.float32)
    for i in range(n):
        s = (s * 6364136223846793005 + 1442695040888963407) & ((1<<64)-1)
        arr[i] = float(np.int64((s >> 32)) ) / float(1<<31)
    return arr

N, H, d = 256, 1, 128
Q = lcg_fill(42+1, H*N*d).reshape(H, N, d).astype(np.float32)
K = lcg_fill(42+2, H*N*d).reshape(H, N, d).astype(np.float32)
V = lcg_fill(42+3, H*N*d).reshape(H, N, d).astype(np.float32)

def dense_attn(Q, K, V, causal=False):
    O = np.zeros_like(Q)
    for h in range(Q.shape[0]):
        S = Q[h] @ K[h].T / np.sqrt(d)
        if causal:
            S = np.where(np.arange(N)[None,:] <= np.arange(N)[:,None], S, -np.inf)
        M = S.max(axis=-1, keepdims=True)
        P = np.exp(S - M)
        L = P.sum(axis=-1, keepdims=True)
        L = np.where(L > 0, L, 1)
        O[h] = (P / L) @ V[h]
    return O

# Pick the kernel-appropriate reference
kernel = "$KERNEL"
if "windowed" in kernel:
    w = 64
    def attn(Q, K, V):
        O = np.zeros_like(Q)
        for h in range(Q.shape[0]):
            S = Q[h] @ K[h].T / np.sqrt(d)
            mask = np.abs(np.arange(N)[None,:] - np.arange(N)[:,None]) > w
            S = np.where(mask, -np.inf, S)
            M = S.max(axis=-1, keepdims=True); P = np.exp(S - M)
            L = np.where(P.sum(-1, keepdims=True)>0, P.sum(-1, keepdims=True), 1)
            O[h] = (P / L) @ V[h]
        return O
    ref = attn(Q, K, V)
elif "blocksparse" in kernel:
    # rho=1.0 dense pattern -- we requested --rho=1.0 below for the smoke
    ref = dense_attn(Q, K, V, causal=False)
else:
    ref = dense_attn(Q, K, V, causal=False)

got = np.fromfile("$OUT_FP32", dtype=np.float32).reshape(H, N, d)
diff = ref - got
m = float(np.max(np.abs(diff)))
r = float(np.linalg.norm(diff) / max(np.linalg.norm(ref), 1e-12))
print(f"  max_abs={m:.3e}  rel_l2={r:.3e}")
if r > 1e-3:
    sys.exit(f"  FAILED: rel_l2={r}")
print(f"  $TAG OK")
EOF
}

run_check "4.1a-dense"        dense       ""
run_check "4.1b-windowed"     windowed    "--w=64 --G=0"
run_check "4.1c-blocksparse"  blocksparse "--rho=1.0 --B=32"

echo ""
echo "=== Wall time at N=4096, H=16, d=128 ==="
echo "--- dense d=128 ---"
build/attn --kernel=dense       --N=4096 --H=16 --d=128 \
    --warmup=3 --iters=10 --seed=42
echo "--- windowed d=128 (w=256, G=0) ---"
build/attn --kernel=windowed    --N=4096 --H=16 --d=128 --w=256 --G=0 \
    --warmup=3 --iters=10 --seed=42
echo "--- blocksparse d=128 (rho=0.10, B=32) ---"
build/attn --kernel=blocksparse --N=4096 --H=16 --d=128 --rho=0.10 --B=32 \
    --warmup=3 --iters=10 --seed=42

echo ""
echo "=== Done ==="
