#!/bin/bash
#SBATCH --job-name=attn_d128_san
#SBATCH --partition=gpu-turing
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=slurms/logs/icme_d128_sanitizer_%A.out
#SBATCH --error=slurms/logs/icme_d128_sanitizer_%A.err

# Run windowed_fp16 at d=128 under compute-sanitizer to get a precise
# memory-access error site. We pin the failing config from prior runs
# (N=256, H=1, d=128, w=64), and ask the sanitizer for line-info.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurms/logs report/data build

module load course/cme213/nvhpc/24.1

SM_ARCH=${SM_ARCH:-sm_75}

echo "=== Host: $(hostname)  Date: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
nvidia-smi --query-gpu=index,name --format=csv,noheader

echo ""
echo "=== Locate compute-sanitizer ==="
which compute-sanitizer || echo "not on PATH"
ls -la $(dirname $(which nvcc))/../bin/ 2>/dev/null | grep -i sanit || true
find / -name compute-sanitizer 2>/dev/null | head -3

echo ""
echo "=== make clean && make all ==="
make clean SM_ARCH=$SM_ARCH || true
make all SM_ARCH=$SM_ARCH

SAN=$(command -v compute-sanitizer || find / -name compute-sanitizer 2>/dev/null | head -1)
if [ -z "$SAN" ]; then
    echo "compute-sanitizer not found; running plain to confirm crash still reproduces"
    SAN=""
fi

echo ""
echo "=== windowed_fp16 d=128 N=256 H=1 under sanitizer ==="
$SAN ${SAN:+--tool=memcheck --show-backtrace=device} \
    build/attn --kernel=windowed_fp16 --N=256 --H=1 --d=128 \
        --w=64 --G=0 --warmup=0 --iters=1 --seed=42 2>&1 | tail -100

echo ""
echo "=== blocksparse_fp16 d=128 N=256 H=1 under sanitizer ==="
$SAN ${SAN:+--tool=memcheck --show-backtrace=device} \
    build/attn --kernel=blocksparse_fp16 --N=256 --H=1 --d=128 \
        --rho=0.25 --B=32 --warmup=0 --iters=1 --seed=42 2>&1 | tail -100

echo ""
echo "=== Done ==="
