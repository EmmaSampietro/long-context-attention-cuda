#!/bin/bash
#SBATCH --job-name=attn_e2e
#SBATCH --partition=gpu-turing
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:25:00
#SBATCH --output=slurms/logs/icme_end_to_end_%A.out
#SBATCH --error=slurms/logs/icme_end_to_end_%A.err

# End-to-end attention LAYER benchmark: our kernels (via ctypes against
# libattn.so) vs SDPA-FP16 (full layer with projections in PyTorch).
# Asks: does the 21x kernel-only lead survive the inclusion of Q/K/V/O
# projection overhead?

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurms/logs report/data build

module load course/cme213/nvhpc/24.1
# We need PyTorch with CUDA; if the project venv has it, activate it.
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

SM_ARCH=${SM_ARCH:-sm_75}

echo "=== Host: $(hostname)  Date: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
nvidia-smi --query-gpu=index,name --format=csv,noheader
echo ""
python3 -c "import torch; print(f'torch={torch.__version__}  cuda={torch.version.cuda}  is_avail={torch.cuda.is_available()}')"
echo ""

echo "=== Build libattn.so ==="
make clean SM_ARCH=$SM_ARCH || true
make attn-lib SM_ARCH=$SM_ARCH

JOB_TAG=$(hostname)_${SLURM_JOB_ID:-$$}

echo ""
echo "=== End-to-end layer benchmark, d=64 (d_model=1024, H=16) ==="
python3 bench/end_to_end.py \
    --H 16 --d 64 --w 256 \
    --N 1024,2048,4096,8192,16384,32768 \
    --kernels windowed_fp16,dense_fp16 \
    --iters 10 --warmup 3 \
    --out "report/data/end_to_end_d64_${JOB_TAG}.csv"

echo ""
echo "=== End-to-end layer benchmark, d=128 (d_model=2048, H=16) ==="
python3 bench/end_to_end.py \
    --H 16 --d 128 --w 256 \
    --N 1024,2048,4096,8192,16384 \
    --kernels windowed_fp16,dense_fp16 \
    --iters 10 --warmup 3 \
    --out "report/data/end_to_end_d128_${JOB_TAG}.csv"

echo ""
echo "=== Done ==="
