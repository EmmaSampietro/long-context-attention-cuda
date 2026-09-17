#!/bin/bash
#SBATCH --job-name=attn_n128k
#SBATCH --partition=gpu-turing
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --output=slurms/logs/icme_n128k_%A.out
#SBATCH --error=slurms/logs/icme_n128k_%A.err

# Extend the comparison curve to N=131072 (=128K) for the most informative
# kernels. Memory-wise, four FP32 tensors at this N occupy 4 * (128K * 16 *
# 64 * 4) = ~2.1 GB total; fits comfortably on the 23 GB RTX 6000.
#
# Runs:
#   - windowed FP32 chunked  (predicted ~150 ms, O(N) extension of 74.9 ms@N=64K)
#   - windowed FP16 WMMA     (predicted ~15-20 ms, sub-linear from 9.59 ms@N=64K)
#   - dense FP16             (predicted ~8 s, O(N^2) extension of 2034 ms@N=64K)
#   - SDPA-FP16 baseline     (uses torch+FA-2 from cluster .venv)
#
# Saves into n128k_*.csv files in the same schema as n64k_*.csv so the
# existing plotters pick them up automatically.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurms/logs report/data build

module load course/cme213/nvhpc/24.1
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

SM_ARCH=${SM_ARCH:-sm_75}
H=${H:-16}
W=${W:-256}
G=${G:-0}
JOB_TAG=$(hostname)_${SLURM_JOB_ID}

echo "=== Host: $(hostname)  Date: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

echo ""
echo "=== make all ==="
make all SM_ARCH=$SM_ARCH

echo ""
echo "=== windowed FP32 chunked at N=131072 ==="
python3 bench/final_crossover.py \
    --N 131072 --H $H --d 64 \
    --w $W --G $G \
    --warmup 3 --iters 10 \
    --kernels windowed \
    --out "report/data/n128k_windowed_${JOB_TAG}.csv"

echo ""
echo "=== windowed FP16 WMMA at N=131072 ==="
python3 bench/final_crossover.py \
    --N 131072 --H $H --d 64 \
    --w $W --G $G \
    --warmup 3 --iters 10 \
    --kernels windowed_fp16 \
    --out "report/data/n128k_windowed_fp16_${JOB_TAG}.csv"

echo ""
echo "=== dense FP16 WMMA at N=131072 (slow; iters=3) ==="
python3 bench/final_crossover.py \
    --N 131072 --H $H --d 64 \
    --warmup 2 --iters 3 \
    --kernels dense_fp16 \
    --out "report/data/n128k_dense_fp16_${JOB_TAG}.csv"

echo ""
echo "=== SDPA-FP16 baseline at N=131072 (torch / FA-2) ==="
# SDPA-FP16's tensors at N=131K are ~1 GB each (4 GB total) plus FA-2's small
# on-chip workspace; well within the 23 GB device.
python3 bench/sdpa_baseline.py \
    --N 131072 --H $H --d 64 \
    --dtypes fp16 \
    --warmup 2 --iters 3 \
    --out "report/data/n128k_sdpa_fp16_${JOB_TAG}.csv"

echo ""
echo "=== Done ==="
