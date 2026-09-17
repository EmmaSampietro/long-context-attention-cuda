#!/bin/bash
#SBATCH --job-name=attn_dense_fp16
#SBATCH --partition=gpu-turing
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:10:00
#SBATCH --output=slurms/logs/icme_dense_fp16_%A.out
#SBATCH --error=slurms/logs/icme_dense_fp16_%A.err

# Day 2 follow-up: run the existing final_crossover.py harness with the new
# dense_fp16 (WMMA) kernel across the same N values as the FP32 sweep, so
# we have a directly comparable series for the dense-vs-SDPA plot.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurms/logs report/data build

module load course/cme213/nvhpc/24.1

SM_ARCH=${SM_ARCH:-sm_75}
N_LIST=${N_LIST:-"1024,2048,4096,8192,16384,32768"}
H=${H:-16}
JOB_TAG=$(hostname)_${SLURM_JOB_ID}

CSV="report/data/dense_fp16_sweep_${JOB_TAG}.csv"

echo "=== Host: $(hostname)  Date: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "N_LIST=$N_LIST  H=$H"
nvidia-smi --query-gpu=index,name --format=csv,noheader

make all SM_ARCH=$SM_ARCH

# Reuse the existing harness; just restrict to the new kernel.
python3 bench/final_crossover.py \
    --N "$N_LIST" \
    --H $H --d 64 \
    --warmup 5 --iters 10 \
    --kernels dense_fp16 \
    --out "$CSV"

echo ""
echo "CSV written to $CSV"
echo "=== Done ==="
