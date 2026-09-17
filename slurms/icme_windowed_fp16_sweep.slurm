#!/bin/bash
#SBATCH --job-name=attn_wfp16_sweep
#SBATCH --partition=gpu-turing
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:10:00
#SBATCH --output=slurms/logs/icme_windowed_fp16_sweep_%A.out
#SBATCH --error=slurms/logs/icme_windowed_fp16_sweep_%A.err

# Day 3 follow-up: full N-sweep of the new windowed_fp16 (WMMA) kernel,
# matching dense_fp16_sweep_*.csv schema so the existing plot scripts can
# pick it up alongside the FP32 chunked sweep and the SDPA baseline.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurms/logs report/data build

module load course/cme213/nvhpc/24.1

SM_ARCH=${SM_ARCH:-sm_75}
N_LIST=${N_LIST:-"1024,2048,4096,8192,16384,32768"}
H=${H:-16}
W=${W:-256}
G=${G:-0}
JOB_TAG=$(hostname)_${SLURM_JOB_ID}

CSV="report/data/windowed_fp16_sweep_${JOB_TAG}.csv"

echo "=== Host: $(hostname)  Date: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "N_LIST=$N_LIST  H=$H  w=$W"
nvidia-smi --query-gpu=index,name --format=csv,noheader

make all SM_ARCH=$SM_ARCH

python3 bench/final_crossover.py \
    --N "$N_LIST" \
    --H $H --d 64 \
    --w $W --G $G \
    --warmup 5 --iters 10 \
    --kernels windowed_fp16 \
    --out "$CSV"

echo ""
echo "CSV written to $CSV"
echo "=== Done ==="
