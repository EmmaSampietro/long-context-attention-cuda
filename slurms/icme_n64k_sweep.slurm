#!/bin/bash
#SBATCH --job-name=attn_n64k
#SBATCH --partition=gpu-turing
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --output=slurms/logs/icme_n64k_%A.out
#SBATCH --error=slurms/logs/icme_n64k_%A.err

# Extend the comparison curve to N=65536 for the kernels where the result
# is informative: dense_fp16 (WMMA), windowed_fp16 (WMMA), and our FP32
# chunked windowed (algorithmic baseline). SDPA-FP16 at N=64K will also be
# re-run via the existing python harness once we have torch available; for
# now we record what we can on the cluster side.
#
# Dense FP32 chunked at N=64K would take ~10s/iter * 10 iters ~= 100s; over
# budget. We already have N=32K dense which is enough for the asymptotics.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurms/logs report/data build

module load course/cme213/nvhpc/24.1

SM_ARCH=${SM_ARCH:-sm_75}
H=${H:-16}
W=${W:-256}
G=${G:-0}
JOB_TAG=$(hostname)_${SLURM_JOB_ID}

echo "=== Host: $(hostname)  Date: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

make all SM_ARCH=$SM_ARCH

# Three runs, three CSVs. All append to existing schemas so the plotters
# pick them up automatically alongside the smaller-N sweeps.
echo ""
echo "=== windowed FP32 chunked at N=65536 ==="
python3 bench/final_crossover.py \
    --N 65536 --H $H --d 64 \
    --w $W --G $G \
    --warmup 3 --iters 10 \
    --kernels windowed \
    --out "report/data/n64k_windowed_${JOB_TAG}.csv"

echo ""
echo "=== windowed FP16 WMMA at N=65536 ==="
python3 bench/final_crossover.py \
    --N 65536 --H $H --d 64 \
    --w $W --G $G \
    --warmup 3 --iters 10 \
    --kernels windowed_fp16 \
    --out "report/data/n64k_windowed_fp16_${JOB_TAG}.csv"

echo ""
echo "=== dense FP16 WMMA at N=65536 ==="
python3 bench/final_crossover.py \
    --N 65536 --H $H --d 64 \
    --warmup 3 --iters 10 \
    --kernels dense_fp16 \
    --out "report/data/n64k_dense_fp16_${JOB_TAG}.csv"

echo ""
echo "=== Done ==="
