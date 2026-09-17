#!/bin/bash
#SBATCH --job-name=attn_final_crossover
#SBATCH --partition=gpu-turing
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --output=slurms/logs/icme_final_crossover_%A.out
#SBATCH --error=slurms/logs/icme_final_crossover_%A.err

# Final-report three-kernel crossover sweep on Quadro RTX 6000.
# Runs dense / windowed (w=256, G=8) / block-sparse (rho=0.10, B=64) at
# N = 1K..32K, recording the new mem_mb column (peak device footprint
# from cudaMemGetInfo) along with the median/min/max timings.
#
# Submit with:    sbatch slurms/icme_final_crossover.slurm

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurms/logs report/data build

module load course/cme213/nvhpc/24.1

SM_ARCH=${SM_ARCH:-sm_75}
N_LIST=${N_LIST:-"1024,2048,4096,8192,16384,32768"}
H=${H:-16}
W=${W:-256}
G=${G:-8}
RHO=${RHO:-0.10}
B=${B:-64}
JOB_TAG=$(hostname)_${SLURM_JOB_ID}
OUT="report/data/final_crossover_${JOB_TAG}.csv"

echo "=== Host: $(hostname)  Date: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "N_LIST=$N_LIST  H=$H  w=$W  G=$G  rho=$RHO  B=$B"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

make all SM_ARCH=$SM_ARCH

python3 bench/final_crossover.py \
    --N "$N_LIST" \
    --H $H --d 64 \
    --w $W --G $G --rho $RHO --B $B \
    --warmup 5 --iters 10 \
    --out "$OUT"

echo ""
echo "CSV written to $OUT"
echo "=== Done ==="
