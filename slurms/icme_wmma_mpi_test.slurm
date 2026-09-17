#!/bin/bash
#SBATCH --job-name=attn_wmma_mpi
#SBATCH --partition=gpu-turing
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=slurms/logs/icme_wmma_mpi_%A.out
#SBATCH --error=slurms/logs/icme_wmma_mpi_%A.err

# Day 3: combine the project's two strongest wins: WMMA windowed (B2)
# running across 4 GPUs via the existing seq-par halo-exchange driver.
# Strong scaling at N in {4K, 8K, 16K} for ranks in {1, 2, 4}. Emits
# one CSV summarising ms_per_iter per (kernel, N, ranks).
# Also re-runs the FP32 chunked windowed at the same configs so we can
# compare WMMA vs FP32 within the same MPI driver.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurms/logs report/data build

module load course/cme213/nvhpc/24.1

SM_ARCH=${SM_ARCH:-sm_75}
N_LIST=${N_LIST:-"4096 8192 16384"}
H=${H:-16}
W=${W:-256}
JOB_TAG=$(hostname)_${SLURM_JOB_ID}
CSV="report/data/icme_wmma_mpi_${JOB_TAG}.csv"

echo "=== Host: $(hostname)  Date: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
nvidia-smi --query-gpu=index,name --format=csv,noheader

make mpi-seqpar SM_ARCH=$SM_ARCH

echo "ranks,kernel,N,H,w,ms_per_iter" > "$CSV"

extract() {
    awk '/^mpi=seqpar/ {
        for (i = 1; i <= NF; ++i) if ($i ~ /^ms_per_iter=/) {
            split($i, a, "="); print a[2]; exit
        }
    }'
}

for RANKS in 1 2 4; do
    for KERNEL in windowed windowed_fp16; do
        for N in $N_LIST; do
            TAG="K${KERNEL}_N${N}_R${RANKS}"
            OUTFILE="report/data/wmma_mpi_${TAG}_${JOB_TAG}.txt"
            echo ""
            echo "------ kernel=$KERNEL  N=$N  ranks=$RANKS ------"
            mpirun -np $RANKS build/attn_mpi_seqpar \
                --kernel=$KERNEL --N=$N --d=64 --H=$H --w=$W --G=0 \
                --causal=0 --overlap=0 --warmup=5 --iters=20 \
                | tee "$OUTFILE"
            MS=$(extract < "$OUTFILE")
            echo "$RANKS,$KERNEL,$N,$H,$W,$MS" >> "$CSV"
        done
    done
done

echo ""
echo "Summary: $CSV"
echo "=== Done ==="
