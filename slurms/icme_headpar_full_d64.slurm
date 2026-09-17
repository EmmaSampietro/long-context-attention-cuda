#!/bin/bash
#SBATCH --job-name=attn_headpar_d64
#SBATCH --partition=gpu-turing
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=slurms/logs/icme_headpar_full_d64_%A.out
#SBATCH --error=slurms/logs/icme_headpar_full_d64_%A.err

# Head-parallel MPI strong-scaling sweep for the kernels missing from the
# earlier scaling figures: block-sparse (FP32 + FP16) and dense FP16. The
# windowed FP32/FP16 numbers are already in the report (\S8). At d=64.
# N in {4K, 8K, 16K} x ranks in {1, 2, 4} x 3 kernels = 27 runs, ~10 min.
# Emits one CSV summarising ms_per_iter per (kernel, N, ranks).

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurms/logs report/data build

module load course/cme213/nvhpc/24.1

SM_ARCH=${SM_ARCH:-sm_75}
N_LIST=${N_LIST:-"4096 8192 16384"}
H=${H:-16}
JOB_TAG=$(hostname)_${SLURM_JOB_ID}
CSV="report/data/icme_headpar_full_d64_${JOB_TAG}.csv"

echo "=== Host: $(hostname)  Date: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
nvidia-smi --query-gpu=index,name --format=csv,noheader

make mpi-headpar SM_ARCH=$SM_ARCH

echo "ranks,kernel,N,H,d,B_block,rho,ms_per_iter" > "$CSV"

extract() {
    awk '/^mpi=headpar/ {
        for (i = 1; i <= NF; ++i) if ($i ~ /^ms_per_iter=/) {
            split($i, a, "="); print a[2]; exit
        }
    }'
}

D=64
RHO=0.10
B=64

for RANKS in 1 2 4; do
    for KERNEL in blocksparse blocksparse_fp16 dense_fp16; do
        for N in $N_LIST; do
            TAG="K${KERNEL}_N${N}_R${RANKS}"
            OUTFILE="report/data/headpar_${TAG}_${JOB_TAG}.txt"
            echo ""
            echo "------ kernel=$KERNEL  N=$N  ranks=$RANKS ------"
            CMD="mpirun -np $RANKS build/attn_mpi_headpar \
                --kernel=$KERNEL --N=$N --d=$D --H=$H \
                --causal=0 --warmup=3 --iters=10 --seed=42"
            if [[ "$KERNEL" == blocksparse* ]]; then
                CMD="$CMD --rho=$RHO --B=$B"
            fi
            eval "$CMD" | tee "$OUTFILE"
            MS=$(extract < "$OUTFILE")
            echo "$RANKS,$KERNEL,$N,$H,$D,$B,$RHO,$MS" >> "$CSV"
        done
    done
done

echo ""
echo "Summary CSV: $CSV"
echo "=== Done ==="
