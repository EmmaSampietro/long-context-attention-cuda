#!/bin/bash
#SBATCH --job-name=attn_d128_fill
#SBATCH --partition=gpu-turing
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --output=slurms/logs/icme_mpi_d128_fill_%A.out
#SBATCH --error=slurms/logs/icme_mpi_d128_fill_%A.err

# Fill in the missing cells of the d=128 strong-scaling matrix:
#   - Head-par block-sparse FP32 + FP16 at N=16384, P in {1,2,4}
#   - Head-par dense FP32 + FP16 at N=8192, P in {1,2,4}
#
# Appends to the same CSV schema as icme_mpi_d128.slurm so the plot picks
# them up automatically.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurms/logs report/data build

module load course/cme213/nvhpc/24.1

SM_ARCH=${SM_ARCH:-sm_75}
H=${H:-16}
JOB_TAG=$(hostname)_${SLURM_JOB_ID}
CSV="report/data/icme_mpi_d128_fill_${JOB_TAG}.csv"

echo "=== Host: $(hostname)  Date: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
nvidia-smi --query-gpu=index,name --format=csv,noheader

make mpi-headpar SM_ARCH=$SM_ARCH

echo "driver,ranks,kernel,N,H,d,extra,ms_per_iter" > "$CSV"

extract() {
    awk '/^mpi=(seqpar|headpar)/ {
        for (i = 1; i <= NF; ++i) if ($i ~ /^ms_per_iter=/) {
            split($i, a, "="); print a[2]; exit
        }
    }'
}

D=128
B=32
RHO=0.10

# --- 1) Block-sparse FP32 + FP16 at d=128, N=16K -------------------------
echo ""
echo "=========================================="
echo "=== Block-sparse @ d=128, N=16384       ==="
echo "=========================================="
for RANKS in 1 2 4; do
    for KERNEL in blocksparse blocksparse_fp16; do
        N=16384
        TAG="headpar_K${KERNEL}_N${N}_R${RANKS}"
        OUTFILE="report/data/d128_${TAG}_${JOB_TAG}.txt"
        echo ""
        echo "------ headpar  kernel=$KERNEL  N=$N  ranks=$RANKS  d=128 ------"
        mpirun -np $RANKS build/attn_mpi_headpar \
            --kernel=$KERNEL --N=$N --d=$D --H=$H --rho=$RHO --B=$B \
            --causal=0 --warmup=2 --iters=5 --seed=42 \
            | tee "$OUTFILE"
        MS=$(extract < "$OUTFILE")
        echo "headpar,$RANKS,$KERNEL,$N,$H,$D,B=$B/rho=$RHO,$MS" >> "$CSV"
    done
done

# --- 2) Dense FP32 + FP16 at d=128, N=8K --------------------------------
# At d=128 the dense kernel takes ~750ms per call at N=8K with H=16, so we
# keep iters small.
echo ""
echo "=========================================="
echo "=== Dense @ d=128, N=8192               ==="
echo "=========================================="
for RANKS in 1 2 4; do
    for KERNEL in dense dense_fp16; do
        N=8192
        TAG="headpar_K${KERNEL}_N${N}_R${RANKS}"
        OUTFILE="report/data/d128_${TAG}_${JOB_TAG}.txt"
        echo ""
        echo "------ headpar  kernel=$KERNEL  N=$N  ranks=$RANKS  d=128 ------"
        mpirun -np $RANKS build/attn_mpi_headpar \
            --kernel=$KERNEL --N=$N --d=$D --H=$H \
            --causal=0 --warmup=2 --iters=3 --seed=42 \
            | tee "$OUTFILE"
        MS=$(extract < "$OUTFILE")
        echo "headpar,$RANKS,$KERNEL,$N,$H,$D,,$MS" >> "$CSV"
    done
done

echo ""
echo "Summary CSV: $CSV"
echo "=== Done ==="
