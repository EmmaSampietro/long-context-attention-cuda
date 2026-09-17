#!/bin/bash
#SBATCH --job-name=attn_mpi_d128
#SBATCH --partition=gpu-turing
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --output=slurms/logs/icme_mpi_d128_%A.out
#SBATCH --error=slurms/logs/icme_mpi_d128_%A.err

# d=128 MPI strong-scaling sweep. Two drivers:
#   - mpi-seqpar for windowed FP32 + FP16 WMMA (the headline)
#   - mpi-headpar for blocksparse FP32 + FP16 (uses B=32 at d=128)
# At d=128 the dense kernel is slow (~187 ms single-GPU at N=4K), so we cap
# its head-par coverage to N=4K and a single rank to confirm it works.
# Emits one CSV summarising ms_per_iter per (driver, kernel, N, ranks).

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurms/logs report/data build

module load course/cme213/nvhpc/24.1

SM_ARCH=${SM_ARCH:-sm_75}
N_LIST_WIN=${N_LIST_WIN:-"4096 8192 16384"}
N_LIST_BSP=${N_LIST_BSP:-"4096 8192"}
H=${H:-16}
W=${W:-256}
JOB_TAG=$(hostname)_${SLURM_JOB_ID}
CSV="report/data/icme_mpi_d128_${JOB_TAG}.csv"

echo "=== Host: $(hostname)  Date: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
nvidia-smi --query-gpu=index,name --format=csv,noheader

make mpi SM_ARCH=$SM_ARCH

echo "driver,ranks,kernel,N,H,d,extra,ms_per_iter" > "$CSV"

extract() {
    awk '/^mpi=(seqpar|headpar)/ {
        for (i = 1; i <= NF; ++i) if ($i ~ /^ms_per_iter=/) {
            split($i, a, "="); print a[2]; exit
        }
    }'
}

D=128

# --- 1) Seq-par windowed FP32 + FP16 at d=128 ----------------------------
echo ""
echo "=========================================="
echo "=== Seq-par windowed @ d=128            ==="
echo "=========================================="
for RANKS in 1 2 4; do
    for KERNEL in windowed windowed_fp16; do
        for N in $N_LIST_WIN; do
            TAG="seqpar_K${KERNEL}_N${N}_R${RANKS}"
            OUTFILE="report/data/d128_${TAG}_${JOB_TAG}.txt"
            echo ""
            echo "------ seqpar  kernel=$KERNEL  N=$N  ranks=$RANKS  d=128 ------"
            mpirun -np $RANKS build/attn_mpi_seqpar \
                --kernel=$KERNEL --N=$N --d=$D --H=$H --w=$W --G=0 \
                --causal=0 --overlap=0 --warmup=3 --iters=10 --seed=42 \
                | tee "$OUTFILE"
            MS=$(extract < "$OUTFILE")
            echo "seqpar,$RANKS,$KERNEL,$N,$H,$D,w=$W,$MS" >> "$CSV"
        done
    done
done

# --- 2) Head-par block-sparse FP32 + FP16 at d=128 (B=32) ---------------
echo ""
echo "=========================================="
echo "=== Head-par block-sparse @ d=128       ==="
echo "=========================================="
B=32
RHO=0.10
for RANKS in 1 2 4; do
    for KERNEL in blocksparse blocksparse_fp16; do
        for N in $N_LIST_BSP; do
            TAG="headpar_K${KERNEL}_N${N}_R${RANKS}"
            OUTFILE="report/data/d128_${TAG}_${JOB_TAG}.txt"
            echo ""
            echo "------ headpar  kernel=$KERNEL  N=$N  ranks=$RANKS  d=128 ------"
            mpirun -np $RANKS build/attn_mpi_headpar \
                --kernel=$KERNEL --N=$N --d=$D --H=$H --rho=$RHO --B=$B \
                --causal=0 --warmup=3 --iters=10 --seed=42 \
                | tee "$OUTFILE"
            MS=$(extract < "$OUTFILE")
            echo "headpar,$RANKS,$KERNEL,$N,$H,$D,B=$B/rho=$RHO,$MS" >> "$CSV"
        done
    done
done

# --- 3) Head-par dense FP32 + FP16 at d=128 (smoke only, N=4K, single rank) ---
# Dense at d=128 is ~187 ms (FP32) per call - doing a full P x N sweep here
# would dominate the wall time. We confirm functional coverage and report
# the single-rank baseline.
echo ""
echo "=========================================="
echo "=== Head-par dense @ d=128 (smoke)      ==="
echo "=========================================="
for RANKS in 1 4; do
    for KERNEL in dense dense_fp16; do
        N=4096
        TAG="headpar_K${KERNEL}_N${N}_R${RANKS}"
        OUTFILE="report/data/d128_${TAG}_${JOB_TAG}.txt"
        echo ""
        echo "------ headpar  kernel=$KERNEL  N=$N  ranks=$RANKS  d=128 ------"
        mpirun -np $RANKS build/attn_mpi_headpar \
            --kernel=$KERNEL --N=$N --d=$D --H=$H \
            --causal=0 --warmup=2 --iters=5 --seed=42 \
            | tee "$OUTFILE"
        MS=$(extract < "$OUTFILE")
        echo "headpar,$RANKS,$KERNEL,$N,$H,$D,,$MS" >> "$CSV"
    done
done

echo ""
echo "Summary CSV: $CSV"
echo "=== Done ==="
