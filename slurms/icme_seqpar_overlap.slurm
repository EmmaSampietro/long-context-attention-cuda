#!/bin/bash
#SBATCH --job-name=attn_seqpar_overlap
#SBATCH --partition=gpu-turing
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=slurms/logs/icme_seqpar_overlap_%A.out
#SBATCH --error=slurms/logs/icme_seqpar_overlap_%A.err

# Sequence-parallel windowed: compare serial halo exchange vs non-blocking
# overlap of halos with inner-region compute. Same shapes, two runs each.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurms/logs report/data build

module load course/cme213/nvhpc/24.1

RANKS=${RANKS:-4}
N_LIST=${N_LIST:-"4096 8192 16384"}
H=${H:-16}
W=${W:-256}
SM_ARCH=${SM_ARCH:-sm_75}
JOB_TAG=$(hostname)_${SLURM_JOB_ID}
CSV="report/data/icme_seqpar_overlap_${JOB_TAG}.csv"

echo "=== Host: $(hostname)  Date: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "RANKS=$RANKS  N_LIST=$N_LIST  H=$H  w=$W"
nvidia-smi --query-gpu=index,name --format=csv,noheader

make mpi-seqpar SM_ARCH=$SM_ARCH

echo "ranks,N,H,w,overlap,ms_per_iter" > "$CSV"

extract() {
    # Pull the ms_per_iter from a "mpi=seqpar ranks=... ms_per_iter=X" line.
    awk '/^mpi=seqpar/ {
        for (i = 1; i <= NF; ++i) if ($i ~ /^ms_per_iter=/) {
            split($i, a, "="); print a[2]; exit
        }
    }'
}

for N in $N_LIST; do
    for OVERLAP in 0 1; do
        TAG="N${N}_ov${OVERLAP}"
        OUTFILE="report/data/icme_seqpar_${TAG}_${JOB_TAG}.txt"
        echo ""
        echo "------ N=$N overlap=$OVERLAP ------"
        mpirun -np $RANKS build/attn_mpi_seqpar \
            --kernel=windowed --N=$N --d=64 --H=$H --w=$W --G=0 \
            --causal=0 --overlap=$OVERLAP --warmup=5 --iters=20 \
            | tee "$OUTFILE"
        MS=$(extract < "$OUTFILE")
        echo "$RANKS,$N,$H,$W,$OVERLAP,$MS" >> "$CSV"
    done
done

echo ""
echo "Summary CSV written to $CSV"
echo "=== Done ==="
