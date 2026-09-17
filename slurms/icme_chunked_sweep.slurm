#!/bin/bash
#SBATCH --job-name=attn_chunked
#SBATCH --partition=gpu-turing
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --output=slurms/logs/icme_chunked_%A.out
#SBATCH --error=slurms/logs/icme_chunked_%A.err

# Re-run after switching the inner online-softmax loop from a full-tile
# s[Bc=64] register buffer to a chunked s_chunk[Cn=16] buffer. Goal:
# fewer registers per thread => higher occupancy => faster.
#
# Emits:
#   - report/data/final_crossover_chunked_<tag>.csv  (full crossover sweep)
#   - report/data/roofline_chunked_<tag>.txt         (ncu register count
#                                                    + Roofline metrics)

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

CSV="report/data/final_crossover_chunked_${JOB_TAG}.csv"
NCU="report/data/roofline_chunked_${JOB_TAG}.txt"

NCU_TMP=$HOME/.cache/nsight-compute-${SLURM_JOB_ID}
mkdir -p "$NCU_TMP"
export TMPDIR="$NCU_TMP"

echo "=== Host: $(hostname)  Date: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "N_LIST=$N_LIST  H=$H  w=$W  G=$G  rho=$RHO  B=$B"
nvidia-smi --query-gpu=index,name --format=csv,noheader

# Clean build so utils.cuh template / kernel changes definitely take.
make clean SM_ARCH=$SM_ARCH || true
make all SM_ARCH=$SM_ARCH

# 1) Crossover sweep with chunked kernels.
python3 bench/final_crossover.py \
    --N "$N_LIST" \
    --H $H --d 64 \
    --w $W --G $G --rho $RHO --B $B \
    --warmup 5 --iters 10 \
    --out "$CSV"

# 2) Roofline metrics + the launch__registers_per_thread metric, which is the
#    direct measure of the register pressure we are trying to reduce.
METRICS="smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,\
smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,\
smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,\
dram__bytes.sum,\
gpu__time_duration.avg,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,\
launch__registers_per_thread,\
launch__occupancy_limit_registers"

: > "$NCU"
run_ncu() {
    local tag="$1"; shift
    echo "" | tee -a "$NCU"
    echo "================ $tag ================" | tee -a "$NCU"
    ncu --target-processes all \
        --replay-mode application \
        --cache-control none \
        --metrics "$METRICS" --csv "$@" 2>&1 | tee -a "$NCU"
}

run_ncu "dense" \
    build/attn --kernel=dense --N=4096 --d=64 --H=$H \
        --warmup=0 --iters=1
run_ncu "windowed" \
    build/attn --kernel=windowed --N=4096 --d=64 --H=$H --w=$W --G=$G \
        --warmup=0 --iters=1
run_ncu "blocksparse" \
    build/attn --kernel=blocksparse --N=4096 --d=64 --H=$H --rho=$RHO \
        --warmup=0 --iters=1

rm -rf "$NCU_TMP"

echo ""
echo "Crossover CSV: $CSV"
echo "Roofline TXT:  $NCU"
echo "=== Done ==="
