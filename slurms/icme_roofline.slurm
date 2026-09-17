#!/bin/bash
#SBATCH --job-name=attn_roofline
#SBATCH --partition=gpu-turing
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=slurms/logs/icme_roofline_%A.out
#SBATCH --error=slurms/logs/icme_roofline_%A.err

# Roofline data collection on Stanford ICME gpu-turing (Quadro RTX 6000).
# For each of dense / windowed / block-sparse at the canonical M3 config
# (N=4096, H=16, d=64, FP32), use ncu to measure:
#   - FP32 FLOPs (fadd + fmul + 2x ffma)
#   - DRAM bytes read+written
#   - Kernel elapsed time
# Then we compute achieved AI = FLOPs / bytes and achieved TFLOPS =
# FLOPs / time, on the host side.
#
# Submit with:    sbatch slurms/icme_roofline.slurm

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurms/logs analysis/profiling/runs report/data build

module load course/cme213/nvhpc/24.1

SM_ARCH=${SM_ARCH:-sm_75}
N=${N:-4096}
H=${H:-16}
W=${W:-256}
G=${G:-8}
RHO=${RHO:-0.1}
JOB_TAG=$(hostname)_${SLURM_JOB_ID}

echo "=== Host: $(hostname)  Date: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
nvidia-smi --query-gpu=index,name --format=csv,noheader
which ncu && ncu --version | head -2

make all SM_ARCH=$SM_ARCH

# Metrics we collect — same set for all three kernels.
METRICS="smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,\
smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,\
smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,\
dram__bytes.sum,\
gpu__time_duration.avg,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"

SUMMARY_OUT="report/data/roofline_${JOB_TAG}.txt"
: > "$SUMMARY_OUT"

run_ncu() {
    local tag="$1"; shift
    echo "" | tee -a "$SUMMARY_OUT"
    echo "================ $tag ================" | tee -a "$SUMMARY_OUT"
    ncu --target-processes all --metrics "$METRICS" --csv "$@" 2>&1 \
        | tee -a "$SUMMARY_OUT"
}

# Single iteration each (warmup=0, iters=1) so we get clean per-launch metrics.
# ncu replays the kernel internally; warmup is unnecessary.
run_ncu "dense" \
    build/attn --kernel=dense --N=$N --d=64 --H=$H \
        --warmup=0 --iters=1

run_ncu "windowed" \
    build/attn --kernel=windowed --N=$N --d=64 --H=$H --w=$W --G=$G \
        --warmup=0 --iters=1

run_ncu "blocksparse" \
    build/attn --kernel=blocksparse --N=$N --d=64 --H=$H --rho=$RHO \
        --warmup=0 --iters=1

echo ""
echo "Summary written to $SUMMARY_OUT"
echo "=== Done ==="
