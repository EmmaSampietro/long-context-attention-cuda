#!/bin/bash
#SBATCH --job-name=attn_roofline_N
#SBATCH --partition=gpu-turing
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=slurms/logs/icme_roofline_N_%A.out
#SBATCH --error=slurms/logs/icme_roofline_N_%A.err

# Roofline at multiple sequence lengths on Quadro RTX 6000.
# Same metric set as icme_roofline.slurm, but loops over N to expose how
# achieved AI grows for dense (quadratic compute) vs stays flat for
# windowed and block-sparse.
#
# Section tags in the .txt are "<kernel>_N<N>" so the parser can recover
# both axes. Submit with:    sbatch slurms/icme_roofline_N_sweep.slurm

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurms/logs analysis/profiling/runs report/data build

module load course/cme213/nvhpc/24.1

SM_ARCH=${SM_ARCH:-sm_75}
H=${H:-16}
W=${W:-256}
G=${G:-8}
RHO=${RHO:-0.1}
N_LIST=${N_LIST:-"1024 4096 16384"}
JOB_TAG=$(hostname)_${SLURM_JOB_ID}

echo "=== Host: $(hostname)  Date: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "N_LIST=$N_LIST  H=$H  w=$W  G=$G  rho=$RHO"
nvidia-smi --query-gpu=index,name --format=csv,noheader
which ncu && ncu --version | head -2

make all SM_ARCH=$SM_ARCH

METRICS="smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,\
smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,\
smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,\
dram__bytes.sum,\
gpu__time_duration.avg,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"

SUMMARY_OUT="report/data/roofline_N_sweep_${JOB_TAG}.txt"
: > "$SUMMARY_OUT"

run_ncu() {
    local tag="$1"; shift
    echo "" | tee -a "$SUMMARY_OUT"
    echo "================ $tag ================" | tee -a "$SUMMARY_OUT"
    ncu --target-processes all --metrics "$METRICS" --csv "$@" 2>&1 \
        | tee -a "$SUMMARY_OUT"
}

for N in $N_LIST; do
    echo ""
    echo "########## N=$N ##########" | tee -a "$SUMMARY_OUT"

    run_ncu "dense_N${N}" \
        build/attn --kernel=dense --N=$N --d=64 --H=$H \
            --warmup=0 --iters=1

    run_ncu "windowed_N${N}" \
        build/attn --kernel=windowed --N=$N --d=64 --H=$H --w=$W --G=$G \
            --warmup=0 --iters=1

    run_ncu "blocksparse_N${N}" \
        build/attn --kernel=blocksparse --N=$N --d=64 --H=$H --rho=$RHO \
            --warmup=0 --iters=1
done

echo ""
echo "Summary written to $SUMMARY_OUT"
echo "=== Done ==="
