#!/bin/bash
#SBATCH --job-name=attn_roofline_vec
#SBATCH --partition=gpu-turing
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=slurms/logs/icme_roofline_vec_%A.out
#SBATCH --error=slurms/logs/icme_roofline_vec_%A.err

# Re-run ncu metrics ONLY (the crossover sweep already ran fine). Tries a
# few lock-avoidance options to dodge the /tmp/nsight-compute-lock perm
# problem on a shared node. If ncu still fails, the script captures the
# diagnostic output and exits non-zero so the failure is loud.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurms/logs report/data build

module load course/cme213/nvhpc/24.1

SM_ARCH=${SM_ARCH:-sm_75}
H=${H:-16}
W=${W:-256}
G=${G:-8}
RHO=${RHO:-0.1}
JOB_TAG=$(hostname)_${SLURM_JOB_ID}
NCU=report/data/roofline_vec3_${JOB_TAG}.txt

# Per-job scratch dir under $HOME (we own it).
NCU_TMP=$HOME/.cache/nsight-compute-${SLURM_JOB_ID}
mkdir -p "$NCU_TMP"
export TMPDIR="$NCU_TMP"   # generic temp redirect — some ncu versions honor this

echo "=== Host: $(hostname)  Date: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "TMPDIR=$TMPDIR"
nvidia-smi --query-gpu=index,name --format=csv,noheader

# Diagnose what this ncu version actually supports BEFORE relying on flags.
echo ""
echo "=== ncu version + lock-related flags ==="
ncu --version | head -3 || true
echo "--- lock / replay / temp options found in --help ---"
ncu --help 2>&1 | grep -iE 'lock|temp|replay|cache-control' || echo "(none found)"
echo ""

# Inspect the system-wide lock (read-only).
echo "=== /tmp/nsight-compute-lock state ==="
ls -la /tmp/nsight-compute-lock 2>&1 || echo "(no lock file present)"
echo ""

make all SM_ARCH=$SM_ARCH

METRICS="smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,\
smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,\
smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,\
dram__bytes.sum,\
gpu__time_duration.avg,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"

: > "$NCU"
run_ncu() {
    local tag="$1"; shift
    echo "" | tee -a "$NCU"
    echo "================ $tag ================" | tee -a "$NCU"
    # --replay-mode application re-runs the binary instead of the kernel for
    # each metric group, which on some ncu versions sidesteps the kernel-replay
    # lock. --cache-control none disables ncu's metric cache. Either may be
    # rejected by this version; failures are captured in the output file.
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

# Tidy up the per-job temp dir.
rm -rf "$NCU_TMP"

# Fail loudly if ncu produced only error messages.
if ! grep -q 'gpu__time_duration' "$NCU"; then
    echo ""
    echo "WARNING: $NCU contains no gpu__time_duration rows. ncu likely failed; see above."
    exit 1
fi

echo ""
echo "Roofline TXT: $NCU"
echo "=== Done ==="
