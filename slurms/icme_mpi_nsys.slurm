#!/bin/bash
#SBATCH --job-name=attn_mpi_nsys
#SBATCH --partition=gpu-turing
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=slurms/logs/icme_mpi_nsys_%A.out
#SBATCH --error=slurms/logs/icme_mpi_nsys_%A.err

# Nsight Systems profile of the MPI drivers (head-parallel and sequence-parallel)
# on the ICME gpu-turing partition. We profile rank 0 only — that captures all
# of rank 0's kernel launches AND its MPI events (sends, receives, collectives),
# which is what we need for the comm/compute breakdown in the M4 report.
#
# Submit with:  sbatch slurms/icme_mpi_nsys.slurm

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurms/logs analysis/profiling/runs report/data build

module load course/cme213/nvhpc/24.1

SM_ARCH=${SM_ARCH:-sm_75}
N=${N:-4096}
H=${H:-16}
W=${W:-256}
RANKS=${RANKS:-4}
JOB_TAG=$(hostname)_${SLURM_JOB_ID}

echo "=== Host: $(hostname)  Date: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
nvidia-smi --query-gpu=index,name --format=csv,noheader
which nsys && nsys --version | head -2

make mpi SM_ARCH=$SM_ARCH

# Helper: launch under mpirun, but only invoke nsys on rank 0. Other ranks run
# the binary plain, contributing to the MPI flow but not generating trace files.
RANK_VAR='${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-0}}'
profile_rank0_wrapper() {
    local outpath="$1"; shift
    local script="/tmp/nsys_wrap_$$.sh"
    cat > "$script" <<EOF
#!/bin/bash
if [ "\$RANK_VAR_RESOLVED" -eq 0 ]; then
    exec nsys profile -t cuda,nvtx,mpi --mpi-impl=openmpi \\
        -o "$outpath" --force-overwrite=true \\
        "\$@"
else
    exec "\$@"
fi
EOF
    chmod +x "$script"
    echo "$script"
}

# Inline alternative: use a bash -c with explicit rank check.
run_one() {
    local tag="$1"; shift
    local outpath="analysis/profiling/runs/${tag}_${JOB_TAG}_rank0"
    echo "=== nsys ${tag} → ${outpath}.nsys-rep ==="
    mpirun -np "$RANKS" bash -c '
        rank=${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-0}}
        if [ "$rank" -eq 0 ]; then
            exec nsys profile -t cuda,nvtx,mpi --mpi-impl=openmpi \
                -o "'"$outpath"'" --force-overwrite=true "$@"
        else
            exec "$@"
        fi
        ' bash "$@"
}

# 1) Head-parallel, dense
run_one "m4_headpar_dense_N${N}_R${RANKS}" \
    build/attn_mpi_headpar --kernel=dense --N=$N --d=64 --H=$H \
        --warmup=1 --iters=3

# 2) Head-parallel, windowed
run_one "m4_headpar_windowed_N${N}_R${RANKS}" \
    build/attn_mpi_headpar --kernel=windowed --N=$N --d=64 --H=$H \
        --w=$W --G=8 --warmup=1 --iters=3

# 3) Sequence-parallel, windowed
run_one "m4_seqpar_windowed_N${N}_R${RANKS}" \
    build/attn_mpi_seqpar --kernel=windowed --N=$N --d=64 --H=$H \
        --w=$W --warmup=1 --iters=3

# Extract stats summaries — kernel time, MPI event time, totals.
echo ""
echo "=== Stats summaries (rank 0) ==="
SUMMARY_OUT="report/data/m4_nsys_summary_${JOB_TAG}.txt"
: > "$SUMMARY_OUT"
for TAG in m4_headpar_dense m4_headpar_windowed m4_seqpar_windowed; do
    REP="analysis/profiling/runs/${TAG}_N${N}_R${RANKS}_${JOB_TAG}_rank0.nsys-rep"
    if [[ -f "$REP" ]]; then
        echo ""                                           | tee -a "$SUMMARY_OUT"
        echo "================ ${TAG} ================"  | tee -a "$SUMMARY_OUT"
        echo ""                                           | tee -a "$SUMMARY_OUT"
        echo "--- CUDA kernel summary ---"                | tee -a "$SUMMARY_OUT"
        nsys stats --report cuda_gpu_kern_sum \
            --format table --force-export=true "$REP" 2>&1 \
            | head -40 | tee -a "$SUMMARY_OUT"
        echo ""                                           | tee -a "$SUMMARY_OUT"
        echo "--- MPI event summary ---"                  | tee -a "$SUMMARY_OUT"
        nsys stats --report mpi_event_sum \
            --format table --force-export=true "$REP" 2>&1 \
            | head -40 | tee -a "$SUMMARY_OUT"
    else
        echo "NOT FOUND: $REP" | tee -a "$SUMMARY_OUT"
    fi
done

echo ""
echo "Summary written to $SUMMARY_OUT"
echo "=== Done ==="
