#!/bin/bash
#SBATCH --job-name=attn_d128_xover
#SBATCH --partition=gpu-turing
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurms/logs/icme_d128_crossover_%A.out
#SBATCH --error=slurms/logs/icme_d128_crossover_%A.err

# Single-GPU sweep at d=128 across N ∈ {1K, 2K, 4K, 8K, 16K} for all six
# kernels (FP32 + FP16 WMMA × dense / windowed / block-sparse). Generates
# the d=128 analogue of the existing d=64 crossover figure.
#
# Dense FP32 at d=128 is slow (~187 ms at N=4K, ~3000 ms at N=16K) so we
# cap dense FP32 at N=8K and skip dense at N=16K. Other kernels go the
# full ladder.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurms/logs report/data build

module load course/cme213/nvhpc/24.1

SM_ARCH=${SM_ARCH:-sm_75}
H=${H:-16}
W=${W:-256}
RHO=${RHO:-0.10}
B=32                  # required for d=128 block-sparse (host wrapper enforces)
JOB_TAG=$(hostname)_${SLURM_JOB_ID}

echo "=== Host: $(hostname)  Date: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
nvidia-smi --query-gpu=index,name --format=csv,noheader

echo ""
echo "=== make all ==="
make all SM_ARCH=$SM_ARCH

# We emit one CSV per kernel for easy joining; the plot script merges them.
emit_header() {
    local F="$1"
    if [ ! -f "$F" ]; then
        echo "kernel,N,d,H,w,G,causal,ms_median,ms_min,ms_max,mem_mb" > "$F"
    fi
}

run_and_record() {
    local KERNEL="$1"; shift
    local N="$1"; shift
    local EXTRA="$1"; shift   # extra CLI args
    local CSV="$1"; shift

    local LINE
    LINE=$(build/attn --kernel=$KERNEL --N=$N --H=$H --d=128 $EXTRA \
        --warmup=3 --iters=10 --seed=42 2>&1 | grep '^kernel=' || true)
    echo "$LINE"
    if [ -z "$LINE" ]; then
        echo "  (skip: kernel returned no summary line)"
        return
    fi
    # Parse `key=val key=val ...` into CSV
    python3 -c "
import re, sys
s = '''$LINE'''
kv = dict(re.findall(r'(\w+)=([^\s]+)', s))
def f(k): return kv.get(k, '')
print(','.join([f('kernel'), f('N'), f('d'), f('H'), f('w'), f('G'),
                f('causal'), f('ms_median'), f('ms_min'), f('ms_max'),
                f('mem_mb')]))
" >> "$CSV"
}

# --- Dense FP32 + FP16 ---
CSV_DENSE="report/data/d128_crossover_dense_${JOB_TAG}.csv"
emit_header "$CSV_DENSE"
echo ""
echo "=== Dense d=128 ==="
for N in 1024 2048 4096 8192; do
    run_and_record dense      $N "" "$CSV_DENSE"
done
for N in 1024 2048 4096 8192 16384; do
    run_and_record dense_fp16 $N "" "$CSV_DENSE"
done

# --- Windowed FP32 + FP16 ---
CSV_WIN="report/data/d128_crossover_windowed_${JOB_TAG}.csv"
emit_header "$CSV_WIN"
echo ""
echo "=== Windowed d=128 (w=256) ==="
for KERNEL in windowed windowed_fp16; do
    for N in 1024 2048 4096 8192 16384; do
        run_and_record $KERNEL $N "--w=$W --G=0" "$CSV_WIN"
    done
done

# --- Block-sparse FP32 + FP16 ---
CSV_BSP="report/data/d128_crossover_blocksparse_${JOB_TAG}.csv"
emit_header "$CSV_BSP"
echo ""
echo "=== Block-sparse d=128 (rho=$RHO, B=$B) ==="
for KERNEL in blocksparse blocksparse_fp16; do
    for N in 1024 2048 4096 8192 16384; do
        run_and_record $KERNEL $N "--rho=$RHO --B=$B" "$CSV_BSP"
    done
done

echo ""
echo "Summary CSVs:"
echo "  $CSV_DENSE"
echo "  $CSV_WIN"
echo "  $CSV_BSP"
echo "=== Done ==="
