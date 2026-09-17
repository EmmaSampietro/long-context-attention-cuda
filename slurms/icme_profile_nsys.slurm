#!/bin/bash
#SBATCH --job-name=attn_icme_nsys
#SBATCH --partition=gpu-turing
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=slurms/logs/icme_nsys_%A.out
#SBATCH --error=slurms/logs/icme_nsys_%A.err

# Nsight Systems timeline of one kernel at one configuration on Stanford ICME.
# At H>1 this captures the per-head stream overlap (or lack thereof) — the
# pre-fix vs post-fix comparison figure the lab notes describe.
#
# Examples:
#   sbatch --export=ALL,KERNEL=dense,N=4096,D=64,H=16 slurms/icme_profile_nsys.slurm
#   sbatch --export=ALL,KERNEL=windowed,N=4096,D=64,H=16,W=256,G=8 slurms/icme_profile_nsys.slurm

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurms/logs analysis/profiling/runs build

module load course/cme213/nvhpc/24.1

KERNEL=${KERNEL:-dense}
N=${N:-4096}
D=${D:-64}
H=${H:-16}
W=${W:-256}
G=${G:-0}
SM_ARCH=${SM_ARCH:-sm_75}
COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo dev)

echo "=== Host: $(hostname)  KERNEL=$KERNEL N=$N D=$D H=$H W=$W G=$G  SM_ARCH=$SM_ARCH ==="
nvidia-smi || true
which nsys && nsys --version || { echo "nsys not on PATH"; exit 2; }

[[ -x build/attn ]] || make all SM_ARCH=$SM_ARCH

if [[ "$KERNEL" == "windowed" ]]; then
    TAG="windowed_N${N}_d${D}_H${H}_w${W}_G${G}_${COMMIT}"
    KERNEL_ARGS="--kernel=windowed --N=$N --d=$D --H=$H --w=$W --G=$G"
else
    TAG="${KERNEL}_N${N}_d${D}_H${H}_${COMMIT}"
    KERNEL_ARGS="--kernel=$KERNEL --N=$N --d=$D --H=$H"
fi

OUT="analysis/profiling/runs/${TAG}"
echo "=== Nsight Systems → ${OUT}.nsys-rep ==="
nsys profile -t cuda,nvtx -o "$OUT" --force-overwrite=true \
    build/attn $KERNEL_ARGS --warmup=2 --iters=5
